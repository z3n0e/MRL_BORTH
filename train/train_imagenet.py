'''
ResNet training entry point modified for MRL.
'''
import sys 
sys.path.append("../") # adding root folder to the path

import os
import torch as ch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

from torchvision import datasets, models
from torchvision.transforms import v2
import torchmetrics
import numpy as np
from tqdm import tqdm

import random
import time
import json
from uuid import uuid4
from pathlib import Path
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from MRL import *

def seed_worker(worker_id):
    worker_seed = ch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


Section('model', 'model details').params(
    arch=Param(And(str, OneOf(models.__dir__())), default='resnet18'),
    pretrained=Param(int, 'is pretrained? (1/0)', default=0),
    efficient=Param(int, "MRL-E?", default=0),
    mrl=Param(int, "MRL?", default=0),
    nesting_start=Param(int, '2**i will be starting dimension for nesting', default=3),
    fixed_feature=Param(int, 'In case we want to do the fixed feature training, by default it is 2048', default=2048),
    bor_mrl=Param(int, 'Use recursive-prefix Block-Orthogonal Residual MRL? (1/0)', default=0),
    bor_block_mrl=Param(int, 'Use independent-block Block-Orthogonal Residual MRL? (1/0)', default=0),
    cascade_stop_gradient_mrl=Param(int, 'Use recursive-prefix stop-gradient MRL without rotations? (1/0)', default=0),
    bor_mode=Param(And(str, OneOf(['orthogonal', 'frozen'])), 'BOR block transform mode', default='orthogonal'),
    bor_orthogonal_map=Param(And(str, OneOf(['matrix_exp', 'cayley', 'householder'])), 'BOR orthogonal parametrization map', default='matrix_exp'),
    bor_use_trivialization=Param(int, 'Use dynamic trivialization for BOR orthogonal maps? (1/0)', default=1),
    bor_stop_gradient=Param(And(int, OneOf([-1, 0, 1])), 'Stop gradients before BOR orthogonal maps? (0 off, 1 on, -1 class default)', default=0),
    bor_residual_orthogonal=Param(int, 'Use gated residual orthogonal adapter for recursive BOR prefixes? (1/0)', default=0),
    bor_residual_alpha_init=Param(float, 'Initial logit for gated residual BOR alpha', default=-3.0),
    cascade_stop_gradient=Param(And(int, OneOf([-1, 0, 1])), 'Stop gradients between cascade prefixes? (0 off, 1 on, -1 class default)', default=-1)
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=160),
    max_res=Param(int, 'the maximum (starting) resolution', default=160),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=0),
    start_ramp=Param(int, 'when to start interpolating resolution', default=0)
)

Section('data', 'data related stuff').params(
    dataset=Param(And(str, OneOf(['imagenet', 'cifar100'])), 'Dataset to train on', default='imagenet'),
    root=Param(str, 'Dataset root directory', default=''),
    num_workers=Param(int, 'The number of dataloader workers', default=8),
    pin_memory=Param(int, 'Pin dataloader memory? (1/0)', default=1),
    prefetch_factor=Param(int, 'Batches prefetched by each worker', default=4)
)

Section('lr', 'lr scheduling').params(
    step_ratio=Param(float, 'learning rate step ratio', default=0.1),
    step_length=Param(int, 'learning rate step length', default=30),
    lr_schedule_type=Param(OneOf(['step', 'cyclic', 'constant']), default='cyclic'),
    lr=Param(float, 'learning rate', default=0.5),
    lr_peak_epoch=Param(int, 'Epoch at which LR peaks', default=2),
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', required=True),
    run_name=Param(str, 'Optional run folder name inside logging.folder', default=''),
    log_level=Param(int, '0 if only at end 1 otherwise', default=1)
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=512),
    resolution=Param(int, 'final resized validation image size', default=224),
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=1)
)

Section('training', 'training hyper param stuff').params(
    eval_only=Param(int, 'eval only?', default=0),
    path=Param(str, 'weight path for trained model', default=None),
    batch_size=Param(int, 'The batch size', default=512),
    optimizer=Param(And(str, OneOf(['sgd'])), 'The optimizer', default='sgd'),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=4e-5),
    epochs=Param(int, 'number of epochs', default=30),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.1),
    use_blurpool=Param(int, 'use blurpool?', default=0),
    seed=Param(int, 'random seed for reproducible training', default=0),
    deterministic=Param(int, 'enable deterministic PyTorch/CUDA behavior? (1/0)', default=0)
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
DATASET_CONFIGS = {
    'imagenet': {
        'num_classes': 1000,
        'mean': IMAGENET_MEAN,
        'std': IMAGENET_STD
    },
    'cifar100': {
        'num_classes': 100,
        'mean': CIFAR100_MEAN,
        'std': CIFAR100_STD
    }
}

def set_reproducibility(seed, deterministic):
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    random.seed(seed)
    np.random.seed(seed)
    ch.manual_seed(seed)
    ch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        ch.backends.cudnn.benchmark = False
        ch.backends.cudnn.deterministic = True
        if hasattr(ch.backends, 'cuda') and hasattr(ch.backends.cuda, 'matmul'):
            ch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(ch.backends.cudnn, 'allow_tf32'):
            ch.backends.cudnn.allow_tf32 = False
        if hasattr(ch, 'use_deterministic_algorithms'):
            ch.use_deterministic_algorithms(True)
    else:
        ch.backends.cudnn.benchmark = True

@param('lr.lr')
@param('lr.step_ratio')
@param('lr.step_length')
@param('training.epochs')
def get_step_lr(epoch, lr, step_ratio, step_length, epochs):
    if epoch >= epochs:
        return 0

    num_steps = epoch // step_length
    return step_ratio**num_steps * lr

@param('lr.lr')
def get_constant_lr(epoch, lr):
    return lr

@param('lr.lr')
@param('training.epochs')
@param('lr.lr_peak_epoch')
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]

class BlurPoolConv2d(ch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = ch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)

class ImageNetTrainer:
    @param('model.efficient')
    @param('model.mrl')
    @param('model.bor_mrl')
    @param('model.bor_block_mrl')
    @param('model.cascade_stop_gradient_mrl')
    @param('model.nesting_start')
    @param('model.fixed_feature')
    @param('data.dataset')
    @param('training.seed')
    @param('training.deterministic')
    def __init__(self, efficient, mrl, bor_mrl, bor_block_mrl,
                 cascade_stop_gradient_mrl,
                 nesting_start, fixed_feature,
                 dataset, seed, deterministic):
        self.all_params = get_current_config(); 
        self.seed = seed
        self.deterministic = deterministic
        set_reproducibility(seed, deterministic)
        self.device = ch.device('cuda' if ch.cuda.is_available() else 'cpu')
        self.num_gpus = ch.cuda.device_count() if self.device.type == 'cuda' else 0
        self.efficient = efficient
        self.bor_mrl = bool(bor_mrl)
        self.bor_block_mrl = bool(bor_block_mrl)
        self.cascade_stop_gradient_mrl = bool(cascade_stop_gradient_mrl)
        exclusive_mrl_variants = [
            self.bor_mrl,
            self.bor_block_mrl,
            self.cascade_stop_gradient_mrl,
        ]
        if sum(exclusive_mrl_variants) > 1:
            raise ValueError(
                "Choose only one custom MRL method: --model.bor_mrl=1, "
                "--model.bor_block_mrl=1, or --model.cascade_stop_gradient_mrl=1."
            )
        if any(exclusive_mrl_variants) and mrl:
            raise ValueError("Custom MRL variants are their own MRL methods; do not combine them with --model.mrl=1.")
        if any(exclusive_mrl_variants) and self.efficient:
            raise ValueError("Custom MRL variants use one classifier per prefix; do not combine them with --model.efficient=1.")
        self.nesting = (self.efficient or mrl or any(exclusive_mrl_variants))
        self.nesting_start = nesting_start
        self.nesting_list = [2**i for i in range(self.nesting_start, 12)] if self.nesting else None
        self.fixed_feature=fixed_feature
        self.dataset = dataset
        self.dataset_config = DATASET_CONFIGS[dataset]
        self.num_classes = self.dataset_config['num_classes']
        self.uid = str(uuid4())

        self.train_loader = self.create_train_loader()
        self.val_loader = self.create_val_loader()
        self.model, self.scaler = self.create_model_and_scaler()
        self.create_optimizer()
        self.initialize_logger()
        

    @param('lr.lr_schedule_type')
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {
            'cyclic': get_cyclic_lr,
            'step': get_step_lr,
            'constant': get_constant_lr
        }

        return lr_schedules[lr_schedule_type](epoch)

    # resolution tools
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    @param('resolution.start_ramp')
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing):
        assert optimizer == 'sgd'

        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if ('bn' in k)]
        other_params = [v for k, v in all_params if not ('bn' in k)]
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        }, {
            'params': other_params,
            'weight_decay': weight_decay
        }]

        self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        # Adding Nesting Case....
        if self.nesting:
            self.loss = Matryoshka_CE_Loss(label_smoothing=label_smoothing)
        else:   
            self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)


    def _dataset_root(self, root):
        return Path(root).expanduser()

    def _make_loader(self, dataset, batch_size, num_workers, pin_memory,
                     prefetch_factor, is_train, seed):
        generator = ch.Generator()
        generator.manual_seed(seed)

        kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': is_train,
            'num_workers': num_workers,
            'pin_memory': bool(pin_memory),
            'persistent_workers': num_workers > 0,
            'drop_last': is_train,
            'worker_init_fn': seed_worker,
            'generator': generator
        }
        if num_workers > 0:
            kwargs['prefetch_factor'] = prefetch_factor

        return DataLoader(**kwargs)

    def _cifar100_dataset(self, root, train, transform):
        root = self._dataset_root(root)
        dataset = datasets.CIFAR100(root=str(root), train=train,
                                    download=True,
                                    transform=transform)
        return dataset

    def _imagenet_dataset(self, root, split, transform):
        split_root = self._dataset_root(root) / split
        if not split_root.is_dir():
            raise FileNotFoundError(
                f'Expected ImageNet {split} directory at {split_root}. '
                'Use a root with train/ and val/ subdirectories.'
            )
        return datasets.ImageFolder(split_root, transform=transform)

    @param('data.root')
    @param('data.num_workers')
    @param('data.pin_memory')
    @param('data.prefetch_factor')
    @param('training.batch_size')
    @param('training.seed')
    def create_train_loader(self, root, num_workers, pin_memory, prefetch_factor,
                            batch_size, seed):
        res = self.get_resolution(epoch=0)
        self.train_resolution = res

        if self.dataset == 'cifar100':
            transform = v2.Compose([
                v2.RandomCrop(32, padding=4),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(ch.float32, scale=True),
                v2.Normalize(CIFAR100_MEAN, CIFAR100_STD)
            ])
            dataset = self._cifar100_dataset(root, train=True, transform=transform)
        else:
            transform = v2.Compose([
                v2.RandomResizedCrop(res),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(ch.float32, scale=True),
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])
            dataset = self._imagenet_dataset(root, 'train', transform)

        return self._make_loader(dataset, batch_size, num_workers, pin_memory,
                                 prefetch_factor, is_train=True, seed=seed)

    @param('data.root')
    @param('data.num_workers')
    @param('data.pin_memory')
    @param('data.prefetch_factor')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.seed')
    def create_val_loader(self, root, num_workers, pin_memory, prefetch_factor,
                          batch_size, resolution, seed):
        if self.dataset == 'cifar100':
            transforms = []
            if resolution != 32:
                transforms.extend([
                    v2.Resize(resolution),
                    v2.CenterCrop(resolution)
                ])
            transforms.extend([
                v2.ToImage(),
                v2.ToDtype(ch.float32, scale=True),
                v2.Normalize(CIFAR100_MEAN, CIFAR100_STD)
            ])
            dataset = self._cifar100_dataset(root, train=False,
                                             transform=v2.Compose(transforms))
        else:
            resize_size = int(resolution / 0.875)
            transform = v2.Compose([
                v2.Resize(resize_size),
                v2.CenterCrop(resolution),
                v2.ToImage(),
                v2.ToDtype(ch.float32, scale=True),
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])
            dataset = self._imagenet_dataset(root, 'val', transform)

        return self._make_loader(dataset, batch_size, num_workers, pin_memory,
                                 prefetch_factor, is_train=False, seed=seed)

    @param('training.epochs')
    @param('logging.log_level')
    def train(self, epochs, log_level):
        epoch = -1
        for epoch in range(epochs):
            # TODO: Dynamic resolution scheduling with TorchVision requires
            # rebuilding the train transform/loader per epoch. Training uses
            # the initial get_resolution(epoch=0) value for now.
            train_loss = self.train_loop(epoch)

            if log_level > 0:
                extra_dict = {
                    'train_loss': train_loss,
                    'epoch': epoch
                }

                self.eval_and_log(extra_dict)

            self.save_checkpoint('latest_weights.pt', epoch=epoch)

        self.eval_and_log({'epoch':epoch})
        self.save_checkpoint('final_weights.pt', epoch=epoch)

    def save_checkpoint(self, filename, epoch=None):
        checkpoint_path = self.log_folder / filename
        ch.save(self.base_model().state_dict(), checkpoint_path)
        metadata = {
            'checkpoint': filename,
            'epoch': epoch,
            'saved_at': time.time()
        }
        with open(self.log_folder / f'{filename}.json', 'w') as handle:
            json.dump(metadata, handle, indent=2)

    def base_model(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def load_model_state(self, path):
        state_dict = ch.load(path, map_location=self.device)
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {
                key.removeprefix('module.'): value
                for key, value in state_dict.items()
            }
        self.base_model().load_state_dict(state_dict)

    def bor_residual_alpha_log_dict(self):
        model = self.base_model()
        fc = getattr(model, 'fc', None)
        if not getattr(fc, 'bor_residual_orthogonal', False):
            return {}

        alphas = fc.alpha_values()
        if alphas is None:
            return {}

        alpha_values = alphas.detach().float().cpu().tolist()
        log_dict = {
            f'bor_residual_alpha_{dim}': float(alpha)
            for dim, alpha in zip(fc.nesting_list[:-1], alpha_values)
        }
        if alpha_values:
            log_dict['bor_residual_alpha_mean'] = float(np.mean(alpha_values))
        return log_dict

    def eval_and_log(self, extra_dict={}):
        start_val = time.time()
        if self.nesting:
            stats = self.val_loop_nesting()
        else:
            stats = self.val_loop()
        val_time = time.time() - start_val

        d = {
            'current_lr': self.optimizer.param_groups[0]['lr'], 'val_time': val_time
        }
        d.update(self.bor_residual_alpha_log_dict())
        for k in stats.keys():
            if k=='loss':
                continue
            else:
                d[k]=stats[k]

        self.log(dict(d, **extra_dict))

        return stats

    @param('model.arch')
    @param('model.pretrained')
    @param('training.use_blurpool') # Later Arguments for nesting/fixed_feat
    @param('model.bor_mode')
    @param('model.bor_orthogonal_map')
    @param('model.bor_use_trivialization')
    @param('model.bor_stop_gradient')
    @param('model.bor_residual_orthogonal')
    @param('model.bor_residual_alpha_init')
    @param('model.cascade_stop_gradient')
    def create_model_and_scaler(self, arch, pretrained, use_blurpool,
                                bor_mode, bor_orthogonal_map, bor_use_trivialization,
                                bor_stop_gradient, bor_residual_orthogonal,
                                bor_residual_alpha_init, cascade_stop_gradient):
        '''
        Nesting Start is just the log_2 {smallest dim} unit. In our work we used powers of two, however this part can be changed easily. 
        If we do not want to use MRL, we just keep both the efficient and mrl flags to 0
        If we want a fixed feature baseline, then we just change fixed_feature={Rep. Size of your choice}

        NOTE: Blurpool follows the original training recipe.
        '''

        scaler = GradScaler(enabled=self.device.type == 'cuda')
        model = getattr(models, arch)(pretrained=pretrained)

        if self.bor_mrl:
            print("Creating classification layer of type :\t BOR-MRL (recursive prefix)")
            model.fc = BlockOrthogonalResidualMRLHead(
                self.nesting_list,
                num_classes=self.num_classes,
                mode=bor_mode,
                orthogonal_map=bor_orthogonal_map,
                use_trivialization=bool(bor_use_trivialization),
                stop_gradient=resolve_stop_gradient_override(bor_stop_gradient),
                bor_residual_orthogonal=bool(bor_residual_orthogonal),
                bor_residual_alpha_init=bor_residual_alpha_init,
            )
        elif self.cascade_stop_gradient_mrl:
            print("Creating classification layer of type :\t Cascade Stop-Gradient MRL")
            model.fc = CascadeStopGradientMRLHead(
                self.nesting_list,
                num_classes=self.num_classes,
                stop_gradient=resolve_stop_gradient_override(cascade_stop_gradient),
            )
        elif self.bor_block_mrl:
            print("Creating classification layer of type :\t BOR-MRL (independent residual blocks)")
            model.fc = IndependentBlockOrthogonalMRLHead(
                self.nesting_list,
                num_classes=self.num_classes,
                mode=bor_mode,
                orthogonal_map=bor_orthogonal_map,
                use_trivialization=bool(bor_use_trivialization),
                stop_gradient=resolve_stop_gradient_override(bor_stop_gradient),
            )
        elif self.nesting:
            ff= "MRL-E" if self.efficient else "MRL"
            print(f"Creating classification layer of type :\t {ff}")
            model.fc = MRL_Linear_Layer(self.nesting_list, num_classes=self.num_classes, efficient=self.efficient)
        elif self.fixed_feature != 2048:
            print("Using Fixed Features.... ")
            model.fc =  FixedFeatureLayer(self.fixed_feature, self.num_classes)
        elif model.fc.out_features != self.num_classes:
            print(f"Creating classification layer for {self.num_classes} classes")
            model.fc = ch.nn.Linear(model.fc.in_features, self.num_classes)
            
        def apply_blurpool(mod: ch.nn.Module):
            for (name, child) in mod.named_children():
                if isinstance(child, ch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
                    setattr(mod, name, BlurPoolConv2d(child))
                else: apply_blurpool(child)
        if use_blurpool: apply_blurpool(model)

        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.device)

        if self.device.type == 'cuda' and self.num_gpus > 1:
            print(f"Using DataParallel on {self.num_gpus} GPUs")
            model = ch.nn.DataParallel(model)

        return model, scaler

    @param('logging.log_level')
    def train_loop(self, epoch, log_level):
        model = self.model
        model.train()
        losses = []

        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        iters = len(self.train_loader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        iterator = tqdm(self.train_loader)
        for ix, (images, target) in enumerate(iterator):
            ### Training start
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lrs[ix]

            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            images = images.contiguous(memory_format=ch.channels_last)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.device.type == 'cuda'):
                output = self.model(images)
                loss_train = self.loss(output, target)

            self.scaler.scale(loss_train).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            ### Training end

            ### Logging start
            if log_level > 0:
                losses.append(loss_train.detach())

                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.3f}')

                names = ['ep', 'iter', 'shape', 'lrs']
                values = [epoch, ix, tuple(images.shape), group_lrs]
                if log_level > 1:
                    names += ['loss']
                    values += [f'{loss_train.item():.3f}']

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)
            ### Logging end

        if log_level > 0:
            loss = ch.stack(losses).mean().cpu()
            assert not ch.isnan(loss), 'Loss is NaN!'
            return loss.item()

    @param('validation.lr_tta')
    def val_loop(self, lr_tta):
        model = self.model
        model.eval()

        with ch.no_grad():
            with autocast(enabled=self.device.type == 'cuda'):
                for images, target in tqdm(self.val_loader):
                    images = images.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    images = images.contiguous(memory_format=ch.channels_last)

                    output = self.model(images)
                    if lr_tta:
                        output += self.model(ch.flip(images, dims=[3]))

                    for k in ['top_1', 'top_5']:
                        self.val_meters[k](output, target)

                    loss_val = self.loss(output, target)
                    self.val_meters['loss'](loss_val)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats


    @param('validation.lr_tta')
    def val_loop_nesting(self, lr_tta):
        '''
        Since Nested Layers will give a tuple of logits, we have a different subroutine for validation.
        '''

        model = self.model
        model.eval()
        with ch.no_grad():
            with autocast(enabled=self.device.type == 'cuda'):
                for images, target in tqdm(self.val_loader):
                    images = images.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    images = images.contiguous(memory_format=ch.channels_last)

                    output = self.model(images); output=ch.stack(output, dim=0)

                    if lr_tta:
                        output += ch.stack(self.model(ch.flip(images, dims=[3])), dim=0) # Just one augmentation.

                    # Logging the accuracies top1/5 for each of nesting...
                    for i in range(len(self.nesting_list)):
                        s = "top_1_{}".format(self.nesting_list[i])
                        self.val_meters[s](output[i], target)
                        s = "top_5_{}".format(self.nesting_list[i])
                        self.val_meters[s](output[i], target)

                    loss_val = self.loss(output, target)
                    self.val_meters['loss'](loss_val)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats


    @param('logging.folder')
    @param('logging.run_name')
    def initialize_logger(self, folder, run_name):
        def accuracy_meter(top_k=1):
            try:
                return torchmetrics.Accuracy(
                    task='multiclass',
                    num_classes=self.num_classes,
                    top_k=top_k
                ).to(self.device)
            except TypeError:
                kwargs = {'top_k': top_k} if top_k != 1 else {}
                try:
                    return torchmetrics.Accuracy(
                        compute_on_step=False,
                        **kwargs
                    ).to(self.device)
                except TypeError:
                    return torchmetrics.Accuracy(**kwargs).to(self.device)

        if self.nesting:
            self.val_meters={}
            for i in self.nesting_list:
                self.val_meters['top_1_{}'.format(i)] = accuracy_meter()

            for i in self.nesting_list:
                self.val_meters['top_5_{}'.format(i)] = accuracy_meter(top_k=5)

            self.val_meters['loss'] = MeanScalarMetric().to(self.device)

        else:   
            self.val_meters = {
                'top_1': accuracy_meter(),
                'top_5': accuracy_meter(top_k=5),
                'loss': MeanScalarMetric().to(self.device)
            }

        folder = (Path(folder) / (run_name if run_name else str(self.uid))).absolute()
        folder.mkdir(parents=True)

        self.log_folder = folder
        self.start_time = time.time()

        print(f'=> Logging in {self.log_folder}')
        params = {
            '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
        }

        with open(folder / 'params.json', 'w+') as handle:
            json.dump(params, handle)

    def log(self, content):
        print(f'=> Log: {content}')
        cur_time = time.time()
        with open(self.log_folder / 'log', 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - self.start_time,
                **content
            }) + '\n')
            fd.flush()

    @classmethod
    @param('training.eval_only')
    @param('training.path')
    def exec(cls, eval_only, path=None):
        trainer = cls()
        if eval_only:
            print("Loading Model.....")
            trainer.load_model_state(path)
            print("Loading Complete!")
            trainer.eval_and_log()
        else:
            trainer.train()

# Utils
class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state('sum', default=ch.tensor(0.))
        self.add_state('count', default=ch.tensor(0))

    def update(self, sample: ch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count

# Running
def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()

if __name__ == "__main__":
    make_config()
    ImageNetTrainer.exec()
