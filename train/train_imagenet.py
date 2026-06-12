"""
ResNet training entry point for CIFAR-100 and ImageNet MRL.
"""
import json
import os
import random
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from uuid import uuid4

sys.path.append("../")

import numpy as np
import torch as ch
import torch.nn.functional as F
from fastargs import Param, Section, get_current_config
from fastargs.decorators import param
from fastargs.validation import And, OneOf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import v2
from tqdm import tqdm

from MRL import FixedFeatureLayer, Matryoshka_CE_Loss, MRL_Linear_Layer


Section("model", "model details").params(
	arch=Param(And(str, OneOf(models.__dir__())), default="resnet50"),
	pretrained=Param(int, "use TorchVision pretrained weights? (1/0)", default=0),
	efficient=Param(int, "use MRL-E shared classifier? (1/0)", default=0),
	mrl=Param(int, "use Matryoshka Representation Learning? (1/0)", default=0),
	nesting_start=Param(int, "2**i will be the smallest nesting dimension", default=3),
	fixed_feature=Param(int, "fixed-feature eval/training prefix size", default=2048),
	prefix_mask_prob=Param(float, "training-only inherited-prefix mask probability", default=0.0),
	prefix_mask_scale=Param(And(str, OneOf(["inverted", "none"])), "prefix mask scaling", default="inverted"),
)

Section("resolution", "resolution scheduling").params(
	min_res=Param(int, "minimum training resolution", default=160),
	max_res=Param(int, "maximum training resolution", default=160),
	end_ramp=Param(int, "when to stop interpolating resolution", default=0),
	start_ramp=Param(int, "when to start interpolating resolution", default=0),
)

Section("data", "data related settings").params(
	dataset=Param(And(str, OneOf(["imagenet", "cifar100"])), "dataset", default="imagenet"),
	root=Param(str, "dataset root directory", default=""),
	num_workers=Param(int, "number of dataloader workers", default=8),
	pin_memory=Param(int, "pin dataloader memory? (1/0)", default=1),
	prefetch_factor=Param(int, "batches prefetched by each worker", default=4),
)

Section("lr", "lr scheduling").params(
	step_ratio=Param(float, "learning rate step ratio", default=0.1),
	step_length=Param(int, "learning rate step length", default=30),
	lr_schedule_type=Param(OneOf(["step", "cyclic", "constant"]), default="cyclic"),
	lr=Param(float, "learning rate", default=0.5),
	lr_peak_epoch=Param(int, "epoch at which LR peaks", default=2),
)

Section("logging", "logging settings").params(
	folder=Param(str, "log location", required=True),
	run_name=Param(str, "optional run folder name", default=""),
	log_level=Param(int, "0 only epoch-end logs, 1 progress, 2 verbose progress", default=1),
)

Section("validation", "validation settings").params(
	batch_size=Param(int, "validation batch size", default=512),
	resolution=Param(int, "validation image size", default=224),
	lr_tta=Param(int, "left-right flip test-time augmentation? (1/0)", default=1),
)

Section("training", "training hyperparameters").params(
	eval_only=Param(int, "eval only? (1/0)", default=0),
	path=Param(str, "weight path for trained model", default=None),
	batch_size=Param(int, "training batch size", default=512),
	optimizer=Param(And(str, OneOf(["sgd"])), "optimizer", default="sgd"),
	momentum=Param(float, "SGD momentum", default=0.9),
	weight_decay=Param(float, "weight decay", default=4e-5),
	bn_wd=Param(int, "apply weight decay to norm layers? (1/0)", default=0),
	epochs=Param(int, "number of epochs", default=30),
	label_smoothing=Param(float, "label smoothing", default=0.1),
	use_blurpool=Param(int, "use blurpool? (1/0)", default=0),
	seed=Param(int, "random seed", default=0),
	deterministic=Param(int, "enable deterministic PyTorch/CUDA behavior? (1/0)", default=0),
	mrl_loss_mode=Param(And(str, OneOf(["all", "sampled_prefix"])), "MRL loss mode", default="all"),
	sampled_prefix_distribution=Param(
		And(str, OneOf(["uniform", "inverse_dim", "inverse_sqrt_dim"])),
		"sampled-prefix MRL distribution",
		default="uniform",
	),
	sampled_prefix_log_interval=Param(int, "sampled-prefix log interval in batches", default=100),
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
DATASET_CONFIGS = {
	"imagenet": {"num_classes": 1000, "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
	"cifar100": {"num_classes": 100, "mean": CIFAR100_MEAN, "std": CIFAR100_STD},
}


def seed_worker(worker_id):
	worker_seed = ch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)


def set_reproducibility(seed, deterministic):
	os.environ.setdefault("PYTHONHASHSEED", str(seed))
	random.seed(seed)
	np.random.seed(seed)
	ch.manual_seed(seed)
	ch.cuda.manual_seed_all(seed)

	if deterministic:
		os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
		ch.backends.cudnn.benchmark = False
		ch.backends.cudnn.deterministic = True
		if hasattr(ch.backends, "cuda") and hasattr(ch.backends.cuda, "matmul"):
			ch.backends.cuda.matmul.allow_tf32 = False
		if hasattr(ch.backends.cudnn, "allow_tf32"):
			ch.backends.cudnn.allow_tf32 = False
		if hasattr(ch, "use_deterministic_algorithms"):
			try:
				ch.use_deterministic_algorithms(True, warn_only=True)
			except TypeError:
				ch.use_deterministic_algorithms(True)
	else:
		ch.backends.cudnn.benchmark = True
		if hasattr(ch.backends, "cuda") and hasattr(ch.backends.cuda, "matmul"):
			ch.backends.cuda.matmul.allow_tf32 = True
		if hasattr(ch.backends.cudnn, "allow_tf32"):
			ch.backends.cudnn.allow_tf32 = True
		if hasattr(ch, "set_float32_matmul_precision"):
			ch.set_float32_matmul_precision("high")


@param("lr.lr")
@param("lr.step_ratio")
@param("lr.step_length")
@param("training.epochs")
def get_step_lr(epoch, lr, step_ratio, step_length, epochs):
	if epoch >= epochs:
		return 0
	return step_ratio ** (epoch // step_length) * lr


@param("lr.lr")
def get_constant_lr(epoch, lr):
	return lr


@param("lr.lr")
@param("training.epochs")
@param("lr.lr_peak_epoch")
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
		self.register_buffer("blur_filter", filt)

	def forward(self, x):
		blurred = F.conv2d(
			x,
			self.blur_filter,
			stride=1,
			padding=(1, 1),
			groups=self.conv.in_channels,
			bias=None,
		)
		return self.conv.forward(blurred)


def apply_blurpool(mod):
	for name, child in mod.named_children():
		if isinstance(child, ch.nn.Conv2d) and np.max(child.stride) > 1 and child.in_channels >= 16:
			setattr(mod, name, BlurPoolConv2d(child))
		else:
			apply_blurpool(child)


def make_torchvision_model(arch, pretrained):
	model_fn = getattr(models, arch)
	if pretrained:
		try:
			return model_fn(weights="DEFAULT")
		except TypeError:
			return model_fn(pretrained=True)
	try:
		return model_fn(weights=None)
	except TypeError:
		return model_fn(pretrained=False)


def topk_correct(logits, target, k):
	k = min(int(k), logits.shape[1])
	pred = logits.topk(k=k, dim=1, largest=True, sorted=True).indices
	return int(pred.eq(target.view(-1, 1)).any(dim=1).sum().item())


class ImageNetTrainer:
	@param("model.efficient")
	@param("model.mrl")
	@param("model.nesting_start")
	@param("model.fixed_feature")
	@param("model.prefix_mask_prob")
	@param("model.prefix_mask_scale")
	@param("data.dataset")
	@param("training.seed")
	@param("training.deterministic")
	def __init__(
		self,
		efficient,
		mrl,
		nesting_start,
		fixed_feature,
		prefix_mask_prob,
		prefix_mask_scale,
		dataset,
		seed,
		deterministic,
	):
		self.all_params = get_current_config()
		self.seed = seed
		self.deterministic = bool(deterministic)
		set_reproducibility(seed, self.deterministic)
		self.device = ch.device("cuda" if ch.cuda.is_available() else "cpu")
		self.num_gpus = ch.cuda.device_count() if self.device.type == "cuda" else 0
		self.efficient = bool(efficient)
		self.mrl = bool(mrl)
		self.nesting = bool(self.mrl or self.efficient)
		self.nesting_start = int(nesting_start)
		self.nesting_list = None
		self.fixed_feature = int(fixed_feature)
		self.prefix_mask_prob = float(prefix_mask_prob)
		self.prefix_mask_scale = prefix_mask_scale
		self.dataset = dataset
		self.dataset_config = DATASET_CONFIGS[dataset]
		self.num_classes = self.dataset_config["num_classes"]
		self.uid = str(uuid4())

		if self.prefix_mask_prob > 0.0 and not self.nesting:
			raise ValueError("--model.prefix_mask_prob requires --model.mrl=1 or --model.efficient=1")

		self.train_loader = self.create_train_loader()
		self.val_loader = self.create_val_loader()
		self.model, self.scaler = self.create_model_and_scaler()
		self.create_optimizer()
		self.initialize_logger()

	@param("lr.lr_schedule_type")
	def get_lr(self, epoch, lr_schedule_type):
		lr_schedules = {
			"cyclic": get_cyclic_lr,
			"step": get_step_lr,
			"constant": get_constant_lr,
		}
		return lr_schedules[lr_schedule_type](epoch)

	@param("resolution.min_res")
	@param("resolution.max_res")
	@param("resolution.end_ramp")
	@param("resolution.start_ramp")
	def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
		assert min_res <= max_res
		if epoch <= start_ramp:
			return min_res
		if epoch >= end_ramp:
			return max_res
		interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
		return int(np.round(interp[0] / 32)) * 32

	def build_nesting_list(self, feature_dim):
		start_dim = 2 ** self.nesting_start
		if start_dim > feature_dim:
			raise ValueError(
				f"smallest nesting dimension {start_dim} exceeds feature dimension {feature_dim}"
			)
		dims = []
		dim = start_dim
		while dim < feature_dim:
			dims.append(dim)
			dim *= 2
		if not dims or dims[-1] != feature_dim:
			dims.append(feature_dim)
		return dims

	@param("training.momentum")
	@param("training.optimizer")
	@param("training.weight_decay")
	@param("training.bn_wd")
	@param("training.label_smoothing")
	@param("training.mrl_loss_mode")
	@param("training.sampled_prefix_distribution")
	def create_optimizer(
		self,
		momentum,
		optimizer,
		weight_decay,
		bn_wd,
		label_smoothing,
		mrl_loss_mode,
		sampled_prefix_distribution,
	):
		assert optimizer == "sgd"
		decay_params, no_decay_params = [], []
		norm_types = (
			ch.nn.BatchNorm1d,
			ch.nn.BatchNorm2d,
			ch.nn.BatchNorm3d,
			ch.nn.GroupNorm,
			ch.nn.LayerNorm,
		)
		for module in self.model.modules():
			for name, param in module.named_parameters(recurse=False):
				if not param.requires_grad:
					continue
				if name.endswith("bias") or (not bn_wd and isinstance(module, norm_types)):
					no_decay_params.append(param)
				else:
					decay_params.append(param)

		param_groups = [
			{"params": no_decay_params, "weight_decay": 0.0},
			{"params": decay_params, "weight_decay": weight_decay},
		]
		try:
			self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum, foreach=True)
		except TypeError:
			self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)

		if self.nesting:
			self.loss = Matryoshka_CE_Loss(
				label_smoothing=label_smoothing,
				mrl_loss_mode=mrl_loss_mode,
				nesting_list=self.nesting_list,
				sampled_prefix_distribution=sampled_prefix_distribution,
			)
			self.val_loss = Matryoshka_CE_Loss(label_smoothing=label_smoothing)
			if mrl_loss_mode == "sampled_prefix":
				probs = self.loss.sampled_prefix_probabilities()
				print(
					"Sampled-prefix MRL distribution "
					f"({sampled_prefix_distribution}): {dict(zip(self.nesting_list, probs))}"
				)
		else:
			if mrl_loss_mode != "all":
				raise ValueError("--training.mrl_loss_mode=sampled_prefix requires an MRL model")
			self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
			self.val_loss = self.loss

	def _dataset_root(self, root):
		return Path(root).expanduser()

	def _make_loader(self, dataset, batch_size, num_workers, pin_memory, prefetch_factor, is_train, seed):
		generator = ch.Generator()
		generator.manual_seed(seed)
		kwargs = {
			"dataset": dataset,
			"batch_size": batch_size,
			"shuffle": is_train,
			"num_workers": num_workers,
			"pin_memory": bool(pin_memory),
			"persistent_workers": num_workers > 0,
			"drop_last": is_train,
			"worker_init_fn": seed_worker,
			"generator": generator,
		}
		if num_workers > 0:
			kwargs["prefetch_factor"] = prefetch_factor
		return DataLoader(**kwargs)

	def _cifar100_dataset(self, root, train, transform):
		return datasets.CIFAR100(
			root=str(self._dataset_root(root)),
			train=train,
			download=True,
			transform=transform,
		)

	def _imagenet_dataset(self, root, split, transform):
		split_root = self._dataset_root(root) / split
		if not split_root.is_dir():
			raise FileNotFoundError(
				f"Expected ImageNet {split} directory at {split_root}. "
				"Use a root with train/ and val/ subdirectories."
			)
		return datasets.ImageFolder(split_root, transform=transform)

	@param("data.root")
	@param("data.num_workers")
	@param("data.pin_memory")
	@param("data.prefetch_factor")
	@param("training.batch_size")
	@param("training.seed")
	def create_train_loader(self, root, num_workers, pin_memory, prefetch_factor, batch_size, seed):
		res = self.get_resolution(epoch=0)
		self.train_resolution = res
		if self.dataset == "cifar100":
			transform = v2.Compose([
				v2.RandomCrop(32, padding=4),
				v2.RandomHorizontalFlip(),
				v2.ToImage(),
				v2.ToDtype(ch.float32, scale=True),
				v2.Normalize(CIFAR100_MEAN, CIFAR100_STD),
			])
			dataset = self._cifar100_dataset(root, train=True, transform=transform)
		else:
			transform = v2.Compose([
				v2.RandomResizedCrop(res),
				v2.RandomHorizontalFlip(),
				v2.ToImage(),
				v2.ToDtype(ch.float32, scale=True),
				v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
			])
			dataset = self._imagenet_dataset(root, "train", transform)
		return self._make_loader(dataset, batch_size, num_workers, pin_memory, prefetch_factor, True, seed)

	@param("data.root")
	@param("data.num_workers")
	@param("data.pin_memory")
	@param("data.prefetch_factor")
	@param("validation.batch_size")
	@param("validation.resolution")
	@param("training.seed")
	def create_val_loader(self, root, num_workers, pin_memory, prefetch_factor, batch_size, resolution, seed):
		if self.dataset == "cifar100":
			transforms = []
			if resolution != 32:
				transforms.extend([v2.Resize(resolution), v2.CenterCrop(resolution)])
			transforms.extend([
				v2.ToImage(),
				v2.ToDtype(ch.float32, scale=True),
				v2.Normalize(CIFAR100_MEAN, CIFAR100_STD),
			])
			dataset = self._cifar100_dataset(root, train=False, transform=v2.Compose(transforms))
		else:
			resize_size = int(resolution / 0.875)
			transform = v2.Compose([
				v2.Resize(resize_size),
				v2.CenterCrop(resolution),
				v2.ToImage(),
				v2.ToDtype(ch.float32, scale=True),
				v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
			])
			dataset = self._imagenet_dataset(root, "val", transform)
		return self._make_loader(dataset, batch_size, num_workers, pin_memory, prefetch_factor, False, seed)

	@param("model.arch")
	@param("model.pretrained")
	@param("training.use_blurpool")
	def create_model_and_scaler(self, arch, pretrained, use_blurpool):
		scaler = GradScaler(enabled=self.device.type == "cuda")
		model = make_torchvision_model(arch, bool(pretrained))
		feature_dim = model.fc.in_features

		if self.nesting:
			self.nesting_list = self.build_nesting_list(feature_dim)
			head_name = "MRL-E" if self.efficient else "MRL"
			print(f"Creating classification layer of type:\t{head_name}")
			print(f"MRL nesting dimensions: {self.nesting_list}")
			if self.prefix_mask_prob > 0.0:
				print(
					"MRL prefix masking enabled: "
					f"p={self.prefix_mask_prob}, scale={self.prefix_mask_scale}"
				)
			model.fc = MRL_Linear_Layer(
				self.nesting_list,
				num_classes=self.num_classes,
				efficient=self.efficient,
				prefix_mask_prob=self.prefix_mask_prob,
				prefix_mask_scale=self.prefix_mask_scale,
			)
		elif self.fixed_feature != 2048:
			if self.fixed_feature > feature_dim:
				raise ValueError(
					f"fixed_feature={self.fixed_feature} exceeds model feature dimension {feature_dim}"
				)
			print(f"Using fixed feature prefix: {self.fixed_feature}")
			model.fc = FixedFeatureLayer(self.fixed_feature, self.num_classes)
		elif model.fc.out_features != self.num_classes:
			print(f"Creating classification layer for {self.num_classes} classes")
			model.fc = ch.nn.Linear(feature_dim, self.num_classes)

		if use_blurpool:
			apply_blurpool(model)

		if self.device.type == "cuda":
			model = model.to(memory_format=ch.channels_last)
		model = model.to(self.device)

		if self.device.type == "cuda" and self.num_gpus > 1:
			print(f"Using DataParallel on {self.num_gpus} GPUs")
			model = ch.nn.DataParallel(model)

		return model, scaler

	def base_model(self):
		return self.model.module if hasattr(self.model, "module") else self.model

	def load_model_state(self, path):
		state_dict = ch.load(path, map_location=self.device)
		if any(key.startswith("module.") for key in state_dict.keys()):
			state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
		self.base_model().load_state_dict(state_dict)

	def sampled_prefix_active(self):
		return self.nesting and getattr(self.loss, "mrl_loss_mode", None) == "sampled_prefix"

	def sampled_prefix_counts_log_dict(self):
		if not self.sampled_prefix_active():
			return {}
		counts = self.loss.sample_counts_list()
		return {
			"sampled_prefix_counts": counts,
			"sampled_prefix_counts_by_dim": {
				str(dim): int(count)
				for dim, count in zip(self.nesting_list, counts)
			},
		}

	def sampled_prefix_startup_log_dict(self):
		if not self.sampled_prefix_active():
			return {}
		return {
			"mrl_loss_mode": "sampled_prefix",
			"sampled_prefix_distribution": self.loss.sampled_prefix_distribution,
			"sampled_prefix_dims": self.nesting_list,
			"sampled_prefix_probs": self.loss.sampled_prefix_probabilities(),
		}

	def sampled_prefix_step_log_dict(self):
		if not self.sampled_prefix_active() or self.loss.last_sampled_idx is None:
			return {}
		log_dict = {
			"mrl_loss_mode": "sampled_prefix",
			"sampled_prefix_idx": int(self.loss.last_sampled_idx),
			"sampled_prefix_dim": int(self.loss.last_sampled_dim),
			"sampled_prefix_ce_loss": float(self.loss.last_selected_ce.float().cpu().item()),
		}
		log_dict.update(self.sampled_prefix_counts_log_dict())
		return log_dict

	@param("training.epochs")
	@param("logging.log_level")
	def train(self, epochs, log_level):
		epoch = -1
		for epoch in range(epochs):
			train_loss = self.train_loop(epoch)
			if log_level > 0:
				self.eval_and_log({"train_loss": train_loss, "epoch": epoch})
			self.save_checkpoint("latest_weights.pt", epoch=epoch)
		self.eval_and_log({"epoch": epoch})
		self.save_checkpoint("final_weights.pt", epoch=epoch)

	def save_checkpoint(self, filename, epoch=None):
		checkpoint_path = self.log_folder / filename
		ch.save(self.base_model().state_dict(), checkpoint_path)
		metadata = {"checkpoint": filename, "epoch": epoch, "saved_at": time.time()}
		with open(self.log_folder / f"{filename}.json", "w") as handle:
			json.dump(metadata, handle, indent=2)

	@param("logging.log_level")
	@param("training.sampled_prefix_log_interval")
	def train_loop(self, epoch, log_level, sampled_prefix_log_interval):
		self.model.train()
		lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
		iters = len(self.train_loader)
		lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])
		total_loss = 0.0
		total_seen = 0

		iterator = tqdm(self.train_loader, disable=log_level <= 0)
		for ix, (images, target) in enumerate(iterator):
			for param_group in self.optimizer.param_groups:
				param_group["lr"] = lrs[ix]

			images = images.to(self.device, non_blocking=True)
			target = target.to(self.device, non_blocking=True)
			if self.device.type == "cuda":
				images = images.contiguous(memory_format=ch.channels_last)

			self.optimizer.zero_grad(set_to_none=True)
			with autocast(enabled=self.device.type == "cuda"):
				output = self.model(images)
				loss_train = self.loss(output, target)

			self.scaler.scale(loss_train).backward()
			self.scaler.step(self.optimizer)
			self.scaler.update()

			batch_size = target.numel()
			total_loss += float(loss_train.detach().cpu().item()) * batch_size
			total_seen += batch_size

			if self.sampled_prefix_active() and sampled_prefix_log_interval > 0:
				if ix == 0 or (ix + 1) % sampled_prefix_log_interval == 0:
					self.log({"epoch": epoch, "iter": ix, **self.sampled_prefix_step_log_dict()})

			if log_level > 0:
				group_lrs = [f"{group['lr']:.3f}" for group in self.optimizer.param_groups]
				names = ["ep", "iter", "shape", "lrs"]
				values = [epoch, ix, tuple(images.shape), group_lrs]
				if log_level > 1:
					names += ["loss"]
					values += [f"{loss_train.item():.3f}"]
				iterator.set_description(", ".join(f"{n}={v}" for n, v in zip(names, values)))

		loss = total_loss / max(total_seen, 1)
		assert not np.isnan(loss), "Loss is NaN!"
		return loss

	@param("validation.lr_tta")
	def val_loop(self, lr_tta):
		self.model.eval()
		total_loss = 0.0
		total_seen = 0
		top1 = 0
		top5 = 0
		with ch.inference_mode():
			for images, target in tqdm(self.val_loader, leave=False):
				images = images.to(self.device, non_blocking=True)
				target = target.to(self.device, non_blocking=True)
				if self.device.type == "cuda":
					images = images.contiguous(memory_format=ch.channels_last)
				with autocast(enabled=self.device.type == "cuda"):
					output = self.model(images)
					if lr_tta:
						output = output + self.model(ch.flip(images, dims=[3]))
					loss_val = self.val_loss(output, target)

				batch_size = target.numel()
				total_loss += float(loss_val.detach().cpu().item()) * batch_size
				total_seen += batch_size
				top1 += topk_correct(output, target, 1)
				top5 += topk_correct(output, target, 5)

		return {
			"top_1": top1 / total_seen,
			"top_5": top5 / total_seen,
			"loss": total_loss / total_seen,
		}

	@param("validation.lr_tta")
	def val_loop_nesting(self, lr_tta):
		self.model.eval()
		total_loss = 0.0
		total_seen = 0
		top1 = {dim: 0 for dim in self.nesting_list}
		top5 = {dim: 0 for dim in self.nesting_list}

		with ch.inference_mode():
			for images, target in tqdm(self.val_loader, leave=False):
				images = images.to(self.device, non_blocking=True)
				target = target.to(self.device, non_blocking=True)
				if self.device.type == "cuda":
					images = images.contiguous(memory_format=ch.channels_last)
				with autocast(enabled=self.device.type == "cuda"):
					output = ch.stack(self.model(images), dim=0)
					if lr_tta:
						output = output + ch.stack(self.model(ch.flip(images, dims=[3])), dim=0)
					loss_val = self.val_loss(output, target)

				batch_size = target.numel()
				total_loss += float(loss_val.detach().cpu().item()) * batch_size
				total_seen += batch_size
				for i, dim in enumerate(self.nesting_list):
					top1[dim] += topk_correct(output[i], target, 1)
					top5[dim] += topk_correct(output[i], target, 5)

		stats = {"loss": total_loss / total_seen}
		for dim in self.nesting_list:
			stats[f"top_1_{dim}"] = top1[dim] / total_seen
			stats[f"top_5_{dim}"] = top5[dim] / total_seen
		return stats

	def eval_and_log(self, extra_dict=None):
		extra_dict = extra_dict or {}
		start_val = time.time()
		stats = self.val_loop_nesting() if self.nesting else self.val_loop()
		val_time = time.time() - start_val
		log_dict = {
			"current_lr": self.optimizer.param_groups[0]["lr"],
			"val_time": val_time,
			**self.sampled_prefix_counts_log_dict(),
		}
		log_dict.update({key: value for key, value in stats.items() if key != "loss"})
		log_dict.update(extra_dict)
		self.log(log_dict)
		self.print_validation_summary(log_dict)
		return stats

	def print_validation_summary(self, log_dict):
		epoch = log_dict.get("epoch", "?")
		train_loss = log_dict.get("train_loss")
		prefix = f"=> Val: epoch={epoch}"
		if train_loss is not None:
			prefix += f", train_loss={float(train_loss):.4f}"
		if self.nesting:
			top1_items = []
			for dim in self.nesting_list:
				key = f"top_1_{dim}"
				if key in log_dict:
					top1_items.append(f"{dim}:{100.0 * log_dict[key]:.2f}")
			if top1_items:
				print(f"{prefix}, top1[%]=" + ", ".join(top1_items))
		elif "top_1" in log_dict and "top_5" in log_dict:
			print(
				f"{prefix}, top1={100.0 * log_dict['top_1']:.2f}%, "
				f"top5={100.0 * log_dict['top_5']:.2f}%"
			)

	@param("logging.folder")
	@param("logging.run_name")
	def initialize_logger(self, folder, run_name):
		folder = (Path(folder) / (run_name if run_name else str(self.uid))).absolute()
		folder.mkdir(parents=True)
		self.log_folder = folder
		self.start_time = time.time()
		print(f"=> Logging in {self.log_folder}")

		params = {".".join(k): self.all_params[k] for k in self.all_params.entries.keys()}
		with open(folder / "params.json", "w+") as handle:
			json.dump(params, handle)

		startup_log = self.sampled_prefix_startup_log_dict()
		if self.prefix_mask_prob > 0.0:
			startup_log = {
				**startup_log,
				"prefix_mask_prob": self.prefix_mask_prob,
				"prefix_mask_scale": self.prefix_mask_scale,
			}
		if startup_log:
			self.log(startup_log)

	def log(self, content):
		print(f"=> Log: {content}")
		cur_time = time.time()
		with open(self.log_folder / "log", "a+") as fd:
			fd.write(json.dumps({
				"timestamp": cur_time,
				"relative_time": cur_time - self.start_time,
				**content,
			}) + "\n")
			fd.flush()

	@classmethod
	@param("training.eval_only")
	@param("training.path")
	def exec(cls, eval_only, path=None):
		trainer = cls()
		if eval_only:
			print("Loading model...")
			trainer.load_model_state(path)
			print("Loading complete.")
			trainer.eval_and_log()
		else:
			trainer.train()


def make_config(quiet=False):
	config = get_current_config()
	parser = ArgumentParser(description="Fast CIFAR/ImageNet MRL training")
	config.augment_argparse(parser)
	config.collect_argparse_args(parser)
	config.validate(mode="stderr")
	if not quiet:
		config.summary()


if __name__ == "__main__":
	make_config()
	ImageNetTrainer.exec()
