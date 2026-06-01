'''
Code to evaluate MRL models on different validation benchmarks. 
'''
import sys 
sys.path.append("../") # adding root folder to the path

import json
import os
import random
import torch 
import torchvision
from torchvision import transforms
from torchvision.models import *
from torchvision import datasets
from tqdm import tqdm
from pathlib import Path

from MRL import *
from argparse import ArgumentParser
from utils import *

try:
	from imagenetv2_pytorch import ImageNetV2Dataset
except ImportError:
	ImageNetV2Dataset = None

# nesting list is by default from 8 to 2048 in powers of 2, can be modified from here.
BATCH_SIZE = 256
IMG_SIZE = 256
CENTER_CROP_SIZE = 224
NESTING_LIST=[2**i for i in range(3, 12)]
ROOT="../../IMAGENET/" # path to validation datasets
DATASET_CONFIGS = {
	'imagenet': {
		'num_classes': 1000,
		'img_size': IMG_SIZE,
		'center_crop_size': CENTER_CROP_SIZE,
		'mean': [0.485, 0.456, 0.406],
		'std': [0.229, 0.224, 0.225]
	},
	'cifar100': {
		'num_classes': 100,
		'img_size': 32,
		'center_crop_size': 32,
		'mean': [0.5071, 0.4867, 0.4408],
		'std': [0.2675, 0.2565, 0.2761]
	}
}

def get_dataset_config(dataset):
	return DATASET_CONFIGS['cifar100'] if dataset.lower() == 'cifar100' else DATASET_CONFIGS['imagenet']

def load_imagenet_v2(transform):
	if ImageNetV2Dataset is None:
		raise ImportError("imagenetv2_pytorch is required for --dataset V2")
	return ImageNetV2Dataset("matched-frequency", transform=transform)

def set_eval_reproducibility(seed, deterministic):
	os.environ.setdefault('PYTHONHASHSEED', str(seed))
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	if deterministic:
		os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True
		if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
			torch.backends.cuda.matmul.allow_tf32 = False
		if hasattr(torch.backends.cudnn, 'allow_tf32'):
			torch.backends.cudnn.allow_tf32 = False
		if hasattr(torch, 'use_deterministic_algorithms'):
			try:
				torch.use_deterministic_algorithms(True, warn_only=True)
			except TypeError:
				torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
	worker_seed = args.seed + worker_id
	random.seed(worker_seed)
	torch.manual_seed(worker_seed)

def save_metrics(metrics, path):
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, 'w') as handle:
		json.dump(metrics, handle, indent=2)

parser=ArgumentParser()

# model args
parser.add_argument('--efficient', action='store_true', help='Efficient Flag')
parser.add_argument('--mrl', action='store_true', help='To use MRL')
parser.add_argument('--rep_size', type=int, default=2048, help='Rep. size for fixed feature model')
parser.add_argument('--path', type=str, required=True, help='Path to .pt model checkpoint')
parser.add_argument('--old_ckpt', action='store_true', help='To use our trained checkpoints')
parser.add_argument('--workers', type=int, default=12, help='num workers for dataloader')
parser.add_argument('--bor_mrl', action='store_true', help='Use recursive-prefix Block-Orthogonal Residual MRL')
parser.add_argument('--bor_block_mrl', action='store_true', help='Use independent-block Block-Orthogonal Residual MRL')
parser.add_argument('--bor_mode', type=str, choices=['identity', 'orthogonal'], default='orthogonal', help='BOR block transform mode')
parser.add_argument('--bor_orthogonal_map', type=str, choices=['matrix_exp', 'cayley', 'householder'], default='matrix_exp', help='BOR orthogonal parametrization map')
parser.add_argument('--bor_use_trivialization', type=int, default=1, help='Use dynamic trivialization for BOR orthogonal maps')
# dataset/eval args
parser.add_argument('--tta', action='store_true', help='Test Time Augmentation Flag')
parser.add_argument('--dataset', type=str, default='V1', help='Benchmarks: V1/V2/A/Sketch/R/CIFAR100')
parser.add_argument('--data_root', type=str, default=ROOT, help='Root directory for ImageNet-style datasets or torchvision CIFAR100 data')
parser.add_argument('--seed', type=int, default=0, help='random seed for reproducible evaluation')
parser.add_argument('--deterministic', action='store_true', help='enable deterministic PyTorch/CUDA behavior')
parser.add_argument('--metrics_output', type=str, default='', help='Optional JSON file for evaluation metrics')
parser.add_argument('--save_logits', action='store_true', help='To save logits for model analysis')
parser.add_argument('--save_softmax', action='store_true', help='To save softmax_probs for model analysis')
parser.add_argument('--save_gt', action='store_true', help='To save ground truth for model analysis')
parser.add_argument('--save_predictions', action='store_true', help='To save predicted labels for model analysis')
# retrieval args
parser.add_argument('--retrieval', action='store_true', help='flag for image retrieval array dumps')
parser.add_argument('--random_sample_dim', type=int, default=4202000, help='number of random samples to slice from retrieval database')
parser.add_argument('--retrieval_array_path', default='', help='path to save database and query arrays for retrieval', type=str)


args = parser.parse_args()
if args.bor_mrl and args.bor_block_mrl:
	raise ValueError("Choose only one BOR-MRL method: --bor_mrl or --bor_block_mrl.")
if (args.bor_mrl or args.bor_block_mrl) and args.mrl:
	raise ValueError("BOR-MRL is its own MRL variant; do not combine BOR-MRL with --mrl.")
if (args.bor_mrl or args.bor_block_mrl) and args.efficient:
	raise ValueError("BOR-MRL currently uses one classifier per prefix; do not combine BOR-MRL with --efficient.")
if (args.bor_mrl or args.bor_block_mrl) and args.retrieval:
	raise NotImplementedError("--retrieval with BOR-MRL is not implemented because BOR-MRL transforms residual blocks inside the classification head.")

set_eval_reproducibility(args.seed, args.deterministic)
dataset_config = get_dataset_config(args.dataset)
num_classes = dataset_config['num_classes']
data_root = Path(args.data_root)

model = resnet50(False)
if args.bor_mrl:
	model.fc = BlockOrthogonalResidualMRLHead(
		NESTING_LIST,
		num_classes=num_classes,
		mode=args.bor_mode,
		orthogonal_map=args.bor_orthogonal_map,
		use_trivialization=bool(args.bor_use_trivialization),
	)
elif args.bor_block_mrl:
	model.fc = IndependentBlockOrthogonalMRLHead(
		NESTING_LIST,
		num_classes=num_classes,
		mode=args.bor_mode,
		orthogonal_map=args.bor_orthogonal_map,
		use_trivialization=bool(args.bor_use_trivialization),
	)
elif not args.old_ckpt:
	if args.mrl:
		model.fc = MRL_Linear_Layer(NESTING_LIST, num_classes=num_classes, efficient=args.efficient)
	else:
		model.fc=FixedFeatureLayer(args.rep_size, num_classes)
else:
	if args.mrl:	
		model = load_from_old_ckpt(model, args.efficient, NESTING_LIST, num_classes=num_classes)
	else:
		model.fc=FixedFeatureLayer(args.rep_size, num_classes)

apply_blurpool(model)	
model.load_state_dict(get_ckpt(args.path)) # Since our models have a torch DDP wrapper, we modify keys to exclude first 7 chars. 
model = model.cuda()
model.eval()
is_nested_model = args.mrl or args.bor_mrl or args.bor_block_mrl

normalize = transforms.Normalize(mean=dataset_config['mean'], std=dataset_config['std'])
test_transform = transforms.Compose([
				transforms.Resize(dataset_config['img_size']),
				transforms.CenterCrop(dataset_config['center_crop_size']),
				transforms.ToTensor(),
				normalize])

# Model Eval
if not args.retrieval:
	if args.dataset.lower() == 'cifar100':
		print("Loading CIFAR-100 test set")
		dataset = datasets.CIFAR100(root=str(data_root), train=False, transform=test_transform, download=True)
	elif args.dataset == 'V2':
		print("Loading Robustness Dataset")
		dataset = load_imagenet_v2(test_transform)
	elif args.dataset == 'A':
		print("Loading true Imagenet-A val set")
		dataset = torchvision.datasets.ImageFolder(str(data_root / 'imagenet-a'), transform=test_transform)
	elif args.dataset == 'R':
		print("Loading true Imagenet-R val set")
		dataset = torchvision.datasets.ImageFolder(str(data_root / 'imagenet-r_'), transform=test_transform)
	elif args.dataset == 'sketch':
		print("Loading Imagenet-Sketch dataset")
		dataset = torchvision.datasets.ImageFolder(str(data_root / 'sketch'), transform=test_transform)
	else:
		print("Loading true Imagenet 1K val set")
		dataset = torchvision.datasets.ImageFolder(str(data_root / 'val'), transform=test_transform)

	generator = torch.Generator()
	generator.manual_seed(args.seed)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=args.workers,
			shuffle=False, worker_init_fn=seed_worker, generator=generator)

	if is_nested_model:
		_, top1_acc, top5_acc, total_time, num_images, m_score_dict, softmax_probs, gt, logits = evaluate_model(
				model, dataloader, show_progress_bar=True, nesting_list=NESTING_LIST, tta=args.tta, imagenetA=args.dataset == 'A', imagenetR=args.dataset == 'R')
	else:
		_, top1_acc, top5_acc, total_time, num_images, m_score_dict, softmax_probs, gt, logits = evaluate_model(
				model, dataloader, show_progress_bar=True, nesting_list=None, tta=args.tta, imagenetA=args.dataset == 'A', imagenetR=args.dataset == 'R')

	tqdm.write('Evaluated {} images'.format(num_images))
	confidence, predictions = torch.max(softmax_probs, dim=-1)
	if is_nested_model:
		metric_rows = []
		for i, nesting in enumerate(NESTING_LIST):
			metric_rows.append({
				'rep_size': int(nesting),
				'top1': float(top1_acc[nesting]),
				'top5': float(top5_acc[nesting])
			})
			print("Rep. Size", "\t", nesting, "\n")
			tqdm.write('    Top-1 accuracy for {} : {:.2f}'.format(nesting, 100.0 * top1_acc[nesting]))
			tqdm.write('    Top-5 accuracy for {} : {:.2f}'.format(nesting, 100.0 * top5_acc[nesting]))
			tqdm.write('    Total time: {:.1f}  (average time per image: {:.2f} ms)'.format(total_time, 1000.0 * total_time / num_images))
	else:
		metric_rows = [{
			'rep_size': int(args.rep_size),
			'top1': float(top1_acc),
			'top5': float(top5_acc)
		}]
		print("Rep. Size", "\t", args.rep_size, "\n")
		tqdm.write('    Evaluated {} images'.format(num_images))
		tqdm.write('    Top-1 accuracy: {:.2f}%'.format(100.0 * top1_acc))
		tqdm.write('    Top-5 accuracy: {:.2f}%'.format(100.0 * top5_acc))
		tqdm.write('    Total time: {:.1f}  (average time per image: {:.2f} ms)'.format(total_time, 1000.0 * total_time / num_images))

	metrics = {
		'dataset': args.dataset,
		'checkpoint': args.path,
		'mrl': bool(args.mrl),
		'bor_mrl': bool(args.bor_mrl),
		'bor_block_mrl': bool(args.bor_block_mrl),
		'bor_mode': args.bor_mode,
		'bor_orthogonal_map': args.bor_orthogonal_map,
		'bor_use_trivialization': bool(args.bor_use_trivialization),
		'efficient': bool(args.efficient),
		'rep_size': int(args.rep_size),
		'tta': bool(args.tta),
		'seed': int(args.seed),
		'deterministic': bool(args.deterministic),
		'num_images': int(num_images),
		'total_time': float(total_time),
		'metrics': metric_rows
	}
	if args.metrics_output:
		save_metrics(metrics, args.metrics_output)


	# saving torch tensor for model analysis... 
	if args.save_logits or args.save_softmax or args.save_predictions:
		save_string = f"mrl={args.mrl}_bor_mrl={args.bor_mrl}_bor_block_mrl={args.bor_block_mrl}_efficient={args.efficient}_dataset={args.dataset}_tta={args.tta}"
		if args.save_logits:
			torch.save(logits, save_string+"_logits.pth")
		if args.save_predictions:
			torch.save(predictions, save_string+"_predictions.pth")
		if args.save_softmax:
			torch.save(softmax_probs, save_string+"_softmax.pth")

	if args.save_gt:
		torch.save(gt, f"gt_dataset={args.dataset}.pth")


# Image Retrieval Inference
else:
	if args.dataset.lower() == 'cifar100':
		train_dataset = datasets.CIFAR100(root=str(data_root), train=True, transform=test_transform, download=True)
		test_dataset = datasets.CIFAR100(root=str(data_root), train=False, transform=test_transform, download=True)
	elif args.dataset == '1K':
		train_dataset = datasets.ImageFolder(str(data_root / "train"), transform=test_transform)
		test_dataset = datasets.ImageFolder(str(data_root / "val"), transform=test_transform)
	elif args.dataset == 'V2':
		train_dataset = None  # V2 has only a test set
		test_dataset = load_imagenet_v2(test_transform)
	elif args.dataset == '4K':
		train_path = 'path_to_imagenet4k_train/'
		test_path = 'path_to_imagenet4k_test/'
		train_dataset = datasets.ImageFolder(train_path, transform=test_transform)
		test_dataset = datasets.ImageFolder(test_path, transform=test_transform)
	else:
		print("Error: unsupported dataset!")

	database_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=False)
	queryset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=False)

	config = args.dataset + "_val_mrl" + str(int(args.mrl)) + "_e" + str(int(args.efficient)) + "_ff" + str(int(args.rep_size))
	print("Retrieval Config: " + config)
	generate_retrieval_data(model, queryset_loader, config, args.random_sample_dim, args.rep_size, args.retrieval_array_path)
	config = args.dataset + "_train_mrl" + str(int(args.mrl)) + "_e" + str(int(args.efficient)) + "_ff" + str(int(args.rep_size))
	print("Retrieval Config: " + config)
	generate_retrieval_data(model, database_loader, config, args.random_sample_dim, args.rep_size, args.retrieval_array_path)
