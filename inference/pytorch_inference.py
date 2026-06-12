"""
Evaluate MRL checkpoints and dump retrieval feature arrays.
"""
import json
import multiprocessing as mp
import os
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append("../")

import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet50
from tqdm import tqdm

from MRL import FixedFeatureLayer, MRL_Linear_Layer
from utils import apply_blurpool, evaluate_model, generate_retrieval_data, get_ckpt, load_from_old_ckpt


def configure_multiprocessing_start_method():
	if sys.version_info < (3, 14) or os.name != "posix" or sys.platform == "darwin":
		return
	start_method = os.environ.get("MRL_MULTIPROCESSING_START_METHOD", "fork")
	if not start_method:
		return
	if start_method not in mp.get_all_start_methods():
		raise ValueError(
			f"Unsupported MRL_MULTIPROCESSING_START_METHOD={start_method!r}. "
			f"Available methods: {mp.get_all_start_methods()}"
		)
	current = mp.get_start_method(allow_none=True)
	if current != start_method:
		mp.set_start_method(start_method, force=True)


configure_multiprocessing_start_method()

try:
	from imagenetv2_pytorch import ImageNetV2Dataset
except ImportError:
	ImageNetV2Dataset = None


BATCH_SIZE = 256
IMG_SIZE = 256
CENTER_CROP_SIZE = 224
NESTING_LIST = [2**i for i in range(3, 12)]
ROOT = "../../IMAGENET/"
DATASET_CONFIGS = {
	"imagenet": {
		"num_classes": 1000,
		"img_size": IMG_SIZE,
		"center_crop_size": CENTER_CROP_SIZE,
		"mean": [0.485, 0.456, 0.406],
		"std": [0.229, 0.224, 0.225],
	},
	"cifar100": {
		"num_classes": 100,
		"img_size": 32,
		"center_crop_size": 32,
		"mean": [0.5071, 0.4867, 0.4408],
		"std": [0.2675, 0.2565, 0.2761],
	},
}


def get_dataset_config(dataset):
	return DATASET_CONFIGS["cifar100"] if dataset.lower() == "cifar100" else DATASET_CONFIGS["imagenet"]


def load_imagenet_v2(transform):
	if ImageNetV2Dataset is None:
		raise ImportError("imagenetv2_pytorch is required for --dataset V2")
	return ImageNetV2Dataset("matched-frequency", transform=transform)


def set_eval_reproducibility(seed, deterministic):
	os.environ.setdefault("PYTHONHASHSEED", str(seed))
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	if deterministic:
		os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True
		if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
			torch.backends.cuda.matmul.allow_tf32 = False
		if hasattr(torch.backends.cudnn, "allow_tf32"):
			torch.backends.cudnn.allow_tf32 = False
		if hasattr(torch, "use_deterministic_algorithms"):
			try:
				torch.use_deterministic_algorithms(True, warn_only=True)
			except TypeError:
				torch.use_deterministic_algorithms(True)
	else:
		torch.backends.cudnn.benchmark = True
		if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
			torch.backends.cuda.matmul.allow_tf32 = True
		if hasattr(torch.backends.cudnn, "allow_tf32"):
			torch.backends.cudnn.allow_tf32 = True


def seed_worker(worker_id):
	worker_seed = torch.initial_seed() % 2**32
	random.seed(worker_seed + worker_id)
	torch.manual_seed(worker_seed + worker_id)


def save_metrics(metrics, path):
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w") as handle:
		json.dump(metrics, handle, indent=2)


def method_name(args):
	if args.mrl or args.efficient:
		return "mrl_e" if args.efficient else "mrl"
	return "fixed_feature"


def make_loader(dataset, workers, seed):
	generator = torch.Generator()
	generator.manual_seed(seed)
	kwargs = {
		"batch_size": BATCH_SIZE,
		"num_workers": workers,
		"shuffle": False,
		"worker_init_fn": seed_worker,
		"generator": generator,
		"pin_memory": torch.cuda.is_available(),
		"persistent_workers": workers > 0,
	}
	return torch.utils.data.DataLoader(dataset, **kwargs)


def classification_dataset(args, data_root, test_transform):
	if args.dataset.lower() == "cifar100":
		print("Loading CIFAR-100 test set")
		return datasets.CIFAR100(root=str(data_root), train=False, transform=test_transform, download=True)
	if args.dataset == "V2":
		print("Loading ImageNetV2")
		return load_imagenet_v2(test_transform)
	if args.dataset == "A":
		print("Loading ImageNet-A")
		return torchvision.datasets.ImageFolder(str(data_root / "imagenet-a"), transform=test_transform)
	if args.dataset == "R":
		print("Loading ImageNet-R")
		return torchvision.datasets.ImageFolder(str(data_root / "imagenet-r_"), transform=test_transform)
	if args.dataset == "sketch":
		print("Loading ImageNet-Sketch")
		return torchvision.datasets.ImageFolder(str(data_root / "sketch"), transform=test_transform)
	print("Loading ImageNet-1K val set")
	return torchvision.datasets.ImageFolder(str(data_root / "val"), transform=test_transform)


def retrieval_datasets(args, data_root, test_transform):
	if args.dataset.lower() == "cifar100":
		train_dataset = datasets.CIFAR100(root=str(data_root), train=True, transform=test_transform, download=True)
		test_dataset = datasets.CIFAR100(root=str(data_root), train=False, transform=test_transform, download=True)
	elif args.dataset == "1K":
		train_dataset = datasets.ImageFolder(str(data_root / "train"), transform=test_transform)
		test_dataset = datasets.ImageFolder(str(data_root / "val"), transform=test_transform)
	elif args.dataset == "V2":
		train_dataset = datasets.ImageFolder(str(data_root / "train"), transform=test_transform)
		test_dataset = load_imagenet_v2(test_transform)
	elif args.dataset == "4K":
		train_path = "path_to_imagenet4k_train/"
		test_path = "path_to_imagenet4k_test/"
		train_dataset = datasets.ImageFolder(train_path, transform=test_transform)
		test_dataset = datasets.ImageFolder(test_path, transform=test_transform)
	else:
		raise ValueError(f"Unsupported retrieval dataset: {args.dataset}")
	return train_dataset, test_dataset


def build_model(args, num_classes, device):
	model = resnet50(weights=None)
	is_mrl_model = args.mrl or args.efficient
	if args.old_ckpt:
		if is_mrl_model:
			model = load_from_old_ckpt(model, args.efficient, NESTING_LIST, num_classes=num_classes)
		else:
			model.fc = FixedFeatureLayer(args.rep_size, num_classes)
	elif is_mrl_model:
		model.fc = MRL_Linear_Layer(NESTING_LIST, num_classes=num_classes, efficient=args.efficient)
	else:
		model.fc = FixedFeatureLayer(args.rep_size, num_classes)

	apply_blurpool(model)
	model.load_state_dict(get_ckpt(args.path))
	model = model.to(device)
	if device.type == "cuda":
		model = model.to(memory_format=torch.channels_last)
	model.eval()
	return model


parser = ArgumentParser()
parser.add_argument("--efficient", action="store_true", help="use MRL-E")
parser.add_argument("--mrl", action="store_true", help="use MRL")
parser.add_argument("--rep_size", type=int, default=2048, help="representation size for fixed-feature model")
parser.add_argument("--path", type=str, required=True, help="path to .pt model checkpoint")
parser.add_argument("--old_ckpt", action="store_true", help="load original MRL checkpoint naming")
parser.add_argument("--workers", type=int, default=12, help="number of dataloader workers")
parser.add_argument("--tta", action="store_true", help="left-right flip test-time augmentation")
parser.add_argument("--dataset", type=str, default="V1", help="Benchmarks: V1/V2/A/Sketch/R/CIFAR100")
parser.add_argument("--data_root", type=str, default=ROOT, help="ImageNet-style root or torchvision CIFAR root")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--deterministic", action="store_true", help="enable deterministic PyTorch/CUDA behavior")
parser.add_argument("--metrics_output", type=str, default="", help="optional JSON file for evaluation metrics")
parser.add_argument("--save_logits", action="store_true", help="save logits for model analysis")
parser.add_argument("--save_softmax", action="store_true", help="save softmax probabilities for model analysis")
parser.add_argument("--save_gt", action="store_true", help="save ground truth for model analysis")
parser.add_argument("--save_predictions", action="store_true", help="save predicted labels for model analysis")
parser.add_argument("--retrieval", action="store_true", help="dump retrieval feature arrays")
parser.add_argument("--random_sample_dim", type=int, default=4202000, help="optional database random sample size")
parser.add_argument("--retrieval_array_path", default="", type=str, help="path to save retrieval arrays")

args = parser.parse_args()
set_eval_reproducibility(args.seed, args.deterministic)
dataset_config = get_dataset_config(args.dataset)
num_classes = dataset_config["num_classes"]
data_root = Path(args.data_root)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(args, num_classes, device)
is_nested_model = args.mrl or args.efficient

normalize = transforms.Normalize(mean=dataset_config["mean"], std=dataset_config["std"])
test_transform = transforms.Compose([
	transforms.Resize(dataset_config["img_size"]),
	transforms.CenterCrop(dataset_config["center_crop_size"]),
	transforms.ToTensor(),
	normalize,
])

if not args.retrieval:
	dataset = classification_dataset(args, data_root, test_transform)
	dataloader = make_loader(dataset, args.workers, args.seed)
	nesting_list = NESTING_LIST if is_nested_model else None
	_, top1_acc, top5_acc, total_time, num_images, _, softmax_probs, gt, logits = evaluate_model(
		model,
		dataloader,
		show_progress_bar=True,
		nesting_list=nesting_list,
		tta=args.tta,
		imagenetA=args.dataset == "A",
		imagenetR=args.dataset == "R",
	)

	tqdm.write(f"Evaluated {num_images} images")
	confidence, predictions = torch.max(softmax_probs, dim=-1)
	if is_nested_model:
		metric_rows = []
		for nesting in NESTING_LIST:
			metric_rows.append({
				"rep_size": int(nesting),
				"top1": float(top1_acc[nesting]),
				"top5": float(top5_acc[nesting]),
			})
			print("Rep. Size", "\t", nesting, "\n")
			tqdm.write(f"    Top-1 accuracy for {nesting} : {100.0 * top1_acc[nesting]:.2f}")
			tqdm.write(f"    Top-5 accuracy for {nesting} : {100.0 * top5_acc[nesting]:.2f}")
	else:
		metric_rows = [{
			"rep_size": int(args.rep_size),
			"top1": float(top1_acc),
			"top5": float(top5_acc),
		}]
		print("Rep. Size", "\t", args.rep_size, "\n")
		tqdm.write(f"    Top-1 accuracy: {100.0 * top1_acc:.2f}%")
		tqdm.write(f"    Top-5 accuracy: {100.0 * top5_acc:.2f}%")

	tqdm.write(
		f"    Total time: {total_time:.1f} "
		f"(average time per image: {1000.0 * total_time / num_images:.2f} ms)"
	)
	metrics = {
		"dataset": args.dataset,
		"checkpoint": args.path,
		"method": method_name(args),
		"mrl": bool(args.mrl or args.efficient),
		"efficient": bool(args.efficient),
		"rep_size": int(args.rep_size),
		"tta": bool(args.tta),
		"seed": int(args.seed),
		"deterministic": bool(args.deterministic),
		"num_images": int(num_images),
		"total_time": float(total_time),
		"metrics": metric_rows,
	}
	if args.metrics_output:
		save_metrics(metrics, args.metrics_output)

	save_string = (
		f"method={method_name(args)}_efficient={args.efficient}_"
		f"dataset={args.dataset}_tta={args.tta}"
	)
	if args.save_logits:
		torch.save(logits, save_string + "_logits.pth")
	if args.save_predictions:
		torch.save(predictions, save_string + "_predictions.pth")
	if args.save_softmax:
		torch.save(softmax_probs, save_string + "_softmax.pth")
	if args.save_gt:
		torch.save(gt, f"gt_dataset={args.dataset}.pth")

else:
	print("Retrieval arrays use raw avgpool encoder features.")
	train_dataset, test_dataset = retrieval_datasets(args, data_root, test_transform)
	database_loader = make_loader(train_dataset, args.workers, args.seed)
	queryset_loader = make_loader(test_dataset, args.workers, args.seed)

	mrl_flag = int(args.mrl or args.efficient)
	config = args.dataset + "_val_mrl" + str(mrl_flag) + "_e" + str(int(args.efficient)) + "_ff" + str(int(args.rep_size))
	print("Retrieval Config: " + config)
	generate_retrieval_data(model, queryset_loader, config, args.random_sample_dim, args.rep_size, args.retrieval_array_path)
	config = args.dataset + "_train_mrl" + str(mrl_flag) + "_e" + str(int(args.efficient)) + "_ff" + str(int(args.rep_size))
	print("Retrieval Config: " + config)
	generate_retrieval_data(model, database_loader, config, args.random_sample_dim, args.rep_size, args.retrieval_array_path)
