import torch.nn as nn
from torchvision import models


CIFAR100_NUM_CLASSES = 100
CIFAR100_INPUT_SIZE = 32
CIFAR100_FEATURE_DIMS = {
	"resnet18": 512,
	"resnet34": 512,
	"resnet50": 2048,
	"resnet101": 2048,
	"resnet152": 2048,
}


def make_torchvision_model(arch, pretrained=False):
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


def apply_cifar_resnet_stem(model):
	"""Use the CIFAR ResNet stem: 3x3 stride-1 conv and no initial maxpool."""
	if not hasattr(model, "conv1") or not hasattr(model, "maxpool"):
		raise ValueError("CIFAR stem requires a ResNet-like model with conv1 and maxpool")
	model.conv1 = nn.Conv2d(
		3,
		64,
		kernel_size=3,
		stride=1,
		padding=1,
		bias=False,
	)
	model.maxpool = nn.Identity()
	return model


def should_use_cifar_stem(dataset, arch):
	return str(dataset).lower() == "cifar100" and str(arch).lower() == "resnet18"


def maybe_apply_cifar_stem(model, dataset, arch):
	if should_use_cifar_stem(dataset, arch):
		apply_cifar_resnet_stem(model)
	return model


def build_power2_prefix_dims(feature_dim, nesting_start=3):
	start_dim = 2 ** int(nesting_start)
	feature_dim = int(feature_dim)
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


def parse_prefix_dims(spec, feature_dim, nesting_start=3):
	if spec is None or str(spec).strip() == "":
		return build_power2_prefix_dims(feature_dim, nesting_start)
	dims = sorted(set(int(part.strip()) for part in str(spec).split(",") if part.strip()))
	feature_dim = int(feature_dim)
	if not dims:
		raise ValueError("prefix dims cannot be empty")
	if dims[-1] != feature_dim:
		dims.append(feature_dim)
	if any(dim <= 0 or dim > feature_dim for dim in dims):
		raise ValueError(f"Invalid prefix dims {dims} for feature_dim={feature_dim}")
	return dims
