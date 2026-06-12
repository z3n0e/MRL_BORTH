from typing import List

import torch
import torch.nn as nn


"""
Core Matryoshka Representation Learning layers and losses.

This file intentionally keeps only the standard MRL/MRL-E heads used by the
CIFAR-100 and ImageNet training code. Experimental method variants were removed
so the method surface stays small and easy to audit.
"""


MRL_LOSS_MODES = {"all", "sampled_prefix"}
SAMPLED_PREFIX_DISTRIBUTIONS = {"uniform", "inverse_dim", "inverse_sqrt_dim"}
PREFIX_MASK_SCALES = {"inverted", "none"}


def validate_nesting_list(nesting_list):
	dims = [int(dim) for dim in nesting_list]
	if len(dims) == 0:
		raise ValueError("nesting_list must contain at least one positive dimension")
	if any(dim <= 0 for dim in dims):
		raise ValueError("nesting_list dimensions must all be positive")
	if any(dim <= prev for prev, dim in zip(dims, dims[1:])):
		raise ValueError("nesting_list must be strictly increasing")
	return dims


def block_widths_from_nesting_list(nesting_list):
	dims = validate_nesting_list(nesting_list)
	return [dims[0]] + [dim - prev_dim for prev_dim, dim in zip(dims, dims[1:])]


def validate_prefix_mask(prefix_mask_prob, prefix_mask_scale):
	prefix_mask_prob = float(prefix_mask_prob)
	if not 0.0 <= prefix_mask_prob < 1.0:
		raise ValueError("prefix_mask_prob must be in [0, 1)")
	if prefix_mask_scale not in PREFIX_MASK_SCALES:
		raise ValueError(
			f"prefix_mask_scale must be one of {sorted(PREFIX_MASK_SCALES)}, "
			f"got {prefix_mask_scale!r}"
		)
	return prefix_mask_prob, prefix_mask_scale


def mask_previous_prefix_features(features, previous_dim, mask_prob, scale="inverted"):
	"""
	Mask inherited coordinates for a larger MRL prefix during training.

	For a prefix h[:d_i], only h[:d_{i-1}] is randomly masked; the new block
	h[d_{i-1}:d_i] remains visible. This matches the MNIST neural-collapse
	implementation while supporting batched CIFAR/ImageNet features.
	"""
	previous_dim = int(previous_dim)
	mask_prob, scale = validate_prefix_mask(mask_prob, scale)
	if previous_dim <= 0 or mask_prob == 0.0:
		return features
	if previous_dim > features.shape[1]:
		raise ValueError(
			f"previous_dim={previous_dim} exceeds feature width {features.shape[1]}"
		)

	keep = torch.rand(features.shape[0], previous_dim, device=features.device) >= mask_prob
	mask = keep.to(dtype=features.dtype)
	if scale == "inverted":
		mask = mask / (1.0 - mask_prob)

	masked = features.clone()
	masked[:, :previous_dim] = masked[:, :previous_dim] * mask
	return masked


def mrl_sampling_probabilities(nesting_list, distribution="uniform"):
	if distribution not in SAMPLED_PREFIX_DISTRIBUTIONS:
		raise ValueError(
			f"sampled prefix distribution must be one of {sorted(SAMPLED_PREFIX_DISTRIBUTIONS)}, "
			f"got {distribution!r}"
		)

	dims = torch.tensor(validate_nesting_list(nesting_list), dtype=torch.float32)
	if distribution == "uniform":
		weights = torch.ones_like(dims)
	elif distribution == "inverse_dim":
		weights = 1.0 / dims
	else:
		weights = torch.rsqrt(dims)

	return weights / weights.sum()


def mrl_output_tuple(output):
	if isinstance(output, torch.Tensor):
		return tuple(output.unbind(dim=0))
	return tuple(output)


class Matryoshka_CE_Loss(nn.Module):
	def __init__(
		self,
		relative_importance: List[float] = None,
		mrl_loss_mode="all",
		nesting_list=None,
		sampled_prefix_distribution="uniform",
		**kwargs,
	):
		super().__init__()
		if mrl_loss_mode not in MRL_LOSS_MODES:
			raise ValueError(
				f"mrl_loss_mode must be one of {sorted(MRL_LOSS_MODES)}, "
				f"got {mrl_loss_mode!r}"
			)
		if sampled_prefix_distribution not in SAMPLED_PREFIX_DISTRIBUTIONS:
			raise ValueError(
				f"sampled_prefix_distribution must be one of "
				f"{sorted(SAMPLED_PREFIX_DISTRIBUTIONS)}, "
				f"got {sampled_prefix_distribution!r}"
			)

		self.criterion = nn.CrossEntropyLoss(**kwargs)
		self.relative_importance = relative_importance
		self.mrl_loss_mode = mrl_loss_mode
		self.nesting_list = None if nesting_list is None else validate_nesting_list(nesting_list)
		self.sampled_prefix_distribution = sampled_prefix_distribution
		self.last_sampled_idx = None
		self.last_sampled_dim = None
		self.last_selected_ce = None

		if self.mrl_loss_mode == "sampled_prefix":
			if self.nesting_list is None:
				raise ValueError("nesting_list is required when mrl_loss_mode='sampled_prefix'")
			probs = mrl_sampling_probabilities(self.nesting_list, self.sampled_prefix_distribution)
			self.register_buffer("sampled_prefix_probs", probs)
			self.register_buffer("sample_counts", torch.zeros(len(self.nesting_list), dtype=torch.long))
		else:
			self.register_buffer("sampled_prefix_probs", torch.empty(0, dtype=torch.float32))
			self.register_buffer("sample_counts", torch.empty(0, dtype=torch.long))

	def per_prefix_losses(self, output, target):
		outputs = mrl_output_tuple(output)
		if len(outputs) == 0:
			raise ValueError("Matryoshka_CE_Loss requires at least one prefix output")
		return torch.stack([self.criterion(output_i, target) for output_i in outputs])

	def relative_importance_weights(self, losses):
		if self.relative_importance is None:
			return losses.new_ones(losses.shape)
		weights = losses.new_tensor(self.relative_importance)
		if weights.numel() != losses.numel():
			raise ValueError(
				f"relative_importance has {weights.numel()} entries, "
				f"but there are {losses.numel()} prefix losses"
			)
		return weights

	def weighted_all_loss(self, losses):
		return (self.relative_importance_weights(losses) * losses).sum()

	def sample_prefix_idx(self):
		if self.sampled_prefix_probs.numel() == 0:
			raise RuntimeError("sample_prefix_idx() requires mrl_loss_mode='sampled_prefix'")
		return int(torch.multinomial(self.sampled_prefix_probs.cpu(), 1).item())

	def sampled_prefix_probabilities(self):
		return self.sampled_prefix_probs.detach().cpu().tolist()

	def sample_counts_list(self):
		return self.sample_counts.detach().cpu().tolist()

	def forward(self, output, target):
		outputs = mrl_output_tuple(output)
		if self.mrl_loss_mode == "all":
			return self.weighted_all_loss(self.per_prefix_losses(outputs, target))

		if len(outputs) != self.sampled_prefix_probs.numel():
			raise ValueError(
				f"Expected {self.sampled_prefix_probs.numel()} prefix logits, "
				f"got {len(outputs)}"
			)

		sampled_idx = self.sample_prefix_idx()
		selected_ce = self.criterion(outputs[sampled_idx], target)
		dummy = selected_ce.new_zeros(())
		for output_i in outputs:
			dummy = dummy + 0.0 * output_i.sum()

		with torch.no_grad():
			self.sample_counts[sampled_idx] += 1

		self.last_sampled_idx = sampled_idx
		self.last_sampled_dim = self.nesting_list[sampled_idx]
		self.last_selected_ce = selected_ce.detach()
		return selected_ce + dummy


class MRL_Linear_Layer(nn.Module):
	def __init__(
		self,
		nesting_list: List,
		num_classes=1000,
		efficient=False,
		prefix_mask_prob=0.0,
		prefix_mask_scale="inverted",
		**kwargs,
	):
		super().__init__()
		self.nesting_list = validate_nesting_list(nesting_list)
		self.num_classes = int(num_classes)
		self.efficient = bool(efficient)
		self.prefix_mask_prob, self.prefix_mask_scale = validate_prefix_mask(
			prefix_mask_prob,
			prefix_mask_scale,
		)
		self.last_prefixes = None
		self.last_raw_prefixes = None

		if self.efficient:
			setattr(
				self,
				"nesting_classifier_0",
				nn.Linear(self.nesting_list[-1], self.num_classes, **kwargs),
			)
		else:
			for i, num_feat in enumerate(self.nesting_list):
				setattr(
					self,
					f"nesting_classifier_{i}",
					nn.Linear(num_feat, self.num_classes, **kwargs),
				)

	def reset_parameters(self):
		if self.efficient:
			self.nesting_classifier_0.reset_parameters()
		else:
			for i in range(len(self.nesting_list)):
				getattr(self, f"nesting_classifier_{i}").reset_parameters()

	def _classifier_input(self, x, dim, previous_dim):
		prefix = x[:, :dim]
		if self.training and self.prefix_mask_prob > 0.0:
			return mask_previous_prefix_features(
				prefix,
				previous_dim=previous_dim,
				mask_prob=self.prefix_mask_prob,
				scale=self.prefix_mask_scale,
			)
		return prefix

	def forward(self, x):
		if x.dim() != 2:
			raise ValueError(f"MRL_Linear_Layer expects [B, D] input, got shape {tuple(x.shape)}")
		if x.shape[1] < self.nesting_list[-1]:
			raise ValueError(
				f"Expected at least {self.nesting_list[-1]} features, got {x.shape[1]}"
			)

		raw_prefixes = [x[:, :num_feat] for num_feat in self.nesting_list]
		prefixes = []
		logits = []
		previous_dim = 0
		for i, num_feat in enumerate(self.nesting_list):
			prefix = self._classifier_input(x, num_feat, previous_dim)
			prefixes.append(prefix)
			if self.efficient:
				classifier = self.nesting_classifier_0
				weight = classifier.weight[:, :num_feat]
				logit = torch.matmul(prefix, weight.t())
				if classifier.bias is not None:
					logit = logit + classifier.bias
			else:
				logit = getattr(self, f"nesting_classifier_{i}")(prefix)
			logits.append(logit)
			previous_dim = num_feat

		self.last_raw_prefixes = raw_prefixes
		self.last_prefixes = prefixes
		return tuple(logits)


class FixedFeatureLayer(nn.Linear):
	"""
	Compatibility layer for fixed-feature checkpoints/evaluation.
	"""

	def __init__(self, in_features, out_features, **kwargs):
		super().__init__(in_features, out_features, **kwargs)

	def forward(self, x):
		prefix = x[:, :self.in_features]
		if self.bias is None:
			return torch.matmul(prefix, self.weight.t())
		return torch.matmul(prefix, self.weight.t()) + self.bias
