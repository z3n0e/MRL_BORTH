import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from contextlib import nullcontext

try:
	from torch.nn.utils.parametrizations import orthogonal
except ImportError:
	orthogonal = None

'''
Loss function for Matryoshka Representation Learning 
'''

MRL_LOSS_MODES = {"all", "sampled_prefix"}
SAMPLED_PREFIX_DISTRIBUTIONS = {"uniform", "inverse_dim", "inverse_sqrt_dim"}
MRL_CONFLICT_MODES = {"none", "block_cascade"}
T_ORTHOGONAL_MAP_ALIASES = {
	"matrix_exp": "matrix_exp",
	"householder": "householder",
	"household": "householder",
}


def mrl_sampling_probabilities(nesting_list, distribution="uniform"):
	if distribution not in SAMPLED_PREFIX_DISTRIBUTIONS:
		raise ValueError(
			f"sampled prefix distribution must be one of {sorted(SAMPLED_PREFIX_DISTRIBUTIONS)}, "
			f"got {distribution!r}"
		)

	dims = torch.tensor([int(dim) for dim in nesting_list], dtype=torch.float32)
	if dims.numel() == 0:
		raise ValueError("nesting_list must contain at least one positive dimension")
	if torch.any(dims <= 0):
		raise ValueError("nesting_list dimensions must all be positive")

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


def mrl_gradient_conflict_stats(losses, shared_tensor, nesting_list=None, eps=1e-12):
	if shared_tensor is None:
		raise ValueError("shared_tensor is required for MRL gradient conflict measurement")

	loss_items = list(losses.unbind(dim=0)) if isinstance(losses, torch.Tensor) else list(losses)
	if len(loss_items) == 0:
		raise ValueError("losses must contain at least one prefix loss")

	grads = []
	for loss in loss_items:
		grad = torch.autograd.grad(
			loss,
			shared_tensor,
			retain_graph=True,
			allow_unused=True,
		)[0]
		if grad is None:
			grad = torch.zeros_like(shared_tensor)
		grads.append(grad.detach().float().reshape(-1))

	grad_matrix = torch.stack(grads)
	norms = grad_matrix.norm(p=2, dim=1).clamp_min(eps)
	cosine = (grad_matrix @ grad_matrix.t()) / (norms[:, None] * norms[None, :])
	cosine = cosine.clamp(min=-1.0, max=1.0)
	num_losses = cosine.shape[0]

	if num_losses > 1:
		pair_i, pair_j = torch.triu_indices(num_losses, num_losses, offset=1, device=cosine.device)
		pair_cosine = cosine[pair_i, pair_j]
		conflicting = pair_cosine < 0
		worst_pos = int(pair_cosine.argmin().item())
		worst_i = int(pair_i[worst_pos].item())
		worst_j = int(pair_j[worst_pos].item())
		conflict_count = int(conflicting.sum().item())
		pair_count = int(pair_cosine.numel())
		negative_mean = (
			float(pair_cosine[conflicting].mean().item())
			if conflict_count > 0 else 0.0
		)
		mean_cosine = float(pair_cosine.mean().item())
		min_cosine = float(pair_cosine.min().item())
	else:
		worst_i = worst_j = 0
		conflict_count = 0
		pair_count = 0
		negative_mean = 0.0
		mean_cosine = 0.0
		min_cosine = 0.0

	stats = {
		"mrl_grad_conflict_mean_cosine": mean_cosine,
		"mrl_grad_conflict_min_cosine": min_cosine,
		"mrl_grad_conflict_negative_mean_cosine": negative_mean,
		"mrl_grad_conflict_fraction": float(conflict_count / pair_count) if pair_count else 0.0,
		"mrl_grad_conflict_count": conflict_count,
		"mrl_grad_conflict_pair_count": pair_count,
		"mrl_grad_conflict_worst_i": worst_i,
		"mrl_grad_conflict_worst_j": worst_j,
		"mrl_grad_conflict_cosine_matrix": cosine.cpu().tolist(),
	}

	if nesting_list is not None and len(nesting_list) == num_losses:
		stats["mrl_grad_conflict_worst_dim_i"] = int(nesting_list[worst_i])
		stats["mrl_grad_conflict_worst_dim_j"] = int(nesting_list[worst_j])

	return stats


def mrl_relative_importance_weights(losses, relative_importance=None):
	loss_items = list(losses.unbind(dim=0)) if isinstance(losses, torch.Tensor) else list(losses)
	if len(loss_items) == 0:
		raise ValueError("losses must contain at least one prefix loss")

	ref = loss_items[0]
	if relative_importance is None:
		return ref.new_ones(len(loss_items))

	weights = ref.new_tensor(relative_importance)
	if weights.numel() != len(loss_items):
		raise ValueError(
			f"relative_importance has {weights.numel()} entries, "
			f"but there are {len(loss_items)} prefix losses"
		)
	return weights


def mrl_per_prefix_feature_grads(losses, shared_tensor):
	if shared_tensor is None:
		raise ValueError("shared_tensor is required for MRL conflict gating")

	loss_items = list(losses.unbind(dim=0)) if isinstance(losses, torch.Tensor) else list(losses)
	if len(loss_items) == 0:
		raise ValueError("losses must contain at least one prefix loss")

	grads = []
	for loss in loss_items:
		grad = torch.autograd.grad(
			loss,
			shared_tensor,
			retain_graph=True,
			allow_unused=True,
		)[0]
		if grad is None:
			grad = torch.zeros_like(shared_tensor)
		grads.append(grad.detach().clone())
	return grads


def block_cascade_conflict_gating(feature_grads, nesting_list, alpha=0.5, eps=1e-8):
	if len(feature_grads) != len(nesting_list):
		raise ValueError(
			f"Expected one feature gradient per nesting dim: "
			f"{len(feature_grads)} gradients for {len(nesting_list)} dims"
		)
	if len(feature_grads) == 0:
		raise ValueError("feature_grads must contain at least one gradient")
	if feature_grads[0].dim() != 2:
		raise ValueError(f"Expected feature gradients with shape [B, D], got {tuple(feature_grads[0].shape)}")

	dims = [int(dim) for dim in nesting_list]
	if any(dim <= 0 for dim in dims):
		raise ValueError("nesting_list dimensions must all be positive")
	if any(dim <= prev for prev, dim in zip(dims, dims[1:])):
		raise ValueError("nesting_list must be strictly increasing")
	if dims[-1] > feature_grads[0].shape[1]:
		raise ValueError(f"Largest nesting dim {dims[-1]} exceeds feature dim {feature_grads[0].shape[1]}")

	filtered = [grad.detach().clone() for grad in feature_grads]
	pair_stats = []
	cosine_values = []
	conflict_values = []
	projection_values = []

	for idx, (small_dim, large_dim) in enumerate(zip(dims, dims[1:])):
		for grad in filtered:
			if grad.shape != filtered[0].shape:
				raise ValueError("All feature gradients must have the same shape")

		g_small = filtered[idx][:, :small_dim].float()
		g_large_shared = filtered[idx + 1][:, :small_dim].float()
		dot = (g_large_shared * g_small).sum(dim=1, keepdim=True)
		small_norm_sq = g_small.pow(2).sum(dim=1, keepdim=True)
		large_norm = g_large_shared.norm(p=2, dim=1, keepdim=True)
		small_norm = small_norm_sq.sqrt()
		cosine = dot / (large_norm * small_norm + eps)
		conflicts = dot < 0

		coeff = torch.where(
			conflicts,
			float(alpha) * dot / (small_norm_sq + eps),
			torch.zeros_like(dot),
		)
		projection = coeff * g_small
		filtered[idx + 1][:, :small_dim] = (
			g_large_shared - projection
		).to(dtype=filtered[idx + 1].dtype)

		projection_magnitude = projection.norm(p=2, dim=1)
		pair_label = f"{large_dim}<-{small_dim}"
		pair_stats.append({
			"pair": pair_label,
			"small_dim": small_dim,
			"large_dim": large_dim,
			"mean_cosine": float(cosine.mean().detach().cpu().item()),
			"conflict_fraction": float(conflicts.float().mean().detach().cpu().item()),
			"avg_projection_magnitude": float(projection_magnitude.mean().detach().cpu().item()),
		})
		cosine_values.append(cosine.reshape(-1).detach())
		conflict_values.append(conflicts.float().reshape(-1).detach())
		projection_values.append(projection_magnitude.reshape(-1).detach())

	if cosine_values:
		all_cosines = torch.cat(cosine_values)
		all_conflicts = torch.cat(conflict_values)
		all_projections = torch.cat(projection_values)
		mean_cosine = float(all_cosines.mean().cpu().item())
		conflict_fraction = float(all_conflicts.mean().cpu().item())
		avg_projection = float(all_projections.mean().cpu().item())
	else:
		mean_cosine = 0.0
		conflict_fraction = 0.0
		avg_projection = 0.0

	stats = {
		"mrl_conflict_mode": "block_cascade",
		"mrl_conflict_mean_adjacent_cosine": mean_cosine,
		"mrl_conflict_fraction": conflict_fraction,
		"mrl_conflict_avg_projection_magnitude": avg_projection,
		"mrl_conflict_pairs": pair_stats,
	}
	return filtered, stats


def mrl_block_cascade_filtered_feature_gradient(
	losses,
	shared_tensor,
	nesting_list,
	relative_importance=None,
	alpha=0.5,
	eps=1e-8,
):
	feature_grads = mrl_per_prefix_feature_grads(losses, shared_tensor)
	filtered_grads, stats = block_cascade_conflict_gating(
		feature_grads,
		nesting_list,
		alpha=alpha,
		eps=eps,
	)
	weights = mrl_relative_importance_weights(losses, relative_importance)
	combined = torch.zeros_like(shared_tensor)
	for weight, grad in zip(weights, filtered_grads):
		combined = combined + weight.to(device=grad.device, dtype=grad.dtype) * grad
	return combined, stats


def _process_procrustes_features(x, center=True, normalize=True, eps=1e-6):
	x = x.float()
	if center:
		x = x - x.mean(dim=0, keepdim=True)
	if normalize:
		x = F.normalize(x, p=2, dim=1, eps=eps)
	return x


def _disabled_autocast_context(device):
	device_type = getattr(device, "type", str(device))
	if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
		try:
			return torch.amp.autocast(device_type=device_type, enabled=False)
		except (TypeError, RuntimeError):
			pass
	if device_type == "cuda" and hasattr(torch.cuda, "amp"):
		return torch.cuda.amp.autocast(enabled=False)
	return nullcontext()


def procrustes_cascade_distillation_loss(
	prefixes,
	normalize=True,
	center=True,
	detach_teacher=True,
	eps=1e-6,
	max_svd_dim=1024,
):
	prefixes = list(prefixes)
	if len(prefixes) == 0:
		raise ValueError("prefixes must contain at least one tensor")
	if len(prefixes) == 1:
		return prefixes[0].new_zeros(())

	losses = []
	for small, large in zip(prefixes, prefixes[1:]):
		if small.dim() != 2 or large.dim() != 2:
			raise ValueError("PCD prefixes must be 2D tensors with shape [B, d]")
		if small.shape[0] != large.shape[0]:
			raise ValueError("PCD adjacent prefixes must have the same batch size")
		if large.shape[1] < small.shape[1]:
			raise ValueError("PCD requires nondecreasing adjacent prefix dimensions")
		if max_svd_dim is not None and max(small.shape[1], large.shape[1]) > int(max_svd_dim):
			continue

		with _disabled_autocast_context(small.device):
			large_teacher = large.detach() if detach_teacher else large
			processed_small = _process_procrustes_features(
				small,
				center=center,
				normalize=normalize,
				eps=eps,
			)
			processed_large = _process_procrustes_features(
				large_teacher,
				center=center,
				normalize=normalize,
				eps=eps,
			)

			with torch.no_grad():
				a = processed_large.detach().float()
				b = processed_small.detach().float()
				m = (a.t() @ b).float()
				u, _, vh = torch.linalg.svd(m, full_matrices=False)
				q = u @ vh

			target = processed_large @ q.to(device=processed_large.device, dtype=processed_large.dtype)
			losses.append(F.mse_loss(processed_small, target))

	if not losses:
		return prefixes[0].new_zeros(())

	return torch.stack(losses).mean()


def adjacent_residual_prefix_cosine_stats(prefixes, nesting_list, eps=1e-8):
	prefixes = list(prefixes)
	dims = [int(dim) for dim in nesting_list]
	if len(prefixes) != len(dims):
		raise ValueError(
			f"Expected {len(dims)} prefixes for nesting_list, got {len(prefixes)}"
		)

	pair_stats = []
	cosine_values = []
	distance_values = []
	for idx, (prev_dim, dim) in enumerate(zip(dims, dims[1:])):
		residual_width = dim - prev_dim
		if residual_width != prev_dim:
			continue

		prefix = prefixes[idx + 1]
		if prefix.dim() != 2:
			raise ValueError("prefixes must be 2D tensors with shape [B, d]")
		if prefix.shape[1] < dim:
			raise ValueError(
				f"Prefix for dim {dim} has width {prefix.shape[1]}, expected at least {dim}"
			)

		previous = prefix[:, :prev_dim].float()
		residual = prefix[:, prev_dim:dim].float()
		cosine = F.cosine_similarity(residual, previous, dim=1, eps=eps)
		distance = 1.0 - cosine
		mean_cosine = float(cosine.detach().mean().cpu().item())
		mean_distance = float(distance.detach().mean().cpu().item())
		pair_label = f"{dim}<-{prev_dim}"

		pair_stats.append({
			"pair": pair_label,
			"small_dim": prev_dim,
			"large_dim": dim,
			"mean_cosine": mean_cosine,
			"mean_cosine_distance": mean_distance,
		})
		cosine_values.append(cosine.detach())
		distance_values.append(distance.detach())

	if not cosine_values:
		return {
			"residual_prefix_alignment_pair_count": 0,
			"residual_prefix_cosine_mean": 0.0,
			"residual_prefix_cosine_distance_mean": 0.0,
			"residual_prefix_cosine_min": 0.0,
			"residual_prefix_cosine_max": 0.0,
			"residual_prefix_alignment_pairs": [],
		}

	all_cosines = torch.cat([values.reshape(-1) for values in cosine_values])
	all_distances = torch.cat([values.reshape(-1) for values in distance_values])
	stats = {
		"residual_prefix_alignment_pair_count": len(pair_stats),
		"residual_prefix_cosine_mean": float(all_cosines.mean().cpu().item()),
		"residual_prefix_cosine_distance_mean": float(all_distances.mean().cpu().item()),
		"residual_prefix_cosine_min": float(all_cosines.min().cpu().item()),
		"residual_prefix_cosine_max": float(all_cosines.max().cpu().item()),
		"residual_prefix_alignment_pairs": pair_stats,
	}
	for pair in pair_stats:
		label = f"{pair['large_dim']}_vs_{pair['small_dim']}"
		stats[f"residual_prefix_cosine_{label}"] = pair["mean_cosine"]
		stats[f"residual_prefix_cosine_distance_{label}"] = pair["mean_cosine_distance"]
	return stats


class Matryoshka_CE_Loss(nn.Module):
	def __init__(self, relative_importance: List[float]=None,
	             mrl_loss_mode="all", nesting_list=None,
	             sampled_prefix_distribution="uniform", **kwargs):
		super(Matryoshka_CE_Loss, self).__init__()
		if mrl_loss_mode not in MRL_LOSS_MODES:
			raise ValueError(f"mrl_loss_mode must be one of {sorted(MRL_LOSS_MODES)}, got {mrl_loss_mode!r}")
		if sampled_prefix_distribution not in SAMPLED_PREFIX_DISTRIBUTIONS:
			raise ValueError(
				f"sampled_prefix_distribution must be one of {sorted(SAMPLED_PREFIX_DISTRIBUTIONS)}, "
				f"got {sampled_prefix_distribution!r}"
			)

		self.criterion = nn.CrossEntropyLoss(**kwargs)
		# relative importance shape: [G]
		self.relative_importance = relative_importance
		self.mrl_loss_mode = mrl_loss_mode
		self.nesting_list = None if nesting_list is None else [int(dim) for dim in nesting_list]
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

	def weighted_all_loss(self, losses):
		# Set relative_importance to 1 if not specified
		rel_importance = losses.new_ones(losses.shape) if self.relative_importance is None else losses.new_tensor(self.relative_importance)
		
		# Apply relative importance weights
		weighted_losses = rel_importance * losses
		return weighted_losses.sum()

	def sample_prefix_idx(self):
		if self.sampled_prefix_probs.numel() == 0:
			raise RuntimeError("sample_prefix_idx() requires mrl_loss_mode='sampled_prefix'")
		return int(torch.multinomial(self.sampled_prefix_probs.cpu(), 1).item())

	def sampled_prefix_probabilities(self):
		return self.sampled_prefix_probs.detach().cpu().tolist()

	def sample_counts_list(self):
		return self.sample_counts.detach().cpu().tolist()

	def forward(self, output, target):
		# output shape: [G granularities, N batch size, C number of classes]
		# target shape: [N batch size]
		outputs = mrl_output_tuple(output)

		if self.mrl_loss_mode == "all":
			# Calculate losses for each output and stack them. This is still O(N)
			losses = self.per_prefix_losses(outputs, target)
			return self.weighted_all_loss(losses)

		if len(outputs) != self.sampled_prefix_probs.numel():
			raise ValueError(
				f"Expected {self.sampled_prefix_probs.numel()} prefix logits, got {len(outputs)}"
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


def block_widths_from_nesting_list(nesting_list):
	if len(nesting_list) == 0:
		raise ValueError("nesting_list must contain at least one positive dimension")

	dims = [int(dim) for dim in nesting_list]
	if any(dim <= 0 for dim in dims):
		raise ValueError("nesting_list dimensions must all be positive")

	for prev_dim, dim in zip(dims, dims[1:]):
		if dim <= prev_dim:
			raise ValueError("nesting_list must be strictly increasing")

	return [dims[0]] + [dim - prev_dim for prev_dim, dim in zip(dims, dims[1:])]


def maybe_stop_gradient(x, stop_gradient):
	return x.detach() if stop_gradient else x


def resolve_stop_gradient_override(value):
	if value is None:
		return None
	value = int(value)
	if value not in {-1, 0, 1}:
		raise ValueError("stop-gradient override must be -1, 0, or 1")
	return None if value == -1 else bool(value)


def resolve_stop_gradient(value, default):
	if value is None:
		return bool(default)
	return bool(value)


def resolve_t_orthogonal_map(value):
	if value not in T_ORTHOGONAL_MAP_ALIASES:
		raise ValueError(
			"T orthogonal map must be one of ['householder', 'matrix_exp'], "
			f"got {value!r}"
		)
	return T_ORTHOGONAL_MAP_ALIASES[value]


def make_orthogonal_linear_layer(dim, mode="orthogonal",
                                 orthogonal_map="matrix_exp",
                                 use_trivialization=True,
                                 identity_init=False):
	allowed_modes = {"orthogonal", "frozen"}
	allowed_maps = {"matrix_exp", "cayley", "householder"}
	if mode not in allowed_modes:
		raise ValueError(f"mode must be one of {sorted(allowed_modes)}, got {mode!r}")
	if orthogonal_map not in allowed_maps:
		raise ValueError(f"orthogonal_map must be one of {sorted(allowed_maps)}, got {orthogonal_map!r}")
	if mode == "orthogonal" and orthogonal is None:
		raise RuntimeError(
			"torch.nn.utils.parametrizations.orthogonal is required "
			"for mode='orthogonal'"
		)

	layer = nn.Linear(dim, dim, bias=False)
	if mode == "frozen":
		with torch.no_grad():
			if identity_init:
				layer.weight.copy_(torch.eye(dim))
			else:
				nn.init.orthogonal_(layer.weight)
		layer.weight.requires_grad_(False)
		return layer

	with torch.no_grad():
		layer.weight.copy_(torch.eye(dim))
	try:
		return orthogonal(
			layer,
			"weight",
			orthogonal_map=orthogonal_map,
			use_trivialization=use_trivialization,
		)
	except TypeError as exc:
		raise RuntimeError(
			"Installed PyTorch does not support the requested "
			"orthogonal parametrization options. Use a modern "
			"PyTorch with torch.nn.utils.parametrizations.orthogonal."
		) from exc


class GatedResidualOrthogonalAdapter(nn.Module):
	def __init__(self, dim, mode="orthogonal",
	             orthogonal_map="matrix_exp", use_trivialization=True,
	             alpha_init=-3.0):
		super().__init__()
		self.in_features = int(dim)
		self.out_features = int(dim)
		self.mode = mode
		self.orthogonal_map = orthogonal_map
		self.use_trivialization = bool(use_trivialization)
		self.orthogonal = make_orthogonal_linear_layer(
			self.in_features,
			mode=mode,
			orthogonal_map=orthogonal_map,
			use_trivialization=use_trivialization,
		)
		self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_init)))

	@property
	def weight(self):
		return self.orthogonal.weight

	def alpha(self):
		return torch.sigmoid(self.alpha_logit)

	def forward(self, x):
		alpha = self.alpha().to(dtype=x.dtype, device=x.device)
		rotated = self.orthogonal(x)
		return (1.0 - alpha) * x + alpha * rotated

	def extra_repr(self):
		return (
			f"in_features={self.in_features}, out_features={self.out_features}, "
			f"mode={self.mode}, orthogonal_map={self.orthogonal_map}, "
			f"use_trivialization={self.use_trivialization}, "
			f"alpha={self.alpha().item():.6f}"
		)


class BlockOrthogonalLayer(nn.Module):
	def __init__(self, nesting_list, mode="orthogonal",
	             orthogonal_map="matrix_exp", use_trivialization=True,
	             stop_gradient=False):
		super().__init__()
		allowed_modes = {"orthogonal", "frozen"}
		allowed_maps = {"matrix_exp", "cayley", "householder"}
		if mode not in allowed_modes:
			raise ValueError(f"mode must be one of {sorted(allowed_modes)}, got {mode!r}")
		if orthogonal_map not in allowed_maps:
			raise ValueError(f"orthogonal_map must be one of {sorted(allowed_maps)}, got {orthogonal_map!r}")

		self.nesting_list = [int(dim) for dim in nesting_list]
		self.block_widths = block_widths_from_nesting_list(self.nesting_list)
		self.mode = mode
		self.orthogonal_map = orthogonal_map
		self.use_trivialization = bool(use_trivialization)
		self.stop_gradient = resolve_stop_gradient(stop_gradient, default=False)
		self.total_dim = self.nesting_list[-1]

		self.blocks = nn.ModuleList()
		if self.mode in {"orthogonal", "frozen"}:
			if self.mode == "orthogonal" and orthogonal is None:
				raise RuntimeError(
					"torch.nn.utils.parametrizations.orthogonal is required "
					"for BlockOrthogonalLayer(mode='orthogonal')"
				)
			for width in self.block_widths:
				layer = make_orthogonal_linear_layer(
					width,
					mode=self.mode,
					orthogonal_map=self.orthogonal_map,
					use_trivialization=self.use_trivialization,
				)
				self.blocks.append(layer)

	def get_block_widths(self):
		return list(self.block_widths)

	def forward(self, x, return_blocks=False):
		if x.dim() != 2:
			raise ValueError(f"BlockOrthogonalLayer expects [B, D] input, got shape {tuple(x.shape)}")
		if x.shape[1] != self.total_dim:
			raise ValueError(f"Expected feature dimension {self.total_dim}, got {x.shape[1]}")

		raw_blocks = torch.split(x, self.block_widths, dim=1)
		transformed_blocks = [
			layer(maybe_stop_gradient(block, self.stop_gradient))
			for layer, block in zip(self.blocks, raw_blocks)
		]

		z = torch.cat(transformed_blocks, dim=1)
		if return_blocks:
			return z, transformed_blocks
		return z

	def prefix_gram_error(self, x):
		z = self(x)
		errors = []
		for dim in self.nesting_list:
			raw_gram = x[:, :dim] @ x[:, :dim].t()
			transformed_gram = z[:, :dim] @ z[:, :dim].t()
			errors.append((raw_gram - transformed_gram).abs().max())
		return torch.stack(errors).max()

	def extra_repr(self):
		return (
			f"nesting_list={self.nesting_list}, mode={self.mode}, "
			f"orthogonal_map={self.orthogonal_map}, "
			f"use_trivialization={self.use_trivialization}, "
			f"stop_gradient={self.stop_gradient}"
		)


class IndependentBlockOrthogonalMRLHead(nn.Module):
	def __init__(self, nesting_list, num_classes, mode="orthogonal",
	             orthogonal_map="matrix_exp", use_trivialization=True,
	             stop_gradient=False):
		super().__init__()
		self.nesting_list = [int(dim) for dim in nesting_list]
		self.num_classes = int(num_classes)
		self.block_transform = BlockOrthogonalLayer(
			self.nesting_list,
			mode=mode,
			orthogonal_map=orthogonal_map,
			use_trivialization=use_trivialization,
			stop_gradient=stop_gradient,
		)
		self.stop_gradient = self.block_transform.stop_gradient
		self.classifiers = nn.ModuleList([
			nn.Linear(dim, self.num_classes) for dim in self.nesting_list
		])
		self.last_z = None
		self.last_blocks = None
		self.last_raw_blocks = None
		self.last_prefixes = None
		self.last_input = None
		self.capture_input_for_gradient_conflict = False

	def forward(self, x):
		if x.dim() != 2:
			raise ValueError(f"IndependentBlockOrthogonalMRLHead expects [B, D] input, got shape {tuple(x.shape)}")
		if x.shape[1] != self.nesting_list[-1]:
			raise ValueError(f"Expected feature dimension {self.nesting_list[-1]}, got {x.shape[1]}")

		self.last_input = x if self.capture_input_for_gradient_conflict else None
		self.last_raw_blocks = list(torch.split(x, self.block_transform.block_widths, dim=1))
		z, transformed_blocks = self.block_transform(x, return_blocks=True)
		self.last_z = z
		self.last_blocks = transformed_blocks
		self.last_prefixes = [z[:, :dim] for dim in self.nesting_list]

		return tuple(
			classifier(prefix)
			for classifier, prefix in zip(self.classifiers, self.last_prefixes)
		)

	def prefix_gram_error(self, x):
		self(x)
		errors = []
		for dim, prefix in zip(self.nesting_list, self.last_prefixes):
			raw_gram = x[:, :dim] @ x[:, :dim].t()
			transformed_gram = prefix @ prefix.t()
			errors.append((raw_gram - transformed_gram).abs().max())
		return torch.stack(errors).max()

	def block_norms(self):
		if self.last_blocks is None:
			raise RuntimeError("block_norms() requires a previous forward pass")
		return torch.stack([block.norm(p=2, dim=1).mean() for block in self.last_blocks])


class BlockOrthogonalResidualMRLHead(nn.Module):
	def __init__(self, nesting_list, num_classes, mode="orthogonal",
	             orthogonal_map="matrix_exp", use_trivialization=True,
	             stop_gradient=False, bor_residual_orthogonal=False,
	             bor_residual_alpha_init=-3.0):
		super().__init__()
		self.nesting_list = [int(dim) for dim in nesting_list]
		self.num_classes = int(num_classes)
		allowed_modes = {"orthogonal", "frozen"}
		allowed_maps = {"matrix_exp", "cayley", "householder"}
		if mode not in allowed_modes:
			raise ValueError(f"mode must be one of {sorted(allowed_modes)}, got {mode!r}")
		if orthogonal_map not in allowed_maps:
			raise ValueError(f"orthogonal_map must be one of {sorted(allowed_maps)}, got {orthogonal_map!r}")
		if mode == "orthogonal" and orthogonal is None:
			raise RuntimeError(
				"torch.nn.utils.parametrizations.orthogonal is required "
				"for BlockOrthogonalResidualMRLHead(mode='orthogonal')"
			)

		self.mode = mode
		self.orthogonal_map = orthogonal_map
		self.use_trivialization = bool(use_trivialization)
		self.stop_gradient = resolve_stop_gradient(stop_gradient, default=False)
		self.bor_residual_orthogonal = bool(bor_residual_orthogonal)
		self.bor_residual_alpha_init = float(bor_residual_alpha_init)
		self.block_widths = block_widths_from_nesting_list(self.nesting_list)

		self.classifiers = nn.ModuleList()
		for dim in self.nesting_list:
			self.classifiers.append(nn.Linear(dim, self.num_classes))

		self.prefix_orthogonal_layers = nn.ModuleList()
		if self.mode in {"orthogonal", "frozen"}:
			for dim in self.nesting_list[:-1]:
				if self.bor_residual_orthogonal:
					layer = GatedResidualOrthogonalAdapter(
						dim,
						mode=self.mode,
						orthogonal_map=self.orthogonal_map,
						use_trivialization=self.use_trivialization,
						alpha_init=self.bor_residual_alpha_init,
					)
				else:
					layer = make_orthogonal_linear_layer(
						dim,
						mode=self.mode,
						orthogonal_map=self.orthogonal_map,
						use_trivialization=self.use_trivialization,
					)
				self.prefix_orthogonal_layers.append(layer)

		self.last_z = None
		self.last_blocks = None
		self.last_prefixes = None
		self.last_input = None
		self.capture_input_for_gradient_conflict = False

	def forward(self, x):
		if x.dim() != 2:
			raise ValueError(f"BlockOrthogonalResidualMRLHead expects [B, D] input, got shape {tuple(x.shape)}")
		if x.shape[1] != self.nesting_list[-1]:
			raise ValueError(f"Expected feature dimension {self.nesting_list[-1]}, got {x.shape[1]}")

		self.last_input = x if self.capture_input_for_gradient_conflict else None
		blocks = list(torch.split(x, self.block_widths, dim=1))
		self.last_blocks = blocks

		prefix = blocks[0]
		prefixes = [prefix]
		logits = [self.classifiers[0](prefix)]
		for i, block in enumerate(blocks[1:]):
			previous_prefix = self.prefix_orthogonal_layers[i](
				maybe_stop_gradient(prefix, self.stop_gradient)
			)
			prefix = torch.cat([previous_prefix, block], dim=1)
			prefixes.append(prefix)
			logits.append(self.classifiers[i + 1](prefix))

		self.last_prefixes = prefixes
		self.last_z = prefixes[-1]

		return tuple(logits)

	def prefix_gram_error(self, x):
		self(x)
		errors = []
		for dim, prefix in zip(self.nesting_list, self.last_prefixes):
			raw_gram = x[:, :dim] @ x[:, :dim].t()
			transformed_gram = prefix @ prefix.t()
			errors.append((raw_gram - transformed_gram).abs().max())
		return torch.stack(errors).max()

	def block_norms(self):
		if self.last_blocks is None:
			raise RuntimeError("block_norms() requires a previous forward pass")
		return torch.stack([block.norm(p=2, dim=1).mean() for block in self.last_blocks])

	def alpha_values(self):
		if not self.bor_residual_orthogonal:
			return None
		return torch.stack([layer.alpha() for layer in self.prefix_orthogonal_layers])


class ResidualAlignedMRLHead(nn.Module):
	def __init__(self, nesting_list, num_classes, mode="orthogonal",
	             orthogonal_map="matrix_exp", use_trivialization=True,
	             mse_weight=1.0, cosine_weight=1.0,
	             detach_prefix_target=True, eps=1e-8):
		super().__init__()
		self.nesting_list = [int(dim) for dim in nesting_list]
		self.num_classes = int(num_classes)
		allowed_modes = {"orthogonal", "frozen"}
		allowed_maps = {"matrix_exp", "cayley", "householder"}
		if mode not in allowed_modes:
			raise ValueError(f"mode must be one of {sorted(allowed_modes)}, got {mode!r}")
		if orthogonal_map not in allowed_maps:
			raise ValueError(f"orthogonal_map must be one of {sorted(allowed_maps)}, got {orthogonal_map!r}")
		if mode == "orthogonal" and orthogonal is None:
			raise RuntimeError(
				"torch.nn.utils.parametrizations.orthogonal is required "
				"for ResidualAlignedMRLHead(mode='orthogonal')"
			)

		self.mode = mode
		self.orthogonal_map = orthogonal_map
		self.use_trivialization = bool(use_trivialization)
		self.mse_weight = float(mse_weight)
		self.cosine_weight = float(cosine_weight)
		self.detach_prefix_target = bool(detach_prefix_target)
		self.eps = float(eps)
		self.block_widths = block_widths_from_nesting_list(self.nesting_list)

		for prev_dim, block_width in zip(self.nesting_list[:-1], self.block_widths[1:]):
			if block_width != prev_dim:
				raise ValueError(
					"ResidualAlignedMRLHead requires each new residual block to "
					"have the same width as the previous prefix. This holds for "
					"doubling nesting lists such as [8, 16, 32, ...]."
				)

		self.classifiers = nn.ModuleList([
			nn.Linear(dim, self.num_classes)
			for dim in self.nesting_list
		])
		self.residual_orthogonal_layers = nn.ModuleList([
			make_orthogonal_linear_layer(
				prev_dim,
				mode=self.mode,
				orthogonal_map=self.orthogonal_map,
				use_trivialization=self.use_trivialization,
				identity_init=True,
			)
			for prev_dim in self.nesting_list[:-1]
		])

		self.last_blocks = None
		self.last_prefixes = None
		self.last_residuals = None
		self.last_rotated_residuals = None
		self.last_auxiliary_loss_stats = None
		self.last_input = None
		self.capture_input_for_gradient_conflict = False

	def forward(self, x):
		if x.dim() != 2:
			raise ValueError(f"ResidualAlignedMRLHead expects [B, D] input, got shape {tuple(x.shape)}")
		if x.shape[1] != self.nesting_list[-1]:
			raise ValueError(f"Expected feature dimension {self.nesting_list[-1]}, got {x.shape[1]}")

		self.last_input = x if self.capture_input_for_gradient_conflict else None
		blocks = list(torch.split(x, self.block_widths, dim=1))
		prefixes = [x[:, :dim] for dim in self.nesting_list]
		residuals = blocks[1:]
		rotated_residuals = [
			layer(residual.detach())
			for layer, residual in zip(self.residual_orthogonal_layers, residuals)
		]

		self.last_blocks = blocks
		self.last_prefixes = prefixes
		self.last_residuals = residuals
		self.last_rotated_residuals = rotated_residuals
		self.last_auxiliary_loss_stats = None

		return tuple(
			classifier(prefix)
			for classifier, prefix in zip(self.classifiers, prefixes)
		)

	def auxiliary_loss(self, targets=None):
		if self.last_prefixes is None or self.last_rotated_residuals is None:
			raise RuntimeError("auxiliary_loss() requires a previous forward pass")
		if len(self.last_prefixes) < 2:
			return self.classifiers[0].weight.new_zeros(())

		mse_losses = []
		cosine_losses = []
		pair_stats = []
		for prev_dim, dim, previous_prefix, residual, rotated in zip(
			self.nesting_list[:-1],
			self.nesting_list[1:],
			self.last_prefixes[:-1],
			self.last_residuals,
			self.last_rotated_residuals,
		):
			target_prefix = previous_prefix.detach() if self.detach_prefix_target else previous_prefix
			mse_loss = F.mse_loss(rotated.float(), target_prefix.float())
			cosine = F.cosine_similarity(residual.float(), rotated.float(), dim=1, eps=self.eps)
			cosine_loss = (1.0 - cosine).mean()
			mse_losses.append(mse_loss)
			cosine_losses.append(cosine_loss)
			pair_stats.append({
				"pair": f"{dim}<-{prev_dim}",
				"small_dim": prev_dim,
				"large_dim": dim,
				"mse": float(mse_loss.detach().cpu().item()),
				"cosine_distance": float(cosine_loss.detach().cpu().item()),
				"cosine": float(cosine.detach().mean().cpu().item()),
			})

		mse = torch.stack(mse_losses).mean()
		cosine_distance = torch.stack(cosine_losses).mean()
		loss = self.mse_weight * mse + self.cosine_weight * cosine_distance
		self.last_auxiliary_loss_stats = {
			"residual_aligned_mrl_loss": float(loss.detach().cpu().item()),
			"residual_aligned_mrl_mse": float(mse.detach().cpu().item()),
			"residual_aligned_mrl_cosine_distance": float(cosine_distance.detach().cpu().item()),
			"residual_aligned_mrl_mse_weight": self.mse_weight,
			"residual_aligned_mrl_cosine_weight": self.cosine_weight,
			"residual_aligned_mrl_pairs": pair_stats,
		}
		for pair in pair_stats:
			label = f"{pair['large_dim']}_vs_{pair['small_dim']}"
			self.last_auxiliary_loss_stats[f"residual_aligned_mrl_mse_{label}"] = pair["mse"]
			self.last_auxiliary_loss_stats[f"residual_aligned_mrl_cosine_distance_{label}"] = pair["cosine_distance"]
			self.last_auxiliary_loss_stats[f"residual_aligned_mrl_cosine_{label}"] = pair["cosine"]
		return loss

	def auxiliary_log_dict(self):
		return dict(self.last_auxiliary_loss_stats or {})

	def block_norms(self):
		if self.last_blocks is None:
			raise RuntimeError("block_norms() requires a previous forward pass")
		return torch.stack([block.norm(p=2, dim=1).mean() for block in self.last_blocks])


class CascadeStopGradientMRLHead(nn.Module):
	def __init__(self, nesting_list, num_classes, stop_gradient=True):
		super().__init__()
		self.nesting_list = [int(dim) for dim in nesting_list]
		self.num_classes = int(num_classes)
		self.block_widths = block_widths_from_nesting_list(self.nesting_list)
		self.stop_gradient = resolve_stop_gradient(stop_gradient, default=True)

		self.classifiers = nn.ModuleList([
			nn.Linear(dim, self.num_classes)
			for dim in self.nesting_list
		])

		self.last_blocks = None
		self.last_prefixes = None
		self.last_input = None
		self.capture_input_for_gradient_conflict = False

	def forward(self, x):
		if x.dim() != 2:
			raise ValueError(f"CascadeStopGradientMRLHead expects [B, D] input, got shape {tuple(x.shape)}")
		if x.shape[1] != self.nesting_list[-1]:
			raise ValueError(f"Expected feature dimension {self.nesting_list[-1]}, got {x.shape[1]}")

		self.last_input = x if self.capture_input_for_gradient_conflict else None
		blocks = list(torch.split(x, self.block_widths, dim=1))
		self.last_blocks = blocks

		prefix = blocks[0]
		prefixes = [prefix]
		logits = [self.classifiers[0](prefix)]

		for i, block in enumerate(blocks[1:]):
			old_prefix = maybe_stop_gradient(prefix, self.stop_gradient)
			prefix = torch.cat([old_prefix, block], dim=1)

			prefixes.append(prefix)
			logits.append(self.classifiers[i + 1](prefix))

		self.last_prefixes = prefixes
		return tuple(logits)

	def prefix_gram_error(self, x):
		self(x)
		errors = []
		for dim, prefix in zip(self.nesting_list, self.last_prefixes):
			raw_gram = x[:, :dim] @ x[:, :dim].t()
			transformed_gram = prefix @ prefix.t()
			errors.append((raw_gram - transformed_gram).abs().max())
		return torch.stack(errors).max()

	def block_norms(self):
		if self.last_blocks is None:
			raise RuntimeError("block_norms() requires a previous forward pass")
		return torch.stack([block.norm(p=2, dim=1).mean() for block in self.last_blocks])


class RecursiveLinkBlock(nn.Module):
	def __init__(self, in_dim, out_dim, hidden_ratio=0.5, dropout=0.0):
		super().__init__()
		self.in_dim = int(in_dim)
		self.out_dim = int(out_dim)
		self.hidden_ratio = float(hidden_ratio)
		self.dropout = float(dropout)
		hidden_dim = max(self.out_dim, int(self.in_dim * self.hidden_ratio))

		self.net = nn.Sequential(
			nn.LayerNorm(self.in_dim),
			nn.Linear(self.in_dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(self.dropout),
			nn.Linear(hidden_dim, self.out_dim),
		)

	def forward(self, x):
		return self.net(x)


class RecursiveLinkMRLHead(nn.Module):
	def __init__(
		self,
		nesting_list,
		num_classes,
		recursive_link_hidden_ratio=0.5,
		recursive_link_dropout=0.0,
		recursive_link_alpha_init=-4.0,
		recursive_link_stop_gradient=False,
	):
		super().__init__()
		self.nesting_list = [int(dim) for dim in nesting_list]
		self.num_classes = int(num_classes)
		self.block_widths = block_widths_from_nesting_list(self.nesting_list)
		self.recursive_link_hidden_ratio = float(recursive_link_hidden_ratio)
		self.recursive_link_dropout = float(recursive_link_dropout)
		self.recursive_link_alpha_init = float(recursive_link_alpha_init)
		self.recursive_link_stop_gradient = bool(recursive_link_stop_gradient)

		self.classifiers = nn.ModuleList([
			nn.Linear(dim, self.num_classes)
			for dim in self.nesting_list
		])
		self.links = nn.ModuleList([
			RecursiveLinkBlock(
				in_dim=prev_dim,
				out_dim=block_width,
				hidden_ratio=self.recursive_link_hidden_ratio,
				dropout=self.recursive_link_dropout,
			)
			for prev_dim, block_width in zip(self.nesting_list[:-1], self.block_widths[1:])
		])
		self.alpha_logits = nn.Parameter(
			torch.full((len(self.nesting_list) - 1,), self.recursive_link_alpha_init)
		)

		self.last_blocks = None
		self.last_prefixes = None
		self.last_z = None
		self.last_input = None
		self.capture_input_for_gradient_conflict = False

	def forward(self, x):
		if x.dim() != 2:
			raise ValueError(f"RecursiveLinkMRLHead expects [B, D] input, got shape {tuple(x.shape)}")
		if x.shape[1] != self.nesting_list[-1]:
			raise ValueError(f"Expected feature dimension {self.nesting_list[-1]}, got {x.shape[1]}")

		self.last_input = x if self.capture_input_for_gradient_conflict else None
		blocks = list(torch.split(x, self.block_widths, dim=1))
		self.last_blocks = blocks

		prefix = blocks[0]
		prefixes = [prefix]
		logits = [self.classifiers[0](prefix)]

		for i, block in enumerate(blocks[1:]):
			link_input = maybe_stop_gradient(prefix, self.recursive_link_stop_gradient)
			correction = self.links[i](link_input)
			alpha = torch.sigmoid(self.alpha_logits[i]).to(dtype=x.dtype, device=x.device)
			new_block = block + alpha * correction
			prefix = torch.cat([prefix, new_block], dim=1)
			prefixes.append(prefix)
			logits.append(self.classifiers[i + 1](prefix))

		self.last_prefixes = prefixes
		self.last_z = prefixes[-1]
		return tuple(logits)

	def prefix_gram_error(self, x):
		self(x)
		errors = []
		for dim, prefix in zip(self.nesting_list, self.last_prefixes):
			raw_gram = x[:, :dim] @ x[:, :dim].t()
			transformed_gram = prefix @ prefix.t()
			errors.append((raw_gram - transformed_gram).abs().max())
		return torch.stack(errors).max()

	def block_norms(self):
		if self.last_blocks is None:
			raise RuntimeError("block_norms() requires a previous forward pass")
		return torch.stack([block.norm(p=2, dim=1).mean() for block in self.last_blocks])

	def alpha_values(self):
		return torch.sigmoid(self.alpha_logits)


class OrthogonalTransitionLayer(nn.Module):
	def __init__(self, nesting_list, mode="orthogonal",
	             orthogonal_map="matrix_exp", use_trivialization=True,
	             stop_gradient=False):
		super().__init__()
		allowed_modes = {"orthogonal", "frozen"}
		if mode not in allowed_modes:
			raise ValueError(f"mode must be one of {sorted(allowed_modes)}, got {mode!r}")
		orthogonal_map = resolve_t_orthogonal_map(orthogonal_map)
		if mode == "orthogonal" and orthogonal is None:
			raise RuntimeError(
				"torch.nn.utils.parametrizations.orthogonal is required "
				"for OrthogonalTransitionLayer(mode='orthogonal')"
			)

		self.nesting_list = [int(dim) for dim in nesting_list]
		self.block_widths = block_widths_from_nesting_list(self.nesting_list)
		self.mode = mode
		self.orthogonal_map = orthogonal_map
		self.use_trivialization = bool(use_trivialization)
		self.stop_gradient = resolve_stop_gradient(stop_gradient, default=False)
		self.total_dim = self.nesting_list[-1]

		self.t_layers = nn.ModuleList([
			make_orthogonal_linear_layer(
				dim,
				mode=self.mode,
				orthogonal_map=self.orthogonal_map,
				use_trivialization=self.use_trivialization,
				identity_init=True,
			)
			for dim in self.nesting_list[1:]
		])

	def get_block_widths(self):
		return list(self.block_widths)

	def get_t_dims(self):
		return list(self.nesting_list[1:])

	def forward(self, x, return_details=False):
		if x.dim() != 2:
			raise ValueError(f"OrthogonalTransitionLayer expects [B, D] input, got shape {tuple(x.shape)}")
		if x.shape[1] != self.total_dim:
			raise ValueError(f"Expected feature dimension {self.total_dim}, got {x.shape[1]}")

		blocks = list(torch.split(x, self.block_widths, dim=1))
		prefixes = [x[:, :self.nesting_list[0]]]
		t_outputs = []

		for t_layer, dim in zip(self.t_layers, self.nesting_list[1:]):
			raw_prefix = x[:, :dim]
			prefix = t_layer(maybe_stop_gradient(raw_prefix, self.stop_gradient))
			t_outputs.append(prefix)
			prefixes.append(prefix)

		if return_details:
			return prefixes, blocks, t_outputs
		return prefixes

	def prefix_gram_error(self, x):
		prefixes = self(x)
		errors = []
		for dim, prefix in zip(self.nesting_list, prefixes):
			raw_gram = x[:, :dim] @ x[:, :dim].t()
			transformed_gram = prefix @ prefix.t()
			errors.append((raw_gram - transformed_gram).abs().max())
		return torch.stack(errors).max()

	def extra_repr(self):
		return (
			f"nesting_list={self.nesting_list}, mode={self.mode}, "
			f"orthogonal_map={self.orthogonal_map}, "
			f"use_trivialization={self.use_trivialization}, "
			f"stop_gradient={self.stop_gradient}"
		)


class TOrthogonalMRLHead(nn.Module):
	def __init__(self, nesting_list, num_classes, mode="orthogonal",
	             orthogonal_map="matrix_exp", use_trivialization=True,
	             stop_gradient=False):
		super().__init__()
		self.transition_layer = OrthogonalTransitionLayer(
			nesting_list,
			mode=mode,
			orthogonal_map=orthogonal_map,
			use_trivialization=use_trivialization,
			stop_gradient=stop_gradient,
		)
		self.nesting_list = list(self.transition_layer.nesting_list)
		self.num_classes = int(num_classes)
		self.mode = self.transition_layer.mode
		self.orthogonal_map = self.transition_layer.orthogonal_map
		self.use_trivialization = self.transition_layer.use_trivialization
		self.stop_gradient = self.transition_layer.stop_gradient

		self.classifiers = nn.ModuleList([
			nn.Linear(dim, self.num_classes)
			for dim in self.nesting_list
		])

		self.last_z = None
		self.last_blocks = None
		self.last_prefixes = None
		self.last_t_outputs = None
		self.last_input = None
		self.capture_input_for_gradient_conflict = False

	@property
	def block_widths(self):
		return self.transition_layer.block_widths

	@property
	def t_layers(self):
		return self.transition_layer.t_layers

	@property
	def prefix_orthogonal_layers(self):
		return self.transition_layer.t_layers

	def get_t_dims(self):
		return self.transition_layer.get_t_dims()

	def forward(self, x):
		if x.dim() != 2:
			raise ValueError(f"TOrthogonalMRLHead expects [B, D] input, got shape {tuple(x.shape)}")
		if x.shape[1] != self.nesting_list[-1]:
			raise ValueError(f"Expected feature dimension {self.nesting_list[-1]}, got {x.shape[1]}")

		self.last_input = x if self.capture_input_for_gradient_conflict else None
		prefixes, blocks, t_outputs = self.transition_layer(x, return_details=True)
		self.last_blocks = blocks
		self.last_prefixes = prefixes
		self.last_t_outputs = t_outputs
		self.last_z = prefixes[-1]

		return tuple(
			classifier(prefix)
			for classifier, prefix in zip(self.classifiers, prefixes)
		)

	def prefix_gram_error(self, x):
		return self.transition_layer.prefix_gram_error(x)

	def block_norms(self):
		if self.last_blocks is None:
			raise RuntimeError("block_norms() requires a previous forward pass")
		return torch.stack([block.norm(p=2, dim=1).mean() for block in self.last_blocks])


class BidirectionalMRLHead(nn.Module):
	def __init__(self, nesting_list, num_classes):
		super().__init__()
		self.nesting_list = [int(dim) for dim in nesting_list]
		self.num_classes = int(num_classes)
		if len(self.nesting_list) == 0:
			raise ValueError("nesting_list must contain at least one positive dimension")
		if any(dim <= 0 for dim in self.nesting_list):
			raise ValueError("nesting_list dimensions must all be positive")
		for prev_dim, dim in zip(self.nesting_list, self.nesting_list[1:]):
			if dim <= prev_dim:
				raise ValueError("nesting_list must be strictly increasing")

		self.full_dim = self.nesting_list[-1]
		for dim in self.nesting_list[:-1]:
			if dim % 2 != 0:
				raise ValueError("All non-full bidirectional nesting dimensions must be even")

		self.classifiers = nn.ModuleList([
			nn.Linear(dim, self.num_classes)
			for dim in self.nesting_list
		])
		self.last_prefixes = None
		self.last_bidirectional_prefixes = None
		self.last_input = None
		self.capture_input_for_gradient_conflict = False

	def forward(self, x):
		if x.dim() != 2:
			raise ValueError(f"BidirectionalMRLHead expects [B, D] input, got shape {tuple(x.shape)}")
		if x.shape[1] != self.full_dim:
			raise ValueError(f"Expected feature dimension {self.full_dim}, got {x.shape[1]}")

		self.last_input = x if self.capture_input_for_gradient_conflict else None
		prefixes = []
		for dim in self.nesting_list:
			if dim == self.full_dim:
				prefixes.append(x)
			else:
				half = dim // 2
				prefixes.append(torch.cat([x[:, :half], x[:, -half:]], dim=1))

		self.last_prefixes = prefixes
		self.last_bidirectional_prefixes = prefixes
		return tuple(
			classifier(prefix)
			for classifier, prefix in zip(self.classifiers, prefixes)
		)


class MRL_Linear_Layer(nn.Module):
	def __init__(self, nesting_list: List, num_classes=1000, efficient=False, **kwargs):
		super(MRL_Linear_Layer, self).__init__()
		self.nesting_list = nesting_list
		self.num_classes = num_classes # Number of classes for classification
		self.efficient = efficient
		self.last_input = None
		self.last_prefixes = None
		self.capture_input_for_gradient_conflict = False
		if self.efficient:
			setattr(self, f"nesting_classifier_{0}", nn.Linear(nesting_list[-1], self.num_classes, **kwargs))		
		else:	
			for i, num_feat in enumerate(self.nesting_list):
				setattr(self, f"nesting_classifier_{i}", nn.Linear(num_feat, self.num_classes, **kwargs))	

	def reset_parameters(self):
		if self.efficient:
			self.nesting_classifier_0.reset_parameters()
		else:
			for i in range(len(self.nesting_list)):
				getattr(self, f"nesting_classifier_{i}").reset_parameters()


	def forward(self, x):
		self.last_input = x if self.capture_input_for_gradient_conflict else None
		self.last_prefixes = [x[:, :num_feat] for num_feat in self.nesting_list]
		nesting_logits = ()
		for i, (num_feat, prefix) in enumerate(zip(self.nesting_list, self.last_prefixes)):
			if self.efficient:
				if self.nesting_classifier_0.bias is None:
					nesting_logits += (torch.matmul(prefix, (self.nesting_classifier_0.weight[:, :num_feat]).t()), )
				else:
					nesting_logits += (torch.matmul(prefix, (self.nesting_classifier_0.weight[:, :num_feat]).t()) + self.nesting_classifier_0.bias, )
			else:
				nesting_logits +=  (getattr(self, f"nesting_classifier_{i}")(prefix),)

		return nesting_logits


class SuffixBalancedMRLHead(nn.Module):
	def __init__(self, nesting_list, num_classes, include_full_suffix=False):
		super().__init__()
		self.nesting_list = [int(dim) for dim in nesting_list]
		self.num_classes = int(num_classes)
		self.include_full_suffix = bool(include_full_suffix)
		if len(self.nesting_list) == 0:
			raise ValueError("nesting_list must contain at least one positive dimension")
		if any(dim <= 0 for dim in self.nesting_list):
			raise ValueError("nesting_list dimensions must all be positive")
		for prev_dim, dim in zip(self.nesting_list, self.nesting_list[1:]):
			if dim <= prev_dim:
				raise ValueError("nesting_list must be strictly increasing")

		self.prefix_head = MRL_Linear_Layer(
			self.nesting_list,
			num_classes=self.num_classes,
			efficient=False,
		)
		self.suffix_nesting_list = (
			list(self.nesting_list)
			if self.include_full_suffix
			else list(self.nesting_list[:-1])
		)
		self.suffix_classifiers = nn.ModuleList([
			nn.Linear(dim, self.num_classes)
			for dim in self.suffix_nesting_list
		])
		self.last_prefix_logits = None
		self.last_suffix_logits = None
		self.last_prefixes = None
		self.last_suffixes = None
		self.last_input = None
		self.capture_input_for_gradient_conflict = False

	@property
	def num_suffix_heads(self):
		return len(self.suffix_classifiers)

	def forward(self, x):
		if x.dim() != 2:
			raise ValueError(f"SuffixBalancedMRLHead expects [B, D] input, got shape {tuple(x.shape)}")
		if x.shape[1] != self.nesting_list[-1]:
			raise ValueError(f"Expected feature dimension {self.nesting_list[-1]}, got {x.shape[1]}")

		self.last_input = x if self.capture_input_for_gradient_conflict else None
		prefix_logits = self.prefix_head(x)
		self.last_prefix_logits = prefix_logits
		self.last_prefixes = self.prefix_head.last_prefixes
		self.last_suffixes = [x[:, -dim:] for dim in self.suffix_nesting_list]
		if self.training:
			self.last_suffix_logits = tuple(
				classifier(suffix)
				for classifier, suffix in zip(self.suffix_classifiers, self.last_suffixes)
			)
		else:
			self.last_suffix_logits = None
		return prefix_logits

	def auxiliary_loss(self, targets):
		if self.num_suffix_heads == 0:
			return torch.zeros((), device=targets.device, dtype=torch.float32)
		if self.last_suffix_logits is None:
			raise RuntimeError("auxiliary_loss() requires a previous forward pass")

		return torch.stack([
			F.cross_entropy(logits, targets)
			for logits in self.last_suffix_logits
		]).sum()


class FixedFeatureLayer(nn.Linear):
    '''
    For our fixed feature baseline, we just replace the classification layer with the following. 
    It effectively just look at the first "in_features" for the classification. 
    '''

    def __init__(self, in_features, out_features, **kwargs):
        super(FixedFeatureLayer, self).__init__(in_features, out_features, **kwargs)

    def forward(self, x):
        if not (self.bias is None):
            out = torch.matmul(x[:, :self.in_features], self.weight.t()) + self.bias
        else:
            out = torch.matmul(x[:, :self.in_features], self.weight.t())
        return out
        
