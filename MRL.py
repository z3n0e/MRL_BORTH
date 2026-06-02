import torch
import torch.nn as nn
from typing import List

try:
	from torch.nn.utils.parametrizations import orthogonal
except ImportError:
	orthogonal = None

'''
Loss function for Matryoshka Representation Learning 
'''

class Matryoshka_CE_Loss(nn.Module):
	def __init__(self, relative_importance: List[float]=None, **kwargs):
		super(Matryoshka_CE_Loss, self).__init__()
		self.criterion = nn.CrossEntropyLoss(**kwargs)
		# relative importance shape: [G]
		self.relative_importance = relative_importance

	def forward(self, output, target):
		# output shape: [G granularities, N batch size, C number of classes]
		# target shape: [N batch size]

		# Calculate losses for each output and stack them. This is still O(N)
		losses = torch.stack([self.criterion(output_i, target) for output_i in output])
		
		# Set relative_importance to 1 if not specified
		rel_importance = losses.new_ones(losses.shape) if self.relative_importance is None else losses.new_tensor(self.relative_importance)
		
		# Apply relative importance weights
		weighted_losses = rel_importance * losses
		return weighted_losses.sum()


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


class BlockOrthogonalLayer(nn.Module):
	def __init__(self, nesting_list, mode="orthogonal",
	             orthogonal_map="matrix_exp", use_trivialization=True):
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
		self.total_dim = self.nesting_list[-1]

		self.blocks = nn.ModuleList()
		if self.mode in {"orthogonal", "frozen"}:
			if self.mode == "orthogonal" and orthogonal is None:
				raise RuntimeError(
					"torch.nn.utils.parametrizations.orthogonal is required "
					"for BlockOrthogonalLayer(mode='orthogonal')"
				)
			for width in self.block_widths:
				layer = nn.Linear(width, width, bias=False)
				if self.mode == "frozen":
					nn.init.orthogonal_(layer.weight)
					layer.weight.requires_grad_(False)
				else:
					with torch.no_grad():
						layer.weight.copy_(torch.eye(width))
					try:
						layer = orthogonal(
							layer,
							"weight",
							orthogonal_map=self.orthogonal_map,
							use_trivialization=self.use_trivialization,
						)
					except TypeError as exc:
						raise RuntimeError(
							"Installed PyTorch does not support the requested "
							"orthogonal parametrization options. Use a modern "
							"PyTorch with torch.nn.utils.parametrizations.orthogonal."
						) from exc
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
			layer(block) for layer, block in zip(self.blocks, raw_blocks)
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
			f"use_trivialization={self.use_trivialization}"
		)


class IndependentBlockOrthogonalMRLHead(nn.Module):
	def __init__(self, nesting_list, num_classes, mode="orthogonal",
	             orthogonal_map="matrix_exp", use_trivialization=True):
		super().__init__()
		self.nesting_list = [int(dim) for dim in nesting_list]
		self.num_classes = int(num_classes)
		self.block_transform = BlockOrthogonalLayer(
			self.nesting_list,
			mode=mode,
			orthogonal_map=orthogonal_map,
			use_trivialization=use_trivialization,
		)
		self.classifiers = nn.ModuleList([
			nn.Linear(dim, self.num_classes) for dim in self.nesting_list
		])
		self.last_z = None
		self.last_blocks = None
		self.last_raw_blocks = None
		self.last_prefixes = None

	def forward(self, x):
		if x.dim() != 2:
			raise ValueError(f"IndependentBlockOrthogonalMRLHead expects [B, D] input, got shape {tuple(x.shape)}")
		if x.shape[1] != self.nesting_list[-1]:
			raise ValueError(f"Expected feature dimension {self.nesting_list[-1]}, got {x.shape[1]}")

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
	             orthogonal_map="matrix_exp", use_trivialization=True):
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
		self.block_widths = block_widths_from_nesting_list(self.nesting_list)

		self.classifiers = nn.ModuleList()
		for dim in self.nesting_list:
			self.classifiers.append(nn.Linear(dim, self.num_classes))

		self.prefix_orthogonal_layers = nn.ModuleList()
		if self.mode in {"orthogonal", "frozen"}:
			for dim in self.nesting_list[:-1]:
				layer = nn.Linear(dim, dim, bias=False)
				if self.mode == "frozen":
					nn.init.orthogonal_(layer.weight)
					layer.weight.requires_grad_(False)
				else:
					with torch.no_grad():
						layer.weight.copy_(torch.eye(dim))
					try:
						layer = orthogonal(
							layer,
							"weight",
							orthogonal_map=self.orthogonal_map,
							use_trivialization=self.use_trivialization,
						)
					except TypeError as exc:
						raise RuntimeError(
							"Installed PyTorch does not support the requested "
							"orthogonal parametrization options. Use a modern "
							"PyTorch with torch.nn.utils.parametrizations.orthogonal."
						) from exc
				self.prefix_orthogonal_layers.append(layer)

		self.last_z = None
		self.last_blocks = None
		self.last_prefixes = None

	def forward(self, x):
		if x.dim() != 2:
			raise ValueError(f"BlockOrthogonalResidualMRLHead expects [B, D] input, got shape {tuple(x.shape)}")
		if x.shape[1] != self.nesting_list[-1]:
			raise ValueError(f"Expected feature dimension {self.nesting_list[-1]}, got {x.shape[1]}")

		blocks = list(torch.split(x, self.block_widths, dim=1))
		self.last_blocks = blocks

		prefix = blocks[0]
		prefixes = [prefix]
		logits = [self.classifiers[0](prefix)]
		for i, block in enumerate(blocks[1:]):
			previous_prefix = self.prefix_orthogonal_layers[i](prefix)
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


class MRL_Linear_Layer(nn.Module):
	def __init__(self, nesting_list: List, num_classes=1000, efficient=False, **kwargs):
		super(MRL_Linear_Layer, self).__init__()
		self.nesting_list = nesting_list
		self.num_classes = num_classes # Number of classes for classification
		self.efficient = efficient
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
		nesting_logits = ()
		for i, num_feat in enumerate(self.nesting_list):
			if self.efficient:
				if self.nesting_classifier_0.bias is None:
					nesting_logits += (torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()), )
				else:
					nesting_logits += (torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()) + self.nesting_classifier_0.bias, )
			else:
				nesting_logits +=  (getattr(self, f"nesting_classifier_{i}")(x[:, :num_feat]),)

		return nesting_logits


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
        
