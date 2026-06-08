import torch
import torch.nn.functional as F
import pytest
import numpy as np

from MRL import (
    BidirectionalMRLHead,
    BlockOrthogonalLayer,
    BlockOrthogonalResidualMRLHead,
    CascadeStopGradientMRLHead,
    FixedFeatureLayer,
    GatedResidualOrthogonalAdapter,
    IndependentBlockOrthogonalMRLHead,
    Matryoshka_CE_Loss,
    MRL_Linear_Layer,
    OrthogonalTransitionLayer,
    RecursiveLinkMRLHead,
    ResidualAlignedMRLHead,
    SuffixBalancedMRLHead,
    TOrthogonalMRLHead,
    adjacent_residual_prefix_cosine_stats,
    block_cascade_conflict_gating,
    block_widths_from_nesting_list,
    mrl_block_cascade_filtered_feature_gradient,
    mrl_gradient_conflict_stats,
    mrl_sampling_probabilities,
    procrustes_cascade_distillation_loss,
)
from retrieval.metrics import (
    compute_retrieval_metrics_at_k,
    relevant_counts_by_label,
    top1_accuracy,
)

def test_forward():
    # Create a Matryoshka_CE_Loss instance
    loss_fn = Matryoshka_CE_Loss()

    # Create some dummy input and target tensors
    # shape: [G, N batch size, C number of classes]
    output = torch.randn(2, 3, 5, requires_grad=True)
    # shape: [N batch size]
    target = torch.empty(3, dtype=torch.long).random_(5)

    # Calculate the loss
    loss = loss_fn.forward(output, target)
    print(loss)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_relative_importance():
    # shape: [G]
    relative_importance = [0.1, 0.9]

    # Create a Matryoshka_CE_Loss instance with relative_importance
    loss_fn = Matryoshka_CE_Loss(relative_importance=relative_importance)

    # Create some dummy input and target tensors
    # shape: [G, N batch size, C number of classes]
    output = torch.randn(2, 3, 5, requires_grad=True)
    # shape: [N batch size]
    target = torch.empty(3, dtype=torch.long).random_(5)
    loss = loss_fn.forward(output, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_default_all_mode_matches_sum_of_ce_losses():
    loss_fn = Matryoshka_CE_Loss()
    output = torch.randn(3, 4, 5, requires_grad=True)
    target = torch.empty(4, dtype=torch.long).random_(5)

    loss = loss_fn(output, target)
    expected = sum(F.cross_entropy(output_i, target) for output_i in output)

    assert torch.allclose(loss, expected)


def test_sampled_prefix_loss_returns_scalar_and_valid_index():
    torch.manual_seed(0)
    nesting_list = [2, 4, 8]
    loss_fn = Matryoshka_CE_Loss(
        mrl_loss_mode="sampled_prefix",
        nesting_list=nesting_list,
    )
    output = torch.randn(3, 4, 5, requires_grad=True)
    target = torch.empty(4, dtype=torch.long).random_(5)

    loss = loss_fn(output, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert 0 <= loss_fn.last_sampled_idx < len(nesting_list)
    assert loss_fn.last_sampled_dim == nesting_list[loss_fn.last_sampled_idx]


def test_sampling_probabilities_sum_and_inverse_dim_prefers_smaller_prefixes():
    nesting_list = [8, 16, 32]
    uniform = mrl_sampling_probabilities(nesting_list, "uniform")
    inverse_dim = mrl_sampling_probabilities(nesting_list, "inverse_dim")
    inverse_sqrt_dim = mrl_sampling_probabilities(nesting_list, "inverse_sqrt_dim")

    assert torch.allclose(uniform.sum(), torch.tensor(1.0))
    assert torch.allclose(inverse_dim.sum(), torch.tensor(1.0))
    assert torch.allclose(inverse_sqrt_dim.sum(), torch.tensor(1.0))
    assert inverse_dim[0] > inverse_dim[1] > inverse_dim[2]
    assert inverse_sqrt_dim[0] > inverse_sqrt_dim[1] > inverse_sqrt_dim[2]


def test_sampled_prefix_dummy_does_not_change_loss_and_touches_all_logits():
    loss_fn = Matryoshka_CE_Loss(
        mrl_loss_mode="sampled_prefix",
        nesting_list=[2, 4, 8],
    )
    loss_fn.sampled_prefix_probs.copy_(torch.tensor([0.0, 1.0, 0.0]))
    outputs = tuple(torch.randn(4, 5, requires_grad=True) for _ in range(3))
    target = torch.empty(4, dtype=torch.long).random_(5)

    loss = loss_fn(outputs, target)
    expected = F.cross_entropy(outputs[1], target)

    assert torch.allclose(loss, expected)
    loss.backward()
    assert outputs[0].grad is not None
    assert outputs[1].grad is not None
    assert outputs[2].grad is not None
    assert torch.allclose(outputs[0].grad, torch.zeros_like(outputs[0]))
    assert outputs[1].grad.abs().sum() > 0
    assert torch.allclose(outputs[2].grad, torch.zeros_like(outputs[2]))


def test_all_prefix_logits_are_produced_in_sampled_mode():
    nesting_list = [2, 4, 8]
    layer = MRL_Linear_Layer(nesting_list, num_classes=5)
    loss_fn = Matryoshka_CE_Loss(
        mrl_loss_mode="sampled_prefix",
        nesting_list=nesting_list,
    )
    output = layer(torch.randn(4, nesting_list[-1]))
    target = torch.empty(4, dtype=torch.long).random_(5)
    loss = loss_fn(output, target)

    assert len(output) == len(nesting_list)
    assert loss.dim() == 0


def test_mrl_gradient_conflict_stats_detects_negative_cosine():
    shared = torch.randn(2, 3, requires_grad=True)
    losses = torch.stack([shared.sum(), -shared.sum()])

    stats = mrl_gradient_conflict_stats(losses, shared, [2, 4])

    assert stats["mrl_grad_conflict_pair_count"] == 1
    assert stats["mrl_grad_conflict_count"] == 1
    assert stats["mrl_grad_conflict_fraction"] == pytest.approx(1.0)
    assert stats["mrl_grad_conflict_min_cosine"] == pytest.approx(-1.0)
    assert stats["mrl_grad_conflict_worst_dim_i"] == 2
    assert stats["mrl_grad_conflict_worst_dim_j"] == 4


def test_block_cascade_conflict_gating_only_changes_shared_conflict_block():
    small_grad = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    large_grad = torch.tensor([[-1.0, 0.0, 2.0, 3.0]])

    filtered, stats = block_cascade_conflict_gating(
        [small_grad, large_grad],
        [2, 4],
        alpha=1.0,
        eps=1e-8,
    )

    assert torch.equal(filtered[0], small_grad)
    assert torch.allclose(filtered[1][:, :2], torch.zeros(1, 2))
    assert torch.equal(filtered[1][:, 2:4], large_grad[:, 2:4])
    assert stats["mrl_conflict_mean_adjacent_cosine"] == pytest.approx(-1.0)
    assert stats["mrl_conflict_fraction"] == pytest.approx(1.0)
    assert stats["mrl_conflict_pairs"][0]["pair"] == "4<-2"


def test_block_cascade_filtered_step_keeps_encoder_and_head_gradients():
    torch.manual_seed(0)
    encoder = torch.nn.Linear(5, 4)
    head = MRL_Linear_Layer([2, 4], num_classes=3)
    loss_fn = Matryoshka_CE_Loss()
    x = torch.randn(6, 5)
    target = torch.empty(6, dtype=torch.long).random_(3)

    z = encoder(x)
    output = head(z)
    prefix_losses = loss_fn.per_prefix_losses(output, target)
    filtered_grad, stats = mrl_block_cascade_filtered_feature_gradient(
        prefix_losses,
        z,
        [2, 4],
        alpha=0.5,
        eps=1e-8,
    )

    head_loss = loss_fn(head(z.detach()), target)
    head_loss.backward()
    z.backward(gradient=filtered_grad)

    assert filtered_grad.shape == z.shape
    assert stats["mrl_conflict_mode"] == "block_cascade"
    assert encoder.weight.grad is not None
    assert torch.isfinite(encoder.weight.grad).all()
    for idx in range(2):
        classifier = getattr(head, f"nesting_classifier_{idx}")
        assert classifier.weight.grad is not None
        assert torch.isfinite(classifier.weight.grad).all()


def test_custom_num_classes_layers():
    num_classes = 100
    nesting_list = [2, 4]
    batch_size = 3

    mrl_layer = MRL_Linear_Layer(nesting_list, num_classes=num_classes)
    mrl_outputs = mrl_layer(torch.randn(batch_size, nesting_list[-1]))

    assert len(mrl_outputs) == len(nesting_list)
    for output in mrl_outputs:
        assert output.shape == (batch_size, num_classes)

    fixed_layer = FixedFeatureLayer(nesting_list[-1], num_classes)
    fixed_output = fixed_layer(torch.randn(batch_size, nesting_list[-1]))

    assert fixed_output.shape == (batch_size, num_classes)


def test_bidirectional_mrl_head_prefixes_and_output_shapes():
    batch_size = 3
    num_classes = 10
    nesting_list = [8, 16, 32, 64]
    x = torch.randn(batch_size, 64)
    head = BidirectionalMRLHead(nesting_list, num_classes=num_classes)

    logits = head(x)

    assert isinstance(logits, tuple)
    assert len(logits) == 4
    for output in logits:
        assert output.shape == (batch_size, num_classes)
    assert torch.equal(
        head.last_bidirectional_prefixes[0],
        torch.cat([x[:, :4], x[:, -4:]], dim=1),
    )
    assert torch.equal(
        head.last_bidirectional_prefixes[1],
        torch.cat([x[:, :8], x[:, -8:]], dim=1),
    )
    assert torch.equal(head.last_bidirectional_prefixes[-1], x)


def test_suffix_balanced_mrl_head_prefix_logits_and_prefixes():
    x = torch.randn(4, 64)
    head = SuffixBalancedMRLHead([8, 16, 32, 64], num_classes=10)

    logits = head(x)

    assert len(logits) == 4
    assert logits[0].shape == (4, 10)
    assert logits[1].shape == (4, 10)
    assert logits[2].shape == (4, 10)
    assert logits[3].shape == (4, 10)
    assert torch.equal(head.last_prefixes[0], x[:, :8])
    assert torch.equal(head.last_prefixes[1], x[:, :16])
    assert torch.equal(head.last_prefixes[2], x[:, :32])
    assert torch.equal(head.last_prefixes[3], x[:, :64])


def test_suffix_balanced_mrl_head_suffixes_exclude_full_by_default():
    x = torch.randn(4, 64)
    head = SuffixBalancedMRLHead([8, 16, 32, 64], num_classes=10)

    head(x)

    assert len(head.last_suffixes) == 3
    assert torch.equal(head.last_suffixes[0], x[:, -8:])
    assert torch.equal(head.last_suffixes[1], x[:, -16:])
    assert torch.equal(head.last_suffixes[2], x[:, -32:])


def test_suffix_balanced_mrl_auxiliary_loss_is_finite_scalar():
    x = torch.randn(4, 64)
    y = torch.randint(0, 10, (4,))
    head = SuffixBalancedMRLHead([8, 16, 32, 64], num_classes=10)

    head(x)
    aux = head.auxiliary_loss(y)
    expected = sum(F.cross_entropy(logits, y) for logits in head.last_suffix_logits)

    assert aux.dim() == 0
    assert torch.isfinite(aux)
    assert aux.item() >= 0
    assert torch.allclose(aux, expected)


def test_suffix_balanced_mrl_include_full_suffix():
    x = torch.randn(4, 64)
    head = SuffixBalancedMRLHead(
        nesting_list=[8, 16, 32, 64],
        num_classes=10,
        include_full_suffix=True,
    )

    head(x)

    assert len(head.last_suffixes) == 4
    assert torch.equal(head.last_suffixes[0], x[:, -8:])
    assert torch.equal(head.last_suffixes[1], x[:, -16:])
    assert torch.equal(head.last_suffixes[2], x[:, -32:])
    assert torch.equal(head.last_suffixes[3], x[:, -64:])


def test_suffix_balanced_mrl_has_no_suffix_weight_argument():
    with pytest.raises(TypeError):
        SuffixBalancedMRLHead(
            nesting_list=[8, 16, 32, 64],
            num_classes=10,
            suffix_weight=0.25,
        )


def test_idempotency():
	"""Tests losses of newer implementations are equal to the original for-loop
	implementation.
	"""
	original_forward = Matryoshka_CE_Loss.forward

	def forward_loop(self, output, target):
		"""Original implementation of forward() using for-loop
		"""
		loss=0
		N = len(output)
		for i in range(N):
			rel = 1.0 if self.relative_importance is None else self.relative_importance[i] 
			loss += rel*self.criterion(output[i], target)
		return loss

	try:
		relative_importance = [0.1, 0.9]

		# Current implementation
		torch.manual_seed(0)
		loss_fn = Matryoshka_CE_Loss(relative_importance=relative_importance)
		output_bc = torch.randn(2, 3, 5, requires_grad=True)
		# shape: [N batch size]
		target_bc = torch.empty(3, dtype=torch.long).random_(5)
		loss_broadcast = loss_fn(output_bc, target_bc)

		# Monkeypatching Original for-loop implementation
		torch.manual_seed(0)
		Matryoshka_CE_Loss.forward = forward_loop
		loss_org = Matryoshka_CE_Loss(relative_importance=relative_importance)
		output_org = torch.randn(2, 3, 5, requires_grad=True)
		# shape: [N batch size]
		target_org = torch.empty(3, dtype=torch.long).random_(5)
		loss_loop = loss_org(output_org, target_org)
	finally:
		Matryoshka_CE_Loss.forward = original_forward

	# Ensure the inputs to the loss fn are equal
	assert torch.equal(output_bc, output_org)
	assert torch.equal(target_bc, target_org)

	# Ensure the outputs are mostly equal
	assert torch.allclose(loss_loop, loss_broadcast)


def test_block_widths_from_nesting_list():
	assert block_widths_from_nesting_list([8, 16, 32]) == [8, 8, 16]

	for invalid in ([], [8, 8, 16], [16, 8], [0, 8]):
		with pytest.raises(ValueError):
			block_widths_from_nesting_list(invalid)


def test_block_orthogonal_layer_shapes():
	nesting_list = [8, 16, 32]
	x = torch.randn(4, 32)

	for mode in ("frozen", "orthogonal"):
		layer = BlockOrthogonalLayer(
			nesting_list,
			mode=mode,
			orthogonal_map="matrix_exp",
		)
		z, blocks = layer(x, return_blocks=True)
		assert z.shape == (4, 32)
		assert [block.shape for block in blocks] == [(4, 8), (4, 8), (4, 16)]


def test_block_orthogonal_prefix_gram_preservation():
	torch.manual_seed(0)
	nesting_list = [8, 16, 32]
	layer = BlockOrthogonalLayer(
		nesting_list,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
	)
	x = torch.randn(6, 32)
	z = layer(x)

	for dim in nesting_list:
		raw_gram = x[:, :dim] @ x[:, :dim].t()
		transformed_gram = z[:, :dim] @ z[:, :dim].t()
		assert torch.allclose(raw_gram, transformed_gram, atol=1e-4, rtol=1e-4)


def test_parametrized_blocks_are_orthogonal():
	layer = BlockOrthogonalLayer(
		[8, 16, 32],
		mode="orthogonal",
		orthogonal_map="matrix_exp",
	)
	for block in layer.blocks:
		weight = block.weight
		eye = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
		assert torch.allclose(weight @ weight.t(), eye, atol=1e-4, rtol=1e-4)
		assert torch.allclose(weight.t() @ weight, eye, atol=1e-4, rtol=1e-4)


def test_frozen_blocks_are_orthogonal_and_not_trainable():
	layer = BlockOrthogonalLayer(
		[8, 16, 32],
		mode="frozen",
		orthogonal_map="matrix_exp",
	)
	for block in layer.blocks:
		weight = block.weight
		eye = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
		assert torch.allclose(weight @ weight.t(), eye, atol=1e-4, rtol=1e-4)
		assert torch.allclose(weight.t() @ weight, eye, atol=1e-4, rtol=1e-4)
		assert not weight.requires_grad


def test_bor_mrl_head_output_shapes():
	head = BlockOrthogonalResidualMRLHead(
		[8, 16, 32],
		num_classes=10,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
	)
	output = head(torch.randn(5, 32))
	assert isinstance(output, tuple)
	assert len(output) == 3
	for logits in output:
		assert logits.shape == (5, 10)


def test_independent_block_bor_mrl_head_output_shapes():
	head = IndependentBlockOrthogonalMRLHead(
		[8, 16, 32],
		num_classes=10,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
	)
	output = head(torch.randn(5, 32))
	assert isinstance(output, tuple)
	assert len(output) == 3
	assert [prefix.shape for prefix in head.last_prefixes] == [(5, 8), (5, 16), (5, 32)]
	for logits in output:
		assert logits.shape == (5, 10)


def test_cascade_stop_gradient_mrl_head_output_shapes_and_prefixes():
	head = CascadeStopGradientMRLHead([8, 16, 32], num_classes=10)
	x = torch.randn(5, 32)

	output = head(x)

	assert isinstance(output, tuple)
	assert len(output) == 3
	assert head.block_widths == [8, 8, 16]
	assert [block.shape for block in head.last_blocks] == [(5, 8), (5, 8), (5, 16)]
	assert [prefix.shape for prefix in head.last_prefixes] == [(5, 8), (5, 16), (5, 32)]
	for dim, prefix in zip(head.nesting_list, head.last_prefixes):
		assert torch.equal(prefix, x[:, :dim])
	for logits in output:
		assert logits.shape == (5, 10)


def test_residual_aligned_mrl_head_output_shapes_and_auxiliary_loss():
	head = ResidualAlignedMRLHead(
		[8, 16, 32],
		num_classes=10,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
	)
	x = torch.randn(5, 32)

	output = head(x)
	aux = head.auxiliary_loss()
	log_dict = head.auxiliary_log_dict()

	assert isinstance(output, tuple)
	assert len(output) == 3
	assert [prefix.shape for prefix in head.last_prefixes] == [(5, 8), (5, 16), (5, 32)]
	assert [residual.shape for residual in head.last_residuals] == [(5, 8), (5, 16)]
	assert [rotated.shape for rotated in head.last_rotated_residuals] == [(5, 8), (5, 16)]
	assert aux.dim() == 0
	assert torch.isfinite(aux)
	assert log_dict["residual_aligned_mrl_mse"] >= 0.0
	assert log_dict["residual_aligned_mrl_cosine_distance"] >= 0.0
	for logits in output:
		assert logits.shape == (5, 10)


def test_residual_aligned_mrl_requires_doubling_nesting():
	with pytest.raises(ValueError):
		ResidualAlignedMRLHead([8, 24], num_classes=10)


def test_residual_aligned_mse_detaches_residual_input():
	head = ResidualAlignedMRLHead(
		[2, 4],
		num_classes=3,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
		mse_weight=1.0,
		cosine_weight=0.0,
		detach_prefix_target=True,
	)
	x = torch.randn(5, 4, requires_grad=True)

	head(x)
	loss = head.auxiliary_loss()
	loss.backward()

	if x.grad is not None:
		assert torch.allclose(x.grad, torch.zeros_like(x.grad))
	assert any(
		param.grad is not None
		for layer in head.residual_orthogonal_layers
		for param in layer.parameters()
		if param.requires_grad
	)


def test_residual_aligned_cosine_loss_trains_residual_direct_branch():
	head = ResidualAlignedMRLHead(
		[2, 4],
		num_classes=3,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
		mse_weight=0.0,
		cosine_weight=1.0,
		detach_prefix_target=True,
	)
	x = torch.randn(5, 4, requires_grad=True)

	head(x)
	loss = head.auxiliary_loss()
	loss.backward()

	assert x.grad is not None
	assert torch.allclose(x.grad[:, :2], torch.zeros_like(x.grad[:, :2]))
	assert x.grad[:, 2:].abs().sum() > 0


def test_adjacent_residual_prefix_cosine_stats():
	prefixes = [
		torch.tensor([[1.0, 0.0]]),
		torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
	]

	stats = adjacent_residual_prefix_cosine_stats(prefixes, [2, 4])

	assert stats["residual_prefix_alignment_pair_count"] == 1
	assert stats["residual_prefix_cosine_4_vs_2"] == pytest.approx(0.0)
	assert stats["residual_prefix_cosine_distance_4_vs_2"] == pytest.approx(1.0)


def test_recursive_link_preserves_exact_prefixes():
	nesting_list = [4, 8, 16]
	x = torch.randn(3, 16)
	head = RecursiveLinkMRLHead(nesting_list, num_classes=5)

	logits = head(x)

	assert len(logits) == 3
	for output in logits:
		assert output.shape == (3, 5)
	assert torch.equal(head.last_prefixes[1][:, :4], head.last_prefixes[0])
	assert torch.equal(head.last_prefixes[2][:, :8], head.last_prefixes[1])


def test_recursive_link_approximates_standard_mrl_with_tiny_alpha():
	nesting_list = [4, 8, 16]
	x = torch.randn(3, 16)
	head = RecursiveLinkMRLHead(
		nesting_list,
		num_classes=5,
		recursive_link_alpha_init=-20.0,
	)

	head(x)

	for dim, prefix in zip(nesting_list, head.last_prefixes):
		assert torch.allclose(prefix, x[:, :dim], atol=1e-6, rtol=1e-6)


def test_procrustes_cascade_distillation_loss_returns_finite_scalar():
	x = torch.randn(3, 16)
	prefixes = [x[:, :4], x[:, :8], x[:, :16]]

	loss = procrustes_cascade_distillation_loss(prefixes)

	assert loss.shape == torch.Size([])
	assert torch.isfinite(loss)
	assert loss.item() >= 0.0


def test_procrustes_cascade_distillation_loss_uses_float32_svd_under_autocast(monkeypatch):
	if not hasattr(torch, "autocast"):
		pytest.skip("torch.autocast is not available")

	seen_dtypes = []
	original_svd = torch.linalg.svd

	def wrapped_svd(input_tensor, *args, **kwargs):
		seen_dtypes.append(input_tensor.dtype)
		return original_svd(input_tensor, *args, **kwargs)

	monkeypatch.setattr(torch.linalg, "svd", wrapped_svd)
	x = torch.randn(4, 16)
	prefixes = [x[:, :4], x[:, :8], x[:, :16]]

	with torch.autocast("cpu", dtype=torch.bfloat16):
		loss = procrustes_cascade_distillation_loss(prefixes)

	assert seen_dtypes
	assert all(dtype == torch.float32 for dtype in seen_dtypes)
	assert torch.isfinite(loss)


def test_procrustes_cascade_distillation_loss_skips_large_pairs():
	x = torch.randn(3, 16)
	prefixes = [x[:, :4], x[:, :8], x[:, :16]]

	loss = procrustes_cascade_distillation_loss(prefixes, max_svd_dim=4)

	assert loss.shape == torch.Size([])
	assert torch.isfinite(loss)
	assert loss.item() >= 0.0


def test_orthogonal_transition_layer_builds_t_prefixes():
	layer = OrthogonalTransitionLayer(
		[8, 16, 32, 64],
		mode="orthogonal",
		orthogonal_map="matrix_exp",
	)
	x = torch.randn(4, 64)

	prefixes, blocks, t_outputs = layer(x, return_details=True)

	assert layer.get_block_widths() == [8, 8, 16, 32]
	assert layer.get_t_dims() == [16, 32, 64]
	assert [block.shape for block in blocks] == [(4, 8), (4, 8), (4, 16), (4, 32)]
	assert [prefix.shape for prefix in prefixes] == [(4, 8), (4, 16), (4, 32), (4, 64)]
	assert [t_output.shape for t_output in t_outputs] == [(4, 16), (4, 32), (4, 64)]
	assert torch.equal(prefixes[0], x[:, :8])
	for prefix, t_output in zip(prefixes[1:], t_outputs):
		assert torch.equal(prefix, t_output)


def test_orthogonal_transition_layer_t_layers_start_as_identity():
	layer = OrthogonalTransitionLayer(
		[8, 16, 32],
		mode="orthogonal",
		orthogonal_map="matrix_exp",
	)

	for t_layer in layer.t_layers:
		weight = t_layer.weight
		eye = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
		assert torch.allclose(weight, eye, atol=1e-4, rtol=1e-4)


def test_orthogonal_transition_layer_supports_householder_t_map():
	layer = OrthogonalTransitionLayer(
		[8, 16, 32],
		mode="orthogonal",
		orthogonal_map="householder",
	)
	x = torch.randn(4, 32)
	prefixes = layer(x)

	assert [prefix.shape for prefix in prefixes] == [(4, 8), (4, 16), (4, 32)]
	for t_layer in layer.t_layers:
		weight = t_layer.weight
		eye = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
		assert torch.allclose(weight, eye, atol=1e-4, rtol=1e-4)
		assert torch.allclose(weight @ weight.t(), eye, atol=1e-4, rtol=1e-4)


def test_frozen_orthogonal_transition_layer_t_layers_start_as_identity():
	layer = OrthogonalTransitionLayer([8, 16, 32], mode="frozen")

	for t_layer in layer.t_layers:
		weight = t_layer.weight
		eye = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
		assert torch.allclose(weight, eye, atol=1e-6, rtol=1e-6)
		assert not weight.requires_grad


def test_orthogonal_transition_layer_applies_each_t_to_raw_h_prefix():
	layer = OrthogonalTransitionLayer([8, 16, 32], mode="frozen")
	with torch.no_grad():
		layer.t_layers[0].weight.copy_(-torch.eye(16))
		layer.t_layers[1].weight.copy_(torch.eye(32))

	x = torch.randn(4, 32)
	prefixes = layer(x)

	assert torch.allclose(prefixes[0], x[:, :8])
	assert torch.allclose(prefixes[1], -x[:, :16])
	assert torch.allclose(prefixes[2], x[:, :32])


def test_t_orthogonal_mrl_head_output_shapes_and_orthogonal_t_layers():
	head = TOrthogonalMRLHead(
		[8, 16, 32, 64],
		num_classes=10,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
	)
	output = head(torch.randn(5, 64))

	assert isinstance(output, tuple)
	assert len(output) == 4
	assert head.get_t_dims() == [16, 32, 64]
	assert len(head.t_layers) == 3
	assert [layer.in_features for layer in head.t_layers] == [16, 32, 64]
	assert [layer.out_features for layer in head.t_layers] == [16, 32, 64]
	assert [prefix.shape for prefix in head.last_prefixes] == [(5, 8), (5, 16), (5, 32), (5, 64)]
	for logits in output:
		assert logits.shape == (5, 10)

	for layer in head.t_layers:
		weight = layer.weight
		eye = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
		assert torch.allclose(weight @ weight.t(), eye, atol=1e-4, rtol=1e-4)
		assert torch.allclose(weight.t() @ weight, eye, atol=1e-4, rtol=1e-4)


def test_t_orthogonal_mrl_backward():
	head = TOrthogonalMRLHead(
		[8, 16, 32],
		num_classes=10,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
	)
	x = torch.randn(5, 32, requires_grad=True)
	target = torch.empty(5, dtype=torch.long).random_(10)
	output = head(x)
	loss = sum(F.cross_entropy(logits, target) for logits in output)
	loss.backward()

	assert x.grad is not None
	assert torch.isfinite(x.grad).all()
	assert any(
		param.grad is not None
		for layer in head.t_layers
		for param in layer.parameters()
		if param.requires_grad
	)


def test_bor_mrl_uses_prefix_orthogonal_layers():
	head = BlockOrthogonalResidualMRLHead(
		[8, 16, 32],
		num_classes=10,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
	)
	assert len(head.prefix_orthogonal_layers) == 2
	assert [layer.in_features for layer in head.prefix_orthogonal_layers] == [8, 16]
	assert [layer.out_features for layer in head.prefix_orthogonal_layers] == [8, 16]


def test_gated_residual_orthogonal_adapter_preserves_shape():
	adapter = GatedResidualOrthogonalAdapter(
		8,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
	)
	x = torch.randn(4, 8)
	y = adapter(x)
	assert y.shape == x.shape


def test_gated_residual_orthogonal_adapter_alpha_init():
	adapter = GatedResidualOrthogonalAdapter(
		8,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
		alpha_init=-3.0,
	)
	assert adapter.alpha().item() == pytest.approx(torch.sigmoid(torch.tensor(-3.0)).item())


def test_gated_residual_orthogonal_adapter_alpha_gradients_flow():
	torch.manual_seed(0)
	adapter = GatedResidualOrthogonalAdapter(
		8,
		mode="frozen",
		alpha_init=-3.0,
	)
	x = torch.randn(4, 8)
	loss = adapter(x).pow(2).sum()
	loss.backward()

	assert adapter.alpha_logit.grad is not None
	assert torch.isfinite(adapter.alpha_logit.grad).item()
	assert adapter.alpha_logit.grad.abs().item() > 0


def test_bor_mrl_gated_residual_adapter_integration():
	head = BlockOrthogonalResidualMRLHead(
		[8, 16, 32],
		num_classes=10,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
		bor_residual_orthogonal=True,
		bor_residual_alpha_init=-3.0,
	)
	output = head(torch.randn(5, 32))

	assert isinstance(output, tuple)
	assert len(output) == 3
	assert [prefix.shape for prefix in head.last_prefixes] == [(5, 8), (5, 16), (5, 32)]
	assert all(
		isinstance(layer, GatedResidualOrthogonalAdapter)
		for layer in head.prefix_orthogonal_layers
	)
	assert head.alpha_values().shape == (2,)
	for logits in output:
		assert logits.shape == (5, 10)


def test_bor_mrl_hard_orthogonal_still_used_when_residual_flag_disabled():
	head = BlockOrthogonalResidualMRLHead(
		[8, 16, 32],
		num_classes=10,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
		bor_residual_orthogonal=False,
	)
	x = torch.randn(6, 32)
	head(x)

	assert not head.bor_residual_orthogonal
	assert head.alpha_values() is None
	assert not any(
		isinstance(layer, GatedResidualOrthogonalAdapter)
		for layer in head.prefix_orthogonal_layers
	)
	for dim, prefix in zip(head.nesting_list, head.last_prefixes):
		raw_gram = x[:, :dim] @ x[:, :dim].t()
		transformed_gram = prefix @ prefix.t()
		assert torch.allclose(raw_gram, transformed_gram, atol=1e-4, rtol=1e-4)


def test_independent_block_bor_mrl_uses_residual_block_layers():
	head = IndependentBlockOrthogonalMRLHead(
		[8, 16, 32],
		num_classes=10,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
	)
	assert head.block_transform.get_block_widths() == [8, 8, 16]
	assert len(head.block_transform.blocks) == 3
	assert [layer.in_features for layer in head.block_transform.blocks] == [8, 8, 16]
	assert [layer.out_features for layer in head.block_transform.blocks] == [8, 8, 16]


def test_bor_mrl_prefix_gram_preservation():
	head = BlockOrthogonalResidualMRLHead(
		[8, 16, 32],
		num_classes=10,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
	)
	x = torch.randn(6, 32)
	head(x)
	for dim, prefix in zip(head.nesting_list, head.last_prefixes):
		raw_gram = x[:, :dim] @ x[:, :dim].t()
		transformed_gram = prefix @ prefix.t()
		assert torch.allclose(raw_gram, transformed_gram, atol=1e-4, rtol=1e-4)


def test_independent_block_bor_mrl_prefix_gram_preservation():
	head = IndependentBlockOrthogonalMRLHead(
		[8, 16, 32],
		num_classes=10,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
	)
	x = torch.randn(6, 32)
	head(x)
	for dim, prefix in zip(head.nesting_list, head.last_prefixes):
		raw_gram = x[:, :dim] @ x[:, :dim].t()
		transformed_gram = prefix @ prefix.t()
		assert torch.allclose(raw_gram, transformed_gram, atol=1e-4, rtol=1e-4)


def test_bor_mrl_backward():
	head = BlockOrthogonalResidualMRLHead(
		[8, 16, 32],
		num_classes=10,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
	)
	x = torch.randn(5, 32, requires_grad=True)
	target = torch.empty(5, dtype=torch.long).random_(10)
	output = head(x)
	loss = sum(F.cross_entropy(logits, target) for logits in output)
	loss.backward()
	assert x.grad is not None
	assert torch.isfinite(x.grad).all()


def test_independent_block_bor_mrl_backward():
	head = IndependentBlockOrthogonalMRLHead(
		[8, 16, 32],
		num_classes=10,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
	)
	x = torch.randn(5, 32, requires_grad=True)
	target = torch.empty(5, dtype=torch.long).random_(10)
	output = head(x)
	loss = sum(F.cross_entropy(logits, target) for logits in output)
	loss.backward()
	assert x.grad is not None
	assert torch.isfinite(x.grad).all()


def test_bor_mrl_stop_gradient_blocks_larger_loss_from_earlier_prefix():
	head = BlockOrthogonalResidualMRLHead(
		[2, 4],
		num_classes=3,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
		stop_gradient=True,
	)
	with torch.no_grad():
		head.classifiers[-1].weight.fill_(1.0)
		head.classifiers[-1].bias.zero_()

	x = torch.randn(5, 4, requires_grad=True)
	loss = head(x)[-1].sum()
	loss.backward()

	assert x.grad is not None
	assert torch.allclose(x.grad[:, :2], torch.zeros_like(x.grad[:, :2]))
	assert x.grad[:, 2:].abs().sum() > 0


def test_bor_mrl_stop_gradient_can_be_disabled():
	head = BlockOrthogonalResidualMRLHead(
		[2, 4],
		num_classes=3,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
		stop_gradient=False,
	)
	with torch.no_grad():
		head.classifiers[-1].weight.fill_(1.0)
		head.classifiers[-1].bias.zero_()

	x = torch.randn(5, 4, requires_grad=True)
	loss = head(x)[-1].sum()
	loss.backward()

	assert x.grad is not None
	assert x.grad[:, :2].abs().sum() > 0
	assert x.grad[:, 2:].abs().sum() > 0


def test_bor_mrl_stop_gradient_defaults_to_disabled():
	head = BlockOrthogonalResidualMRLHead(
		[2, 4],
		num_classes=3,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
	)
	with torch.no_grad():
		head.classifiers[-1].weight.fill_(1.0)
		head.classifiers[-1].bias.zero_()

	x = torch.randn(5, 4, requires_grad=True)
	loss = head(x)[-1].sum()
	loss.backward()

	assert not head.stop_gradient
	assert x.grad is not None
	assert x.grad[:, :2].abs().sum() > 0
	assert x.grad[:, 2:].abs().sum() > 0


def test_cascade_stop_gradient_defaults_to_enabled():
	head = CascadeStopGradientMRLHead([2, 4], num_classes=3)
	with torch.no_grad():
		head.classifiers[-1].weight.fill_(1.0)
		head.classifiers[-1].bias.zero_()

	x = torch.randn(5, 4, requires_grad=True)
	loss = head(x)[-1].sum()
	loss.backward()

	assert head.stop_gradient
	assert x.grad is not None
	assert torch.allclose(x.grad[:, :2], torch.zeros_like(x.grad[:, :2]))
	assert x.grad[:, 2:].abs().sum() > 0


def test_cascade_stop_gradient_can_be_disabled():
	head = CascadeStopGradientMRLHead([2, 4], num_classes=3, stop_gradient=False)
	with torch.no_grad():
		head.classifiers[-1].weight.fill_(1.0)
		head.classifiers[-1].bias.zero_()

	x = torch.randn(5, 4, requires_grad=True)
	loss = head(x)[-1].sum()
	loss.backward()

	assert not head.stop_gradient
	assert x.grad is not None
	assert x.grad[:, :2].abs().sum() > 0
	assert x.grad[:, 2:].abs().sum() > 0


def test_independent_block_stop_gradient_can_be_enabled():
	head = IndependentBlockOrthogonalMRLHead(
		[2, 4],
		num_classes=3,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
		stop_gradient=True,
	)
	with torch.no_grad():
		head.classifiers[-1].weight.fill_(1.0)
		head.classifiers[-1].bias.zero_()

	x = torch.randn(5, 4, requires_grad=True)
	loss = head(x)[-1].sum()
	loss.backward()

	assert x.grad is None
	assert any(
		param.grad is not None
		for block in head.block_transform.blocks
		for param in block.parameters()
		if param.requires_grad
	)


def test_independent_block_stop_gradient_can_be_disabled():
	head = IndependentBlockOrthogonalMRLHead(
		[2, 4],
		num_classes=3,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
		stop_gradient=False,
	)
	with torch.no_grad():
		head.classifiers[-1].weight.fill_(1.0)
		head.classifiers[-1].bias.zero_()

	x = torch.randn(5, 4, requires_grad=True)
	loss = head(x)[-1].sum()
	loss.backward()

	assert x.grad is not None
	assert x.grad.abs().sum() > 0


def test_independent_block_stop_gradient_defaults_to_disabled():
	head = IndependentBlockOrthogonalMRLHead(
		[2, 4],
		num_classes=3,
		mode="orthogonal",
		orthogonal_map="matrix_exp",
	)
	with torch.no_grad():
		head.classifiers[-1].weight.fill_(1.0)
		head.classifiers[-1].bias.zero_()

	x = torch.randn(5, 4, requires_grad=True)
	loss = head(x)[-1].sum()
	loss.backward()

	assert not head.stop_gradient
	assert x.grad is not None
	assert x.grad.abs().sum() > 0


def test_retrieval_metrics_use_database_relevant_counts():
	db_labels = np.array([0, 0, 1, 1, 1, 2])
	query_labels = np.array([0, 1, 2])
	neighbors = np.array([
		[1, 2, 0],
		[0, 2, 4],
		[3, 5, 0],
	])

	metrics = compute_retrieval_metrics_at_k(query_labels, db_labels, neighbors, k=2)

	assert metrics["mAP"] == pytest.approx((0.5 + 0.25 + 0.25) / 3)
	assert metrics["precision"] == pytest.approx(0.5)
	assert metrics["recall"] == pytest.approx(((1 / 2) + (1 / 3) + 1) / 3)
	assert metrics["topk"] == pytest.approx(1.0)
	assert top1_accuracy(query_labels, db_labels, neighbors) == pytest.approx(1 / 3)


def test_retrieval_metrics_accept_column_labels():
	db_labels = np.array([[0], [0], [1], [1], [1], [2]], dtype=np.float16)
	query_labels = np.array([[0], [1], [2]], dtype=np.float16)
	neighbors = np.array([
		[1, 2, 0],
		[0, 2, 4],
		[3, 5, 0],
	])

	assert relevant_counts_by_label(db_labels) == {0: 2, 1: 3, 2: 1}
	metrics = compute_retrieval_metrics_at_k(query_labels, db_labels, neighbors, k=2)

	assert metrics["recall"] == pytest.approx(((1 / 2) + (1 / 3) + 1) / 3)


def test_bor_frozen_uses_nontrainable_orthogonal_prefix_layers():
	head = BlockOrthogonalResidualMRLHead(
		[8, 16, 32],
		num_classes=10,
		mode="frozen",
	)
	x = torch.randn(5, 32)
	output = head(x)
	for layer in head.prefix_orthogonal_layers:
		weight = layer.weight
		eye = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
		assert torch.allclose(weight @ weight.t(), eye, atol=1e-4, rtol=1e-4)
		assert torch.allclose(weight.t() @ weight, eye, atol=1e-4, rtol=1e-4)
		assert not weight.requires_grad
	assert [block.shape[1] for block in head.last_blocks] == [8, 8, 16]
	assert len(output) == 3
	for logits in output:
		assert logits.shape == (5, 10)


def test_identity_mode_is_not_supported():
	with pytest.raises(ValueError):
		BlockOrthogonalLayer([8, 16, 32], mode="identity")
	with pytest.raises(ValueError):
		BlockOrthogonalResidualMRLHead([8, 16, 32], num_classes=10, mode="identity")
