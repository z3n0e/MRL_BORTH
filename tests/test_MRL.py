import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cifar_resnet import apply_cifar_resnet_stem, build_power2_prefix_dims, parse_prefix_dims
from MRL import (
    FixedFeatureLayer,
    Matryoshka_CE_Loss,
    MRL_Linear_Layer,
    block_widths_from_nesting_list,
    mask_previous_prefix_features,
    mrl_sampling_probabilities,
)
from retrieval.metrics import (
    cmc_at_k,
    compute_retrieval_metrics_at_k,
    relevant_counts_by_label,
    top1_accuracy,
)
from neural_collapse.metrics import nc_metrics


def test_forward():
    loss_fn = Matryoshka_CE_Loss()
    output = torch.randn(2, 3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)

    loss = loss_fn(output, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_relative_importance():
    loss_fn = Matryoshka_CE_Loss(relative_importance=[0.1, 0.9])
    output = torch.randn(2, 3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)

    loss = loss_fn(output, target)

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


def test_block_widths_from_nesting_list():
    assert block_widths_from_nesting_list([8, 16, 32]) == [8, 8, 16]
    for invalid in ([], [8, 8, 16], [16, 8], [0, 8]):
        with pytest.raises(ValueError):
            block_widths_from_nesting_list(invalid)


def test_prefix_mask_masks_only_previous_prefix(monkeypatch):
    def fake_rand(*shape, device=None):
        return torch.tensor([[0.25, 0.75]], device=device).expand(shape)

    monkeypatch.setattr(torch, "rand", fake_rand)
    features = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    masked = mask_previous_prefix_features(
        features,
        previous_dim=2,
        mask_prob=0.5,
        scale="inverted",
    )

    assert torch.equal(masked[:, :2], torch.tensor([[0.0, 4.0]]))
    assert torch.equal(masked[:, 2:], features[:, 2:])
    assert torch.equal(features, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))


def test_mrl_linear_layer_prefix_mask_training_only(monkeypatch):
    def fake_rand(*shape, device=None):
        return torch.zeros(shape, device=device)

    monkeypatch.setattr(torch, "rand", fake_rand)
    layer = MRL_Linear_Layer(
        [2, 4],
        num_classes=1,
        efficient=False,
        prefix_mask_prob=0.5,
        bias=False,
    )
    for idx in range(2):
        getattr(layer, f"nesting_classifier_{idx}").weight.data.fill_(1.0)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    train_logits = layer(x)
    layer.eval()
    eval_logits = layer(x)

    assert train_logits[0].item() == pytest.approx(3.0)
    assert train_logits[1].item() == pytest.approx(7.0)
    assert eval_logits[0].item() == pytest.approx(3.0)
    assert eval_logits[1].item() == pytest.approx(10.0)


def test_mrl_efficient_layer_prefix_mask(monkeypatch):
    def fake_rand(*shape, device=None):
        return torch.zeros(shape, device=device)

    monkeypatch.setattr(torch, "rand", fake_rand)
    layer = MRL_Linear_Layer(
        [2, 4],
        num_classes=1,
        efficient=True,
        prefix_mask_prob=0.5,
        bias=False,
    )
    layer.nesting_classifier_0.weight.data.fill_(1.0)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    logits = layer(x)

    assert logits[0].item() == pytest.approx(3.0)
    assert logits[1].item() == pytest.approx(7.0)


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


def test_resnet18_cifar_prefix_dims():
    assert build_power2_prefix_dims(512, nesting_start=3) == [8, 16, 32, 64, 128, 256, 512]
    assert parse_prefix_dims("8,16,32", feature_dim=512) == [8, 16, 32, 512]


def test_apply_cifar_resnet_stem_replaces_conv_and_pool():
    class TinyResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    model = apply_cifar_resnet_stem(TinyResNet())

    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert model.conv1.padding == (1, 1)
    assert isinstance(model.maxpool, nn.Identity)


def test_idempotency():
    original_forward = Matryoshka_CE_Loss.forward

    def forward_loop(self, output, target):
        loss = 0
        for i in range(len(output)):
            rel = 1.0 if self.relative_importance is None else self.relative_importance[i]
            loss += rel * self.criterion(output[i], target)
        return loss

    try:
        relative_importance = [0.1, 0.9]
        torch.manual_seed(0)
        loss_fn = Matryoshka_CE_Loss(relative_importance=relative_importance)
        output_bc = torch.randn(2, 3, 5, requires_grad=True)
        target_bc = torch.empty(3, dtype=torch.long).random_(5)
        loss_broadcast = loss_fn(output_bc, target_bc)

        torch.manual_seed(0)
        Matryoshka_CE_Loss.forward = forward_loop
        loss_org = Matryoshka_CE_Loss(relative_importance=relative_importance)
        output_org = torch.randn(2, 3, 5, requires_grad=True)
        target_org = torch.empty(3, dtype=torch.long).random_(5)
        loss_loop = loss_org(output_org, target_org)
    finally:
        Matryoshka_CE_Loss.forward = original_forward

    assert torch.equal(output_bc, output_org)
    assert torch.equal(target_bc, target_org)
    assert torch.allclose(loss_loop, loss_broadcast)


def test_relevant_counts_by_label_counts_database_labels():
    db_labels = np.array([[0], [0], [1], [1], [1]])

    counts = relevant_counts_by_label(db_labels)

    assert counts == {0: 2, 1: 3}


def test_retrieval_metrics_at_k():
    query_labels = np.array([[0], [1]])
    db_labels = np.array([[0], [0], [1], [1]])
    neighbors = np.array([
        [1, 2, 3],
        [2, 0, 3],
    ])

    metrics = compute_retrieval_metrics_at_k(query_labels, db_labels, neighbors, k=3)

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.75)
    assert metrics["topk"] == pytest.approx(1.0)
    assert metrics["mAP"] == pytest.approx((1.0 / 3.0 + (1.0 + 2.0 / 3.0) / 3.0) / 2.0)


def test_cmc_at_k():
    query_labels = np.array([[0], [1], [2]])
    db_labels = np.array([[1], [0], [2], [1]])
    neighbors = np.array([
        [0, 1, 2],
        [1, 3, 0],
        [0, 3, 2],
    ])

    assert cmc_at_k(query_labels, db_labels, neighbors, 1) == pytest.approx(0.0)
    assert cmc_at_k(query_labels, db_labels, neighbors, 2) == pytest.approx(2.0 / 3.0)
    assert cmc_at_k(query_labels, db_labels, neighbors, 3) == pytest.approx(1.0)


def test_top1_accuracy():
    query_labels = np.array([[0], [1], [2]])
    db_labels = np.array([[1], [0], [2]])
    neighbors = np.array([
        [1, 0],
        [0, 1],
        [2, 1],
    ])

    assert top1_accuracy(query_labels, db_labels, neighbors) == pytest.approx(1.0)


def test_nc_metrics_collapsed_features_have_low_nc1_and_zero_nc4():
    features = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [-0.5, 0.8660254],
        [-0.5, 0.8660254],
        [-0.5, -0.8660254],
        [-0.5, -0.8660254],
    ])
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    classifier_weight = torch.tensor([
        [1.0, 0.0],
        [-0.5, 0.8660254],
        [-0.5, -0.8660254],
    ])
    logits = features @ classifier_weight.t()

    metrics = nc_metrics(features, labels, classifier_weight, num_classes=3, logits=logits)

    assert metrics["nc1_within_to_between"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["nc4_ncc_mismatch"] == pytest.approx(0.0)
    assert metrics["nc2_etf_feasible"] == pytest.approx(1.0)
    assert "nc1" in metrics
    assert "nc3_align_mean" in metrics
    assert "gnc2_weight_margin" in metrics
    assert "gnc2_class_mean_margin" in metrics
    assert metrics["class_mean_effective_rank"] == pytest.approx(2.0)


def test_nc_metrics_noncollapsed_features_increase_nc1():
    collapsed = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [-0.5, 0.8660254],
        [-0.5, 0.8660254],
        [-0.5, -0.8660254],
        [-0.5, -0.8660254],
    ])
    noise = torch.tensor([
        [0.30, 0.10],
        [-0.25, -0.20],
        [0.20, -0.15],
        [-0.30, 0.12],
        [0.15, 0.25],
        [-0.18, -0.22],
    ])
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    classifier_weight = torch.tensor([
        [1.0, 0.0],
        [-0.5, 0.8660254],
        [-0.5, -0.8660254],
    ])

    collapsed_metrics = nc_metrics(
        collapsed,
        labels,
        classifier_weight,
        num_classes=3,
        logits=collapsed @ classifier_weight.t(),
    )
    noisy = collapsed + noise
    noisy_metrics = nc_metrics(
        noisy,
        labels,
        classifier_weight,
        num_classes=3,
        logits=noisy @ classifier_weight.t(),
    )

    assert noisy_metrics["nc1_within_to_between"] > collapsed_metrics["nc1_within_to_between"]


def test_nc2_etf_infeasible_below_class_threshold():
    torch.manual_seed(0)
    num_classes = 10
    features = torch.randn(num_classes * 2, 8)
    labels = torch.arange(num_classes).repeat_interleave(2)
    classifier_weight = torch.randn(num_classes, 8)

    metrics = nc_metrics(features, labels, classifier_weight, num_classes=num_classes)

    assert metrics["nc2_etf_feasible"] == pytest.approx(0.0)
    assert np.isnan(metrics["nc2_etf_error"])
