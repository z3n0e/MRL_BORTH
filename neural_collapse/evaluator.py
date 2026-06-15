from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from MRL import FixedFeatureLayer, MRL_Linear_Layer
from .metrics import nc_metrics


def base_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def topk_correct(logits: torch.Tensor, target: torch.Tensor, k: int) -> int:
    k = min(int(k), logits.shape[1])
    pred = logits.topk(k=k, dim=1, largest=True, sorted=True).indices
    return int(pred.eq(target.view(-1, 1)).any(dim=1).sum().item())


def classifier_weight(model: nn.Module, dim: int) -> torch.Tensor:
    fc = base_model(model).fc
    if isinstance(fc, MRL_Linear_Layer):
        if fc.efficient:
            return fc.nesting_classifier_0.weight[:, :dim]
        idx = fc.nesting_list.index(dim)
        return getattr(fc, f"nesting_classifier_{idx}").weight
    if isinstance(fc, FixedFeatureLayer):
        return fc.weight[:, :dim]
    return fc.weight[:, :dim]


def _output_tuple(output) -> Tuple[torch.Tensor, ...]:
    if isinstance(output, torch.Tensor):
        return (output,)
    return tuple(output)


def collect_features_and_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    prefix_dims: Iterable[int],
    is_nested: bool,
    *,
    desc: str = "nc collect",
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[int, torch.Tensor],
    Dict[int, float],
    Dict[int, float],
    Dict[int, float],
]:
    model.eval()
    prefix_dims = [int(dim) for dim in prefix_dims]
    features_by_batch: List[torch.Tensor] = []
    labels_by_batch: List[torch.Tensor] = []
    logits_by_dim: Dict[int, List[torch.Tensor]] = {dim: [] for dim in prefix_dims}
    loss_sum = {dim: 0.0 for dim in prefix_dims}
    top1_sum = {dim: 0 for dim in prefix_dims}
    top5_sum = {dim: 0 for dim in prefix_dims}
    seen = 0
    activation: Dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        activation["avgpool"] = output.detach()

    handle = base_model(model).avgpool.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for images, target in tqdm(loader, desc=desc, leave=False):
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                if device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)

                output = model(images)
                features = activation.pop("avgpool").flatten(1)
                features_by_batch.append(features.cpu())
                labels_by_batch.append(target.cpu())

                outputs = _output_tuple(output) if is_nested else (output,)
                if len(outputs) != len(prefix_dims):
                    raise RuntimeError(
                        f"Expected {len(prefix_dims)} prefix logits, got {len(outputs)}"
                    )

                batch_size = target.numel()
                seen += batch_size
                for dim, logits in zip(prefix_dims, outputs):
                    logits_by_dim[dim].append(logits.cpu())
                    loss_sum[dim] += float(F.cross_entropy(logits, target, reduction="sum").item())
                    top1_sum[dim] += topk_correct(logits, target, 1)
                    top5_sum[dim] += topk_correct(logits, target, 5)
    finally:
        handle.remove()

    features = torch.cat(features_by_batch, dim=0)
    labels = torch.cat(labels_by_batch, dim=0)
    logits_cat = {dim: torch.cat(parts, dim=0) for dim, parts in logits_by_dim.items()}
    losses = {dim: loss_sum[dim] / seen for dim in prefix_dims}
    top1 = {dim: top1_sum[dim] / seen for dim in prefix_dims}
    top5 = {dim: top5_sum[dim] / seen for dim in prefix_dims}
    return features, labels, logits_cat, losses, top1, top5


def evaluate_nc_rows(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    prefix_dims: Iterable[int],
    is_nested: bool,
    split: str,
    num_classes: int,
    dataset: str,
    arch: str,
    mode: str,
    feature_dim: int,
    cifar_stem: bool = False,
    epoch: int | None = None,
    train_loss: float | None = None,
    event: str | None = None,
) -> List[Dict[str, float | int | str | bool]]:
    prefix_dims = [int(dim) for dim in prefix_dims]
    features, labels, logits_by_dim, loss_by_dim, top1_by_dim, top5_by_dim = collect_features_and_logits(
        model,
        loader,
        device,
        prefix_dims,
        is_nested,
    )

    rows: List[Dict[str, float | int | str | bool]] = []
    for dim in prefix_dims:
        metrics = nc_metrics(
            features=features[:, :dim],
            labels=labels,
            classifier_weight=classifier_weight(model, dim).detach().cpu(),
            num_classes=num_classes,
            logits=logits_by_dim[dim],
        )
        name_parts = [mode, split]
        if epoch is not None:
            name_parts.append(f"epoch{epoch}")
        name_parts.append(f"d{dim}")
        row: Dict[str, float | int | str | bool] = {
            "name": "_".join(name_parts),
            "dataset": dataset,
            "arch": arch,
            "cifar_stem": bool(cifar_stem),
            "mode": mode,
            "split": split,
            "prefix_dim": int(dim),
            "model_feature_dim": int(feature_dim),
            "loss": float(loss_by_dim[dim]),
            "accuracy": float(top1_by_dim[dim]),
            "top5": float(top5_by_dim[dim]),
        }
        if event is not None:
            row["event"] = event
        if epoch is not None:
            row["epoch"] = int(epoch)
        if train_loss is not None:
            row["train_loss"] = float(train_loss)
        row.update(metrics)
        rows.append(row)
    return rows
