# Image Retrieval

Retrieval uses raw ResNet avgpool features saved by [pytorch_inference.py](/home/sricci/Desktop/MRL_BORTH/inference/pytorch_inference.py). MRL performance is evaluated by slicing those saved features to each prefix dimension.

## Dump Feature Arrays

ImageNet-1K:

```bash
cd ../inference

python pytorch_inference.py \
  --retrieval \
  --path path_to_model/final_weights.pt \
  --dataset 1K \
  --data_root /path/to/imagenet \
  --retrieval_array_path ../retrieval_arrays \
  --mrl
```

CIFAR-100:

```bash
python pytorch_inference.py \
  --retrieval \
  --path path_to_model/final_weights.pt \
  --dataset CIFAR100 \
  --data_root /path/to/cifar100/root \
  --retrieval_array_path ../retrieval_arrays \
  --mrl
```

This writes `*-X.npy` feature arrays and `*-y.npy` label arrays for database and query splits.

## Build Neighbors

```bash
cd ../retrieval

python faiss_nn.py \
  --root ../retrieval_arrays \
  --dataset CIFAR100 \
  --model mrl \
  --index-type exactl2 \
  --k 2048 \
  --dims 8 16 32 64 128 256 512 1024 2048
```

The script builds one nearest-neighbor shortlist per requested prefix dimension.

## Compute Metrics

```bash
python compute_metrics.py \
  --root ../retrieval_arrays \
  --dataset CIFAR100 \
  --model mrl \
  --eval-config vanilla \
  --index-type exactl2 \
  --shortlist 10 25 50 100
```

Metrics include mAP@k, precision@k, recall@k, and top-k retrieval accuracy. Recall is computed from the number of database examples with the query label, so the same code supports ImageNet and CIFAR-100.

Use `--model mrl_e` for MRL-E feature arrays. Fixed-feature compatibility is available with `--model ff --rep-size <dim>`.
