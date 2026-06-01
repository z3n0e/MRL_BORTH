#!/bin/bash

# Serialize CIFAR-100 into FFCV files using the same writer used for ImageNet.

set -e

: "${CIFAR100_DIR:=$HOME/.cache/torchvision}"
: "${WRITE_DIR:?WRITE_DIR must be set}"

MAX_RESOLUTION=${1:-32}
JPEG_QUALITY=${2:-90}

write_dataset () {
    split=$1
    write_path=$WRITE_DIR/cifar100_${split}_${MAX_RESOLUTION}_raw.ffcv
    echo "Writing CIFAR-100 ${split} dataset to ${write_path}"
    python write_imagenet.py \
        --cfg.dataset=cifar100 \
        --cfg.split=${split} \
        --cfg.data_dir=$CIFAR100_DIR \
        --cfg.write_path=$write_path \
        --cfg.max_resolution=$MAX_RESOLUTION \
        --cfg.write_mode=raw \
        --cfg.jpeg_quality=$JPEG_QUALITY
}

write_dataset train
write_dataset val
