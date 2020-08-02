#!/bin/bash
# shell script to serialize lrs3 dataset.
#
# Author : NoUnique (kofmap@gmail.com)
# Copyright 2020 NoUnique. All Rights Reserved

TRAINSET_PART_FILES=(
    "lrs3_pretrain_partaa"
    "lrs3_pretrain_partab"
    "lrs3_pretrain_partac"
    "lrs3_pretrain_partad"
    "lrs3_pretrain_partae"
    "lrs3_pretrain_partaf"
    "lrs3_pretrain_partag"
)
TRAINSET_FILE="lrs3_pretrain.zip"
VALIDSET_FILE="lrs3_trainval.zip"
TESTSET_FILE="lrs3_test_v0.4.zip"

SCRIPT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SRC_DIR=${1:-"/data/lrs3"}
DST_DIR=${2:-${SRC_DIR}}

set -x  # echo on

cd ${SRC_DIR}

# Check if train(pretrain) set exists
# If there is no concatenated train set zip file, concatenate it using part files
if [[ ! -f ${TRAINSET_FILE} ]]; then
    echo "${TRAINSET_FILE} does not exist. Try to concatenate part files."
    for file in ${TRAINSET_PART_FILES[@]}; do
        if [[ ! -f ${file} ]]; then
          echo "${file} does not exist."
          exit
        fi
    done
    echo "All part files are exist. Start to concatenate part files."
    cat ${TRAINSET_PART_FILES[@]} > ${TRAINSET_FILE}
    echo "${TRAINSET_FILE} is generated"
fi

# Check if valid(trainval) set exists
if [[ ! -f ${VALIDSET_FILE} ]]; then
    echo "${VALIDSET_FILE} does not exist."
fi

# Check if test set exists
if [[ ! -f ${TESTSET_FILE} ]]; then
    echo "${TESTSET_FILE} does not exist."
fi

cd ${SCRIPT_DIR}
python -m tensorflow_datasets.scripts.download_and_prepare \
    --datasets=lrs3 \
    --module_import=lrs3 \
    --manual_dir=${SRC_DIR} \
    --data_dir=${DST_DIR}
