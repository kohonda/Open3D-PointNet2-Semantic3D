#!/bin/bash

# Build the project

SCRIPT_DIR=$(
    cd $(dirname $0)
    pwd
)

BUILD_DIR=${SCRIPT_DIR}"/build"

mkdir -p ${BUILD_DIR} && cd $_

source ~/venv/python3.7/pointnet/bin/activate &&
    cmake .. -DPYTHON_EXECUTABLE=~/venv/python3.7/pointnet/bin/python&&
    make -j
