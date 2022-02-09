#!/bin/bash

# Build the project

SCRIPT_DIR=$(
    cd $(dirname $0)
    pwd
)

BUILD_DIR=${SCRIPT_DIR}"/build"

mkdir -p ${BUILD_DIR} && cd $_

# Change gcc version to 7 because cuda10.0 is not supported by gcc7~
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 100

source ~/venv/python3.7/pointnet/bin/activate &&
    cmake .. -DCMAKE_C_COMPILER=/usr/bin/gcc &&
    make

# change gcc version back to default
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 70
