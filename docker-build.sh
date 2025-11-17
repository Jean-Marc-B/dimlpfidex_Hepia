#!/bin/bash

set -e

rm -rf build bin dist
mkdir -p build
cd build 
cmake .. 
cmake --build . -j$(nproc)
cd ..
python3 -m build