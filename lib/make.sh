#!/usr/bin/env bash


python setup_bbox.py build_ext --inplace --user
rm -rf build

python setup_layers.py build develop --user

