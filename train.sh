#!/usr/bin/env bash

cd ./LidarDetector/tools
./scripts/dist_train.sh 2 --cfg_file ./cfgs/kitti_models/pp-gt.yaml --batch_size 24
./scripts/dist_train.sh 2 --cfg_file ./cfgs/kitti_models/pp.yaml --batch_size 24
