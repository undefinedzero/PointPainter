# PointPainter
## Segmentation Guidance for 3D object Detection

## Overview
This project uses segmentation information to guide the Lidar 3D detection both in training and inference. This guidance may come from stereo images, Lidar point cloud itself, or ground truth information. The model we implemented consists of four main stages:
1. Semantic segmentation: an image or Lidar based semantic segmentation network which computes the pixel-wise or point-wise segmentation scores;
2. Fusion: Lidar points are “painted” by projected image segmentation scores or directly Lidar segmentation scores. These segmentation scores will serve as additional feature coordinates to spatial positions of Lidar points;
3. Ground truth sampling: when training, use the ground truth 3D boxes to the segment point clouds so that we could augment the training data;
4. 3D object detection: a Lidar based 3D detection network to estimate oriented 3D boxes from these "painted" point clouds.
![flow](./EvalResults/flow.png)

## Related Publication
PointPainting:
   
    @inproceedings{vora2020pointpainting,
    title={Pointpainting: Sequential fusion for 3d object detection},
    author={Vora, Sourabh and et al},
    booktitle={CVPR},
    pages={4604--4612},
    year={2020}
    }
   
SECOND:
   
    @article{yan2018second,
    title={Second: Sparsely embedded convolutional detection},
    author={Yan, Yan and et al},
    journal={Sensors},
    volume={18},
    number={10},
    pages={3337},
    year={2018},
    publisher={Multidisciplinary Digital Publishing Institute}
    }

## Usage
### Lidar detectors
We use the codebase named [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) to implement our 3D detection networks. Follow instrcutions of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) to install.

### Mask generators
We use different models to generate semantic segmentations. Please follow the `README` in `MaskGenerator` folder.

## Results

### Painted point cloud and detection results
![painted68](./EvalResults/demo68all.png)
![painted145](./EvalResults/demo145all.png)

### Evaluation results on KITTI validation set
![eval](./EvalResults/eva.png)
 We calculate the AP in BEV(bird's eye view) metrics. V, P, C represents the vehicle, pedestrian, cyclist class, and e, m, h represents easy, moderate, and hard.

 ## Video


