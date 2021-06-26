# \[CVPR 2021\] Robust Consistent Video Depth Estimation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YOLXsb4JUOD1wt5TXB2ln78koocm-bu8?usp=sharing)

This repository contains Python and C++ implementation of Robust Consistent Video Depth, as described in the paper

`Johannes Kopf, Xuejian Rong, and Jia-Bin Huang. Robust Consistent Video Despth Estimation. CVPR 2021`

###  [Project](https://robust-cvd.github.io/) | [Paper](https://arxiv.org/pdf/2012.05901.pdf) | [Video](https://www.youtube.com/watch?v=x-wHrYHJSm8) | [Colab](https://colab.research.google.com/drive/1YOLXsb4JUOD1wt5TXB2ln78koocm-bu8?usp=sharing)

We present an algorithm for estimating consistent dense depth maps and camera poses from a monocular video. We integrate a learning-based depth prior, in the form of a convolutional neural network trained for single-image depth estimation, with geometric optimization, to estimate a smooth camera trajectory as well as detailed and stable depth reconstruction.

[![teaser](https://robust-cvd.github.io/Robust_Consistent_Video_Depth_Estimation_files/teaser.png)](https://www.youtube.com/watch?v=x-wHrYHJSm8)



## Changelog

- `[June 2021]` Released the companion Colab notebook.
- `[June 2021]` Initial release of Robust CVD.

## Installation

Please refer to the colab notebook for how to install the dependencies.

## Running

Please refer to the colab notebook for how to run the cli tool for now.

## Result Folder Structure

```bash
frames.txt              # meta data about number of frames, image resolution and timestamps for each frame
color_full/             # extracted frames in the original resolution
color_down/             # extracted frames in the resolution for disparity estimation 
color_down_png/      
color_flow/             # extracted frames in the resolution for flow estimation
flow_list.json          # indices of frame pairs to finetune the model with
flow/                   # optical flow 
mask/                   # mask of consistent flow estimation between frame pairs.
vis_flow/               # optical flow visualization. Green regions contain inconsistent flow. 
vis_flow_warped/        # visualzing flow accuracy by warping one frame to another using the estimated flow. e.g., frame_000000_000032_warped.png warps frame_000032 to frame_000000.
depth_${model_type}/    # initial disparity estimation using the original monocular depth model before test-time training
R_hierarchical2_${model_type}/ 
    flow_list_0.20.json                 # indices of frame pairs passing overlap ratio test of threshold 0.2. Same content as ../flow_list.json.
    videos/                             # video visualization of results 
    B0.1_R1.0_PL1-0_LR0.0004_BS4_Oadam/
        checkpoints/                    # checkpoint after each epoch
        depth/                          # final disparity map results after finishing test-time training
        eval/                           # intermediate losses and disparity maps after each epoch 
        tensorboard/                    # tensorboard log for the test-time training process

```

## Citation
If you find our work useful in your research, please consider citing:
```BibTeX
@inproceedings{kopf2021rcvd,
 title={Robust Consistent Video Depth Estimation},
 author={Kopf, Johannes and Rong, Xuejian and Huang, Jia-Bin},
 year={2021},
 booktitle=IEEE/CVF Conference on Computer Vision and Pattern Recognition
}
```
## License
See the [LICENSE](LICENSE) for more details.

## Issues & Help
For help or issues using Robust CVD, please submit a GitHub issue or a PR request.

Before you do this, make sure you have checked [CODE_OF_CONDUCT](./CODE_OF_CONDUCT.md), [CONTRIBUTING](./CONTRIBUTING.md), [ISSUE_TEMPLATE](docs/.github/ISSUE_TEMPLATE.md), and [PR_TEMPLATE](docs/.github/PR_TEMPLATE.md).

## Acknowledgements
Check our previous work on [Consistent Video Depth Estimation](https://github.com/facebookresearch/consistent_depth).

We also thank the authors for releasing [PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch), [Ceres Solver](http://ceres-solver.org/), [OpenCV](http://opencv.org/), [Eigen](https://eigen.tuxfamily.org/), [MiDaS](https://github.com/intel-isl/MiDaS), [RAFT](https://github.com/princeton-vl/RAFT), and [detectron2](https://github.com/facebookresearch/detectron2).

