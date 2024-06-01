# OVER-NAV: Iterative Vision-Language Navigation with Open-Vocabulary Detection and StructurEd Representation

This is the official implementation of IVLN part of our paper [OVER-NAV: Iterative Vision-Language Navigation with Open-Vocabulary Detection and StructurEd Representation](https://arxiv.org/abs/2403.17334).

### Installation

1. Following [IVLN](https://github.com/Bill1235813/IVLN) to prepare the environment.

2. Download the neccessary files from [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/zhaogl_connect_hku_hk/Eus7RS6c20xMii2wVvBPRWIBDR0bqRO3h4q5OZSS05MVng?e=JRRsMP).

3. Generate the images of the Matterport3D environments with ``mattersim_to_image.py`` and ``mattersim_to_image_viewpoints_not_in_path.py``.

4. Run ``train.sh`` to train the model. 

### Citation

If you find this repo useful, please consider citing:

```
@article{zhao2024over,
  title={OVER-NAV: Elevating Iterative Vision-and-Language Navigation with Open-Vocabulary Detection and StructurEd Representation},
  author={Zhao, Ganlong and Li, Guanbin and Chen, Weikai and Yu, Yizhou},
  journal={arXiv preprint arXiv:2403.17334},
  year={2024}
}
```