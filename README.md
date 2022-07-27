# [Details](https://github.com/rtharungowda/navigate-surroundings/blob/main/KYS%20slides_ed.pdf)
# Obj-Det-Stereo-Depth-estimation

Help visually impaired navigate their way using stereo images obatined and depth estimation using triangulation and object detectoin and localization using state of the art YOLOv5 models.

![Hitnet stereo depth estimation](https://github.com/ibaiGorordo/HITNET-Stereo-Depth-estimation/blob/main/doc/img/out.jpg)
*Stereo depth estimation on the cones images from the Middlebury dataset (https://vision.middlebury.edu/stereo/data/scenes2003/)*

## Run

Import and use in google colab to avoid any dependancy issuses and enable gpu for faster inference.
Run the following in shell to execute code.

```
python3 obj_det_depth.py
```

# References:
* Hitnet model: https://github.com/google-research/google-research/tree/master/hitnet
* DrivingStereo dataset: https://drivingstereo-dataset.github.io/
* Original paper: https://arxiv.org/abs/2007.12140
