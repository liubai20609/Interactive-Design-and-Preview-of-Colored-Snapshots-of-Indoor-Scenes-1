# Interactive-Design-and-Preview-of-Colored-Snapshots-of-Indoor-Scenes
![](https://github.com/liubai20609/Interactive-Design-and-Preview-of-Colored-Snapshots-of-Indoor-Scenes-1/blob/master/show.png?raw=true)

We first describe the system <b>(0) Prerequisities</b>.<b>(1) Getting started</b>. <b>(2) User interactions </b>. <b>(3) Supplement </b>. <b>(4) Acknowledgement </b>. 

### (0) Prerequisites
- Linux or OSX
- PyTorch
- CPU or NVIDIA GPU + CUDA CuDNN.
- Install PyTorch and 3rd party Python libraries (OpenCV, scikit-learn and scikit-image). See the https://github.com/junyanz/interactive-deep-colorization for more details.

### (1) Getting Started
- Clone this repo:
```bash
git clone git@github.com:liubai20609/Interactive-Design-and-Preview-of-Colored-Snapshots-of-Indoor-Scenes-1.git
```

- Download the reference model
```
Please refer to https://github.com/junyanz/interactive-deep-colorization
```
### (2) User interactions
- Run the UI: `python ideepcolor.py`. 
![](https://github.com/liubai20609/Interactive-Design-and-Preview-of-Colored-Snapshots-of-Indoor-Scenes-1/blob/master/project/myfile/UI.png?raw=true)

- <b>Adding points</b>: Left-click somewhere on the input pad
- <b>Moving points</b>: Left-click and hold on a point on the input pad, drag to desired location, and let go
- <b>Changing colors</b>: For currently selected point, choose a recommended color (middle-left) or choose a color on the ab color gamut (top-left)
- <b>Removing points</b>: Right-click on a point on the input pad
- <b>Changing patch size</b>: Mouse wheel changes the patch size from 1x1 to 9x9
- <b>Load image</b>: Click the load image button and choose desired scene image
- <b>ChangeColoStyle</b>: Click on the button and change Color gamut.
- <b>Restart</b>: Click on the restart button. All points on the pad will be removed.
- <b>Ok</b>: Click on the ok button. After user interaction, this will Will run the recommendation and rendering program, and then show the rendering result.
- <b>Quit</b>: Click on the quit button.


### (3) Supplement
1. pre-processing：Expert priori preprocessing rendering scheme
2. project: Interactive rendering project
3. renderdesign：rendering scheme
- <b>object-material-color combinations</b>

The furniture, materials and colors used in the program can be modified according to the relationship in the picture below
![](https://github.com/liubai20609/Interactive-Design-and-Preview-of-Colored-Snapshots-of-Indoor-Scenes-1/blob/master/pre-processing/connection%20relation.png?raw=true)
- <b>In the preprocessing, we use the hexadecimal color code</b>
- <b>There are readme.md documentation in each folder.</b>
### (4) Acknowledgement
The colorization part of our code is built upon https://github.com/junyanz/interactive-deep-colorization.
