
## Advanced Lane Finding Project

The goal of this project is to write a software pipeline to identify the lanes in a set of images and a video feed, taken from a camera mounted in a fixed position in front of a moving car. The area between the lanes is then shown on the original image/video stream, together with information about the radius of curvature of the lane (averaged between left and right) and the relative position of the car with respect the center of the lane.

In terms of general steps that the SW will take on the images we can identify:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms and gradients' analysis to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

In the following of this writeup, a section will be dedicated to each steps, clarifying the solution implemented and showing examples of the reults. All the steps will make reference to the code in the Python [Jupyter Notebook](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/blob/master/advanced_lane_finds.ipynb) accompanying this project: this is used to process individual images. The final section of the writep will give more details on the analysis of the video, that is executed through a dedicated [Python script](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/blob/master/AdvLineFinder.py).

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Distorted Chessboard"
[image2]: ./results/cal_results/calibration1_und.jpg "Undistorted Chessboard"
[image3]: ./test_images/test2.jpg "Test Image"
[image4]: ./results/processed_imgs/test2_und.jpg "Undistorted Test Image"
[image5]: ./results/processed_imgs/test3_und.jpg "Undistorted Test Image"
[image6]: ./results/processed_imgs/test3_und_bin.jpg "Binary Test Image"
[image7]: ./results/processed_imgs/straight_lines1_und.jpg "Undistorted Test Image"
[image8]: ./results/processed_imgs/straight_lines1_und_bin.jpg "Binary Test Image"
[image9]: ./results/processed_imgs/straight_lines1_und_bin_warp.jpg "Warped Binary Test Image"
[image10]: ./results/processed_imgs/test1_und_bin_warp.jpg "Warped Binary Test Image"
[image11]: ./results/processed_imgs/test1_und_bin_warp_lanes.jpg "Decorated Warped Binary Test Image"

[video1]: ./project_video.mp4 "Video"


---
## Image Analysis

### 1. Camera Calibration

The code for this step is contained in the sections 1/2 of the aforementioned [Python notebook](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/blob/master/advanced_lane_finds.ipynb). Documentation for the OpenCV functions that will be referenced here below can be found [here](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html).

In order to calibrate the camera some [reference images](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/tree/master/camera_cal) of a chessboard are used. The first step is to obtain "object points", which will be the (x, y, z) coordinates of the reference chessboard corners (NOTE: here we assume a 2D image, hence z will be = 0).
"Image points" will then be collected from the images in the list through the `cv2.findChessboardCorners()` function. 

Once vectors for Image/Object points are built, the calibration coefficients can be calculated through the `cv2.calibrateCamera()` function. The coefficients are then used to correct the images using the `cv2.undistort()` function.
All the calibrated images (together with intermediate steps) are saved in the specific [results](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/tree/master/results/cal_results) folder. The calibration coefficients are also saved as a dictionary in a pickle file for further use.

An example of the effects of the calibration can be seen here below, where we can see the "before" and "after" for one of the chessboard images:

Distorted Chessboard             |  Undistorted Chessboard
:-------------------------:|:-------------------------:
![alt text][image1] |  ![alt text][image2]


### 2. Distortion Correction for Reference Images

The code for this step is contained in the section 2 of the [Python notebook](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/blob/master/advanced_lane_finds.ipynb), and simply consists of the application of the coefficients calculated in the first step to the [reference images](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/tree/master/test_images) provided with the project. The corrected images are saved in a specific [results](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/tree/master/results/processed_imgs) folder.

An example of the results is here below:

Distorted Image             |  Undistorted Image
:-------------------------:|:-------------------------:
![alt text][image3] |  ![alt text][image4]


### 3. Gradient/Color Threshold Analysis

After undistorting the test images, they can be converted in a different color space in order to identify what channel would be the best to apply threshold for the lanes identification. Moreover, a threshold can be applied on the gradients of the image (in the x or y direction) to help identify lines/segments. These steps are documented in section 3 of the project [Python notebook](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/blob/master/advanced_lane_finds.ipynb). 

The working assumptions followed in the analysis are:

* Gradient calculated on the x direction only - this mostly because of the near vertical nature of the lanes;
* Use of the HLS color space, focusing on the S channel. In experimenting. this seemed to provide the better overall performances.

The gradients are expressed by Sobel operator, calculated using the `cv2.Sobel()` function (details [here](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html)); the change in color space is executed through the `cv2.cvtColor()` function (documentation [here](https://docs.opencv.org/2.4.13.7/modules/imgproc/doc/miscellaneous_transformations.html)). 
Some trial-and-error was required in order to assess the values for the right thresholds. Final choices are visible in the code: an example of the outcome achievable with the final selection is visible here below:

Undistorted Image             |  Binary Image
:-------------------------:|:-------------------------:
![alt text][image5] |  ![alt text][image6]

All the binary images are saved in the specific [results](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/tree/master/results/processed_imgs) folder.

### 4. Perspective Transform

The binary images just obtained can the be transformed to be looked at from a "bird's eye" perspective. 

This is documented in section 4 of the project [Python notebook](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/blob/master/advanced_lane_finds.ipynb); specific steps are:

1. Definition of a "mask" on the images, identifying a reference area on which operate the transformation;
2. Definition of a "destination mask", describing the shape that the reference area should present in the warped perspective;
3. Application of the `cv2.getPerspectiveTransform()` function to obtain transformation matrixes (direct and inverse) between the two shapes (documentation on the function can be found [here](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html);
3. "Warping" of the binary images obtained at the previous step by applying the `cv2.warpPerspective()` function.  

Here too, few attempts were necessary to assess a mask (and subsequent transformation) that would provide reasonably acceptable results across the reference images: the final choice is shown in the Notebook. 
The transformed images are saved in the usual [results](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/tree/master/results/processed_imgs) folder, while the transformation matrixes are stored as a dictionary in a pickle file for eventual further reuse. An example of the process is visible here below, where we show the original image and the final binary warped perspective:

Test Image             |  Warped Binary Image
:-------------------------:|:-------------------------:
![alt text][image7] |  ![alt text][image9]


### 5. Lanes Detection

In order to detect the lanes on the warped binary images, a "Sliding Windows" approach is used: the image is "sliced" in horizontal layers (from bottom to top) and windows of established width are defined in each layer. Pixels from every window are aggregated together and a best fit of their final distribution is calculated using a second order polynomial.

The main steps can be summarized as:

1. Take a histogram of the bottom part of the image (for this project, the bottom third);
2. Find the peak of the left and right halves of the histogram: these will be the starting point for the left and right lanes, and they will identify the position of two windows blonging to the first layer;
3. Aggregate the nonzero pixels contained in the two windows;
4. Move to the layer above and evaluate the nonzero pixels in the windows immediately above the previous ones. Aggregate them with the ones coming from the previous layer, and then compare the number of pixels in the windows with a threshold. If the threshold is crossed, shift the position of the next windows (to be used by the subsequent layer) on the new mean;
5. Keep moving upwards layer after layer;
6. Once the whole inage has been evaluated, two vectors of pixels (from the left and right sequence of windows) will have been defined. For each of them a best-fitting second order polynomial is calculated. 

Note that after calculating the best-fitting polynomial in terms of pixel coordinates, we will transform that in meter space. This will be necessary in order to calculate quantities that are related to real space, like the radius of curvature of the lanes: these steps will be detailed in the next section.
In order to convert from pixel space to meter space the appropriate coefficients are calculated starting from the warped images (the one shown in the previous paragraph can be used) and the assumption that the lane is about 30 meters long and 3.7 meters wide. 

The code detailing the process and the parameters used is part of section 5 of the [Python notebook](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/blob/master/advanced_lane_finds.ipynb). For every test image a "decorated" warped binary, showing the sliding windows, the aggregated pixels for the left and right lane, and the best fitting polynomials has been saved in the [results](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/tree/master/results/processed_imgs) folder. An example of the input/output of the process is visible here below:

Warped Binary Image             |  'Decorated" Warped Binary Image
:-------------------------:|:-------------------------:
![alt text][image10] |  ![alt text][image11]


### 6. Calculation of Lanes Curvature and Vehicle Position

The radius of curvature is calculated from the formula `((1+f'(x)^2)^3/2)/f"(x)`, where f' and f" are the first and second derivative of the equation for the curve (in this case the second order polynomial for which the coefficients have been previously calculated).
Note that, even if we calculate the radius of curvature for both left and right lanes, on the final processed image only the average will be shown.

The offset from the center of the lane is calculated assuming the car is located at the center of the warped image. 
The position of the center of the lane is calculated as the midpoint between the intersections of the lanes with the bottom of the image: the relative distance to the image midpoint is the desired offset. When the result is > than 0 the vehicle is at the right of the center of the lane.

These steps are contained in section 5.2 of the [Python notebook](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/blob/master/advanced_lane_finds.ipynb).




### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
