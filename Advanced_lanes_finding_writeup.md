
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

In the following of this writeup, a section will be dedicated to each steps, clarifying the solution implemented and showing examples of the reults. All the steps will make reference to the code in the Python [Jupyter Notebook](./advanced_lane_finds.ipynb) accompanying this project: this is used to process individual images. The final section of the writep will give more details on the analysis of the video, that is executed through a dedicated [Python script](./AdvLineFinder.py).

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
[image12]: ./results/processed_imgs/test1_und.jpg "Undistorted Test Image"
[image13]: ./results/processed_imgs/test1_und_lanes.jpg "Undistorted Test Image with Lanes"


[video1]: ./results/video/processed_project_video.mp4 "Video"


---
## Image Analysis

### 1. Camera Calibration

The code for this step is contained in the sections 1/2 of the aforementioned [Python notebook](./advanced_lane_finds.ipynb). Documentation for the OpenCV functions that will be referenced here below can be found [here](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html).

In order to calibrate the camera some [reference images](./camera_cal) of a chessboard are used. The first step is to obtain "object points", which will be the (x, y, z) coordinates of the reference chessboard corners (NOTE: here we assume a 2D image, hence z will be = 0).
"Image points" will then be collected from the images in the list through the `cv2.findChessboardCorners()` function. 

Once vectors for Image/Object points are built, the calibration coefficients can be calculated through the `cv2.calibrateCamera()` function. The coefficients are then used to correct the images using the `cv2.undistort()` function.
All the calibrated images (together with intermediate steps) are saved in the specific [results](./results/cal_results) folder. The calibration coefficients are also saved as a dictionary in a pickle file for further use.

An example of the effects of the calibration can be seen here below, where we can see the "before" and "after" for one of the chessboard images:

Distorted Chessboard             |  Undistorted Chessboard
:-------------------------:|:-------------------------:
![alt text][image1] |  ![alt text][image2]


### 2. Distortion Correction for Reference Images

The code for this step is contained in the section 2 of the [Python notebook](./advanced_lane_finds.ipynb), and simply consists of the application of the coefficients calculated in the first step to the [reference images](./test_images) provided with the project. The corrected images are saved in a specific [results](./results/processed_imgs) folder.

An example of the results is here below:

Distorted Image             |  Undistorted Image
:-------------------------:|:-------------------------:
![alt text][image3] |  ![alt text][image4]


### 3. Gradient/Color Threshold Analysis

After undistorting the test images, they can be converted in a different color space in order to identify what channel would be the best to apply a threshold for the lanes identification. Moreover, thresholds can be applied on the gradients of the image (in the x or y direction) to help identify lines/segments. These steps are documented in section 3 of the project [Python notebook](./advanced_lane_finds.ipynb). 

The working assumptions followed in the analysis are:

* Gradient calculated on the x direction only - this mostly because of the near vertical nature of the lanes;
* Use of the HLS color space, focusing on the S channel. In experimenting. this seemed to provide the better overall performances.

The gradients are expressed by Sobel operator, calculated using the `cv2.Sobel()` function (details [here](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html)); the change in color space is executed through the `cv2.cvtColor()` function (documentation [here](https://docs.opencv.org/2.4.13.7/modules/imgproc/doc/miscellaneous_transformations.html)). 
Some trial-and-error was required in order to assess the values for the right thresholds. Final choices are visible in the code: an example of the outcome achievable with the final selection is visible here below:

Undistorted Image             |  Binary Image
:-------------------------:|:-------------------------:
![alt text][image5] |  ![alt text][image6]

All the binary images are saved in the specific [results](./results/processed_imgs) folder.

### 4. Perspective Transform

The binary images just obtained can the be transformed to be looked at from a "bird's eye" perspective. 

This is documented in section 4 of the project [Python notebook](./advanced_lane_finds.ipynb); specific steps are:

1. Definition of a "mask" on the images, identifying a reference area on which operate the transformation;
2. Definition of a "destination mask", describing the shape that the reference area should present in the warped perspective;
3. Application of the `cv2.getPerspectiveTransform()` function to obtain transformation matrixes (direct and inverse) between the two shapes (documentation on the function can be found [here](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html);
3. "Warping" of the binary images obtained at the previous step by applying the `cv2.warpPerspective()` function.  

Here too, few attempts were necessary to assess a mask (and subsequent transformation) that would provide reasonably acceptable results across the reference images: the final choice is shown in the Notebook. 
The transformed images are saved in the usual [results](./results/processed_imgs) folder, while the transformation matrixes are stored as a dictionary in a pickle file for eventual further reuse. An example of the process is visible here below, where we show the original image and the final binary warped perspective:

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
6. Once the whole image has been evaluated, two vectors of pixels (from the left and right sequence of windows) will have been defined. For each of them a best-fitting second order polynomial is calculated. 

Note that after calculating the best-fitting polynomial in terms of pixel coordinates, we will transform that in meter space. This will be necessary in order to calculate quantities that are related to real space, like the radius of curvature of the lanes: these steps will be detailed in the next section.
In order to convert from pixel space to meter space the appropriate coefficients are calculated starting from the warped images (the one shown in the previous paragraph can be used) and the assumption that the lane is about 30 meters long and 3.7 meters wide. 

The code detailing the process and the parameters used is part of section 5 of the [Python notebook](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/blob/master/advanced_lane_finds.ipynb). For every test image a "decorated" warped binary, showing the sliding windows, the aggregated pixels for the left and right lane, and the best fitting polynomials has been saved in the [results](./results/processed_imgs) folder. An example of the input/output of the process is visible here below:

Warped Binary Image             |  'Decorated" Warped Binary Image
:-------------------------:|:-------------------------:
![alt text][image10] |  ![alt text][image11]


### 6. Calculation of Lanes Curvature and Vehicle Position

The radius of curvature is calculated from the formula `((1+f'(x)^2)^3/2)/f"(x)`, where f' and f" are the first and second derivative of the equation for the curve (in this case the second order polynomial for which the coefficients have been previously calculated).
Note that, even if we calculate the radius of curvature for both left and right lanes, on the final processed image only the average will be shown.

The offset from the center of the lane is calculated assuming the car is located at the center of the warped image. 
The position of the center of the lane is calculated as the midpoint between the intersections of the lanes with the bottom of the image: the relative distance to the image midpoint is the desired offset. When the result is > than 0 the vehicle is at the right of the center of the lane.

These steps are contained in section 5.2 of the [Python notebook](./advanced_lane_finds.ipynb).


### 7. Revert Back and Information Display

The picture of the warped lanes obtained so far gets reverted back to the original perspective by using the inverse transformation matrix calculated at Point 4. 

All the necessary steps are included in the final sections of the project [Python notebook](./advanced_lane_finds.ipynb). 
First, a "blank" image is drawn containing only the lanes and the envelope between them. Then, this blank image is reverted back to the original perspective using again the `cv2.warpPerspective()` function. Finally, this envelope is overlapped on the original image, and some text is dispalyed with the information on curvature and offset calculated as described in the previous Point 6.

An example for one of the test image is here below:

Undistorted Test Image             |  "Annotated" Test Image
:-------------------------:|:-------------------------:
![alt text][image12] |  ![alt text][image13]


The process is finally applied on the test images, and the processed outputs are saved in the [results](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/tree/master/results/processed_imgs) folder.

---

## Video Analysis

In order to analyze a video feed, the whole pipeline described so far can be reused. However, rather than working with a Jupyter Notebook, a specific [Python script](./AdvLineFinder.py) is provided with this project.

The reasons to move to the script are basically:

1. Better possibilities to isolate methods: the majority of the ~600 lines of code are a collection of methods that implement the steps to calibrate an image, warp the perspective, find lanes and calculate geometric parameters and revert back to original perspective;
2. Better possibilities to define the variables' scope and control the flow. In the script, for example, some variables are defined as global (line 494/513) and shared across functions, while others are defined as part of the methods' I/O.


### Frame Smoothing

On top of the steps defined to analyze images, for the video feed some considerations are needed in order to ensure a smooth transition between frames. To this end, two main features have been added:

1. The coefficients for the second-order polynomials defining the lanes are smoothed across frames. In order to obtain this, for each frame a list of the coefficients for the `n` previous frames (including the current one) is considered, and the lanes that are plotted are the ones obtained by **averaging** the coefficients of the list. After some experiment, given the fps (frame per second) value for the input video (25) a list of 10 frames is considered in the code. Of course, this parameter is adjustable.

2. In some challenging frame (for example in case of sudden changes in luminosity or contrast) the pipeline might actually fail in properly identifying the lanes, and return incorrect coefficients for the polynomials. These cases are identified by evaluating the intersection of the polynomials calculated for each frame with the top and bottom and image. These points are then compared with the intersection of the _averaged_ polynomial coming from the `n` previous frames, again with the top/bottom of the image. At that point:
* If the minimum distances between the intersections is higher than a given threshold, the frame-specific lanes are deemed to be wrong and discarded: for this frame the code will hold the lanes of the previous one; 
* If the maximum distance between intersection is below the threshold, the frame-specific lanes are considerd good, and the list of coefficients is updated;
* If only one side (right/left) has distances below the given threshold, coefficients for that side only are updated accordingly.

The steps above are contained in the `smooth_lanes()` function, defined at line 164 of the script.

In order to provide some graphical clue of the behaviour of the pipeline, the envelope is plotted in Green when the lanes are correctly identified, Red when they are discarded for the current frame, and Yellow when only on lane has been updated.

The result of the application of the script to the project video is [here](./results/video).

---

## Conclusions and Further Developments

As a first design for image recognition, the pipeline developed here seems so produce satisfatory results on the set of images proposed, and it does also seems to behave acceptably on the video. However, the video analysis itself highlights some of the challenging aspects of a solution like this: from it it's possible to see how, for example between 38 and 42 seconds the pipeline fails to properly identify the lanes at times (the envelope is yellow or red), and this is most likely due to the change in the color of the pavement, and in general ambient conditions.

The first fundamental problem that such a system must face is indeed how to provide consistent results in a variety of conditions, especially in terms of luminosity and contrast: light, rain, shadows, color of the pavement/lanes etc.

This translated in a challenge in finding the best parameters for Gradient and Color thresholds: what is used, even if a good compromise might not be the optimal solution always. Some different combinations of color spaces and channels, for example, might be used. Some more sophisticated solutions might even change these thresholding mechanisms dynamically, based on other sensors providing info on things like car speed, or wether is day/night, for example.
The same could be said for things like the mask used for warping the perspective: identifying one took time in this exercise, and still this might not be the best compromise. Here too some dynamic solution (changing the shape of the mask with speed, for example) could be experimented.

Beyond that, another area for improvement is the actual lane identification itself: for this even the Udacity training introduced more than one approach. The sliding Windows here used works, but might not be the best option every time.

Finally, in terms of performances the video analysis could benefit from some improvements: the solution implemented is lower than real time, requiring a couple of minutes to process a 50 seconds video, on an average machine.

The code implemented can definitely be optimized, and there is an influence from steps that are specific to this project (accessing/writing a video on a file system) that would not necessarily apply to a deployment on a vehicle. However, some other things could be further exlored and improved. For example, at the moment no information is exchanged between frames on the position of the lanes, so this process starts from scratch at every frame. This could be improved by forcing the search to start at the position found in the previous frame, and focusing in an area around that. This would also probably improve the resilience of the algorithm, limiting the tendency to get "lost" at times.

As an even more stretched experiment to optimize performaces, it would be interesting to evaluate the behaviour of a C++ implementation of the same pipeline. Most of the code, in fact, relies on the OpenCV library, that is available for C++ also. 
