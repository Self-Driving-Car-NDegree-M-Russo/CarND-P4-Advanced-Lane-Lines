
## Advanced Lane Finding Project

The goal of this project is to write a software pipeline to identify the lanes in a set of images and a video feed, taken from a camera mounted in a fixed position in front of a moving car. The area between the lanes is then shown on the original image/video stream, together with information about the radius of curvature of the lane (averaged between left and right) and the relative position of the car with respect the center of the lane.

In terms of general steps that the SW will take on the images we can identify::

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
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
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


---
## Image Analysis

### 1. Camera Calibration

The code for this step is contained in the cell 1/2 of the aforementioned [Python notebook](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/blob/master/advanced_lane_finds.ipynb). Documentation for the OpenCV functions that will be referenced here below can be found [here](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html).

In order to calibrate the camera some [reference images](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/tree/master/camera_cal) of a chessboard are used. The first step is to obtain "object points", which will be the (x, y, z) coordinates of the reference chessboard corners (NOTE: here we assume a 2D image, hence z will be = 0).
"Image points" will then be collected from the images in the list through the `cv2.findChessboardCorners()` function. 

Once vectors for Image/Object points are built, the calibration coefficients can be calculated through the `cv2.calibrateCamera()` function. The coefficients are then used to correct the images using the `cv2.undistort()` function.
All the calibrated images (together with intermediate steps) are saved in the specific [results](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/tree/master/results/cal_results) folder. The calibration coefficients are also saved as a dictionary in a pickle file for further use.

An example of the effects of the calibration can be seen here below, where we can see the "before" and "after" for one of the chessboard images:

Distorted Chessboard             |  Undistorted Chessboard
:-------------------------:|:-------------------------:
![alt text][image1] |  ![alt text][image2]

### 2. Distortion Correction for Reference Images

The code for this step is contained in the cell 2 of the [Python notebook](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/blob/master/advanced_lane_finds.ipynb), and simply consists of the application of the coefficients calculated in the first step to the [reference images](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/tree/master/test_images) provided with the project. The corrected images are saved in a specific [results](https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/tree/master/results/processed_imgs) folder.

An example of the results is here below:

Distorted Image             |  Undistorted Image
:-------------------------:|:-------------------------:
![alt text][image3] |  ![alt text][image4]






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
