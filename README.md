## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview:
---

This Project is submitted as part of the Udacity Self-Driving Car Nanodegree.

For it, the goal is to write a software pipeline to identify the lanes in a set of images and a video feed, taken from a camera mounted in a fixed position in front of a moving car. The area between the lanes is then shown on the original image/video stream, together with information about the radius of curvature of the lane (averaged between left and right) and the relative position of the car with respect the center of the lane.

In terms of general steps that the SW pipeline takes on the images we can identify:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

To complete the project, few files are are submitted as part of this Git repo: 

1. An annotated writeup describing the fundamental aspects and limitations of the solution implemented.
2. A [Python Jupyther notebook], that is used to analize images (https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/blob/master/advanced_lane_finds.ipynb).
3. A [Python script], that is used to analyse the video. The script reuses most of the pipe defined for the images, but allows better methods definition and flow control (https://github.com/russom/CarND-Advanced-Lane-Lines-RussoM/blob/master/AdvLineFinder.py)
