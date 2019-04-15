## Generic imports

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import os


## Methods
def calibrate_one_image(image_in, g_t_min, g_t_max, s_t_min, s_t_max):
    """
    Function that processes an undistorted, BGR image and returns a binary image, applying thresholds on x-gradient
    and S color channel

    Inputs:
        image_in - the (undistorted) input image, in BGR format
        g_t_min - minimum gradient threshold
        g_t_max - maximum gradient threshold
        s_t_min - minimum s-channel threshold
        s_t_max - maximum s-channel threshold

    Outputs:
        x_binary - binary image where pixels are 1 if  gradient threshold is crossed, 0 otherwise
        s_binary - binary image where pixels are 1 if  color threshold is crossed, 0 otherwise
        combined_binary - binary image where pixels are 1 if both gradient and color thresholds are crossed, 0 otherwise
    """

    # 1. Convert the image in gray scale
    gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

    # 2. Apply the Sobel operator along x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # 3. Threshold on x gradient
    x_binary = np.zeros_like(scaled_sobel)
    x_binary[(scaled_sobel >= g_t_min) & (scaled_sobel <= g_t_max)] = 1

    # 4. Convert to HLS color space, and pick S
    hls = cv2.cvtColor(image_in, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # 5.  Threshold on the S color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_t_min) & (s_channel <= s_t_max)] = 1

    # 6. Define final binary image
    combined_binary = np.zeros_like(x_binary)
    combined_binary[(s_binary == 1) | (x_binary == 1)] = 1

    return x_binary, s_binary, combined_binary


def find_lanes(binary_warped, nwindows, margin, minpix, xm_per_pix, ym_per_pix):
    """
    Function that processes binary warped image and returns left and right lane and second order fit for them.

    Inputs:
        binary_warped - binary warped image - this is a binary bird's eye view of the road in front of the camera,
                        where pixels are 1/0 according to some prepocessing steps
        nwindows - number of sliding windows used to "slice" the original image in the height direction. For each
                   "slice" a histogram of the image will be calculated, and the peaks will identify left and right lane
        margin - +/- width of the image
        minpix - Minimum number of pixels found to recenter window
        xm_per_pix - meters per pixel in x dimension
        xm_per_pix - meters per pixel in y dimension

    Outputs:
        out_img - "decorated" image showing the lanes and the search windows on top of the original binary
        left_fit  - coefficients for the second-order polynomianl fitting the left lane
        right_fit - coefficients for the second-order polynomianl fitting the right lane
        left_fit_cr  - coefficients for the second-order polynomianl fitting the left lane, in meter space
        right_fit_cr - coefficients for the second-order polynomianl fitting the right lane, in meter space
    """

    # 1. Create an output image from the binary
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # 2. Take a histogram of the bottom third of the image
    histogram = np.sum(binary_warped[2 * binary_warped.shape[0] // 3:, :], axis=0)
    # plt.plot(histogram)

    # 3. Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # 4. Set height of the "slicing" windows
    window_height = np.int(binary_warped.shape[0] // nwindows)

    # 5. Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    # 6. Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # 7. Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # 8. Step through the windows one by one
    for window in range(nwindows):
        # 8.1 Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # 8.2 Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # 8.3 Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]

        # 8.4 Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 8.5 If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # 11. Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 12. Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # 13. Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # 14. scale pixel position to meters and fit a polinomial again
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # 15. Color the lanes on the original image
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img, left_fit, right_fit, left_fit_cr, right_fit_cr


def smooth_lanes(Left_Coeffs, Left_Coeffs_cr, Right_Coeffs, Right_Coeffs_cr, Smooth_num, l_fit, l_fit_cr, r_fit,
                 r_fit_cr, image_height, max_dist_m):
    """
    Function that performs some smoothing and filtering on the curvature coefficients for the lanes.
    The coefficients are compared with those coming from a number of previous frames. More specifically, the coeff.
    from previous frames are averaged, and the intersections with the top and bottom of the image are found.Those
    intersections are then compared with the ones obtained with current coefficients, and if the distances are below a
    threshold the lists of coefficients are updated with the current ones, keeping their length. If all the distances
    are beyond the threshold, the coefficients are not added to the list, and the current lanes are discarded. If
    distances are good only on one side of the image (left or right) that side only gets updated.
    Finally the updated list of coefficients gets averaged, and the average will be used to plot the lanes on the
    current frame.
    An indicator is provided as output, to give feedback on the process (lanes found, lanes not found, lanes
    partially found).

    Inputs:
        Left_Coeffs - Initial list of polynomial coefficients for left lanes, coming from previously analysed frames.
        Left_Coeffs_cr - Initial list of polynomial coefficients for left lanes, in meter space, coming from previously
            analysed frames.
        Right_Coeffs - Initial list of polynomial coefficients for right lanes, coming from previously analysed frames.
        Right_Coeffs_cr - Initial list of polynomial coefficients for right lanes, in meter space, coming from previously
            analysed frames.
        Smooth_num - Number of frames considered to smooth the lanes. This will also be the length of the previous lists
        l_fit - Polynomial coefficients for left lane, for the current frame
        l_fit_cr - Polynomial coefficients for left lane, in meter space, for the current frame
        r_fit - Polynomial coefficients for right lane, for the current frame
        r_fit_cr - Polynomial coefficients for right lane, in meter space, for the current frame
        image_height - height of the current frame, in pixels
        max_dist_m - maximum distance in meters to be used as a threshold to evaluate the lanes
    Outputs:
        good_lane_found - indicator for lanes found:
            = 1: lanes found, lists of coefficients updated
            = 0: lanes not found, current coefficients discarded
            = 2: lanes partially found, only one list of coefficients updated
        left_fit - Averaged polynomial coefficients for left lane
        left_fit_cr  - Averaged polynomial coefficients for left lane, in meter space
        right_fit - Averaged polynomial coefficients for right lane
        right_fit_cr -- Averaged polynomial coefficients for right lane, in meter space

    """

    if len(Left_Coeffs) < Smooth_num:
        # Fill the list
        Left_Coeffs.append(l_fit)
        Right_Coeffs.append(r_fit)
        Left_Coeffs_cr.append(l_fit_cr)
        Right_Coeffs_cr.append(r_fit_cr)

        # Assign current coeff
        left_fit = l_fit
        right_fit = r_fit
        left_fit_cr = l_fit_cr
        right_fit_cr = r_fit_cr

        good_lane_found = 1

    else:
        # check intersection to assess if to discard
        new_left_bottom = l_fit[0] * image_height ** 2 + l_fit[1] * image_height + l_fit[2]
        new_right_bottom = r_fit[0] * image_height ** 2 + r_fit[1] * image_height + r_fit[2]
        new_left_top = l_fit[2]
        new_right_top = r_fit[2]

        # get current averaged coeffs
        current_left_fit = np.sum(Left_Coeffs, axis=0) / Smooth_num
        current_right_fit = np.sum(Right_Coeffs, axis=0) / Smooth_num

        current_left_bottom = current_left_fit[0] * image_height ** 2 + current_left_fit[1] * image_height + \
                              current_left_fit[2]
        current_right_bottom = current_right_fit[0] * image_height ** 2 + current_right_fit[1] * image_height + \
                               current_right_fit[2]
        current_left_top = current_left_fit[2]
        current_right_top = current_right_fit[2]

        # distances
        abs_left_bottom = abs(new_left_bottom - current_left_bottom)
        abs_right_bottom = abs(new_right_bottom - current_right_bottom)
        abs_left_top = abs(new_left_top - current_left_top)
        abs_right_top = abs(new_right_top - current_right_top)

        # Convert max distances in pixel space
        max_dist_pxl = max_dist_m / xm_per_pix

        # Flag for good lanes
        good_lane_found = 1

        if (max(abs_left_bottom, abs_right_bottom, abs_left_top, abs_right_top) < max_dist_pxl):
            # All distances are below the threshold. These lanes are probably good, so we'll use them
            # Shift and append
            Left_Coeffs.pop(0)
            Left_Coeffs.append(l_fit)
            Right_Coeffs.pop(0)
            Right_Coeffs.append(r_fit)
            Left_Coeffs_cr.pop(0)
            Left_Coeffs_cr.append(l_fit_cr)
            Right_Coeffs_cr.append(0)
            Right_Coeffs_cr.append(r_fit_cr)

            # Assign average
            left_fit = np.sum(Left_Coeffs, axis=0) / Smooth_num
            right_fit = np.sum(Right_Coeffs, axis=0) / Smooth_num
            left_fit_cr = np.sum(Left_Coeffs_cr, axis=0) / Smooth_num
            right_fit_cr = np.sum(Right_Coeffs_cr, axis=0) / Smooth_num

        elif ((abs_left_bottom < max_dist_pxl) and (abs_left_top < max_dist_pxl)):
            # The left distances are good - we'll upadte only the left coeff
            # Shift and append
            print('BAD RIGHT LANE')
            Left_Coeffs.pop(0)
            Left_Coeffs.append(l_fit)
            Left_Coeffs_cr.pop(0)
            Left_Coeffs_cr.append(l_fit_cr)

            # Assign average
            left_fit = np.sum(Left_Coeffs, axis=0) / Smooth_num
            right_fit = np.sum(Right_Coeffs, axis=0) / Smooth_num
            left_fit_cr = np.sum(Left_Coeffs_cr, axis=0) / Smooth_num
            right_fit_cr = np.sum(Right_Coeffs_cr, axis=0) / Smooth_num

            good_lane_found = 2

        elif ((abs_right_bottom < max_dist_pxl) and (abs_right_top < max_dist_pxl)):
            # The right distances are good - we'll upadte only the right coeff
            # Shift and append
            print('BAD LEFT LANE')
            Right_Coeffs.pop(0)
            Right_Coeffs.append(r_fit)
            Right_Coeffs_cr.append(0)
            Right_Coeffs_cr.append(r_fit_cr)

            # Assign average
            left_fit = np.sum(Left_Coeffs, axis=0) / Smooth_num
            right_fit = np.sum(Right_Coeffs, axis=0) / Smooth_num
            left_fit_cr = np.sum(Left_Coeffs_cr, axis=0) / Smooth_num
            right_fit_cr = np.sum(Right_Coeffs_cr, axis=0) / Smooth_num

            good_lane_found = 2

        else:
            # These lanes are probably not good - we'll ignore them
            print('BAD LANES')
            left_fit = np.sum(Left_Coeffs, axis=0) / Smooth_num
            right_fit = np.sum(Right_Coeffs, axis=0) / Smooth_num
            left_fit_cr = np.sum(Left_Coeffs_cr, axis=0) / Smooth_num
            right_fit_cr = np.sum(Right_Coeffs_cr, axis=0) / Smooth_num

            good_lane_found = 0
    return good_lane_found, left_fit, left_fit_cr, right_fit, right_fit_cr


def calculate_curv_offs(left_fit, left_fit_cr, right_fit, right_fit_cr, image_height, image_width):
    """
    Function that calculates the radius of curvature of the lanes (in meters), and the offset of the car from the center
    of the lane.
    The radius of curvature is calculated from the formula sqrt(1+f'(x)^2)/f"(x), where f' and f" are the first and
    second derivative of the equation for the curve (in this case the second order polynomial for which the coefficients
    have been calculated).
    The offset from the center of the lane is calculated assuming the car is located at the center of the warped image.
    The position of the center of the lane is calculated as the midpoint between the intersections of the lanes with the
    bottom of the image.

    Inputs:
        left_fit - Polynomial coefficients for left lane
        left_fit_cr  - Polynomial coefficients for left lane, in meter space
        right_fit - Polynomial coefficients for right lane
        right_fit_cr -- Polynomial coefficients for right lane, in meter space
        image_height - height of the current frame, in pixels
        image_width - width of the current frame, in pixels
    Outputs:
        left_curverad_m - radius of curvature for left lane, in meter space
        left_fitx - vector of points describing the left lane
        right_curverad_m - radius of curvature for right lane, in meter space
        right_fitx - vector of points describing the right lane
        vehicle_offset - offset between vehicle and center lane (in m). Positive means to the right.
        ploty - reference vector for plotting both left and right lane

    """
    # Generate x and y values for plotting
    ploty = np.linspace(0, image_height - 1, image_height)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Define a point where to evaluate
    y_eval = np.max(ploty)

    # Radius of curvature (in meter space)
    left_curverad_m = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad_m = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # Center offset
    left_bottom = left_fit[0] * image_height ** 2 + left_fit[1] * image_height + left_fit[2]
    right_bottom = right_fit[0] * image_height ** 2 + right_fit[1] * image_height + right_fit[2]
    lane_center = (left_bottom + right_bottom) / 2
    vehicle_offset = (image_width / 2 - lane_center) * xm_per_pix

    return left_curverad_m, left_fitx, right_curverad_m, right_fitx, vehicle_offset, ploty


def revert_perspective(left_fitx, right_fitx, ploty, good_lane_found, wpd_img, or_image_height, or_image_width):
    """
    Function that return an image with the lanes that have been found, drawn in real perspective.

    Inputs:
        left_fitx - vector of points describing the left lane
        right_fitx - vector of points describing the right lane
        ploty - reference vector for plotting both left and right lane
        good_lane_found - indicator for lanes found:
            = 1: lanes found, lists of coefficients updated - lanes will  be plotted in green
            = 0: lanes not found, current coefficients discarded - lanes will  be plotted in red
            = 2: lanes partially found, only one list of coefficients updated - lanes will  be plotted in yellow
        wpd_image - distorted ("bird's eye") image
        or_image_height - original (unwarped) image height, in pixels
        or_image_width - original (unwarped) image width, in pixels
    Outputs:
        newwarp - blank image with the new lanes plotted in real perspective

    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(wpd_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    if (good_lane_found == 1):
        # good lanes found here
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    elif (good_lane_found == 2):
        # only one good lane found here
        # Area will be shown in yellow
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 255))
    else:
        # bad lanes found here
        # area will be shown in red
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (or_image_width, or_image_height))

    return newwarp


def process_frame(img, Left_Coeffs, Right_Coeffs, Left_Coeffs_cr, Right_Coeffs_cr, Smooth_num, max_dist_m):
    """
    Function that processes a frame from a video to identify and plot lanes on it

    Inputs:
        img: the frame in input
        Left_Coeffs - Initial list of polynomial coefficients for left lanes, coming from previously analysed frames.
        Right_Coeffs - Initial list of polynomial coefficients for right lanes, coming from previously analysed frames.
        Left_Coeffs_cr - Initial list of polynomial coefficients for left lanes, in meter space, coming from previously
            analysed frames.
        Right_Coeffs_cr - Initial list of polynomial coefficients for right lanes, in meter space, coming from previously
            analysed frames.
        Smooth_num - Number of frames considered to smooth the lanes. This will also be the length of the previous lists
        max_dist_m - maximum distance in meters to be used as a threshold to evaluate the lanes
    Outputs:
        result - an image with the lanes plotted on it, showing also an indication of their radius of curvature and the
            offset of the car with respect to the center of the lane

    """
    # 1. undistort the original image
    udst = cv2.undistort(img, mtx, dist, None, mtx)

    # 2. generate binary
    x_b, s_b, bimg = calibrate_one_image(udst, thresh_min, thresh_max, s_thresh_min, s_thresh_max)

    # 3. warp transform
    wpd_img = cv2.warpPerspective(bimg, M, (bimg.shape[1], bimg.shape[0]))

    # 4.find lanes
    out_img, l_fit, r_fit, l_fit_cr, r_fit_cr = find_lanes(wpd_img, nwindows, margin, minpix, xm_per_pix,
                                                                                    ym_per_pix)
    # 5. smooth the lanes
    good_lane_found, left_fit, left_fit_cr, right_fit, right_fit_cr = smooth_lanes(Left_Coeffs, Left_Coeffs_cr,
                                                                                    Right_Coeffs, Right_Coeffs_cr,
                                                                                    Smooth_num, l_fit, l_fit_cr, r_fit,
                                                                                    r_fit_cr, wpd_img.shape[0],
                                                                                    max_dist_m)

    # 5. Calculate the radius of curvature and center offset
    left_curverad_m, left_fitx, right_curverad_m, right_fitx, vehicle_offset, ploty = calculate_curv_offs(left_fit,
                                                                                                    left_fit_cr,
                                                                                                    right_fit,
                                                                                                    right_fit_cr,
                                                                                                    wpd_img.shape[0],
                                                                                                    wpd_img.shape[1])

    # 6. Revert back to original perspective
    newwarp = revert_perspective(left_fitx, right_fitx, ploty, good_lane_found, wpd_img, udst.shape[0], udst.shape[1])

    # 7. Combine the result with the original image
    result = cv2.addWeighted(udst, 1, newwarp, 0.3, 0)

    # Add also an indication of the radius of curvature
    avg_curverad_m = np.round((left_curverad_m + right_curverad_m) / 2, 2)
    text_to_show_curv = 'Avg. radius of curvature = ' + str(avg_curverad_m) + ' m'
    result = cv2.putText(result, text_to_show_curv, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # And for the center offset
    if (vehicle_offset < 0):
        text_to_show_ofs = 'Offset from lane center = ' + str(np.round(abs(vehicle_offset), 2)) + ' m to the left'
    else:
        text_to_show_ofs = 'Offset from lane center = ' + str(np.round(vehicle_offset, 2)) + ' m to the right'
    result = cv2.putText(result, text_to_show_ofs, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return result

################################
# VIDEO PROCESSING SCRIPT
################################
# Load Dictionaries with calibration and transformation parameters
cal_pickle_in = open("CalibrationParameters.p","rb")
cal_par = pickle.load(cal_pickle_in)

mtx = cal_par["Cal_Mat"]
dist = cal_par["DistCoeffs"]

war_pickle_in = open("WarpingMatrix.p","rb")
war_par = pickle.load(war_pickle_in)

M = war_par["Warp_Mat"]
Minv = war_par["Inv_Warp_Mat"]


## GLOBAL variables
# Gradient and color thresholds:
# Define thresholds on the x gradient
thresh_min = 55
thresh_max = 75

# Define thresholds on the S color channel
s_thresh_min = 170
s_thresh_max = 200

# Finding lanes parameters:
# Set parameters for sliding windows size
nwindows = 10
margin = 75
minpix = 30

# Scale from pixel space to meters. This values are calculated based on the ref. image
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 580  # meters per pixel in x dimension
##


# Define Videos
video_input = 'project_video.mp4'
video_output = 'results/video/processed_ project_video.mp4'

# Open input Video
cap = cv2.VideoCapture(video_input)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output, fourcc, fps, (width,height))

# Counter
count = 0


# Number of frames to smooth, and initial coefficients for the curvature radius
Smoothed_frames = 10
max_dist_m = 1

Left_coeffs = []
Right_coeffs = []
Left_coeffs_cr = []
Right_coeffs_cr =[]

###

# Process video
while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Process the frame
    if (ret == True):
        if ((count%50) == 0):
            print('Read frame: %d ' % count, ret)
        processed_frame = process_frame(frame,Left_coeffs,Right_coeffs,Left_coeffs_cr,Right_coeffs_cr,Smoothed_frames,
                                        max_dist_m)
        out.write(processed_frame)
        count += 1
    else:
         break

# When everything done, release the videos
cap.release()
out.release()

# Show the last processed frame - this step is optional, can be commented away if needed
cv2.imshow('Last frame ',processed_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

