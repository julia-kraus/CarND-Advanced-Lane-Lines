import numpy as np
import matplotlib.pyplot as plt
import cv2
import preprocessing


class Line():
    """Class keeping track of a lane and all the lanes in the previous frame"""

    def __init__(self):
        # was the line detected in the last iteration? If yes, we can search from prior
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None


def search_lane_from_scratch(binary_warped):
    """Searches lane from binary warped image using histogram and sliding window technique.
    `out_img` return value only needed for plotting, not for further lane detection algorithm."""
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    n_windows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    min_pix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // n_windows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(n_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        # Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If found > min_pix pixels, recenter next window
        if len(good_left_inds) > min_pix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > min_pix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # fits polynomial to the detected lane line pixels
    left_fitx, right_fitx, ploty = fit_polynomial(binary_warped.shape, leftx, lefty, rightx, righty)

    # Visualization
    # Colors in the left and right lane region lane line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # better: zip left_fitx and ploty and use opencv.polyLines. This approach doesn't work if there are multiple
    # images in a row.!
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return left_fitx, right_fitx, ploty, out_img


def search_lane_from_prior(binary_warped, left_fit, right_fit):
    """Search lanes by only searching a region around a previous polynomial fit"""
    # HYPERPARAMETER
    # Width of the margin around the previous polynomial to search
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # get activated pixels in region around the previous polynomial
    left_lane_inds = ((nonzerox < (left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy +
                                   left_fit[2] + margin)) & (nonzerox > (left_fit[0] * nonzeroy ** 2 +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] - margin)))
    right_lane_inds = ((nonzerox < (right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy +
                                    right_fit[2] + margin)) & (nonzerox > (right_fit[0] * nonzeroy ** 2 +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] - margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials THIS FUNCTION DOES NOT EXIST ANYMORE!!!!
    left_fitx, right_fitx, ploty = fit_polynomial(binary_warped.shape, leftx, lefty, rightx, righty)

    return left_fitx, right_fitx, ploty


def fit_polynomial(img_shape, leftx, lefty, rightx, righty):
    """Fits polynomials to lane line pixels"""
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    # Calc both polynomials using ploty, left_fit and right_fit ###\n",
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect\n",
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty
    return left_fitx, right_fitx, ploty


# this function MIGHT NOT BE not completely right yet. We need to figure out how to use ym_per_pix and xm_per_pix
def get_curvature_real(fitx, ploty):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # fit new polynomial to x, y in the real world space
    fit_cr = np.polyfit(y * ym_per_pix, fitx * xm_per_pix, 2)

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    y_eval_meters = ym_per_pix * y_eval

    A_l = fit_cr[0]
    B_l = fit_cr[1]
    C_l = fit_cr[2]

    # Radius of curvature in meters
    curverad = ((1 + (2 * A_l * y_eval_meters + B_l) ** 2) ** 1.5) / (abs(2 * A_l))

    return curverad


class LaneFinder():
    """Class for lane finding. """

    def __init__(self):
        # Lines are only once initialized per list of images -> .detected values remain
        self.left_lane = Line()
        self.right_lane = Line()

    def get_result(self, image, warped, left_fitx, right_fitx, ploty, Minv, curvature):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        # Ab hier nochmal neu        
        # Put text on an image
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Radius of Curvature: {} m".format(int(curvature))
        cv2.putText(result, text, (400, 100), font, 1, (255, 255, 255), 2)
        # Find the position of the car
        pts = np.argwhere(newwarp[:, :, 1])
        position = LaneFinder.get_car_position(self, pts)
        if position < 0:
            text = "Vehicle is {:.2f} m left of center".format(-position)
        else:
            text = "Vehicle is {:.2f} m right of center".format(position)

        cv2.putText(result, text, (400, 150), font, 1, (255, 255, 255), 2)

        return result

    def find_lanes(self, image):
        """Finds lanes and lane curvatures in a single images and plots them."""
        binary_warped, M, Minv = preprocessing.preprocess_image(image)

        # find lane pixels
        if self.left_lane.detected and self.right_lane.detected:
            ###TO-DO: ergänze Parameter best fit or WHAT?
            left_fitx, right_fitx, ploty = search_lane_from_prior(binary_warped,
                                                                           self.left_lane.best_fit,
                                                                           self.right_lane.best_fit)
        else:
            left_fitx, right_fitx, ploty, out_img = search_lane_from_scratch(binary_warped)

        # find curvatures 
        left_curve_radius = get_curvature_real(left_fitx, ploty)
        right_curve_radius = get_curvature_real(right_fitx, ploty)

        # Does sanity check really need to output something or is it enought that the class variables are set
        left_fitx, right_fitx = self.sanity_check()

        # left_curve_radius is enough since both curve radii should be about the same
        return self.get_result(image, binary_warped, left_fitx, right_fitx, ploty, Minv, left_curve_radius)

    def get_car_position(self, pts, image_shape=(720, 1280)):
        """Calculates car position from the center of the lane. Returns car's distance from the center of the road in 
        meter"""

        position = image_shape[1] / 2
        left = np.min(pts[(pts[:, 1] < position) & (pts[:, 0] > 700)][:, 1])
        right = np.max(pts[(pts[:, 1] > position) & (pts[:, 0] > 700)][:, 1])
        center = (left + right) / 2

        # Define conversions in x and y from pixels space to meters,
        # assume the lane is about 30 meters long and 3.7 meters wide
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        dist_in_m = (position - center) * xm_per_pix
        return dist_in_m

    def sanity_check(self, lane, curverad, fitx, fit):
        # Sanity check for the lane
        lane.current_fit = fit

        if abs(curverad - 2000) / 2000 < 2:
            lane.detected = True

            # Keep a running average over 3 frames
            if len(lane.recent_xfitted) > 5 and lane.recent_xfitted:
                lane.recent_xfitted.pop()
                lane.recent_fits.pop()

            lane.recent_xfitted.append(fitx.reshape(1, -1))
            lane.recent_fits.append(fit.reshape(1, -1))

            if len(lane.recent_xfitted) > 1:
                lane.bestx = np.mean(np.vstack(lane.recent_xfitted), axis=1)
                lane.best_fit = np.mean(np.vstack(lane.recent_fits), axis=1)

            lane.bestx = fitx
            lane.best_fit = fit

            return lane.bestx

        else:
            lane.detected = False

        return lane.bestx if lane.bestx is not None else lane.current_fit
