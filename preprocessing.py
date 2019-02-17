import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt


############################################################################################
# Helper functions for Loading and Plotting the Test Images                                #
############################################################################################

def load_image(path, to_rgb=True):
    """
    Load image from the given path. By default the returned image is in RGB.
    When to_rgb is set to False the image return is in BGR. Returns by default a RGB image.
    """
    img = cv2.imread(path)

    if not to_rgb:
        return img
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_img_lists(image_lists, image_names=None, title=None, figsize=(20, 20), ticks=False):
    """Helper function that shows several lists of images alongside each other"""
    rows = len(image_lists[0])
    cols = len(image_lists)
    cmap = None

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for j in range(cols):
        for i in range(rows):
            # if there is more than one image in the list --> more than one row
            if rows >= 2:
                ax = axes[i, j]
                image = image_lists[j][i]
                image_name = image_names[j][i]

            else:
                ax = axes[j]
                image = image_lists[j]
                image_name = image_names[j]

            # if image has less than three color channels
            if image.shape[-1] < 3 or len(image.shape) < 3:
                cmap = "gray"
                image = np.reshape(image, (image.shape[0], image.shape[1]))

            if not ticks:
                ax.axis("off")

            ax.imshow(image, cmap=cmap)
            ax.set_title(image_name)

    fig.suptitle(title, fontsize=10, y=1)
    fig.tight_layout()
    plt.show()

    return


##################################################################################
# Camera Calibration                                                             #
##################################################################################

def get_image_object_points(img_paths, nx, ny):
    """Maps object points in 3d real world space to image points in the 2d image.
    Used for camera calibration."""
    # prepare object and image points
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane
    # Prepare object points, like (0, 0, 0), (1, 0, 0), (2, 0, 0),... , (8, 5, 0)
    obj_pts = np.zeros((nx * ny, 3), np.float32)
    obj_pts[:, :2] = np.mgrid[:nx, :ny].T.reshape(-1, 2)

    for img_path in img_paths:
        # read in image
        img = load_image(img_path)
        # convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret == True:
            img_points.append(corners)
            obj_points.append(obj_pts)

    return img_points, obj_points


def calibrate_camera(img, cal_img_path, nx, ny):
    """Calibrates camera using one test image and the calibration images."""
    img_pts, obj_pts = get_image_object_points(cal_img_path, nx, ny)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, gray.shape[::-1], None, None)
    save_camera_coefficients(img_pts, obj_pts, mtx, dist)
    return mtx, dist


def save_camera_coefficients(image_points, object_points, mtx, dst):
    with open('calibration_coefficients/mtx_dst', 'wb') as f:
        pickle.dump('mtx', f)
        pickle.dump('dist', f)
    with open('calibration_coefficients/img_obj_pts', 'wb') as f:
        pickle.dump('image_points', f)
        pickle.dump('object_points', f)


def undistort_img(img, mtx, dist):
    """Undistorts image using a pre-calibrated camera matrix and distortion coefficients."""
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def cal_undistort(img, imgpoints, objpoints):
    """Alternative function for undistorting that calibrates mtx and dist every time anew."""
    # Use cv2.calibrateCamera() and cv2.undistort()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


##############################################################################################
# Image Preprocessing for Lane Line Pixel Detection: Thresholding and Perspective Transform  #
##############################################################################################

def to_grayscale():
    pass


def to_hls():
    pass


def color_threshold(image, channel, low=0, high=255):
    """Applies color thresholding to an image."""
    # image is in grayscale
    if (image.shape[-1] < 3 or len(image.shape) < 3):
        img_channel = image
    img_channel = image[:, :, channel]
    binary = np.zeros_like(img_channel)
    binary[(img_channel > low) & (img_channel <= high)] = 1
    return binary


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255, ksize=5):
    """Applies a sobel operator to the image and filters image pixels based on their absolute sobel value."""

    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if (orient == 'x'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    elif (orient == 'y'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    sobel_mask = np.zeros_like(scaled_sobel)
    sobel_mask[(scaled_sobel >= thresh_min) & (scaled_sobel < thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return sobel_mask


def color_gradient_threshold(img_undist, plot=False):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    # if necessary also gradient direction or 
    hls = cv2.cvtColor(img_undist, cv2.COLOR_RGB2HLS)
    color_mask = color_threshold(hls, channel=2, low=90, high=255)

    sobel_thresh_min = 30
    sobel_thresh_max = 100

    sobel_x_mask = abs_sobel_thresh(img_undist, orient='x',
                                    thresh_min=sobel_thresh_min, thresh_max=sobel_thresh_max, ksize=15)
    sobel_y_mask = abs_sobel_thresh(img_undist, orient='y',
                                    thresh_min=sobel_thresh_min, thresh_max=sobel_thresh_max, ksize=15)

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sobel_x_mask)
    combined_binary[(color_mask == 1) | (sobel_x_mask == 1) | (sobel_y_mask == 1)] = 1

    if plot == False:
        return combined_binary
    else:
        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        color_binary = np.dstack((np.zeros_like(sobel_x_mask), sobel_x_mask, color_mask)) * 255
        return combined_binary, color_binary


def get_perspective_transform(img):
    """Transforms a road image into bird's eye perspective"""

    # the four trapezoid points defined by looking at a test image
    bottom_px = img.shape[0] - 1
    src = np.array([[210, bottom_px], [600, 450], [700, 450], [1110, bottom_px]], np.float32)
    offset = 200
    img_size = (img.shape[1], img.shape[0])

    dst = np.array([[200, bottom_px], [200, 0], [1000, 0], [1000, bottom_px]], np.float32)
    dst_pts = np.float32([[offset, offset], [img_size[0] - offset, offset],
                          [img_size[0] - offset, img_size[1] - offset],
                          [offset, img_size[1] - offset]])
    # we need M as well as Minv to later re/transform the image for plotting
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, M, Minv


# mtx and dist have been calculated above
def preprocess_image(img):
    """Applies all image preprocessing steps before lane detection."""
    # undistort image with the previously calculated `mtx` and `dst`
    undist_img = undistort_img(img, mtx=mtx, dist=dist)
    # apply color and gradient thresholding to get a binary image
    binary_img = color_gradient_threshold(undist_img)
    # apply perspective transform to get a warped image
    binary_warped_img, M, Minv = get_perspective_transform(binary_img)

    return binary_warped_img, M, Minv
