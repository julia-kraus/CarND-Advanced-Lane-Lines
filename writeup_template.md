# Advanced Lane Line Detection

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undist_test_img.png "Undistorted Test Img"
[image2]: ./output_images/orig_test_img.png "Original Distorted Test Img"
[image4]: ./output_images/calib_example.png "Example Calibration Img"
[image5]: ./output_images/detected_corners.png "Detected Corners Calibration Img"
[image6]: ./output_images/undistorted_calib.png "Undistorted Calibration Img"
[image7]: ./output_images/hls_channel.png "HLS Color Space S Channel"
[image8]: ./output_images/hls_thresholding.png "HLS Thresholding Binary"
[image9]: ./output_images/sobel_operator.png "Gradient Thresholding Binary"
[image10]: ./output_images/combined_binary.png "Gradient and Color Thresholding Combined"
[image11]: ./output_images/unwarped.png "Unwarped original image"
[image12]: ./output_images/warped.png "Perspective Transformed image"
[image13]: ./output_images/persp_trafo_img_before.png "Perspective Transform Source"
[image14]: ./output_images/persp_trano_img_after.png "Perspective Transform Destination"
[image15]: ./output_images/before_pipeline.png "Before Pipeline"
[image16]: ./output_images/after_pipeline.png "After Pipeline"
[image17]: ./output_images/lane_lines.png "Lane Lines Sliding Window Method"
[image18]: ./output_images/histogram.png "Histogram"
[image18]: ./output_images/result.png "result"

[video1]: ./out_project_video.mp4 "Video"


### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration


The code for this step is contained in the first code cell of the IPython notebook located in "./examples/solution.ipynb" (or in lines 65 through 234 of the file called `preprocessing.py`).  

First of all, I load the calibration images that are stored in the ./camera_cal folder. These are 20 images and we can see that there are 9x6 chessboard corners. Then, I apply the opencv function `cv2.get_chessboard_corners` which finds the coordinates of the chessboard corners in the image. The result can be visualized witht he opencv function `cv2.draw_chessboard_corners`Once they are found, we need a function that maps the 2D image points to the "object points", which will be the (x, y, z) coordinates of the chessboard corners in the real world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

An example of a distorted calibration image can be seen here:

![alt text][image4]

Finding the chessboard corners:

![alt text][image5]


I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  The function returned a camera matrix `mtx` and distortion coefficients `dist`, which I saved in the folder , so I wouldn't have to calculate them every time anew. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image6]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
This is an example distorted test image:
![alt text][image2]

Next, I applied distortion correction to all the test images. Here is one distortion corrected test image:
![alt text][image1]

One can see that after the distortion correction, one can see the image more from the front.

#### 2. Color and Gradient Thresholding

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 125 through 200 in `preprocessing.py`).  First of all, I considered different color spaces and found that the lines were most clearly visible in the saturation channel of the HLS color space. For a comparison of different channels and color spaces, see `solution.ipynb`. The saturation channel of our example pic can be seen here:

![alt text][image7]

Next, I applied a color threshold on this channel based on which I converted our test images to binary. Here's an example of my output for this step:

![alt text][image8]

In the next step, I applied a Sobel transformation in x-directions on the image. The Sobel transformation highlights pixels with steep gradients on the image and thus can help us detect the lane line pixels. Then I selected a threshold on the Sobel transform and got the following binary image: 

![alt text][image9]

Applying both gradient and color threshold resulted in the following test image:

![alt text][image10]

We can see that the lane lines are clearly visible. 

#### 3. Perspective Transform.

It is easier to fit a curve to the lane lines, if we can see the street from a bird's eye view. Therefore, we transformed our image into bird's eye view. The code for my perspective transform can be found in lines 202 through 222 in the file `preprocessing.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `get_perspective_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the following test image to determine the source and destination points manually:

![alt text][image13]

Into this picture, I fit a polygon that matches the lane lines: 
![alt text][image14]

I chose the following:

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

Then, I transformed the test images. Here is an example of an unwarped and a warped image. Before warping:

![alt text][image11]

After warping:
![alt text][image12]

I combined all these preprocessing steps into the `preprocess_image` pipeline function that can be found in `preprocessing.py`, lines 223 to 234. The output of this image pipeline on our test image was:

![alt text][image16]

We can again see that the lane lines are visible. However, there are still many pixels activated that don't belong to the lane line. Our next steps are therefore to identify only the lane line pixels and fit a curve to the lane line.

#### 4. Identifiying Lane Line Pixels
This section is implemented in `lane_finding.py`.

First of all, we want to find out, where our lines begin at the bottom of the image. Therefore, we plot a histogram in x-direction on the test images. This means, we step in x-direction over the image. At each step we count up how many pixel activations there across the lower half of the image in y-direction. Then, we plot the histogram. The histogram's two highest peaks are where the most activated pixels are, and therefore the lanes are most likely located:

![alt text][image18]

Once we have found the two lane bases at the bottom of the window, we use a sliding window approach to track the lane lines to the top of the image. As first is to split the histogram into two sides, one for each lane line. Then we move a fixed size sliding window up. We determine the center of the activated pixels inside this sliding window. If there are enough activated pixels, we re-center the window. This is repeated until the top of the image, for both lanes separately. This window search is implemented in `lane_finding.py`'s function `search_lane_from_scratch`. After having thus found the lane pixels for the left and right lane, we fit a second order polynomial (of form `A * y**2 + B*y + C`) for each of them. The result can be seen here:

![alt text][image17]

In case of a video we don't have to do the sliding window search for each frame because subsequent frames have the lanes in similar positions. In the next frame of video you don't need to do a blind search again, but instead you can just search in a margin around the previous line position. This strategy is implemented in the function `search_lane_from_prior`.

This is equivalent to using a customized region of interest for each frame of video, and should help you track the lanes through sharp curves and tricky conditions. If you lose track of the lines, go back to your sliding windows search or other method to rediscover them. Further, make the algortihm more stable, the fitted polynomials were averaged over the last three detected polynomials.

#### 5. Calculating Curvature

After we identified the lane line pixels, we need to calculate the road's curvature radius. We can calculate the radius of curvature directly from the polynomials that we fitted to the lane lines with the equation `((1 + (2 * A * y + B) ** 2) ** 1.5) / abs(2 * A)`. However, we still need to convert the result from pixels to meters. This was done assuming the road is 30m long and 3.7m wide, using the conversion

`ym_per_pix = 30/720 # meters per pixel in y dimension` and

`xm_per_pix = 3.7/700 # meters per pixel in x dimension`

I did this in lines # through # in my code in the function `get_curvature_real`.

#### 6. Result Sample Image

In the last step, I calculated the car's position from the curvature. Therefore, I determined the leftmost and rightmost lane pixel. The lane center thus is the average over the left and right pixel. 
Then I calculated how far the lane center is apart from the image center in pixels. Again the same pixel-to-meter conversion as above was applied to obtain the distance in meters. 

![alt text][image19]

---

### Pipeline (video)


Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
