## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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

[image1]: ./assets/calibration1.jpg "Undistorted"
[image2]: ./assets/straight_lines1.jpg "Road Transformed"
[image3]: ./assets/straight_lines1_combined_thresh.png "Binary Example"
[image4]: ./assets/straight_lines1_perspective.jpg "Warp Example"
[image5]: ./assets/straight_lines1_birds_eye.jpg "Bird's eye Example"
[image6]: ./assets/straight_lines1_combined_thresh_lanes.png "Fit Visual"
[image7]: ./assets/straight_lines1_lanes.png "Output Visual"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained ``` calibration.py ```

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

Tha code for rectification stage in the pipeline is contained in ```rectify.py``` The calibration matrix is loaded from the pickle file, aleready saved after calibration on chessboard corners images. Then the input image is undistorted and saved, on disk or returned for next stage in the pipeline. Once we have the camera matrix, given the image was captured with the same camera, we can use it to rectify any image from the camera. 

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps are in `threshold.py`).  Here's an example of my output for this step:

![alt text][image3]

I used a combiantion of:
 * color thresholding of white and yellow
 * HSV thresholding
 * Sobel horiozntal absolute
 * Sobel magnitude
 * Sobel direction

While all of them may be useful, after careful parameter tunning, I dediced to use only  color, hsv, and horizontal sobel. 
This is by far the most difficult part of the project since line detection is based on the quality of the masks extracted here.

```python 
combined_binary[((color_binary == 1) | (horiz_binary == 1))] = 1 
```


To accelerate processing time, I've only applied threhsolding on the warped image (bird's eye view image), with the selected ROI in front of car.


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is found in ```perspective.py``` file.
The points are defiened in the following manner:

```python

# top left, bottom left, bottom right, top right
    self.src = np.float32(
        [[img_size[0] / 2 - 63, img_size[1] / 2 + 95],
         [180, img_size[1]],
         [img_size[0] - 160, img_size[1]],
         [img_size[0] / 2 + 65, img_size[1] / 2 + 95]])

# top left, bottom left, bottom right, top right
    self.dst = np.float32(
        [[200, 0],
         [200, 720], 
         [1000, 720],
         [1000, 0]])

```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 577, 455      | 200, 0        | 
| 180, 720      | 200, 720      |
| 1120, 720     | 100, 720      |
| 705, 455      |  1000, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

The top down view of the image above looks like this:

![alt text][image5]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the method presented in course lectures with sliding window, and computing histograms. This method seems pretty good, but has some problems when the binary image input is bad.
The complete code for finding the lane lines is in ```line_detectro.py``` file.
Even with window time avearging it has some issues given bad binary image.

The operation has 2 cases:

* initial line finding, when we don't know exactly where the histogram peaks reside.
* line finding after one initial detection, we restric the search space for consecutive frames. There is a margin of 150 pixels around the histogram peaks. I also tested with a margin of 50 pixels around the histogram peaks.

The file contains 2 classes:
* ```LineDetector```
* ```Line```

```LineDetector``` provides methods for finding lane points (left and right), fitting a polynomial, averaging in time and drawing lane overalays, or line fits.

The main method for lane finding is: ``` def detect_lines(self, binary_warped)```. Here I distinguish 2 cases. If the input is a video, we use time averaging to smooth detections, and stabilize line fits. If the input is single image. We only compute for current frame.

Method ``` def detect_lines_initial(self, binary_warped)``` is used for single frame detection and initialzing a video input detection.

Method ```def track_lines(self, binary_warped)``` is used for tracking lines in video frames, using information from previous detection.

Method ```def smooth_detections(self, binary_warped)``` is useful for time fitlering of fits in consecutive frames in videos.

The fit is performed as in course's lectures, using a 2nd order polynomial.

Class ```Line``` is used to abstract data about the 2 fitted lines of the lane.

I provide 3 output videos:
* ```lanes_video_no_smoothing.mp4``` contains lane finding without any time averaging
* ```lanes_video_margin_150.mp4``` contains lane finding with time averaging
* ```lanes_video_margin_50.mp4``` contains lane finding with any time averaging

Here is an example on line fits fot the lane in top down view using a window of 150 pixels:

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The ```line_detector.py``` file contains code for estimating lane curvature, and car position on the lane.

 Method ```def get_lane_curvature(self, binary_warped):``` implements the idea from the course lecture. I used the meteres per pixel transformation scales from the lecture in order to transform to real world coordinates expressed in meters.
 
```python
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
```


The lane curvature is estimated by computing left and right line fit curvatures, and then averaging the 2 curvatures. The radius is expressed in meters using the meters per pixels in y direction. The algorithm is similar to the one presented in the course's lectures.

In order to estimate the car position on the lane, I used the fact that the camera center (image center) is the positon of the car on the lane.

Then I computed the base points of the lanes, using line fits for left and right lanes, respectively.

The difference between the midpoint of the lane (average between the base points of left line fit and right line fit), and the camera center (1280/2) was the car offset from the lane center expressed in pixels.

Then I transformed the difference into meters by scaling it using metersp per pixels in x dimension.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for running the complete pipeline is found in ```pipeline.py```.
It has to methods of operation:

* on frames
* on video

The code for unwarping the image, and overlaying the binary image onto input is found in ```merger.py``` file.

Here is an example of the final output:


![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./lanes_video_margin_150.mp4)

The code is written using some design patterns.
I designed it by taking into account that we can divide the computation processes into stages.

The ```prop_config.py``` file contains global configuration properties for the entire project.
The ```mapper.py``` file contains a generic interface that can run a pipeline stage given the function of the stage, an input image buffer and returning an output. It's used to write stage intermediarey results on disk in different directories for inspecting the pipeline flow.
Each pipeline stage is structured into class and contained in a file.
Each pipeline stage has a testing method on the given test_image.
It is assumed the the stage has the intermediary input avialable form the previous stage when running the tests. All intermediary stage buffers directories are configurablfe from the config file.

The complete pipeline is implemented in ```pipeline.py``` python file.
Method ```def build_and_run_pipeline_on_frames(test_imgs_dir, lanes_dir)``` is used to run the complete pipeline on the test images.
Method ```def build_and_run_pipeline_on_video(test_video, out_video):``` is used to write the pipeline on video input.

The lane overlay is implemented in the  Merger class contained in ```merger.py``` file.

In pepeline.py I defined 2 clases:

```python
class Stage:

    def __init__(self, process_fn, name="", verbose=False):
        self.verbose = verbose
        self.name = name
        self.processor = process_fn
        self.input = None
        self.output = None

    def run(self, buf_input):
        if self.verbose:
            print("Running stage {}".format(self.name))

        self.input = buf_input
        self.output = self.processor.run(buf_input)
        return self.output

```

```python
class Pipeline:

    def __init__(self):
        self.stages = []

    def add_stage(self, stage):
        self.stages.append(stage)
        return self

    def run(self, img):
        input_buffer = img
        for stage in self.stages:
            out_buffer = stage.run(input_buffer)
            input_buffer = out_buffer

        return out_buffer
```

These 2 classes allow generic execution of a pipeline that has multiple stages.
Each stage must read input from an image buffer, and write output to an output buffer.

In order to read lane curvature and car position I had to circumvent this limitation. I used a lateral effect and read data directly form the object contained in hte line detection stage:
 
 ```python
lane_curvature = line_detector.avg_curve_rad.round(3)
car_x_pos = line_detector.car_x_position.round(3)
is_straight_lane = line_detector.is_straight_lane()
```

The pipeline is as follows:

In code:

```python

    pipeline = Pipeline()
    pipeline.add_stage(rectify_stage)
    pipeline.add_stage(perspective_stage)
    pipeline.add_stage(threshold_stage)
    pipeline.add_stage(line_detector_stage)

```
In words:

1. Undistort Frame
2. Transform to bird's eye view
3. Threshold the bird's eye view
4. Detect lines for lane

Then, ```Merger``` class is used to unwarp the image with the detected lane, and overlay it on the input.


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I've encountered difficulties in regions whre thte road changes the aspect.
I overcame this difficulties using 2 approaches:

* drop some irrelevant information from Sobel metrics - like sobel direction.
* add extra color thresholding to improve color filtering.
 
I've also encountered an issue when estimating lane curvature for straight lanes. In ```line_detector.py``` the code for testing if a line is straight is found in  method ```def is_straight_lane(self)```.
 
 The idea is to plot the left line x coordinates and compute a difference between the bottom most one and the top most one like so:
 
 ```   line_x_max_diff = abs(left_line_x_m[0] - left_line_x_m[-1]) ```
 
If this difference is small enough, then the lane is straight.

Being based on the binary image mask, and window sarch around a maximum here are some failure cases of hte approach:

1. It may fail bracuse of binary image when there are some artifacts on the road that are detected by sobel, or when the lane color changes, or when the road texture changes. Sobel detects contrast regions (edges) so it may detect false positives on road artifacts.
 
2. When the road has very tight curves, the window search will fail

3. When the left and right lines don't reside in the ranges whre the window search looks.

Here are some ideas to improve it:

1. Find a more robust approach that is not based on hand coded and hard coded features, but let the data lead you - have a data driven approach. Use this data to bootstrap a deep learning algorithm and predict lane with a convolutional neural network.
2. Ideaally, drop the window search approach because has miss cases and use a global search with non maximum suppresion.





