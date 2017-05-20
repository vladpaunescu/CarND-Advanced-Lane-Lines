import numpy as np
import cv2
import os

from prop_config import cfg
from mapper import Mapper


DEBUG = False

class Line:
    def __init__(self):
        self.x_points = None
        self.y_points = None
        self.poly_fit = None
        self.xfit = None

class LineDetector:
    def __init__(self, video_mode=False):
        self.left_line = Line()
        self.right_line = Line()
        self.is_fitted = False
        self.video_mode = video_mode


        self.out_img = None


    def get_lane_curvature(self, binary_warped):
        y_eval = binary_warped.shape[0]

        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        left_x_points = self.left_line.xfit
        left_y_points = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        right_x_points = self.right_line.xfit
        right_y_points = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        left_fit = self.left_line.poly_fit
        right_fit = self.right_line.poly_fit

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(left_y_points * ym_per_pix, left_x_points * xm_per_pix, 2)
        right_fit_cr = np.polyfit(right_y_points * ym_per_pix, right_x_points * xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix +
                               left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix +
                                right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

        center_camera_x = binary_warped.shape[1] / 2
        bottom_line_y = binary_warped.shape[0] - 1
        left_lane_base_x = left_fit[0] * bottom_line_y ** 2 + left_fit[1] * bottom_line_y + left_fit[2]
        right_lane_base_x = right_fit[0] * bottom_line_y ** 2 + right_fit[1] * bottom_line_y + right_fit[2]

        car_x_position = (center_camera_x - (left_lane_base_x + right_lane_base_x) / 2) * xm_per_pix

        y_pts_m = left_y_points * ym_per_pix
        left_line_x_m = left_fit_cr[0] * y_pts_m ** 2 + left_fit_cr[1] * y_pts_m + left_fit_cr[2]

        line_x_max_diff = abs(left_line_x_m[0] - left_line_x_m[-1])
        avg_curve_rad = (left_curverad + right_curverad) / 2.0

        self.avg_curve_rad = avg_curve_rad
        self.line_x_max_diff = line_x_max_diff
        self.car_x_position = car_x_position

        return avg_curve_rad, line_x_max_diff, car_x_position

    def is_straight_lane(self):
        return self.line_x_max_diff < 0.4

    def draw_lines(self, binary_warped):
        left_fit = self.left_line.poly_fit
        right_fit = self.right_line.poly_fit
        left_x_points, left_y_points = self.left_line.x_points, self.left_line.y_points
        right_x_points, right_y_points = self.right_line.x_points, self.right_line.y_points

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        self.left_line.xfit = left_fitx
        self.right_line.xfit = right_fitx

        self.out_img[left_y_points, left_x_points] = [255, 0, 0]
        self.out_img[right_y_points, right_x_points] = [0, 0, 255]

        left_plt = np.dstack((left_fitx, ploty))
        right_plt = np.dstack((right_fitx, ploty))

        cv2.polylines(self.out_img, [left_plt.astype(np.int32)], False, (0, 255, 255), thickness=2)
        cv2.polylines(self.out_img, [right_plt.astype(np.int32)], False, (0, 255, 255), thickness=2)

        # plt.imshow(out_img)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)

    def fill_lane(self, binary_warped):
        left_fit = self.left_line.poly_fit
        right_fit = self.right_line.poly_fit
        left_x_points, left_y_points = self.left_line.x_points, self.left_line.y_points
        right_x_points, right_y_points = self.right_line.x_points, self.right_line.y_points

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        self.out_img[left_y_points, left_x_points] = [255, 0, 0]
        self.out_img[right_y_points, right_x_points] = [0, 0, 255]

        left_plt = np.dstack((left_fitx, ploty))
        right_plt = np.dstack((right_fitx, ploty))

        cv2.polylines(self.out_img, [left_plt.astype(np.int32)], False, (0, 255, 255), thickness=2)
        cv2.polylines(self.out_img, [right_plt.astype(np.int32)], False, (0, 255, 255), thickness=2)

        left_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((left_pts, right_pts))

        cv2.fillPoly(self.out_img, [pts.astype(np.int32)], (0, 255, 0))

        # plt.imshow(out_img)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)

    def draw_lane_overlay(self, binary_warped):

        left_fit = self.left_line.poly_fit
        right_fit = self.right_line.poly_fit
        left_x_points, left_y_points = self.left_line.x_points, self.left_line.y_points
        right_x_points, right_y_points = self.right_line.x_points, self.right_line.y_points

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        self.left_line.xfit = left_fitx
        self.right_line.xfit = right_fitx

        lane_overlay = np.zeros_like(binary_warped).astype(np.uint8)
        lane_overlay = np.dstack((lane_overlay, lane_overlay, lane_overlay))

        left_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((left_pts, right_pts))

        cv2.fillPoly(lane_overlay, [pts.astype(np.int32)], (0, 255, 0))
        return lane_overlay

    def detect_lines(self, binary_warped):

        if not self.is_fitted:
            self.detect_lines_initial(binary_warped)
        else:
            self.track_lines(binary_warped)

        if self.video_mode:
            self.is_fitted = True

        if DEBUG:
            # self.draw_lines(binary_warped)
            self.fill_lane(binary_warped)
            return self.out_img

        # prepare a blank lane overlay
        self.out_img = self.draw_lane_overlay(binary_warped)
        self.get_lane_curvature(binary_warped)

        return self.out_img


    def detect_lines_initial(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[np.int(binary_warped.shape[0] / 2):, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 150
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each and store the result
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # store data
        self.left_line.x_points,  self.left_line.y_points = leftx, lefty
        self.right_line.x_points,  self.right_line.y_points = rightx, righty
        self.left_line.poly_fit = left_fit
        self.right_line.poly_fit = right_fit

        self.out_img = out_img

    def track_lines(self, binary_warped):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!

        left_fit = self.left_line.poly_fit
        right_fit = self.right_line.poly_fit

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = (
            (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
                nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
                nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # store data
        self.left_line.x_points, self.left_line.y_points = leftx, lefty
        self.right_line.x_points, self.right_line.y_points = rightx, righty
        self.left_line.poly_fit = left_fit
        self.right_line.poly_fit = right_fit

    def run(self, img):
        return self.detect_lines(img)


def run_on_test_images(input_dir, output_dir):
    imgs = os.listdir(input_dir)
    print("Test images {}".format(imgs))
    line_detector = LineDetector()

    line_detector_mapper = Mapper(input_dir, output_dir,
                                  fn=line_detector.detect_lines,
                                  is_binary=True)
    mappers = [line_detector_mapper]

    for idx, img in enumerate(imgs):
        print("Image {}".format(idx))
        for mapper in mappers:
            mapper.process_frame(img)

if __name__ == "__main__":
    print("Find lines on birds eye images")
    run_on_test_images(cfg.TEST_BIRDS_EYE_THRESH_IMGS_DIR,
                       cfg.TEST_BIRDS_EYE_BINARY_LANE_IMGS_DIR)
