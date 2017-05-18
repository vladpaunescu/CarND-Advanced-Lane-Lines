import cv2
import numpy as np
import os

from prop_config import cfg
from mapper import Mapper

class Threshold:
    def __init__(self):
        self.H_COLOR_THRESH = (15, 100)
        self.S_COLOR_THRESH = (170, 255)
        self.SOBEL_X_THRESHOLD = (20, 100)
        self.SOBEL_MAG_THRESHOLD = (50, 100)
        self.SOBEL_DIR_THRESHOLD = (0.7, 1.3)
        self.KSIZE = 15


    def color_thresh(self, image, thresh=None):
        thresh = self.S_COLOR_THRESH if thresh is None else thresh

        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        h_channel = hls[:, :, 0]

        color_mask = np.zeros_like(s_channel)
        color_mask[(s_channel >= thresh[0]) & (s_channel <= thresh[1]) &
                   (h_channel >= self.H_COLOR_THRESH[0]) &
                   (h_channel <= self.H_COLOR_THRESH[1])] = 1

        return color_mask


    def abs_sobel_dir_thresh(self, image, dx = 1, dy = 0, ksize=None, thresh = None):

        ksize = self.KSIZE if ksize is None else ksize
        thresh = self.SOBEL_X_THRESHOLD if thresh is None else thresh

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = hsv[:,:,2]
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        s_binary = np.zeros_like(scaled_sobel)
        s_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return s_binary


    def abs_horiz_sobel_thresh(self, image):
        return self.abs_sobel_dir_thresh(image, dx=1, dy=0,
                                         ksize=self.KSIZE, thresh=self.SOBEL_X_THRESHOLD)


    def magn_sobel_thresh(self, image, ksize=None, thresh=None):

        ksize = self.KSIZE if ksize is None else ksize
        thresh = self.SOBEL_MAG_THRESHOLD if thresh is None else thresh

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = hsv[:,:,2]

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

        gradmag = np.sqrt(sobelx**2 + sobely**2)

        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)

        # Create a binary image of ones where threshold is met, zeros otherwise
        s_binary = np.zeros_like(gradmag)
        s_binary[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

        return s_binary

    def dir_sobel_thresh(self, image, ksize=None, thresh=None):

        ksize = self.KSIZE if ksize is None else ksize
        thresh = self.SOBEL_DIR_THRESHOLD if thresh is None else thresh

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = hsv[:,:,2]

        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    """ this method combines all thresholds """
    def threshold_image(self, img):
        color_binary = self.color_thresh(img)
        horiz_binary = self.abs_horiz_sobel_thresh(img)
        magn_binary = self.magn_sobel_thresh(img)
        dir_binary = self.dir_sobel_thresh(img)

        combined_binary = np.zeros_like(color_binary)
        combined_binary[((color_binary == 1) | (horiz_binary == 1)) & ((magn_binary == 1) | (dir_binary == 1))] = 1

        return combined_binary

    def run(self, img):
        return self.threshold_image(img)


DEBUG = False

def run_on_test_images(input_dir, output_dir):
        imgs = os.listdir(input_dir)
        print("Test images {}".format(imgs))
        threshold = Threshold()

        color_threshold_mapper = Mapper(input_dir, output_dir,
                                        fn=threshold.color_thresh, file_part="_color_thresh")
        sobel_x_threshold_mapper = Mapper(input_dir, output_dir, fn=threshold.abs_horiz_sobel_thresh,
                                          file_part="_sobel_x_thresh")
        sobel_magn_threshold_mapper = Mapper(input_dir, output_dir, fn=threshold.magn_sobel_thresh,
                                             file_part="_sobel_magn_thresh")
        sobel_dir_threshold_mapper = Mapper(input_dir, output_dir, fn=threshold.dir_sobel_thresh,
                                            file_part="_sobel_dir_thresh")
        combined_threshold_mapper = Mapper(input_dir, output_dir, fn=threshold.threshold_image,
                                           file_part="_combined_thresh")

        if DEBUG:
            mappers = [color_threshold_mapper, sobel_x_threshold_mapper,
                   sobel_magn_threshold_mapper, sobel_dir_threshold_mapper, combined_threshold_mapper]
        else:
            mappers = [combined_threshold_mapper]

        for idx,img in enumerate(imgs):
            print("Image {}".format(idx))
            for mapper in mappers:
                mapper.process_frame(img, scale = 255)


if __name__ == "__main__":
        print("Threshold on normal images")
        # run_on_test_images(cfg.TEST_UNDISTORTED_IMGS_DIR, cfg.TEST_THRESH_IMGS_DIR)
        print("Threshold on birds eye images")
        run_on_test_images(cfg.TEST_BIRDS_EYE_IMGS_DIR, cfg.TEST_BIRDS_EYE_THRESH_IMGS_DIR)

