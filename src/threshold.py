import cv2
import numpy as np
import os

from prop_config import cfg


class Threshold:
    def __init__(self):
        self.H_COLOR_THRESH = (15, 100)
        self.S_COLOR_THRESH = (170, 255)
        self.SOBEL_DIR_THRESHOLD = (20, 100)
        self.SOBEL_MAG_THRESHOLD = (50, 100)
        self.SOBEL_DIR_THRESHOLD = (0.7, 1.3)
        self.KSIZE = 15


    def color_thresh(self, image, thresh=None):
        thresh = self.S_COLOR_THRESH if thresh is None else thresh

        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]
        h_channel = hls[:, :, 0]

        color_mask = np.zeros_like(s_channel)
        color_mask[(s_channel >= thresh[0]) & (s_channel <= thresh[1]) &
                     (h_channel >= 15) & (h_channel <= 100)] = 1

        return color_mask


    def abs_sobel_dir_thresh(self, image, dx, dy, ksize=None, thresh = None):

        ksize = self.KSIZE if ksize is None else ksize
        thresh = self.SOBEL_DIR_THRESHOLD if thresh is None else thresh

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        gray = hsv[:,:,2]
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

        s_binary = np.zeros_like(scaled_sobel)
        s_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return s_binary


    def abs_horiz_sobel_thresh(self, image):
        return self.abs_sobel_dir_thresh(image, dx=1, dy=0)


    def magn_sobel_thresh(self, image, ksize=None, thresh=None):

        ksize = self.KSIZE if ksize is None else ksize
        thresh = self.SOBEL_MAG_THRESHOLD if thresh is None else thresh

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
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

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
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


if __name__ == "__main__":

    if not os.path.exists(cfg.PIPELINE_TESTS_DIR):
        os.makedirs(cfg.PIPELINE_TESTS_DIR)

    t = Threshold()
    print(os.path.join(cfg.TEST_IMGS_DIR, "test1.jpg"))
    img = cv2.imread(os.path.join(cfg.TEST_IMGS_DIR, "test1.jpg"))
    color_binary = t.color_thresh(img)
    cv2.imwrite(os.path.join(cfg.PIPELINE_TESTS_DIR, "color_threshold.png"), color_binary * 255)

    sobel_abs_binary = t.abs_horiz_sobel_thresh(img)
    cv2.imwrite(os.path.join(cfg.PIPELINE_TESTS_DIR, "sobel_horiz_abs_binary.png"), sobel_abs_binary * 255)

    sobel_mag_binary = t.magn_sobel_thresh(img)
    cv2.imwrite(os.path.join(cfg.PIPELINE_TESTS_DIR, "sobel_mag_binary.png"), sobel_mag_binary * 255)

    sobel_dir_binary = t.dir_sobel_thresh(img)
    cv2.imwrite(os.path.join(cfg.PIPELINE_TESTS_DIR, "sobel_dir_binary.png"), sobel_dir_binary * 255)

    combined_binary = t.threshold_image(img)
    cv2.imwrite(os.path.join(cfg.PIPELINE_TESTS_DIR, "combined_binary.png"), combined_binary * 255)
