""" config file """

from os.path import join

class Property(object): pass

__C = Property()

__C.CAMERA_CALIB_DIR = "./camera_cal"
__C.BX = 9
__C.BY = 6


__C.CHESS_DIR = "./corners_imgs"
__C.CAMERA_MATRIX_FILE = 'camera_matrix.p'

__C.UNDISTORT_IMGS_DIR = "./undistorted_imgs"

__C.TEST_IMGS_DIR = "./test_images"
__C.TEST_UNDISTORTED_IMGS_DIR = "./test_undistorted_imgs"

__C.TEST_ROI_IMGS_DIR = "./test_roi_imgs"
__C.TEST_BIRDS_EYE_IMGS_DIR = "./test_birds_eye_imgs"


__C.TEST_THRESH_IMGS_DIR = "./test_thresh_imgs"
__C.TEST_BIRDS_EYE_THRESH_IMGS_DIR = "./test_birds_eye_thresh_imgs"

__C.TEST_BIRDS_EYE_BINARY_LINE_IMGS_DIR = "./test_birds_eye_binary_line_imgs"
__C.TEST_BIRDS_EYE_BINARY_LANE_IMGS_DIR = "./test_birds_eye_binary_lane_imgs"

__C.TEST_LANES_IMGS_DIR = "./test_lanes_imgs"

__C.TEST_VIDEO = "./project_video.mp4"
__C.OUT_VIDEO = "./lanes_video.mp4"

__C.PIPELINE_TESTS_DIR = "./pipeline_tests"

__C.IMG_HEIGHT = 720
__C.IMG_WIDTH = 1280

__C.IMG_EXT = ".png"

cfg = __C

CAMERA_DIR = './camera_cal'
CORNERS_DIR = './corners_cal'
UNDISTORTED_DIR = './undistorted_cal'
