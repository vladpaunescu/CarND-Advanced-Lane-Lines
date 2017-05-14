""" config file """

from os.path import join

class Property(object): pass

__C = Property()

__C.CAMERA_CALIB_DIR = "./camera_cal"
__C.BX = 9
__C.BY = 6


__C.CHESS_DIR = "./corners_imgs"
__C.UNDISTORT_DIR = "./undistorted_imgs"
__C.CAMERA_MATRIX_FILE = 'camera_matrix.p'

cfg = __C

CAMERA_DIR = './camera_cal'
CORNERS_DIR = './corners_cal'
UNDISTORTED_DIR = './undistorted_cal'
