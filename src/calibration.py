""" utility for camera calibration """
import os
import cv2
import numpy as np
import pickle

from prop_config import cfg


def calibrate():

    if not os.path.exists(cfg.CHESS_DIR):
        os.makedirs(cfg.CHESS_DIR)

    if not os.path.exists(cfg.UNDISTORT_DIR):
        os.makedirs(cfg.UNDISTORT_DIR)

    imgs = os.listdir(cfg.CAMERA_CALIB_DIR)
    print ("Calibration imgs {}".format(imgs))

    # arrays to store object points
    objpoints = []
    imgpoints = []

    objp = np.zeros((cfg.BY * cfg.BX, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cfg.BX, 0:cfg.BY].T.reshape(-1, 2)  # x, y coordinates
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for fimg in imgs:
        img = cv2.imread(os.path.join(cfg.CAMERA_CALIB_DIR, fimg))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cfg.BX, cfg.BY), None)

        if ret == True:
            corners_refined = cv2.cornerSubPix(gray, corners, (4, 4), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            img_chess = cv2.drawChessboardCorners(img, (cfg.BX, cfg.BY), corners, ret)
            chess_fname = os.path.join(cfg.CHESS_DIR, fimg)
            cv2.imwrite(chess_fname, img_chess)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    for fimg in imgs:
        img = cv2.imread(os.path.join(cfg.CAMERA_CALIB_DIR, fimg))
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        undistorted_fname = os.path.join(cfg.UNDISTORT_DIR, fimg)
        cv2.imwrite(undistorted_fname, undist)


    dist_data = { 'mtx': mtx,
                  'dist': dist
                  }
    with open(cfg.CAMERA_MATRIX_FILE, 'wb') as f:
        pickle.dump(dist_data, f)

    return mtx, dist


if __name__ == "__main__":
    calibrate()