import os
import cv2
import pickle
from calibration import find_camera_matrix
from prop_config import cfg
from mapper import Mapper

class Rectify:

    def __init__(self):
        if not os.path.exists(cfg.CAMERA_MATRIX_FILE):
            self.mtx, self.dist = find_camera_matrix()
        else:
            with open(cfg.CAMERA_MATRIX_FILE, 'rb') as f:
                dist_data = pickle.load(f)
                self.mtx = dist_data['mtx']
                self.dist = dist_data['dist']
        print("[RECTIFY] Loaded camera matrix file {}".format(cfg.CAMERA_MATRIX_FILE))

    def rectify(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def run(self, img):
        return self.rectify(img)


def run_on_test_images():
    imgs = os.listdir(cfg.TEST_IMGS_DIR)
    print("Distorted test images {}".format(imgs))
    rectify = Rectify()
    rectfiy_mapper = Mapper(cfg.TEST_IMGS_DIR, cfg.TEST_UNDISTORTED_IMGS_DIR, fn=rectify.rectify)
    for img in imgs:
        rectfiy_mapper.process_frame(img)


if __name__ =="__main__":
    run_on_test_images()

