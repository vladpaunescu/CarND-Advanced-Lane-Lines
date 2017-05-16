import cv2
import numpy as np
import os

from prop_config import cfg
from mapper import Mapper


class Perspective:

    def __init__(self, img_size=None):
        if img_size == None:
            img_size = cfg.IMG_WIDTH, cfg.IMG_HEIGHT

        self.img_size = img_size

        print("Image size {}".format(img_size))

        # top left, bottom left, bottom right, top right
        self.src = np.float32([[img_size[0] / 2 - 63, img_size[1] / 2 + 95], [180, img_size[1]],
                               [img_size[0] - 160, img_size[1]], [img_size[0] / 2 + 65, img_size[1] / 2 + 95]])

        # top left, bottom left, bottom right, top right
        self.dst = np.float32([[200, 0],  [200, 720], [1000, 720], [1000, 0]])

        print("Perspective src points\n{}".format(self.src))
        print("Perspective dst points\n{}".format(self.dst))

        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def birds_eye(self, img):
        return cv2.warpPerspective(img, self.M, self.img_size)

    def inv_birds_eye(self, ipm):
        return cv2.warpPerspective(ipm, self.M_inv, self.img_size)

    def draw_perspective_roi(self, img):
        out = np.copy(img)
        cv2.polylines(out, [self.src.astype(np.int32)], False, (0, 255, 0), thickness=4)
        cv2.polylines(out, [self.dst.astype(np.int32)], False, (0, 0, 255), thickness=4)
        return out


def run_on_test_images():
        imgs = os.listdir(cfg.TEST_UNDISTORTED_IMGS_DIR)
        print("Rectified test images {}".format(imgs))
        perspective = Perspective()

        # draw rois polygons
        ipm_mapper = Mapper(cfg.TEST_IMGS_DIR, cfg.TEST_ROI_IMGS_DIR, fn=perspective.draw_perspective_roi)
        for img in imgs:
            ipm_mapper.process_frame(img)

        # draw rois polygons
        ipm_mapper = Mapper(cfg.TEST_IMGS_DIR, cfg.TEST_BIRDS_EYE_IMGS_DIR, fn=perspective.birds_eye)
        for img in imgs:
            ipm_mapper.process_frame(img)

if __name__ == "__main__":
        run_on_test_images()



