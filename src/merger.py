import cv2
import numpy as np
import os

from prop_config import cfg
from perspective import Perspective


class Merger:

    DIRECT = 0
    INVERSE = 1

    def __init__(self):
        self.inv_perspective = Perspective(op_mode=Perspective.INVERSE)

    def merge(self, lane_warped, rectified):
        lane_overlay = self.inv_perspective.run(lane_warped)
        merged = cv2.addWeighted(rectified, 1, lane_overlay, 0.3, 0)
        return merged


