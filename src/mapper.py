import os
import cv2


class Mapper:
    def __init__(self, input_dir, output_dir, fn):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.fn = fn

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process_frame(self, img_name):
        img = cv2.imread(os.path.join(self.input_dir, img_name))
        out = self.fn(img)
        out_fname = os.path.join(self.output_dir, img_name)
        cv2.imwrite(out_fname, out)

