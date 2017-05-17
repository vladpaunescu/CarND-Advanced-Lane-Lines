import os
import cv2

from prop_config import cfg

class Mapper:
    def __init__(self, input_dir, output_dir, fn, file_part=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.fn = fn
        self.file_part = file_part

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process_frame(self, img_name, scale=1.0):
        img = cv2.imread(os.path.join(self.input_dir, img_name))
        out = self.fn(img)
        # out_fname = ""
        if self.file_part is not None:
            img_name_no_ext = os.path.splitext(img_name)[0]
            out_fname = os.path.join(self.output_dir,
                                     img_name_no_ext + self.file_part + cfg.IMG_EXT)
        else:
            out_fname = os.path.join(self.output_dir, img_name)
        cv2.imwrite(out_fname, out * scale)

