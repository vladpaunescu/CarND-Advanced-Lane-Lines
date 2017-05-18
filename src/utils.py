import os

from prop_config import cfg


def get_output_path(out_dir, img_name):
    img_name_no_ext = os.path.splitext(img_name)[0]
    return os.path.join(out_dir, img_name_no_ext + cfg.IMG_EXT)