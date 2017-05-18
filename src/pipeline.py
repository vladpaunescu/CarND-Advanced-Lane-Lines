import cv2
import os

from prop_config import cfg

from rectify import Rectify
from perspective import Perspective
from threshold import Threshold
from line_detector import LineDetector
from merger import Merger
from utils import get_output_path


from moviepy.editor import VideoFileClip


DEBUG = False

class Stage:

    def __init__(self, process_fn, name="", verbose=False):
        self.verbose = verbose
        self.name = name
        self.processor = process_fn
        self.input = None
        self.output = None

    def run(self, buf_input):
        if self.verbose:
            print("Running stage {}".format(self.name))

        self.input = buf_input
        self.output = self.processor.run(buf_input)
        return self.output


class Pipeline:

    def __init__(self):
        self.stages = []

    def add_stage(self, stage):
        self.stages.append(stage)
        return self

    def run(self, img):
        input_buffer = img
        for stage in self.stages:
            out_buffer = stage.run(input_buffer)
            input_buffer = out_buffer

        return out_buffer


class FrameProcessor:

    def __init__(self):
        rectify = Rectify()
        self.rectify_stage = Stage(rectify, "RECTIFY")

        perspective = Perspective()
        self.perspective_stage = Stage(perspective, "PERSPECTIVE")

        threshold = Threshold()
        self.threshold_stage = Stage(threshold, "THRESHOLD")

        line_detector = LineDetector()
        self.line_detector_stage = Stage(line_detector, "LINE_DETECTOR")

        pipeline = Pipeline()
        pipeline.add_stage(self.rectify_stage)
        pipeline.add_stage(self.perspective_stage)
        pipeline.add_stage(self.threshold_stage)
        pipeline.add_stage(self.line_detector_stage)

        self.pipeline = pipeline
        self.merger = Merger()

    def run_on_frame(self, frame):
        result = self.pipeline.run(frame)
        return self.merger.merge(result, self.rectify_stage.output)


def build_and_run_pipeline_on_frames(test_imgs_dir, lanes_dir):
    if not os.path.exists(lanes_dir):
        os.makedirs(lanes_dir)

    imgs = os.listdir(test_imgs_dir)
    print("Loaded test images {}".format(imgs))

    rectify = Rectify()
    rectify_stage = Stage(rectify, "RECTIFY", True)

    perspective = Perspective()
    perspective_stage = Stage(perspective, "PERSPECTIVE", True)

    threshold = Threshold()
    threshold_stage = Stage(threshold, "THRESHOLD", True)

    line_detector = LineDetector()
    line_detector_stage = Stage(line_detector, "LINE_DETECTOR", True)

    merger = Merger()

    pipeline = Pipeline()
    pipeline.add_stage(rectify_stage)
    pipeline.add_stage(perspective_stage)
    pipeline.add_stage(threshold_stage)
    pipeline.add_stage(line_detector_stage)

    for img_path in imgs:
        img = cv2.imread(os.path.join(test_imgs_dir, img_path))
        result = pipeline.run(img)
        print("[MERGER] Merging lane with original input")
        result = merger.merge(result, rectify_stage.output)

        out_fname = get_output_path(lanes_dir, img_path)
        cv2.imwrite(out_fname, result)


def build_and_run_pipeline_on_video(test_video, out_video):
    frame_processor = FrameProcessor()
    clip1 = VideoFileClip(test_video)
    white_clip = clip1.fl_image(frame_processor.run_on_frame)
    white_clip.write_videofile(out_video, audio=False)


if __name__ == "__main__":
    # build_and_run_pipeline_on_frames(cfg.TEST_IMGS_DIR, cfg.TEST_LANES_IMGS_DIR)
    build_and_run_pipeline_on_video(cfg.TEST_VIDEO, cfg.OUT_VIDEO)

