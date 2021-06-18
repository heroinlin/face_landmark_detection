"""
SlowFast TensorRT model inference output and time test.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
work_root = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(work_root)
from inference import PLFDONNX


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        "--model_path",
        type=str,
        default=os.path.join(
            work_root, "onnx_models/pfld.onnx"),
        help="Path to the ONNX model file",
    )
    parser.add_argument(
        '-i',
        "--image_path",
        type=str,
        default=os.path.join(work_root, "../../../data/image_data/images/00000_1.jpg"),
        help="test image path",
    )
    parser.add_argument(
        '-t',
        '--times',
        type=int,
        default=100,
        help='inference times',
    )
    return parser.parse_args()


def compute_time(func, args, run_num=100):
    start_time = time.time()
    for i in range(run_num):
        func(*args)
    end_time = time.time()
    avg_run_time = (end_time - start_time)*1000/run_num
    return avg_run_time


def compute_inference_time():
    args = parser_args()
    print(args)
    model_path = args.model_path
    run_times = args.times
    input_path = args.image_path
    if not os.path.exists(input_path):
        print(input_path, " is not exists! please check input path!")
        exit(-1)
    
    detector = PLFDONNX()
    # detector = PLFDONNX(model_path=model_path)
    image = cv2.imdecode(np.fromfile(input_path, np.uint8()), 1)
    input_image = detector._preprocess(image)
    inputs = [input_image]
    pred = detector.inference(inputs)
    print(pred)
    outputs = detector._postprocess(pred)
    print(outputs)
    preprocess_time = compute_time(detector._preprocess, [image], run_times)
    print("avg preprocess time is {:02f} ms".format(preprocess_time))

    inference_time = compute_time(detector.inference, [inputs], run_times)
    print("avg inference time is {:02f} ms".format(inference_time))

    postprocess_time = compute_time(detector._postprocess, [pred], run_times)
    print("avg postprocess time is {:02f} ms".format(postprocess_time))

    total_time = compute_time(detector.detect, [image], run_times)
    print("avg total predict time is {:02f} ms".format(total_time))

    detector.destroy()


if __name__ == "__main__":
    compute_inference_time()
