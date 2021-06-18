import json
import os

import cv2
import numpy as np
import torch

from .models import init_model


def image_crop(image, box_dict):
    if box_dict is not None:
        if isinstance(box_dict, dict):
            image = image[box_dict["top"]: box_dict["top"] + box_dict["height"],
                    box_dict["left"]: box_dict["left"] + box_dict["width"],
                    :]
        if isinstance(box_dict, list):
            box_dict = np.array(box_dict)
        if isinstance(box_dict, np.ndarray):
            if box_dict.ndim >= 2:
                box_dict = box_dict[0, :]
            image = image[box_dict[1]: box_dict[3], box_dict[0]: box_dict[2], :]
    # image = cv2.resize(image, dsize=(112, 112))
    return image


def draw_landmarks(image, landmarks, norm=True):
    """

    Parameters
    ----------
    image 展示的原始图片
    landmarks 维度为[106, 2]的列表或者numpy数组
    norm 关键点坐标的归一化标记，为True表示landmark值范围为[0, 1]

    Returns
    -------

    """
    if norm:
        scale_width = image.shape[1]
        scale_height = image.shape[0]
    else:
        scale_width = 1.0
        scale_height = 1.0
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array(landmarks)
    for index in range(landmarks.shape[0]):
        pt1 = (int(scale_width * landmarks[index, 0]), int(scale_height * landmarks[index, 1]))
        cv2.circle(image, pt1, 1, (0, 0, 255), 2)
    plot_line = lambda i1, i2: cv2.line(image,
                                        (int(scale_width * landmarks[i1, 0]),
                                         int(scale_height * landmarks[i1, 1])),
                                        (int(scale_width * landmarks[i2, 0]),
                                         int(scale_height * landmarks[i2, 1])),
                                        (255, 255, 255), 1)
    close_point_list = [0, 33, 42, 51, 55, 66, 74, 76, 84, 86, 98, 106]
    for ind in range(len(close_point_list) - 1):
        l, r = close_point_list[ind], close_point_list[ind + 1]
        for index in range(l, r - 1):
            plot_line(index, index + 1)
        # 将眼部, 嘴部连线闭合
        plot_line(41, 33)  # 左眉毛
        plot_line(50, 42)  # 右眉毛
        plot_line(65, 55)  # 鼻子
        plot_line(73, 66)  # 左眼
        plot_line(83, 76)  # 右眼
        plot_line(97, 86)  # 外唇
        plot_line(105, 98)  # 内唇


def enlarge_box(box_dict, width, height, scale=1.0):
    """将目标框缩放scale倍, 不超过图像边界"""
    if isinstance(box_dict, list):
        box_dict = np.array(box_dict)
    if isinstance(box_dict, np.ndarray):
        x1, y1, x2, y2 = box_dict[:, 0], box_dict[:, 1], box_dict[:, 2], box_dict[:, 3]
        x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        w = w * scale
        h = h * scale
        x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        x1[np.where(x1 < 0)] = 0
        y1[np.where(y1 < 0)] = 0
        x2[np.where(x2 > width - 1)] = width - 1
        y2[np.where(y2 > height - 1)] = height - 1
        box_dict = np.stack([x1, y1, x2, y2]).transpose()
        box_dict = box_dict.astype(np.int)
    elif isinstance(box_dict, dict):
        center_x = box_dict["left"] + box_dict["width"] // 2
        center_y = box_dict["top"] + box_dict["height"] // 2
        box_dict["width"] = int(box_dict["width"] * scale)
        box_dict["height"] = int(box_dict["height"] * scale)
        box_dict["left"] = max(0, center_x - box_dict["width"] // 2)
        box_dict["top"] = max(0, center_y - box_dict["height"] // 2)
        box_dict["width"] = min(box_dict["width"], width - box_dict["left"])
        box_dict["height"] = min(box_dict["height"], height - box_dict["top"])
    return box_dict


def box_landmark_process(box_dict, landmark_dict):
    """
    将关于全图的关键点字典转化为关于box的关键点字典
    Parameters
    ----------
    box
        目标框
        {'width': 270, 'top': 271, 'left': 300, 'height': 270}
    landmark_dict
        关键点字典
        {"
            contour_chin":
            {
                "y": 540,
                "x": 444
            },
            "left_eye_upper_left_quarter":
             {
                    "y": 318,
                    "x": 322
             },...}
    Returns
    -------
        新的关键点字典
    """
    if isinstance(box_dict, list) and isinstance(landmark_dict, list):
        box_dict = np.array(box_dict)
        landmark_dict = np.array(landmark_dict)
    if isinstance(box_dict, np.ndarray) and isinstance(landmark_dict, np.ndarray):
        # box_dict >> [x1, y1, x2, y2]
        landmark_dict[:, 0] = (landmark_dict[:, 0] - box_dict[0]) / (box_dict[2] - box_dict[0])
        landmark_dict[:, 1] = (landmark_dict[:, 1] - box_dict[1]) / (box_dict[3] - box_dict[1])
    elif isinstance(box_dict, dict) and isinstance(landmark_dict, dict):
        for key, value in landmark_dict.items():
            landmark_dict[key]['x'] = (landmark_dict[key]['x'] - box_dict["left"]) / box_dict["width"]
            landmark_dict[key]['y'] = (landmark_dict[key]['y'] - box_dict["top"]) / box_dict["height"]
    return landmark_dict


def box_landmark_process1(box_dict, landmark_dict):
    """
    将关于关于box的关键点字典转化为全图的关键点字典
    Parameters
    ----------
    box
        目标框
        {'width': 270, 'top': 271, 'left': 300, 'height': 270}
    landmark_dict
        关键点字典
        {"
            contour_chin":
            {
                "y": 540,
                "x": 444
            },
            "left_eye_upper_left_quarter":
             {
                    "y": 318,
                    "x": 322
             },...}
    Returns
    -------
        新的关键点字典
    """
    if isinstance(box_dict, list) and isinstance(landmark_dict, list):
        box_dict = np.array(box_dict)
        landmark_dict = np.array(landmark_dict)
    if isinstance(box_dict, np.ndarray) and isinstance(landmark_dict, np.ndarray):
        # box_dict >> [x1, y1, x2, y2]
        landmark_dict[:, 0] = landmark_dict[:, 0] * (box_dict[2] - box_dict[0]) + box_dict[0]
        landmark_dict[:, 1] = landmark_dict[:, 1] * (box_dict[3] - box_dict[1]) + box_dict[1]
    elif isinstance(box_dict, dict) and isinstance(landmark_dict, dict):
        for key, value in landmark_dict.items():
            landmark_dict[key]['x'] = landmark_dict[key]['x'] * box_dict["width"] + box_dict["left"]
            landmark_dict[key]['y'] = landmark_dict[key]['y'] * box_dict["height"] + box_dict["top"]
    return landmark_dict


def align_landmarks(landmark_dict):
    """
        按照人脸关键点顺序对关键点字典进行排序
        Parameters
        ----------
        landmark_dict  未排序的关键点字典

        Returns
        -------
            排序后的关键点字典
        """
    landmarks = []
    # for index, (key_name, value) in enumerate(landmark_dict.items()):
    # landmark_sort = open("landmark_points.txt", 'r').readlines()
    landmark_sort = open("datasets/landmark_points.txt", 'r').readlines()
    landmark_sort_list = [point_name.strip().split()[-1] for point_name in landmark_sort]
    # print(landmark_sort_list)
    for index in range(106):
        # print(index, landmark_sort_list[index])
        value = landmark_dict[landmark_sort_list[index]]
        landmarks.append([value['x'], value['y']])
    landmarks_array = np.array([[x, y] for [x, y] in landmarks])
    # landmarks_array = landmarks_array.reshape([212])
    return landmarks_array


def get_image_from_json(json_file_path, image_root):
    json_file = open(json_file_path, 'r', encoding='utf-8')
    annotation = json.load(json_file)
    json_file.close()
    image_path = os.path.join(image_root, annotation['image']['information']['path'])
    faces_dict = annotation["faces"]
    boxes_dict = faces_dict[0].get("face_rectangle", None)
    landmark_dict = faces_dict[0].get("landmark", None)
    width = annotation['image']['information'].get('width', 320)
    height = annotation['image']['information'].get('height', 288)
    boxes_dict = enlarge_box(boxes_dict, width, height, scale=1.5)
    landmark_dict = box_landmark_process(boxes_dict, landmark_dict)
    landmark_dict = align_landmarks(landmark_dict)
    image = cv2.imread(image_path, 1)
    # image = image_process(image, boxes_dict)
    # draw_landmarks(image, landmark_dict)
    # cv2.imshow("image", image)
    # cv2.waitKey()
    return image, boxes_dict, landmark_dict


class LandmarkDetector(object):
    def __init__(self, checkpoint_file_path=None, model_name="ghost_pfld", device=None):
        self.checkpoint_path = checkpoint_file_path
        self.model_name = model_name
        self.device = device
        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = init_model(name=self.model_name).to(self.device)
        if self.checkpoint_path is None:
            self.checkpoint_path = os.path.join(os.path.split(os.path.realpath(__file__))[0],
                                                "models/ghost_pfld_374_1.0200.pth")
        self.load_model()
        self.model.eval()

    def load_model(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['net'].state_dict())

    def image_process(self, image, box_dict=None):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.247, 0.243, 0.261]
        image = image_crop(image, box_dict)
        image = cv2.resize(image, dsize=(112, 112))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_array = (np.array(image, dtype=np.float32) / 255 - mean) / std
        image_tensor = torch.from_numpy(image_array.transpose([2, 0, 1])).float()
        image_tensor = torch.unsqueeze(image_tensor, dim=0)
        image_tensor = image_tensor.to(self.device)
        return image_tensor

    def detect(self, input_image):
        image = self.image_process(input_image)
        angle, landmark = self.model(image)
        landmark = landmark.data.cpu().numpy()[0]
        # landmark = landmark.reshape([106, 2])
        return landmark


