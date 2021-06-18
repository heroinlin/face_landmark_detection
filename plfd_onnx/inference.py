# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import onnxruntime


work_root = os.path.split(os.path.realpath(__file__))[0]


class ONNXInference(object):
    def __init__(self, model_path=None):
        """
        对ONNXInference进行初始化

        Parameters
        ----------
        onnx_file_path : str
            onnx模型的路径，推荐使用绝对路径
        """
        super().__init__()
        self.model_path = model_path
        if self.model_path is None:
            print("please set onnx model path!\n")
            exit(-1)
        self.session = onnxruntime.InferenceSession(self.model_path)

    def inference(self, inputs:list):
        """
        onnx的推理
        Parameters
        ----------
        x : list
            onnx模型输入

        Returns
        -------
        list
            onnx模型推理结果
        """
        input_num = len(self.session.get_inputs())
        assert input_num == len(inputs)
        output_num = len(self.session.get_outputs())
        output_names = []
        input_feed = {}
        for idx in range(input_num):
            input_name = self.session.get_inputs()[idx].name
            input_feed.update({input_name: inputs[idx].astype(np.float32)})
        for idx in range(output_num):
            output_names.append(self.session.get_outputs()[idx].name)
        # print(input_feed.keys(), output_names)
        outputs = self.session.run(output_names=output_names,
                                   input_feed=input_feed)
        return outputs


class PLFDONNX(ONNXInference):
    def __init__(self, model_path=None):
        """对Detector进行初始化

        Parameters
        ----------
        onnx_file_path : str
            onnx模型的路径，推荐使用绝对路径
        """
        if model_path is None:
            model_path = os.path.join(work_root,
                                          'onnx_models',
                                          "ghost_pfld_374_1.0200.onnx")
        super(PLFDONNX, self).__init__(model_path)
        self.config = {
            'width': 112,
            'height': 112,
            'color_format': 'RGB',
            'mean': [0.4914, 0.4822, 0.4465],
            'stddev': [0.247, 0.243, 0.261],
            'divisor': 255.0,
            'detect_threshold': 0.4,
            'nms_threshold': 0.3
        }

    def set_config(self, key, value):
        self.config[key] = value

    def destroy(self):
        pass

    def _preprocess(self, image: np.ndarray):
        """对图像进行预处理

        Parameters
        ----------
        image : np.ndarray
            输入的原始图像，BGR格式，通常使用cv2.imread读取得到

        Returns
        -------
        np.ndarray
            原始图像经过预处理后得到的数组
        """
        if self.config['color_format'] == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.config['width'] > 0 and self.config['height'] > 0:
            image = cv2.resize(image, (self.config['width'], self.config['height']))
        input_image = (np.array(image, dtype=np.float32) / self.config['divisor'] - self.config['mean']) / self.config['stddev']
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, 0)
        return input_image

    def _postprocess(self, outputs: list):
        """
        对网络输出框进行后处理
        Parameters
        ----------
        boxes: list
            网络推理结果
        Returns
        -------
            np.ndarray
            返回值维度为(n, 5)，其中n表示目标数量，5表示(x1, y1, x2, y2, score)
        """
        angles, landmarks = outputs
        return angles, landmarks

    def detect(self, image: np.ndarray):
        """对图像做人脸关键点检测

        Parameters
        ----------
        image : np.ndarray
            输入图片，BGR格式，通常使用cv2.imread获取得到

        Returns
        -------
        np.ndarray
            返回值维度为(n, 5)，其中n表示目标数量，5表示(x1, y1, x2, y2, score)
        """
        input_image = self._preprocess(image)
        inputs = [input_image]
        outputs = self.inference(inputs)
        outputs = self._postprocess(outputs)
        return outputs
