import os
import cv2
import itertools
import numpy as np
import string
import random
from glob import glob
from cuda import cudart
import tensorrt as trt

import slide_images


class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibrationDataPath, nCalibration, inputShape, cacheFile, image_suffix: str = ".bmp",
                 show_log: bool = False, save_image: bool = False):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.imageList = glob(calibrationDataPath + f"*{image_suffix}")
        self.nCalibration = nCalibration
        self.shape = inputShape  # (N,C,H,W)
        self.buffeSize = trt.volume(inputShape) * trt.float32.itemsize
        self.cacheFile = cacheFile
        _, self.dIn = cudart.cudaMalloc(self.buffeSize)
        self.oneBatch = self.batchGenerator()
        self.cropImageGenerator = self.cropImageGenerator(show_log=show_log, save_crop=save_image)

        # 测试self.cropImageGenerator是否正确
        # for ii in range(1000):
        #     batch_image = next(self.cropImageGenerator)
        #     print(batch_image.shape)

        print(int(self.dIn))

    def __del__(self):
        cudart.cudaFree(self.dIn)

    @staticmethod
    def generate_random_code(length):
        # 生成包含数字和字母的字符串
        characters = string.ascii_letters + string.digits
        # 从字符集合中随机取出指定长度的字符并拼接起来
        random_code = ''.join(random.choice(characters) for i in range(length))
        return random_code

    def cropImageGenerator(self, show_log=False, save_crop=False):
        batch_image = list()
        for index, image_path in enumerate(itertools.cycle(self.imageList)):
            img = cv2.imread(image_path, -1)
            if img.shape[0] < self.shape[1] or img.shape[1] < self.shape[2]:
                print(f"Warning, onnx shape:{self.shape}, and crop image shape:{img.shape}, skip ...")
                continue
            windows1 = slide_images.sliding_window1(img, self.shape[2:], 1)
            if show_log:
                print(f"using {index}: {image_path} get {len(windows1)} crop image")

            save_path = f"crops/calibration/"
            os.makedirs(save_path, exist_ok=True)
            for j, img_ in enumerate(slide_images.crop_image(img, windows1)):
                batch_image.append(np.asarray(img_).transpose((2, 0, 1)))
                if save_crop:
                    cv2.imwrite(f"{save_path}{os.path.basename(image_path)[:-4]}_crop_{j}"
                                f"_{self.generate_random_code(10)}.jpg", img_)
            if len(batch_image) > self.shape[0]:
                yield np.ascontiguousarray(np.asarray(batch_image[:self.shape[0]]).astype(np.float32))
                batch_image = list()

    def batchGenerator(self):
        for i in range(self.nCalibration):
            print("> calibration %d" % i)
            yield next(self.cropImageGenerator)

    def get_batch_size(self):  # necessary API
        print("---- get batch size ----")
        return self.shape[0]

    def get_batch(self, nameList=None, inputNodeName=None):  # necessary API
        print("---- get batch ----")
        try:
            data = next(self.oneBatch)
            if data.shape[0] != self.shape[0]:
                print("get batch error")
                print(data)
                exit(1)
            cudart.cudaMemcpy(self.dIn, data.ctypes.data, self.buffeSize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            return [int(self.dIn)]
        except StopIteration:
            return None

    def read_calibration_cache(self):  # necessary API
        print("---- read calibration cache ----")
        if os.path.exists(self.cacheFile):
            print("Succeed finding cahce file: %s" % (self.cacheFile))
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return

    def write_calibration_cache(self, cache):  # necessary API
        print("---- write calibration cache ----")
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache!")
        return
