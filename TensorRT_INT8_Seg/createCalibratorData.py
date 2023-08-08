import os

import numpy as np
import cv2
from glob import glob
import mmengine
import mmcv


class createCalibratorData(object):
    def __init__(self, dataset_json: str, npy_path: str = "data.npz", color_type: str = 'color', channel_order: str = 'bgr', percent: float = 0.6):
        super().__init__()

        self.dataset_json = dataset_json
        self.npy_path = npy_path
        self.percent = percent
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, *args, **kwargs):
        self._load_dataset_json(self.dataset_json)
        self._save_npy()

    def _load_dataset_json(self, json_path: str) -> None:
        dataset_json_dict = mmengine.load(json_path)

        url_prefix = dataset_json_dict['meta_info']['url_prefix']

        dataset = dataset_json_dict['dataset']
        dataset = dataset[:int(len(dataset) * self.percent)]
        calibratorData = []
        for index, one_item in enumerate(dataset):
            print(f"parser {index}/{len(dataset)} image ......")
            # 读取图片
            images_path = one_item['data']
            images = []
            for img_path in images_path:
                img_path = os.path.join(url_prefix, img_path['img_path'])
                img_path = img_path.replace('\\', '/')
                img_bytes = mmengine.get(img_path)
                img = mmcv.imfrombytes(img_bytes, flag=self.color_type, channel_order=self.channel_order)
                # img = img.astype(np.float32)
                images.append(img)
            image = np.asarray(images)
            image = image.transpose((1, 2, 0))

            # 生成一个样本
            one_result = dict()
            one_result['data'] = image
            one_result['height'] = one_item['height']
            one_result['width'] = one_item['width']
            one_result['channels'] = len(images)
            calibratorData.append(one_result)

        self.calibratorData = calibratorData

    def _save_npy(self) -> None:
        dataDictionary = {}
        for i, data in enumerate(self.calibratorData):
            dataDictionary[str(i)] = self.calibratorData[i]
        np.savez(self.npy_path, **dataDictionary)


if __name__ == '__main__':
    datasetJson = "D:\\Downloads\\train\\datasets.json"
    createCalibratorData(datasetJson, color_type='grayscale', percent=0.1)()

