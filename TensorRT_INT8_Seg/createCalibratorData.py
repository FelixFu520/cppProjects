import os

import numpy as np
import cv2
from glob import glob
import mmengine
import mmcv


class createCalibratorData(object):
    def __init__(self, dataset_json: str, dst_path: str = "./result", color_type: str = 'color', channel_order: str = 'bgr', percent: float = 0.6):
        super().__init__()

        self.dataset_json = dataset_json
        self.dst_path = dst_path
        self.percent = percent
        self.color_type = color_type
        self.channel_order = channel_order

        self.calibratorData = list()
        self.calibratorDataPath = list()

    def __call__(self, *args, **kwargs):
        self._load_dataset_json(self.dataset_json)
        self._save_image()

    def _load_dataset_json(self, json_path: str) -> None:
        dataset_json_dict = mmengine.load(json_path)

        url_prefix = dataset_json_dict['meta_info']['url_prefix']

        dataset = dataset_json_dict['dataset']
        dataset = dataset[:int(len(dataset) * self.percent)]
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
                images.append(img)
            image = np.asarray(images)
            image = image.transpose((1, 2, 0))

            self.calibratorData.append(image)
            self.calibratorDataPath.append(os.path.basename(images_path[0]['img_path']))

    def _save_image(self) -> None:
        if os.path.exists(self.dst_path):
            os.remove(self.dst_path)
        os.makedirs(self.dst_path, exist_ok=True)
        for img_path, img in zip(self.calibratorDataPath, self.calibratorData):
            dst_path = os.path.join(self.dst_path, img_path)
            cv2.imwrite(dst_path, img)


if __name__ == '__main__':
    datasetJson = "D:\\Downloads\\train\\datasets.json"
    createCalibratorData(datasetJson, color_type='grayscale', percent=0.1)()

