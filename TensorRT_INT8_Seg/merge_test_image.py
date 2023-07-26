import os
import cv2
import numpy as np


def merge_image(src: str, dst: str):
    all_images = os.listdir(src)

    images = list()
    for image_path in all_images:
        img = cv2.imread(os.path.join(src, image_path), -1)
        images.append(img)

    image = np.asarray(images)
    image = image.transpose((1, 2, 0))
    cv2.imwrite(dst, image)


if __name__ == '__main__':
    src = "D:\\Work\\cppProjects\\TensorRT_INT8_Seg\\test"
    dst = "D:\\Work\\cppProjects\\TensorRT_INT8_Seg\\4.bmp"
    merge_image(src, dst)
