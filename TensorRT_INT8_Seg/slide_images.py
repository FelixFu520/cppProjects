import os
import cv2
import copy
import numpy as np


def sliding_window1(image: np.array, window: tuple, step: float, back: bool = True):
    """
    使用window在image上以步长step滑动, 获得所有window大小的图片坐标
    滑动过程中会有以下几种情况:
    1. window宽高 大于 image宽高
    2. window高 大于 image高; window宽 小于 image宽
    3. window高 小于 image高; window宽 大于 image宽
    4. window高宽 小于 image高宽
    :param image: h,w[,c]
    :param window: h,w
    :param step:
    :return: seq
    """
    window_h = window[0]
    window_w = window[1]
    image_h = image.shape[0]
    image_w = image.shape[1]
    windows_result: list = []

    if window_h > image_h and window_w > image_w:
        return

    if window_h > image_h and window_w < image_w:
        for x in range(0, image_w - 1, int(window_w * step)):
            windows_result.append([x, 0, np.clip(x + window_w, 0, image_w), np.clip(0 + window_h, 0, image_h)])
        return np.asarray(windows_result)

    if window_h < image_h and window_w > image_w:
        for y in range(0, image_h - 1, int(window_h * step)):
            windows_result.append([0, y, np.clip(0 + window_w, 0, image_w), np.clip(y + window_h, 0, image_h)])
        return np.asarray(windows_result)

    if window_h < image_h and window_w < image_w:
        for y in range(0, image_h - 1, int(window_h * step)):
            for x in range(0, image_w - 1, int(window_w * step)):
                if back:
                    if x + window_w > image_w:  # 宽超过阈值
                        x_tmp = image_w - window_w
                    else:
                        x_tmp = x
                    if y + window_h > image_h:  # 高超过阈值
                        y_tmp = image_h - window_h
                    else:
                        y_tmp = y
                    windows_result.append([x_tmp, y_tmp, np.clip(x_tmp + window_w, 0, image_w), np.clip(y_tmp + window_h, 0, image_h)])

                else:
                    windows_result.append([x, y, np.clip(x + window_w, 0, image_w), np.clip(y + window_h, 0, image_h)])
        return np.asarray(windows_result)


def sliding_window2(image: np.array, window: tuple, step: float):
    """

    :param image:
    :param window:
    :param step:
    :return:
    """
    window_h = window[0]
    window_w = window[1]
    image_h = image.shape[0]
    image_w = image.shape[1]

    x, y = np.meshgrid(np.arange(0, image_w - 1, int(step * window_w)),
                       np.arange(0, image_h - 1, int(step * window_h)),
                       indexing='xy')

    windows = np.dstack((x, y, np.clip(x + window_w, 0, image_w), np.clip(y + window_h, 0, image_h)))
    return windows.reshape(-1, 4)


def crop_image(image: np.array, coords: np.array):
    """

    :param image:
    :param coords:
    :return:
    """
    for x1, y1, x2, y2 in coords:
        yield image[y1:y2, x1:x2]


def crop_image_pad(image: np.array, coords: np.array, height: int, width: int, mode: str = 'constant'):
    """

    :param image:
    :param coords:
    :return:
    """
    for x1, y1, x2, y2 in coords:
        yield cv2.copyMakeBorder(image[y1:y2, x1:x2], 0, height-(y2-y1), 0, width-(x2-x1), cv2.BORDER_CONSTANT, value=0)


if __name__ == '__main__':
    img = cv2.imread("test.bmp", -1)
    windows1 = sliding_window1(img, (256, 256), 1)
    windows2 = sliding_window2(img, (256, 256), 1)

    for i, img_ in enumerate(crop_image(img, windows1)):
        cv2.imwrite(f"D:/Downloads/back_{i}_{img_.shape[0]}_{img_.shape[1]}.jpg", img_)

    for i, img_ in enumerate(crop_image(img, windows2)):
        cv2.imwrite(f"D:/Downloads/crop_{i}_{img_.shape[0]}_{img_.shape[1]}.jpg", img_)

    for i, img_ in enumerate(crop_image_pad(img, windows1, 256, 256)):
        cv2.imwrite(f"D:/Downloads/pad_{i}_{img_.shape[0]}_{img_.shape[1]}.jpg", img_)

    print("END")
