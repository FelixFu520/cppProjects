import os
from datetime import datetime as dt
from glob import glob
import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart

import calibrator
import slide_images

os.system('chcp 65001')
np.random.seed(31193)


def build_trt(onnxFile: str, trtFile: str, onnxInputShape: list, bUseFP16Mode: bool = None, bUseINT8Mode: bool = None,
              calibrationDataPath: str = None, nCalibration: int = None):
    """
    构建engine
    :param onnxFile:
    :param trtFile:
    :param onnxInputShape:
    :param bUseFP16Mode:
    :param bUseINT8Mode:
    :param calibrationDataPath:
    :param nCalibration:
    :return:
    """
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)

    if os.path.exists(trtFile):
        with open(trtFile, 'rb') as f:
            engineString = f.read()
        runtime = trt.Runtime(logger)
        return runtime.deserialize_cuda_engine(engineString)
    else:
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        if bUseFP16Mode:
            config.set_flag(trt.BuilderFlag.FP16)
        if bUseINT8Mode:
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, nCalibration, onnxInputShape, trtFile)

        parser = trt.OnnxParser(network, logger)
        if not os.path.exists(onnxFile):
            print("Failed finding ONNX file!")
            exit()
        print("Succeeded finding ONNX file!")
        with open(onnxFile, "rb") as model:
            if not parser.parse(model.read()):
                print("Failed parsing .onnx file!")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit()
            print("Succeeded parsing .onnx file!")

        inputTensor = network.get_input(0)
        profile.set_shape(inputTensor.name, onnxInputShape[0], onnxInputShape[1], onnxInputShape[2])
        config.add_optimization_profile(profile)

        engineString = builder.build_serialized_network(network, config)
        if engineString is None:
            print("Failed building engine!")
            exit()
        print("Succeeded building engine!")

        with open(trtFile, "wb") as f:
            f.write(engineString)
        return trt.Runtime(logger).deserialize_cuda_engine(engineString)


def infer(trt_engine: trt.ICudaEngine, batchImage: np.array, num_image: int):
    # get io info
    nIO = trt_engine.num_io_tensors
    lTensorName = [trt_engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [trt_engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    # get context
    context = trt_engine.create_execution_context()
    context.set_input_shape(lTensorName[0], onnx_input_shape[1])
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), trt_engine.get_tensor_dtype(lTensorName[i]),
              trt_engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    # malloc host
    bufferH = list()
    bufferH.append(np.ascontiguousarray(batchImage))
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]),
                                dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    # malloc device
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    # copy to device
    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    # infer
    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))
    context.execute_async_v3(0)

    # copy to host
    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    # free device memory
    for b in bufferD:
        cudart.cudaFree(b)

    # only two output
    confs = bufferH[1]
    labels = bufferH[2]
    confs_result_ = list()
    labels_result_ = list()
    for i in range(num_image):
        label_ = labels[i].transpose((1, 2, 0))
        label_ = np.where(label_ != 0, 255, 0)
        labels_result_.append(label_.squeeze())

        conf_ = confs[i].transpose((1, 2, 0))
        conf_ *= 255
        conf_ = conf_.astype(np.uint8)
        conf_ = np.where(label_ == 255, conf_, 0)
        confs_result_.append(conf_.squeeze())

    return confs_result_, labels_result_


if __name__ == '__main__':
    # 1. build trt
    onnx_file = "./test.onnx"
    trt_file = "./test.trt"
    onnx_input_shape = [(64, 3, 256, 256), (64, 3, 256, 256), (64, 3, 256, 256)]
    engine = build_trt(onnx_file, trt_file, onnx_input_shape, bUseFP16Mode=True)

    # 2. prepare image
    image_path = "2.bmp"
    # get coords
    img = cv2.imread(image_path, -1)
    crop_size = onnx_input_shape[1][2:]
    crop_step = 1
    windows1 = slide_images.sliding_window1(img, crop_size, crop_step)  # 获得coords
    # crop image
    num_images = len(windows1)
    batch_images = []
    img_tmp = None
    for i, img_ in enumerate(slide_images.crop_image(img, windows1)):
        batch_images.append(img_)
        img_tmp = img_
        cv2.imwrite(f"crops/{i}_{windows1[i][0]}_{windows1[i][1]}_{windows1[i][2]}_{windows1[i][3]}_img.jpg", img_)
    while True:
        if len(batch_images) < onnx_input_shape[1][0]:
            batch_images.append(img_tmp)
        else:
            break
    batch_image = np.ascontiguousarray(np.asarray(batch_images).transpose((0, 3, 1, 2))).astype(np.float32)

    # 3. infer
    confs_result, labels_result = infer(engine, batch_image, num_images)
    for i, (conf, label) in enumerate(zip(confs_result, labels_result)):
        cv2.imwrite(f"crops/{i}_{windows1[i][0]}_{windows1[i][1]}_{windows1[i][2]}_{windows1[i][3]}_r_label.jpg", label)
        cv2.imwrite(f"crops/{i}_{windows1[i][0]}_{windows1[i][1]}_{windows1[i][2]}_{windows1[i][3]}_r_conf.jpg", conf)

    # charlet
    charlet_label = np.zeros(img.shape[:2], np.uint8)
    charlet_conf = np.empty(img.shape[:2], np.uint8)
    for coord, conf, label in zip(windows1, confs_result, labels_result):
        x1, y1, x2, y2 = coord[0], coord[1], coord[2], coord[3]

        view_of_charlet_label = charlet_label[y1:y2, x1:x2]
        label = np.where(label > view_of_charlet_label, label, view_of_charlet_label)
        charlet_label[y1:y2, x1:x2] = label

        view_of_charlet_conf = charlet_conf[y1:y2, x1:x2]
        conf = np.where(conf > view_of_charlet_conf, conf, view_of_charlet_conf)
        charlet_conf[y1:y2, x1:x2] = conf
    cv2.imwrite(f"2_label.png", charlet_label)
    cv2.imwrite(f"2_conf.png", charlet_conf)

    print("Succeeded running model in TensorRT!")
