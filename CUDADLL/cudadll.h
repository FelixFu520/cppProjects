#pragma once
#include <opencv2/opencv.hpp>

#ifdef CUDADLL_EXPORTS
#define CUDADLL_API __declspec(dllexport)
#else
#define CUDADLL_API __declspec(dllimport)
#endif

/*!
 * @brief 两个向量相加，这个是个测试接口
 *			c = a + b
 * @param a: 向量a的地址
 * @param b: 向量b的地址
 * @param c: 向量c的地址
 * @param size: 向量长度，a,b,c长度相同
 * @return 错误编码
*/
CUDADLL_API int vectorAdd(int* a, int* b, int* c, int size);

/*!
 * @brief 求CHW三维向量在C维度上的最大值及其位置
 *
 * @param inputImages: 三维向量的地址，尺寸是CHW
 * @param outputMaxValues: 最大值存放地址，尺寸是HW
 * @param outputMaxIndices: 最大值所在下标地址，尺寸是HW
 * @return 错误编码
*/
CUDADLL_API int argmax(const std::vector<cv::Mat>& inputImages, cv::Mat& outputMaxValues, cv::Mat& outputMaxIndices);

/*!
 * @brief 求CHW三维向量在C维度上的最大值及其位置
 *
 * @param inputImages: 三维向量的地址，尺寸是CHW
 * @param outputMaxValues: 最大值存放地址，尺寸是HW
 * @param outputMaxIndices: 最大值所在下标地址，尺寸是HW
 * @return 错误编码
*/
CUDADLL_API int argmax(const std::vector<std::vector<cv::Mat>>& inputImages, std::vector<cv::Mat>& outputMaxValues, std::vector<cv::Mat>& outputMaxIndices);

/*!
 * @brief 求CHW三维向量在C维度上的最大值及其位置
 *
 * @param inputImages: 三维向量的地址，尺寸是CHW
 * @param outputMaxValues: 最大值存放地址，尺寸是HW
 * @param outputMaxIndices: 最大值所在下标地址，尺寸是HW
 * @return 错误编码
*/
CUDADLL_API int argmax(const std::vector<cv::cuda::GpuMat>& inputImages, cv::cuda::GpuMat& outputMaxValues, cv::cuda::GpuMat& outputMaxIndices);