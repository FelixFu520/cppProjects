#pragma once
#include <opencv2/opencv.hpp>

#ifdef CUDADLL_EXPORTS
#define CUDADLL_API __declspec(dllexport)
#else
#define CUDADLL_API __declspec(dllimport)
#endif

/*!
 * @brief ����������ӣ�����Ǹ����Խӿ�
 *			c = a + b
 * @param a: ����a�ĵ�ַ
 * @param b: ����b�ĵ�ַ
 * @param c: ����c�ĵ�ַ
 * @param size: �������ȣ�a,b,c������ͬ
 * @return �������
*/
CUDADLL_API int vectorAdd(int* a, int* b, int* c, int size);

/*!
 * @brief ��CHW��ά������Cά���ϵ����ֵ����λ��
 *
 * @param inputImages: ��ά�����ĵ�ַ���ߴ���CHW
 * @param outputMaxValues: ���ֵ��ŵ�ַ���ߴ���HW
 * @param outputMaxIndices: ���ֵ�����±��ַ���ߴ���HW
 * @return �������
*/
CUDADLL_API int argmax(const std::vector<cv::Mat>& inputImages, cv::Mat& outputMaxValues, cv::Mat& outputMaxIndices);

/*!
 * @brief ��CHW��ά������Cά���ϵ����ֵ����λ��
 *
 * @param inputImages: ��ά�����ĵ�ַ���ߴ���CHW
 * @param outputMaxValues: ���ֵ��ŵ�ַ���ߴ���HW
 * @param outputMaxIndices: ���ֵ�����±��ַ���ߴ���HW
 * @return �������
*/
CUDADLL_API int argmax(const std::vector<std::vector<cv::Mat>>& inputImages, std::vector<cv::Mat>& outputMaxValues, std::vector<cv::Mat>& outputMaxIndices);

/*!
 * @brief ��CHW��ά������Cά���ϵ����ֵ����λ��
 *
 * @param inputImages: ��ά�����ĵ�ַ���ߴ���CHW
 * @param outputMaxValues: ���ֵ��ŵ�ַ���ߴ���HW
 * @param outputMaxIndices: ���ֵ�����±��ַ���ߴ���HW
 * @return �������
*/
CUDADLL_API int argmax(const std::vector<cv::cuda::GpuMat>& inputImages, cv::cuda::GpuMat& outputMaxValues, cv::cuda::GpuMat& outputMaxIndices);