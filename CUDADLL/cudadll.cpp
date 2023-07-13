#include "cudadll.h"
#include "argmax.cuh"

CUDADLL_API int vectorAdd(int* a, int* b, int* c, int size)
{
    vectorAdd(a, b, c, size);
    return 0;
}

CUDADLL_API int argmax(const std::vector<cv::Mat>& inputImages, cv::Mat& outputMaxValues, cv::Mat& outputMaxIndices) {
	if (inputImages.empty()) { return 1; }
	int width = inputImages[0].cols;
	int height = inputImages[0].rows;
	int numImages = inputImages.size();
	outputMaxValues = cv::Mat(inputImages[0].size(), CV_8U, cv::Scalar::all(0));
	outputMaxIndices = cv::Mat(inputImages[0].size(), CV_8U, cv::Scalar::all(0));
	cv::Mat concatedMat;
	cv::vconcat(inputImages, concatedMat);
	// 调用API
	argmaxChannels((float*)concatedMat.data, (unsigned char*)outputMaxValues.data, (unsigned char*)outputMaxIndices.data, width, height, numImages);
    return 0;
}

CUDADLL_API int argmax(const std::vector<std::vector<cv::Mat>>& inputImages, std::vector<cv::Mat>& outputMaxValues, std::vector<cv::Mat>& outputMaxIndices) {
	if (inputImages.empty()) { return 1; }
	int numImages = inputImages.size();
	int width = inputImages[0][0].cols;
	int height = inputImages[0][0].rows;
	int channels = inputImages[0].size();

	outputMaxValues.resize(numImages);
	outputMaxIndices.resize(numImages);
	for (int i = 0; i < numImages; i++) {
		outputMaxValues[i] = cv::Mat(inputImages[0][0].size(), CV_8U, cv::Scalar::all(0));
		outputMaxIndices[i] = cv::Mat(inputImages[0][0].size(), CV_8U, cv::Scalar::all(0));
	}

	// 调用API
	argmaxChannels(inputImages, outputMaxValues, outputMaxIndices, width, height, channels, numImages);
	return 0;
}
CUDADLL_API int argmax(const std::vector<cv::cuda::GpuMat>& inputImages, cv::cuda::GpuMat& outputMaxValues, cv::cuda::GpuMat& outputMaxIndices) {
	if (inputImages.empty()) { return 1; }
	int width = inputImages[0].cols;
	int height = inputImages[0].rows;
	int numImages = inputImages.size();
	outputMaxValues = cv::cuda::GpuMat(inputImages[0].size(), CV_32F, cv::Scalar::all(0));
	outputMaxIndices = cv::cuda::GpuMat(inputImages[0].size(), CV_32F, cv::Scalar::all(0));
	argmaxChannels(inputImages, outputMaxValues, outputMaxIndices, width, height, numImages);
	return 0;

}