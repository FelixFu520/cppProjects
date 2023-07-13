#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <opencv2/opencv.hpp>

// ≤‚ ‘Demo
__global__ void addKernel(int* c, const int* a, const int* b);
int vectorAdd(int* a, int* b, int* c, int size);

// argmax
__global__ void compareMaxValue(float* inputImages, unsigned char* outputMaxValues, unsigned char* outputMaxIndices, int width, int height, int numImages);
__global__ void compareMaxValue(float* inputImages, float* outputMaxValues, float* outputMaxIndices, int width, int height, int numImages);
__global__ void compareMaxValue(float* inputImages, unsigned char* outputMaxValues, unsigned char* outputMaxIndices, int width, int height, int channels, int numImages);

int argmaxChannels(float* inputImages, unsigned char* outputMaxValues, unsigned char* outputMaxIndices, int width, int height, int numImages);
int argmaxChannels(const std::vector<std::vector<cv::Mat>>& inputImages, std::vector<cv::Mat>& outputMaxValues, std::vector<cv::Mat>& outputMaxIndices, int width, int height, int numImages, int channels);
int argmaxChannels(const std::vector<cv::cuda::GpuMat>& inputImages, cv::cuda::GpuMat& outputMaxValues, cv::cuda::GpuMat& outputMaxIndices, int width, int height, int numImages);