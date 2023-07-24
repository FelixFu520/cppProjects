#include "argmax.cuh"

/* ---------------- 向量相加(测试Demo) ----------------*/
// 核函数
__global__ void addKernel(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}
// 向量相加
int myVectorAdd(int* a, int* b, int* c, int size)
{
	int result = -1;
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;

	// 选择用于运行的GPU  
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		result = 1;
		goto Error;
	}

	// 在GPU中为变量dev_a、dev_b、dev_c分配内存空间.  
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		result = 2;
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		result = 3;
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		result = 4;
		goto Error;
	}

	// 从主机内存复制数据到GPU内存中.  
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		result = 5;
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		result = 6;
		goto Error;
	}

	// 启动GPU内核函数  
	addKernel << <1, size >> > (dev_c, dev_a, dev_b);

	// 采用cudaDeviceSynchronize等待GPU内核函数执行完成并且返回遇到的任何错误信息  
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		result = 7;
		goto Error;
	}

	// 从GPU内存中复制数据到主机内存中  
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		result = 8;
		goto Error;
	}

	result = 0;

	// 重置CUDA设备，在退出之前必须调用cudaDeviceReset  
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		return 9;
	}
Error:
	//释放设备中变量所占内存  
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return result;
}

/* ----------------  求CHW在C维度上的最大值及其位置 ----------------*/
// 核函数
__global__ void compareMaxValue(float* inputImages, unsigned char* outputMaxValues, unsigned char* outputMaxIndices, int width, int height, int numImages)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < width * height)
	{
		float maxValue = -FLT_MAX;
		unsigned char maxIndex = 0;

		for (int i = 0; i < numImages; i++)
		{
			float value = inputImages[i * width * height + index];
			if (value > maxValue)
			{
				maxValue = value;
				maxIndex = i;
			}
		}
		
		outputMaxIndices[index] = maxIndex;
		if (maxIndex == 0) {
			outputMaxValues[index] = 0;
		}
		else {
			outputMaxValues[index] = maxValue*255;
		}
	}
}
__global__ void compareMaxValue(float* inputImages, float* outputMaxValues, float* outputMaxIndices, int width, int height, int numImages)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < width * height)
	{
		float maxValue = -FLT_MAX;
		unsigned char maxIndex = 0;

		for (int i = 0; i < numImages; i++)
		{
			float value = inputImages[i * width * height + index];
			if (value > maxValue)
			{
				maxValue = value;
				maxIndex = i;
			}
		}

		outputMaxIndices[index] = maxIndex;
		if (maxIndex == 0) {
			outputMaxValues[index] = 0;
		}
		else {
			outputMaxValues[index] = maxValue * 255;
		}
	}
}
__global__ void compareMaxValue(float* inputImages, unsigned char* outputMaxValues, unsigned char* outputMaxIndices, int width, int height, int channels, int numImages)
{
	//int index = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;  // 每个线程处理一个像素点
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < width * height * numImages)
	{
		float maxValue = -FLT_MAX;
		unsigned char maxIndex = 0;

		for (int i = 0; i < channels; i++)
		{
			float value = inputImages[i * width * height * numImages + index];
			if (value > maxValue)
			{
				maxValue = value;
				maxIndex = i;
			}
		}

		outputMaxIndices[index] = maxIndex;
		if (maxIndex == 0) {
			outputMaxValues[index] = 0;
		}
		else {
			outputMaxValues[index] = (unsigned char)(maxValue * 255);
		}
	}
}
// 求最大值及其位置
int argmaxChannels(float* inputImages, unsigned char* outputMaxValues, unsigned char* outputMaxIndices, int width, int height, int numImages) {
	// Allocate GPU memory for input and output
	float* d_inputImages;
	unsigned char* d_outputMaxValues;
	unsigned char* d_outputMaxIndices;
	cudaMalloc((void**)&d_inputImages, numImages * width * height * sizeof(float));
	cudaMalloc((void**)&d_outputMaxValues, width * height * sizeof(unsigned char));
	cudaMalloc((void**)&d_outputMaxIndices, width * height * sizeof(unsigned char));

	// Copy input data from host to device
	cudaMemcpy(d_inputImages, inputImages, numImages * width * height * sizeof(float), cudaMemcpyHostToDevice);

	// Invoke the CUDA kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
	compareMaxValue << <blocksPerGrid, threadsPerBlock >> > (d_inputImages, d_outputMaxValues, d_outputMaxIndices, width, height, numImages);

	// Wait for kernel to finish
	cudaDeviceSynchronize();

	// Copy output data from device to host
	cudaMemcpy(outputMaxValues, d_outputMaxValues, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaMemcpy(outputMaxIndices, d_outputMaxIndices, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	// Free GPU memory
	cudaFree(d_inputImages);
	cudaFree(d_outputMaxValues);
	cudaFree(d_outputMaxIndices);

	return 0;
}
int argmaxChannels(const std::vector<std::vector<cv::Mat>>& inputImages, std::vector<cv::Mat>& outputMaxValues, std::vector<cv::Mat>& outputMaxIndices, int width, int height, int channels, int numImages) {
	// Allocate GPU memory for input and output
	float* d_inputImages;
	unsigned char* d_outputMaxValues;
	unsigned char* d_outputMaxIndices;
	cudaMalloc((void**)&d_inputImages, numImages * channels * width * height * sizeof(float));
	cudaMalloc((void**)&d_outputMaxValues, numImages * width * height * sizeof(unsigned char));
	cudaMalloc((void**)&d_outputMaxIndices, numImages * width * height * sizeof(unsigned char));

	// Copy input data from host to device
	for (int c = 0; c < channels; c++) {
		for (int n = 0; n < numImages; n++) {
			cudaMemcpy(d_inputImages + c*n*height*width + n*height*width, inputImages[n][c].data, width * height * sizeof(float), cudaMemcpyHostToDevice);
		}
	}


	// Invoke the CUDA kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
	compareMaxValue << <blocksPerGrid, threadsPerBlock >> > (d_inputImages, d_outputMaxValues, d_outputMaxIndices, width, height, channels, numImages);

	// Wait for kernel to finish
	cudaDeviceSynchronize();

	// Copy output data from device to host
	for (int n = 0; n < numImages; n++) {
		cudaMemcpy(outputMaxValues[n].data, d_outputMaxValues + n * width * height, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		cudaMemcpy(outputMaxIndices[n].data, d_outputMaxIndices + n * width * height, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	}


	// Free GPU memory
	cudaFree(d_inputImages);
	cudaFree(d_outputMaxValues);
	cudaFree(d_outputMaxIndices);

	return 0;
}
int argmaxChannels(const std::vector<cv::cuda::GpuMat>& inputImages, cv::cuda::GpuMat& outputMaxValues, cv::cuda::GpuMat& outputMaxIndices, int width, int height, int numImages) {

	float* d_inputImages;
	cudaMalloc((void**)&d_inputImages, numImages * width * height * sizeof(float));

	for (int n = 0; n < inputImages.size(); n++) {
		cudaMemcpy(
			d_inputImages + n * height * width,
			inputImages[n].data,
			height * width * sizeof(float), 
			cudaMemcpyDeviceToDevice
		);
	}

	// Invoke the CUDA kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
	compareMaxValue << <blocksPerGrid, threadsPerBlock >> > (d_inputImages, (float*)outputMaxValues.data, (float*)outputMaxIndices.data, width, height, numImages);

	// Wait for kernel to finish
	cudaDeviceSynchronize();

	// Free GPU memory
	cudaFree(d_inputImages);

	return 0;
}