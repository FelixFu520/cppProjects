#include <stdio.h>
#include <assert.h>
#include <memory>
#include <omp.h>
#include <cuda_runtime.h>					// cuda库
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>	
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <numeric>
#include <cuda_runtime.h>					// cuda库
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>				// opencv库
#include "NvInfer.h"						// TensorRT库
#include "NvInferRuntimeCommon.h"
#include "NvInferRuntime.h"
#include "NvOnnxParser.h"
#include "common.h"							// TensorRT samples中的函数
#include "buffers.h"
#include "logging.h"
#include <omp.h> 							// openMP并行需要
#include "NvInferRuntime.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"


using namespace nvinfer1;
using namespace std;

template <typename T>
using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

namespace sample
{
	Logger gLogger{ Logger::Severity::kINFO };
	LogStreamConsumer gLogVerbose{ LOG_VERBOSE(gLogger) };
	LogStreamConsumer gLogInfo{ LOG_INFO(gLogger) };
	LogStreamConsumer gLogWarning{ LOG_WARN(gLogger) };
	LogStreamConsumer gLogError{ LOG_ERROR(gLogger) };
	LogStreamConsumer gLogFatal{ LOG_FATAL(gLogger) };

	void setReportableSeverity(Logger::Severity severity)
	{
		gLogger.setReportableSeverity(severity);
		gLogVerbose.setReportableSeverity(severity);
		gLogInfo.setReportableSeverity(severity);
		gLogWarning.setReportableSeverity(severity);
		gLogError.setReportableSeverity(severity);
		gLogFatal.setReportableSeverity(severity);
	}
} // namespace sample

int main1() {
	//const int input_widht = 128, input_height = 128, input_channels = 1, input_batch = 1;
	//const int output_widht = 128, output_height = 128, output_channels = 3, output_batch = 1;
	const int input_widht = 256, input_height = 256, input_channels = 1, input_batch = 64;
	const int output_widht = 256, output_height = 256, output_channels = 5, output_batch = 64;
	int threadNum = 1;		// 模拟的线程个数
	int TIMES = 1;			// 每个线程执行多少次推理
	int ALL_TIMES = 10;

	// 读取图片
	vector<vector<cv::Mat>> images;
	//string IMAGES = "D:\\Work\\cppProjects\\Files\\seg_outputChannels_2\\1.bmp";
	string IMAGES = "D:\\Work\\cppProjects\\Files\\Files\\seg\\1.jpg";
	cv::Mat image = cv::imread(IMAGES, cv::IMREAD_UNCHANGED);
	cv::Mat resizeImg;
	cv::resize(image, resizeImg, cv::Size(input_height, input_widht));
	resizeImg.convertTo(resizeImg, CV_32F);
	for(int i=0;i< threadNum;i++){
		vector<cv::Mat> oneBatch;
		for (int i = 0; i < input_batch; i++) {
			oneBatch.push_back(resizeImg.clone());
		}
		images.push_back(oneBatch);
	}

	// 加载序列化的引擎文件
	//string ENGINE_FILE = "D:\\BaiduSyncdisk\\Work\\dlpsdk\\x64\\Files\\seg_outputChannels_2\\2.40.6-RT8-multiType-FP16.trt";
	string ENGINE_FILE = "D:\\Work\\dlpsdk\\x64\\Files\\seg\\jingyuan.trt";
	vector<IExecutionContext*> ctxs;
	for (int i = 0; i < threadNum; i++) {
		std::fstream file;
		file.open(ENGINE_FILE, std::ios::binary | std::ios::in);
		if (!file.is_open()) { return 1; }
		file.seekg(0, std::ios::end);
		int length = file.tellg();
		file.seekg(0, std::ios::beg);
		std::unique_ptr<char[]> data(new char[length]);
		file.read(data.get(), length);
		file.close();

		// 生成mEngine
		nvinfer1::ICudaEngine* engine;
		SampleUniquePtr<nvinfer1::IRuntime> runTime(nvinfer1::createInferRuntime(sample::gLogger));
		engine = runTime->deserializeCudaEngine(data.get(), length, nullptr);

		// 生成Context
		nvinfer1::IExecutionContext* context;
		context = engine->createExecutionContext();
		ctxs.push_back(context);
	}
	
	// 推理
	vector<cv::Mat> results(threadNum);
	for (int aa = 0; aa < ALL_TIMES; aa++) {
		auto timeStart = cv::getTickCount();										// 获取开始时间
		//创建多个流
		vector<cudaStream_t> streams(threadNum);
		for (int i = 0; i < threadNum; i++) cudaStreamCreate(&streams[i]);

		omp_set_num_threads(threadNum);
#pragma omp parallel for
		for (int k = 0; k < threadNum; k++) {
			auto timeOneStart = cv::getTickCount();										// 获取开始时间
			for (int times = 0; times < TIMES; times++) {

				// 分配显存和内存
				auto timeMalloc = cv::getTickCount();										// 获取开始时间
				void* bindings[2];
				cudaMalloc(&bindings[0], input_batch * input_channels * input_height * input_widht * sizeof(float));
				cudaMalloc(&bindings[1], output_batch * output_channels * output_height * output_widht * sizeof(float));
				auto timeMallocUsed = (cv::getTickCount() - timeMalloc) * 1000 / cv::getTickFrequency();
				//std::cout << "1 Time Malloc:" + to_string(timeMallocUsed) << endl;

				// Dims->Vector & Host->Device
				auto timeH2D = cv::getTickCount();										// 获取开始时间
				for (int n = 0; n < output_batch; n++) {
					cudaMemcpyAsync(
						(float*)bindings[0] + n * input_height * input_widht,
						images[k][n].data,
						input_height * input_widht * sizeof(float),
						cudaMemcpyHostToDevice,
						streams[k]);
				}
				auto timeH2DUsed = (cv::getTickCount() - timeH2D) * 1000 / cv::getTickFrequency();
				//std::cout << "2 Time H2D:" + to_string(timeH2DUsed) << endl;

				// 推理
				auto timeExecute = cv::getTickCount();										// 获取开始时间
				ctxs[k]->enqueueV2(bindings, streams[k], nullptr);
				auto timeExecuteUsed = (cv::getTickCount() - timeExecute) * 1000 / cv::getTickFrequency();
				//std::cout << std::string("3 Context Address: " + std::to_string(reinterpret_cast<std::uintptr_t>(ctxs[k]))) << endl;
				//std::cout << "4 Time ExecuteUsed:" + to_string(timeExecuteUsed) << endl;

				// Device->Host
				auto timeD2H = cv::getTickCount();										// 获取开始时间
				void* output = malloc(output_batch * output_channels * output_height * output_widht * sizeof(float));
				cudaMemcpyAsync(
					(float*)output, (float*)bindings[1],
					output_batch * output_channels * output_height * output_widht * sizeof(float),
					cudaMemcpyDeviceToHost,
					streams[k]);
				auto timeD2HUsed = (cv::getTickCount() - timeD2H) * 1000 / cv::getTickFrequency();
				//std::cout << "5 Time D2HUsed:" + to_string(timeD2HUsed) << endl;

				// Vector->Dims
				auto timeVectorDims = cv::getTickCount();										// 获取开始时间
				for (int n = 0; n < output_batch; n++) {
					std::vector<cv::Mat> n_channels;  // 获取每类的概率图（包括多标签和单标签）
					for (int j = 0; j < output_channels; j++) {
						n_channels.push_back(
							cv::Mat(output_height,
								output_widht,
								CV_32F,
								(float*)output + j * output_height * output_widht + n * output_channels * output_height * output_widht)
						);
					}
					cv::Mat res;	// 存储合并后的图片
					cv::merge(n_channels, res);
					results[k] = res;

				}
				auto timeVectorDimsUsed = (cv::getTickCount() - timeVectorDims) * 1000 / cv::getTickFrequency();
				//std::cout << "6 Time VectorDimsUsed:" + to_string(timeVectorDimsUsed) << endl;

				cudaFree(bindings[0]);
				cudaFree(bindings[1]);
				std::free(output);
			}
			
			////同步流
			//for (int ii = 0; ii < threadNum; ii++) {
			//	cudaStreamSynchronize(streams[ii]);
			//}

			auto timeOneEnd = (cv::getTickCount() - timeOneStart) * 1000 / cv::getTickFrequency();
			//std::cout << "------------------------------------Thread Time:" + to_string(timeOneEnd) << endl;
		}
		for (int i = 0; i < threadNum; i++) cudaStreamDestroy(streams[i]);
		auto timeEnd = (cv::getTickCount() - timeStart) * 1000 / cv::getTickFrequency();
		std::cout << "Total Time:" + to_string(timeEnd) << endl;
	}
	std::cout << "END" << endl;

}



int main() {
	const int input_widht = 256, input_height = 256, input_channels = 3, input_batch = 64;
	const int output_widht1 = 256, output_height1 = 256, output_channels1 = 1, output_batch1 = 64;
	const int output_widht2 = 256, output_height2 = 256, output_channels2 = 1, output_batch2 = 64;
	int threadNum = 2;		// 模拟的线程个数
	int TIMES = 1;			// 每个线程执行多少次推理
	int ALL_TIMES = 10;

	// 读取图片
	vector<vector<cv::Mat>> images;
	string IMAGES1 = "D:\\Work\\dlpsdk\\x64\\Files\\seg_multi_output\\test1.bmp";
	string IMAGES2 = "D:\\Work\\dlpsdk\\x64\\Files\\seg_multi_output\\test2.bmp";
	string IMAGES3 = "D:\\BaiduSyncdisk\\Work\\dlpsdk\\x64\\Files\\seg_multi_output\\test3.bmp";
	cv::Mat c1 = cv::imread(IMAGES1, cv::IMREAD_UNCHANGED);
	cv::Mat c2 = cv::imread(IMAGES2, cv::IMREAD_UNCHANGED);
	cv::Mat c3 = cv::imread(IMAGES3, cv::IMREAD_UNCHANGED);
	vector<cv::Mat> channels;
	channels.push_back(c1);
	channels.push_back(c2);
	channels.push_back(c3);
	cv::Mat image;
	cv::merge(channels, image);
	cv::Mat resizeImg;
	cv::resize(image, resizeImg, cv::Size(input_height, input_widht));
	resizeImg.convertTo(resizeImg, CV_32F);
	for (int i = 0; i < threadNum; i++) {
		vector<cv::Mat> oneBatch;
		for (int i = 0; i < input_batch; i++) {
			oneBatch.push_back(resizeImg.clone());
		}
		images.push_back(oneBatch);
	}

	// 加载序列化的引擎文件
	string ENGINE_FILE = "D:\\BaiduSyncdisk\\Work\\dlpsdk\\x64\\Files\\seg_multi_output\\test.trt";
	vector<IExecutionContext*> ctxs;
	for (int i = 0; i < threadNum; i++) {
		std::fstream file;
		file.open(ENGINE_FILE, std::ios::binary | std::ios::in);
		if (!file.is_open()) { return 1; }
		file.seekg(0, std::ios::end);
		int length = file.tellg();
		file.seekg(0, std::ios::beg);
		std::unique_ptr<char[]> data(new char[length]);
		file.read(data.get(), length);
		file.close();

		// 生成mEngine
		nvinfer1::ICudaEngine* engine;
		SampleUniquePtr<nvinfer1::IRuntime> runTime(nvinfer1::createInferRuntime(sample::gLogger));
		engine = runTime->deserializeCudaEngine(data.get(), length, nullptr);

		auto a = engine->getNbBindings();
		for (int k = 0; k < a; k++) {
			Dims dims = engine->getBindingDimensions(k);
			for (int mm = 0; mm < dims.nbDims; mm++) {
				std::cout << dims.d[mm] << " ";
			}
			std::cout << std::endl;
		}
		// 生成Context
		nvinfer1::IExecutionContext* context;
		context = engine->createExecutionContext();
		ctxs.push_back(context);
	}

	// 推理
	vector<vector<cv::Mat>> results(threadNum);
	for (int aa = 0; aa < ALL_TIMES; aa++) {
		auto timeStart = cv::getTickCount();										// 获取开始时间
		//创建多个流
		vector<cudaStream_t> streams(threadNum);
		for (int i = 0; i < threadNum; i++) cudaStreamCreate(&streams[i]);

		omp_set_num_threads(threadNum);
#pragma omp parallel for
		for (int k = 0; k < threadNum; k++) {
			auto timeOneStart = cv::getTickCount();										// 获取开始时间

			for (int times = 0; times < TIMES; times++) {

				// 分配显存和内存
				auto timeMalloc = cv::getTickCount();										// 获取开始时间
				vector<void*> bindings(3);
				cudaMalloc(&bindings[0], input_batch * input_channels * input_height * input_widht * sizeof(float));
				cudaMalloc(&bindings[1], output_batch1 * output_channels1 * output_height1 * output_widht1 * sizeof(float));
				cudaMalloc(&bindings[2], output_batch2 * output_channels2 * output_height2 * output_widht2 * sizeof(float));
				auto timeMallocUsed = (cv::getTickCount() - timeMalloc) * 1000 / cv::getTickFrequency();
				//std::cout << "1 Time Malloc:" + to_string(timeMallocUsed) << endl;

				// Dims->Vector & Host->Device
				auto timeH2D = cv::getTickCount();										// 获取开始时间
				for (int n = 0; n < input_batch; n++) {
					if (input_channels == 1) {
						cudaMemcpyAsync(
							(float*)bindings[0] + n * input_height * input_widht,
							images[k][n].data,
							input_height * input_widht * sizeof(float),
							cudaMemcpyHostToDevice,
							streams[k]);
					}
					else {
						vector<cv::Mat> _channels(input_channels);
						cv::split(images[k][n], _channels);
						for (int c = 0; c < input_channels; c++) {
							cudaMemcpyAsync(
								(float*)bindings[0] + (input_channels * n + c) * input_height * input_widht,
								_channels[c].data,
								input_height * input_widht * sizeof(float),
								cudaMemcpyHostToDevice,
								streams[k]);
						}

					}


				}
				auto timeH2DUsed = (cv::getTickCount() - timeH2D) * 1000 / cv::getTickFrequency();
				//std::cout << "2 Time H2D:" + to_string(timeH2DUsed) << endl;

				// 推理
				auto timeExecute = cv::getTickCount();										// 获取开始时间
				ctxs[k]->enqueueV2(bindings.data(), streams[k], nullptr);
				auto timeExecuteUsed = (cv::getTickCount() - timeExecute) * 1000 / cv::getTickFrequency();
				//std::cout << std::string("3 Context Address: " + std::to_string(reinterpret_cast<std::uintptr_t>(ctxs[k]))) << endl;
				//std::cout << "4 Time ExecuteUsed:" + to_string(timeExecuteUsed) << endl;

				// Device->Host
				auto timeD2H = cv::getTickCount();										// 获取开始时间
				void* output1 = malloc(output_batch1 * output_channels1 * output_height1 * output_widht1 * sizeof(float));
				void* output2 = malloc(output_batch2 * output_channels2 * output_height2 * output_widht2 * sizeof(float));
				cudaMemcpyAsync(
					(float*)output1, (float*)bindings[1],
					output_batch1 * output_channels1 * output_height1 * output_widht1 * sizeof(float),
					cudaMemcpyDeviceToHost,
					streams[k]);
				cudaMemcpyAsync(
					(float*)output2, (float*)bindings[2],
					output_batch2 * output_channels2 * output_height2 * output_widht2 * sizeof(float),
					cudaMemcpyDeviceToHost,
					streams[k]);
				auto timeD2HUsed = (cv::getTickCount() - timeD2H) * 1000 / cv::getTickFrequency();
				//std::cout << "5 Time D2HUsed:" + to_string(timeD2HUsed) << endl;

				// Vector->Dims
				auto timeVectorDims = cv::getTickCount();										// 获取开始时间
				results[k].clear();
				results[k].resize(output_batch1);
				for (int n = 0; n < output_batch1; n++) {
					cv::Mat conf = cv::Mat(output_height1, output_widht1, CV_32F, (float*)output1 + n * output_height1 * output_widht1).clone();
					cv::Mat label = cv::Mat(output_height2, output_widht2, CV_32F, (float*)output2 + n * output_height2 * output_widht2).clone();
					results[k][n] = label;

				}
				auto timeVectorDimsUsed = (cv::getTickCount() - timeVectorDims) * 1000 / cv::getTickFrequency();
				//std::cout << "6 Time VectorDimsUsed:" + to_string(timeVectorDimsUsed) << endl;

				cudaFree(bindings[0]);
				cudaFree(bindings[1]);
				cudaFree(bindings[2]);
				std::free(output1);
				std::free(output2);
			}

			auto timeOneEnd = (cv::getTickCount() - timeOneStart) * 1000 / cv::getTickFrequency();
			std::cout << "------------------------------------Thread Time:" + to_string(timeOneEnd) << endl;
		}
		//同步流
		//for (int ii = 0; ii < threadNum; ii++) { cudaStreamSynchronize(streams[ii]); }

		// 释放流
		for (int i = 0; i < threadNum; i++) cudaStreamDestroy(streams[i]);
		auto timeEnd = (cv::getTickCount() - timeStart) * 1000 / cv::getTickFrequency();
		std::cout << "Total Time:" + to_string(timeEnd) << endl;
	}
	std::cout << "END" << endl;

}