#include <stdio.h>
#include <assert.h>
#include <memory>
#include <omp.h>
#include <cuda_runtime.h>					// cuda��
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>	
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <numeric>
#include <iostream>
#include <thread>

#include "NvInferRuntime.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"

using namespace nvinfer1;
using namespace std;

const int TIMES = 1, THREADS=4;
const unsigned int nElements = 64 * 6 * 256 * 256;
const unsigned int bytes = nElements * sizeof(float);

void testCudaMemcpy(const unsigned int bytes, const unsigned int nElements, int t)
{
    float* h_aPageable, * h_bPageable;  // host arrays
    h_aPageable = (float*)malloc(bytes);                    // host pageable
    h_bPageable = (float*)malloc(bytes);                    // host pageable
    float* d_a;     // device arrays
    cudaMalloc((void**)&d_a, bytes);
    for (int i = 0; i < nElements; ++i) h_aPageable[i] = i;
    for (int j = 0; j < TIMES; j++) {
        auto H2DStart = cv::getTickCount();										// ��ȡ��ʼʱ��
        cudaMemcpy(d_a, h_aPageable, bytes, cudaMemcpyHostToDevice);
        auto H2DUsed = (cv::getTickCount() - H2DStart) * 1000 / cv::getTickFrequency();
        std::cout << std::to_string(t) + std::string(":Multi H2D:" + std::to_string(H2DUsed) + "ms.") << std::endl;

        auto D2HStart = cv::getTickCount();										// ��ȡ��ʼʱ��
        cudaMemcpy(h_bPageable, d_a, bytes, cudaMemcpyDeviceToHost);
        auto D2HUsed = (cv::getTickCount() - D2HStart) * 1000 / cv::getTickFrequency();
        std::cout << std::to_string(t) + std::string(":Multi D2H:" + std::to_string(D2HUsed) + "ms.") << std::endl;
    }
    // cleanup
    cudaFree(d_a);
    free(h_aPageable);
    free(h_bPageable);
}

int main()
{
    // ���̲߳���
    if(true)
    {
        std::cout << "--------------------- ���̡߳���ҳ�ڴ���� -------------------------------" << endl;
        for (int i = 0; i < TIMES; i++) {
            float* h_aPageable, * h_bPageable;  // host arrays
            h_aPageable = (float*)malloc(bytes);                    // host pageable
            h_bPageable = (float*)malloc(bytes);                    // host pageable
            float* d_a;     // device arrays
            cudaMalloc((void**)&d_a, bytes);
            for (int i = 0; i < nElements; ++i) h_aPageable[i] = i;

            auto H2DStart = cv::getTickCount();										// ��ȡ��ʼʱ��
            cudaMemcpy(d_a, h_aPageable, bytes, cudaMemcpyHostToDevice);
            auto H2DUsed = (cv::getTickCount() - H2DStart) * 1000 / cv::getTickFrequency();
            std::cout << std::string("Single H2D:" + std::to_string(H2DUsed) + "ms.") << std::endl;

            auto D2HStart = cv::getTickCount();										// ��ȡ��ʼʱ��
            cudaMemcpy(h_bPageable, d_a, bytes, cudaMemcpyDeviceToHost);
            auto D2HUsed = (cv::getTickCount() - D2HStart) * 1000 / cv::getTickFrequency();
            std::cout << std::string("Single D2H:" + std::to_string(D2HUsed) + "ms.") << std::endl;
            // cleanup
            cudaFree(d_a);
            free(h_aPageable);
            free(h_bPageable);
        }
    }

    // ���̲߳���, ��ҳ�ڴ�
    if (true)
    {
        std::cout << "--------------------- ���̡߳���ҳ�ڴ���� -------------------------------" << endl;
        for (int i = 0; i < TIMES; i++) {
            float* h_aPinned, * h_bPinned;  // host arrays
            cudaMallocHost((void**)&h_aPinned, bytes);
            cudaMallocHost((void**)&h_bPinned, bytes);
            float* d_a;     // device arrays
            cudaMalloc((void**)&d_a, bytes);
            for (int i = 0; i < nElements; ++i) h_aPinned[i] = i;

            auto H2DStart = cv::getTickCount();										// ��ȡ��ʼʱ��
            cudaMemcpy(d_a, h_aPinned, bytes, cudaMemcpyHostToDevice);
            auto H2DUsed = (cv::getTickCount() - H2DStart) * 1000 / cv::getTickFrequency();
            std::cout << std::string("Single H2D:" + std::to_string(H2DUsed) + "ms.") << std::endl;

            auto D2HStart = cv::getTickCount();										// ��ȡ��ʼʱ��
            cudaMemcpy(h_bPinned, d_a, bytes, cudaMemcpyDeviceToHost);
            auto D2HUsed = (cv::getTickCount() - D2HStart) * 1000 / cv::getTickFrequency();
            std::cout << std::string("Single D2H:" + std::to_string(D2HUsed) + "ms.") << std::endl;
            // cleanup
            cudaFree(d_a);
            cudaFreeHost(h_aPinned);
            cudaFreeHost(h_bPinned);
        }
    }
    
    // ���̲߳���
    if(true)
    {
        std::cout << "--------------------- OMP���̡߳���ҳ�ڴ���� -------------------------------" << endl;
        omp_set_num_threads(THREADS);
#pragma omp parallel for
        for (int i = 0; i < THREADS; i++) {
            float* h_aPageable, * h_bPageable;  // host arrays
            h_aPageable = (float*)malloc(bytes);                    // host pageable
            h_bPageable = (float*)malloc(bytes);                    // host pageable
            float* d_a;     // device arrays
            cudaMalloc((void**)&d_a, bytes);
            for (int i = 0; i < nElements; ++i) h_aPageable[i] = i;
            for (int j = 0; j < TIMES; j++) {
                auto H2DStart = cv::getTickCount();										// ��ȡ��ʼʱ��
                cudaMemcpy(d_a, h_aPageable, bytes, cudaMemcpyHostToDevice);
                auto H2DUsed = (cv::getTickCount() - H2DStart) * 1000 / cv::getTickFrequency();
                std::cout << std::to_string(i) + std::string(":Multi H2D:" + std::to_string(H2DUsed) + "ms.") << std::endl;

                auto D2HStart = cv::getTickCount();										// ��ȡ��ʼʱ��
                cudaMemcpy(h_bPageable, d_a, bytes, cudaMemcpyDeviceToHost);
                auto D2HUsed = (cv::getTickCount() - D2HStart) * 1000 / cv::getTickFrequency();
                std::cout << std::to_string(i) + std::string(":Multi D2H:" + std::to_string(D2HUsed) + "ms.") << std::endl;
            }
            // cleanup
            cudaFree(d_a);
            free(h_aPageable);
            free(h_bPageable);
        }
    }
    
    // ���̲߳���
    if (false) {
        std::cout << "--------------------- ���̡߳���ҳ�ڴ���� -------------------------------" << endl;
        vector<thread> threads(THREADS);
        for (int i = 0; i < THREADS; i++) {
            auto th = thread(testCudaMemcpy, ref(bytes), ref(nElements), i); //��һ������Ϊ���������ڶ�������Ϊ�ú����ĵ�һ������������ú������ն������������д�ں��档��ʱ�߳̿�ʼִ�С�
            th.swap(threads[i]);
        }
        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
        }
    }

    // ���̲߳��ԣ���ҳ�ڴ�
    if (true) {
        std::cout << "--------------------- OMP���̡߳���ҳ�ڴ���� -------------------------------" << endl;
        omp_set_num_threads(THREADS);
#pragma omp parallel for
        for (int i = 0; i < THREADS; i++) {
            float* h_aPinned, * h_bPinned;  // host arrays
            cudaMallocHost((void**)&h_aPinned, bytes);
            cudaMallocHost((void**)&h_bPinned, bytes);
            float* d_a;     // device arrays
            cudaMalloc((void**)&d_a, bytes);
            for (int i = 0; i < nElements; ++i) h_aPinned[i] = i;
            for (int j = 0; j < TIMES; j++) {
                auto H2DStart = cv::getTickCount();										// ��ȡ��ʼʱ��
                cudaMemcpy(d_a, h_aPinned, bytes, cudaMemcpyHostToDevice);
                auto H2DUsed = (cv::getTickCount() - H2DStart) * 1000 / cv::getTickFrequency();
                std::cout << std::to_string(i) + std::string(":Multi H2D:" + std::to_string(H2DUsed) + "ms.") << std::endl;

                auto D2HStart = cv::getTickCount();										// ��ȡ��ʼʱ��
                cudaMemcpy(h_bPinned, d_a, bytes, cudaMemcpyDeviceToHost);
                auto D2HUsed = (cv::getTickCount() - D2HStart) * 1000 / cv::getTickFrequency();
                std::cout << std::to_string(i) + std::string(":Multi D2H:" + std::to_string(D2HUsed) + "ms.") << std::endl;
            }
            // cleanup
            cudaFree(d_a);
            cudaFreeHost(h_aPinned);
            cudaFreeHost(h_bPinned);
        }
    }
    
    // ���̲߳��ԣ���ҳ�ڴ�, �ⲿ����ռ�
    if (false) {
        std::cout << "--------------------- OMP���̡߳���ҳ�ڴ桢�ⲿ����ռ���� -------------------------------" << endl;
        vector<float*> h_aPinneds(THREADS), h_bPinneds(THREADS), d_addrs(THREADS);
        for (int i = 0; i < THREADS; i++) {
            cudaMallocHost((void**)&h_aPinneds[i], bytes);
            for (int k = 0; k < nElements; ++k) h_aPinneds[i][k] = k;

            cudaMallocHost((void**)&h_bPinneds[i], bytes);
            cudaMalloc((void**)&d_addrs[i], bytes);
        }

        omp_set_num_threads(THREADS);
#pragma omp parallel for
        for (int i = 0; i < THREADS; i++) {
            for (int j = 0; j < TIMES; j++) {
                auto H2DStart = cv::getTickCount();										// ��ȡ��ʼʱ��
                cudaMemcpy(d_addrs[i], h_aPinneds[i], bytes, cudaMemcpyHostToDevice);
                auto H2DUsed = (cv::getTickCount() - H2DStart) * 1000 / cv::getTickFrequency();
                std::cout << std::to_string(i) + std::string(":Multi H2D:" + std::to_string(H2DUsed) + "ms.") << std::endl;

                auto D2HStart = cv::getTickCount();										// ��ȡ��ʼʱ��
                cudaMemcpy(h_bPinneds[i], d_addrs[i], bytes, cudaMemcpyDeviceToHost);
                auto D2HUsed = (cv::getTickCount() - D2HStart) * 1000 / cv::getTickFrequency();
                std::cout << std::to_string(i) + std::string(":Multi D2H:" + std::to_string(D2HUsed) + "ms.") << std::endl;
            }

        }
        for (int i = 0; i < THREADS; i++) {
            cudaFree(d_addrs[i]);
            cudaFreeHost(h_aPinneds[i]);
            cudaFreeHost(h_bPinneds[i]);
        }
    }

    // ���̲߳��ԣ���ҳ�ڴ�, ʹ����
    if (true) {
        std::cout << "--------------------- OMP���̡߳���ҳ�ڴ桢ʹ�������� -------------------------------" << endl;
        //���������
        cudaStream_t stream[THREADS];
        for (int i = 0; i < THREADS; i++) cudaStreamCreate(&stream[i]);
        
        omp_set_num_threads(THREADS);
#pragma omp parallel for
        for (int i = 0; i < THREADS; i++) {
            float* h_aPinned, * h_bPinned;  // host arrays
            cudaMallocHost((void**)&h_aPinned, bytes);
            cudaMallocHost((void**)&h_bPinned, bytes);
            float* d_a;     // device arrays
            cudaMalloc((void**)&d_a, bytes);
            for (int i = 0; i < nElements; ++i) h_aPinned[i] = i;
            for (int j = 0; j < TIMES; j++) {
                auto H2DStart = cv::getTickCount();										// ��ȡ��ʼʱ��
                cudaMemcpyAsync(d_a, h_aPinned, bytes, cudaMemcpyHostToDevice, stream[i]);
                auto H2DUsed = (cv::getTickCount() - H2DStart) * 1000 / cv::getTickFrequency();
                std::cout << std::to_string(i) + std::string(":Multi H2D:" + std::to_string(H2DUsed) + "ms.") << std::endl;

                std::chrono::milliseconds duration(5);
                std::this_thread::sleep_for(duration);

                auto D2HStart = cv::getTickCount();										// ��ȡ��ʼʱ��
                cudaMemcpyAsync(h_bPinned, d_a, bytes, cudaMemcpyDeviceToHost, stream[i]);
                auto D2HUsed = (cv::getTickCount() - D2HStart) * 1000 / cv::getTickFrequency();
                std::cout << std::to_string(i) + std::string(":Multi D2H:" + std::to_string(D2HUsed) + "ms.") << std::endl;
            }

            //ͬ����
            for (int i = 0; i < THREADS; i++) {
                cudaStreamSynchronize(stream[i]);
            }
            // cleanup
            cudaFree(d_a);
            cudaFreeHost(h_aPinned);
            cudaFreeHost(h_bPinned);
        }
        
        //������ 
        for (int i = 0; i < THREADS; i++) cudaStreamDestroy(stream[i]);
    }
    
    return 0;
}