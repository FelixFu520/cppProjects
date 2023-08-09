#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <io.h>
#include <direct.h>
#include <ctime>

#include "opencv2/opencv.hpp"

#include <cuda_runtime_api.h>
#include <NvOnnxParser.h>
#include <NvInfer.h>



using namespace nvinfer1;

class CNNCalibrator : public IInt8EntropyCalibrator2
{
private:
    std::string mImageSuffix {".bmp"};
    std::vector<std::string> mImagesPath;   // ��������
    unsigned int mImageIndex { 0 };         // ����ʹ���Ǹ����ݽ��н���
    float mStepRatio { 0.9 };               // �ü�Сͼ����
    bool mSaveCalibratorImage{ true };    // �Ƿ�洢����Сͼ
    std::string mDstCalibratorPath {"D:/CalibratorImages/"};    // �Ƿ�洢����Сͼ

    int mCalibrationNum {0};                // �����ܴ���
    int mCalibrationNumTimes {0};           // �����ǵڼ��ν���

    int mElementNum{ 0 };                   // ����Ԫ�ظ���
    size_t mBufferSize{ 0 };                // ����Ԫ����ռ�ռ�
    Dims32 mDim;                            // ����ά��
    float* mBufferD{ nullptr };             // �������Դ��ϵĿռ��ַ

    std::string mCacheFile {""};            // �����ļ�·��

public:
    CNNCalibrator(const std::string& calibrationDataFolder, const int nCalibration, const Dims32 inputShape, const std::string& cacheFile, std::string suffix = ".bmp", bool saveCalibratorImage=true, std::string mDstCalibratorPath="D:/CalibratorImages/");
    ~CNNCalibrator() noexcept;
    int32_t     getBatchSize() const noexcept;
    bool        getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept;
    void const* readCalibrationCache(std::size_t& length) noexcept;
    void        writeCalibrationCache(void const* ptr, std::size_t length) noexcept;
private:
    int        slideCrop(const cv::Mat& roiImg, std::vector<cv::Mat>& cropImgs, float stepRatio=0.9);
};