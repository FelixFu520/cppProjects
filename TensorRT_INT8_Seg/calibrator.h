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
    std::vector<std::string> mImagesPath;   // 矫正数据
    unsigned int mImageIndex { 0 };         // 正在使用那个数据进行矫正
    float mStepRatio { 0.9 };               // 裁剪小图步长
    bool mSaveCalibratorImage{ true };    // 是否存储矫正小图
    std::string mDstCalibratorPath {"D:/CalibratorImages/"};    // 是否存储矫正小图

    int mCalibrationNum {0};                // 矫正总次数
    int mCalibrationNumTimes {0};           // 现在是第几次矫正

    int mElementNum{ 0 };                   // 输入元素个数
    size_t mBufferSize{ 0 };                // 输入元素所占空间
    Dims32 mDim;                            // 输入维度
    float* mBufferD{ nullptr };             // 输入在显存上的空间地址

    std::string mCacheFile {""};            // 缓存文件路径

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