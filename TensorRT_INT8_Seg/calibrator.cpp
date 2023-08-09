#include "calibrator.h"

using namespace nvinfer1;


CNNCalibrator::CNNCalibrator(const std::string& calibrationDataFolder, const int nCalibration, const Dims32 dim, const std::string& cacheFile, std::string suffix, bool saveCalibratorImage, std::string dstCalibratorPath) :
    mCalibrationNum(nCalibration), mDim(dim), mCacheFile(cacheFile), mCalibrationNumTimes(0), mImageSuffix(suffix), mSaveCalibratorImage(saveCalibratorImage), mDstCalibratorPath(dstCalibratorPath)
{
#ifdef DEBUG
    std::cout << "[MyCalibrator::MyCalibrator]" << std::endl;
#endif

    // 获得所有图片路径
    cv::glob(calibrationDataFolder + "*" + mImageSuffix, mImagesPath);

    // 计算输入的元素个数
    mElementNum = 1;
    for (int i = 0; i < dim.nbDims; ++i)
    {
        mElementNum *= dim.d[i];
    }

    // 计算输入元素占用空间
    mBufferSize = sizeof(float) * mElementNum;
    
    // 分配显存
    cudaMalloc((void**)&mBufferD, mBufferSize);

    return;
}

bool CNNCalibrator::getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept
{
#ifdef DEBUG
    std::cout << "[MyCalibrator::getBatch]---------" + std::to_string(mCalibrationNumTimes)  << std::endl;
#endif
    if (mCalibrationNumTimes < mCalibrationNum)
    {
        // 获取一个batchsize的输入数据
        std::vector<cv::Mat> inputs;
        while (inputs.size() < mDim.d[0]) {
            std::string imgPath = this->mImagesPath[mImageIndex];   // 获取图片路径
            cv::Mat img = cv::imread(imgPath, -1);                  // 读取图片

            // 筛选掉图片宽或高 小于 模型宽或高
            if (img.rows < mDim.d[2] || img.cols < mDim.d[3]) {
                std::cout << "Warning, " + imgPath + " width/height is too small." << std::endl;
                // 更新正在矫正图片索引
                mImageIndex = (++mImageIndex) % mImagesPath.size();
                if (mImageIndex == 0) {
                    float oldStepRatio = mStepRatio;
                    mStepRatio -= 0.1;
                    if (mStepRatio < 0.5) {
                        mStepRatio = 0.9;
                    }
                    std::cout << "Change StepRatio from `" + std::to_string(oldStepRatio) + "`  to `" + std::to_string(mStepRatio) + "`." << std::endl;

                }
                continue;
            }
            // 通道数不对直接报错
            if (img.channels() != mDim.d[1]) {
                throw std::invalid_argument("Image Channels Error.");
            }

            // 裁剪图片
            std::vector<cv::Mat> cropImages;
            if (slideCrop(img, cropImages, mStepRatio) != 0) {
                throw std::invalid_argument("Slide Crop Error.");
            }

  
            // 插入图片
            if (cropImages.size() != 0) {
                // 检查裁剪图片
                for (int k = 0; k < cropImages.size(); k++) {
                    if (cropImages[k].rows != mDim.d[2] || cropImages[k].cols != mDim.d[3]) {
                        std::cout << "Using `" + std::to_string(mImageIndex) + "` :" + imgPath + ", Crop `" + std::to_string(cropImages.size()) + "` Images Error." << std::endl;
                        throw std::invalid_argument("Slide Crop Error.");
                    }
                }

                inputs.insert(inputs.end(), cropImages.begin(), cropImages.end());
                std::cout << "Using `" + std::to_string(mImageIndex) + "` :" + imgPath + ", Crop `" + std::to_string(cropImages.size()) + "` Images. and StepRatio `" + std::to_string(mStepRatio) +"`." << std::endl;
            }
            
            // 更新正在矫正图片索引
            mImageIndex = (++mImageIndex) % mImagesPath.size();     
            if (mImageIndex == 0) {
                float oldStepRatio = mStepRatio;
                mStepRatio -= 0.1;
                if (mStepRatio < 0.5) {
                    mStepRatio = 0.9;
                }
                std::cout << "Change StepRatio from `" + std::to_string(oldStepRatio) + "`  to `" + std::to_string(mStepRatio) + "`." << std::endl;

            }

        }
        
        // 存储矫正图
        if (mSaveCalibratorImage) {
            // 创建目录
            if (0 != _access(mDstCalibratorPath.c_str(), 0)) {
                if (0 != _mkdir(mDstCalibratorPath.c_str())) {
                    throw std::invalid_argument("Save Crop Image Error.");
                }
            }

            for (int i = 0; i < mDim.d[0]; i++) {
                time_t nowtime = time(NULL);
                struct tm* p;
                p = gmtime(&nowtime);
                char srcImgPath[64];
                sprintf(srcImgPath, "%d-%d-%d-%d-%d-%d-%d.bmp", 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec, i);
                std::string img_file_save = srcImgPath;
                cv::imwrite(mDstCalibratorPath + img_file_save, inputs[i]);
            }
        }

        // 拷贝到mBufferD
        for (int i = 0; i < mDim.d[0]; i++) {
            cv::Mat convertToFP32;
            inputs[i].convertTo(convertToFP32, CV_32F);
            int input_channels = convertToFP32.channels();
            int input_width = convertToFP32.cols;
            int input_height = convertToFP32.rows;

            if (input_channels == 1) {
                cudaMemcpy(mBufferD + i*input_height*input_width, convertToFP32.data, input_height*input_width*sizeof(float), cudaMemcpyHostToDevice);
            }
            else {
                std::vector<cv::Mat> _channels(input_channels);
                cv::split(convertToFP32, _channels);
                for (int d = 0; d < input_channels; d++) {
                    cudaMemcpy(mBufferD + (i * input_channels + d) * input_height * input_width, _channels[d].data, input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);

                }
            }
        }
        bindings[0] = mBufferD;
        mCalibrationNumTimes++;
        return true;
    }
    else
    {
        return false;
    }
}

int CNNCalibrator::slideCrop(const cv::Mat& roiImg, std::vector<cv::Mat>& cropImgs, float stepRatio) {
	// 1. 如果srcImg为空
	if (roiImg.empty()) {
		return 1;
	}

	// 2. 获取模型输入的宽高dw/dh, 和待分割图像的宽高roiImgW/roiImgH
	int roiImgH = roiImg.rows;
	int roiImgW = roiImg.cols;
	int dh = mDim.d[2];
	int dw = mDim.d[3];
	int x_step = int(dw * stepRatio); // x方向步长， int是向下取整
	int y_step = int(dh * stepRatio); // y方向步长

	// 3. 当roiImg的宽高都小于模型宽高时-----> 情形1：检测图(HW)<滑动框(HW)
	if (roiImgH <= dh && roiImgW <= dw) {
		return 0;
	}
	// 4. 当roiImg的宽度都大于模型宽高时-----> 情形2：检测图(HW)>滑动框(HW)
	else if (roiImgH >= dh && roiImgW >= dw) {
		// 4.1 左上整个区域
		for (int i = 0; i <= roiImgH - dh; i += y_step) {		// y轴方向
			for (int j = 0; j <= roiImgW - dw; j += x_step) {  // x轴方向
				cv::Rect rect(j, i, dw, dh);
				cropImgs.push_back(roiImg(rect).clone());
			}
		}
	}
	// 5. 当roiImg宽小于模型宽，roiImg高大于模型高时-----> 情形3：检测图是竖条状，滑动框的W>检测图的宽，H<检测图的高
	else if (roiImgW < dw && roiImgH >= dh) {
		return 0;
	}
	// 6. 当roiImg宽大于模型宽，roiImg高小于模型高时-----> 情形4：检测图是横条状，滑动框的W<检测图的宽，H>检测图的高
	else if (roiImgW >= dw && roiImgH < dh) {
        return 0;
	}
    return 0;
}

CNNCalibrator::~CNNCalibrator() noexcept
{
#ifdef DEBUG
    std::cout << "[MyCalibrator::~MyCalibrator]" << std::endl;
#endif
    if (mBufferD != nullptr)
    {
        cudaFree(mBufferD);
    }
    return;
}

int32_t CNNCalibrator::getBatchSize() const noexcept
{
#ifdef DEBUG
    std::cout << "[MyCalibrator::getBatchSize]" << std::endl;
#endif
    return mDim.d[0];
}

void const* CNNCalibrator::readCalibrationCache(std::size_t& length) noexcept
{
#ifdef DEBUG
    std::cout << "[MyCalibrator::readCalibrationCache]" << std::endl;
#endif
    std::fstream f;
    f.open(mCacheFile, std::fstream::in);
    if (f.fail())
    {
        std::cout << "Failed finding cache file!" << std::endl;
        return nullptr;
    }
    char* ptr = new char[length];
    if (f.is_open())
    {
        f >> ptr;
    }
    return ptr;
}

void CNNCalibrator::writeCalibrationCache(void const* ptr, std::size_t length) noexcept
{
#ifdef DEBUG
    std::cout << "[MyCalibrator::writeCalibrationCache]" << std::endl;
#endif
    std::ofstream f(mCacheFile, std::ios::binary);
    if (f.fail())
    {
        std::cout << "Failed opening cache file to write!" << std::endl;
        return;
    }
    f.write(static_cast<char const*>(ptr), length);
    if (f.fail())
    {
        std::cout << "Failed saving cache file!" << std::endl;
        return;
    }
    f.close();
}