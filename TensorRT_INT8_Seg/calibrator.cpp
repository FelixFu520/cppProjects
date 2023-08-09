#include "calibrator.h"

using namespace nvinfer1;


CNNCalibrator::CNNCalibrator(const std::string& calibrationDataFolder, const int nCalibration, const Dims32 dim, const std::string& cacheFile, std::string suffix, bool saveCalibratorImage, std::string dstCalibratorPath) :
    mCalibrationNum(nCalibration), mDim(dim), mCacheFile(cacheFile), mCalibrationNumTimes(0), mImageSuffix(suffix), mSaveCalibratorImage(saveCalibratorImage), mDstCalibratorPath(dstCalibratorPath)
{
#ifdef DEBUG
    std::cout << "[MyCalibrator::MyCalibrator]" << std::endl;
#endif

    // �������ͼƬ·��
    cv::glob(calibrationDataFolder + "*" + mImageSuffix, mImagesPath);

    // ���������Ԫ�ظ���
    mElementNum = 1;
    for (int i = 0; i < dim.nbDims; ++i)
    {
        mElementNum *= dim.d[i];
    }

    // ��������Ԫ��ռ�ÿռ�
    mBufferSize = sizeof(float) * mElementNum;
    
    // �����Դ�
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
        // ��ȡһ��batchsize����������
        std::vector<cv::Mat> inputs;
        while (inputs.size() < mDim.d[0]) {
            std::string imgPath = this->mImagesPath[mImageIndex];   // ��ȡͼƬ·��
            cv::Mat img = cv::imread(imgPath, -1);                  // ��ȡͼƬ

            // ɸѡ��ͼƬ���� С�� ģ�Ϳ���
            if (img.rows < mDim.d[2] || img.cols < mDim.d[3]) {
                std::cout << "Warning, " + imgPath + " width/height is too small." << std::endl;
                // �������ڽ���ͼƬ����
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
            // ͨ��������ֱ�ӱ���
            if (img.channels() != mDim.d[1]) {
                throw std::invalid_argument("Image Channels Error.");
            }

            // �ü�ͼƬ
            std::vector<cv::Mat> cropImages;
            if (slideCrop(img, cropImages, mStepRatio) != 0) {
                throw std::invalid_argument("Slide Crop Error.");
            }

  
            // ����ͼƬ
            if (cropImages.size() != 0) {
                // ���ü�ͼƬ
                for (int k = 0; k < cropImages.size(); k++) {
                    if (cropImages[k].rows != mDim.d[2] || cropImages[k].cols != mDim.d[3]) {
                        std::cout << "Using `" + std::to_string(mImageIndex) + "` :" + imgPath + ", Crop `" + std::to_string(cropImages.size()) + "` Images Error." << std::endl;
                        throw std::invalid_argument("Slide Crop Error.");
                    }
                }

                inputs.insert(inputs.end(), cropImages.begin(), cropImages.end());
                std::cout << "Using `" + std::to_string(mImageIndex) + "` :" + imgPath + ", Crop `" + std::to_string(cropImages.size()) + "` Images. and StepRatio `" + std::to_string(mStepRatio) +"`." << std::endl;
            }
            
            // �������ڽ���ͼƬ����
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
        
        // �洢����ͼ
        if (mSaveCalibratorImage) {
            // ����Ŀ¼
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

        // ������mBufferD
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
	// 1. ���srcImgΪ��
	if (roiImg.empty()) {
		return 1;
	}

	// 2. ��ȡģ������Ŀ��dw/dh, �ʹ��ָ�ͼ��Ŀ��roiImgW/roiImgH
	int roiImgH = roiImg.rows;
	int roiImgW = roiImg.cols;
	int dh = mDim.d[2];
	int dw = mDim.d[3];
	int x_step = int(dw * stepRatio); // x���򲽳��� int������ȡ��
	int y_step = int(dh * stepRatio); // y���򲽳�

	// 3. ��roiImg�Ŀ�߶�С��ģ�Ϳ��ʱ-----> ����1�����ͼ(HW)<������(HW)
	if (roiImgH <= dh && roiImgW <= dw) {
		return 0;
	}
	// 4. ��roiImg�Ŀ�ȶ�����ģ�Ϳ��ʱ-----> ����2�����ͼ(HW)>������(HW)
	else if (roiImgH >= dh && roiImgW >= dw) {
		// 4.1 ������������
		for (int i = 0; i <= roiImgH - dh; i += y_step) {		// y�᷽��
			for (int j = 0; j <= roiImgW - dw; j += x_step) {  // x�᷽��
				cv::Rect rect(j, i, dw, dh);
				cropImgs.push_back(roiImg(rect).clone());
			}
		}
	}
	// 5. ��roiImg��С��ģ�Ϳ�roiImg�ߴ���ģ�͸�ʱ-----> ����3�����ͼ������״���������W>���ͼ�Ŀ�H<���ͼ�ĸ�
	else if (roiImgW < dw && roiImgH >= dh) {
		return 0;
	}
	// 6. ��roiImg�����ģ�Ϳ�roiImg��С��ģ�͸�ʱ-----> ����4�����ͼ�Ǻ���״���������W<���ͼ�Ŀ�H>���ͼ�ĸ�
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