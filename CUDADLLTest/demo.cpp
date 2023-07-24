#include <iostream>
#include "CudaDll.h"
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;

int main()
{
	// testΩ”ø⁄≤‚ ‘
	if(false)
	{
		const int size = 6;
		int a[size] = { 1, 2, 3,4,5,6 };
		int b[size] = { 12,14,16,18,20,22 };
		int c[size] = { 0 };
		int result = vectorAdd(a, b, c, size);
		printf("[%d, %d, %d, %d, %d, %d] \n + [%d, %d, %d, %d, %d, %d] \n= [%d, %d, %d, %d, %d, %d]",
			a[0], a[1], a[2], a[3], a[4], a[5],
			b[0], b[1], b[2], b[3], b[4], b[5],
			c[0], c[1], c[2], c[3], c[4], c[5]);
	}
	
	// argmaxΩ”ø⁄≤‚ ‘
	if(false)
	{
		// ∂¡»°Õº∆¨
		vector<Mat> allImagesC(8);	// CPU Mat
		vector<cuda::GpuMat> allImagesG(8);	// GPU Mat
		for (int i = 0; i < 8; i++) {
			Mat t = imread(string("D:\\BaiduSyncdisk\\Work\\cuda\\test\\cpu\\") + to_string(i) + ".png", IMREAD_UNCHANGED);
			t.convertTo(t, CV_32F, 1 / 255.0);
			allImagesC[i] = t;
			allImagesG[i].upload(t);
		}

		// ------------------ CPU ∞Ê±æ≤‚ ‘  ------------------
		cv::Mat output1, output2;
		argmax(allImagesC, output1, output2);
		output2 *= 255;

		// ------------------ GPU ∞Ê±æ≤‚ ‘ ------------------
		cv::cuda::GpuMat output7, output8;
		argmax(allImagesG, output7, output8);
		cv::Mat output77, output88;
		output7.download(output77);
		output8.download(output88);
		output77.convertTo(output77, CV_8U);
		output88.convertTo(output88, CV_8U);
		output88 *= 255;
	}
	if (true) {
		// ∂¡»°Õº∆¨
		int numImage = 1;
		vector<vector<Mat>> allImagesC(numImage);	// CPU Mat
		for (int i = 0; i < numImage; i++) {
			vector<Mat> temp;
			for (int j = 0; j < 8; j++) {
				Mat t = imread(string("D:\\BaiduSyncdisk\\Work\\cuda\\test\\cpu\\") + to_string(i) + ".png", IMREAD_UNCHANGED);
				t.convertTo(t, CV_32F, 1 / 255.0);
				temp.push_back(t);
			}
			allImagesC[i] = temp;
		}


		// ------------------ CPU ∞Ê±æ≤‚ ‘  ------------------
		vector<cv::Mat> output1, output2;
		argmax(allImagesC, output1, output2);

	}
	return 0;
}
