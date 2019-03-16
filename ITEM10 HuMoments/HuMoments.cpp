
//

#include "pch.h"
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main()
{
	//是否进行log转换
	bool showLogTransformedHuMoments = true;

	// Obtain filename 图像地址
	string filename("./image/s0.png");

	// Read Image 读图
	Mat im = imread(filename, IMREAD_GRAYSCALE);

	// Threshold image 阈值分割
	threshold(im, im, 0, 255, THRESH_OTSU);

	// Calculate Moments 计算矩
	//第二个参数True表示非零的像素都会按值1对待，也就是说相当于对图像进行了二值化处理，阈值为1
	Moments moment = moments(im, false);

	// Calculate Hu Moments 计算Hu矩
	double huMoments[7];
	HuMoments(moment, huMoments);

	// Print Hu Moments
	cout << filename << ": ";

	for (int i = 0; i < 7; i++)
	{
		if (showLogTransformedHuMoments)
		{
			// Log transform Hu Moments to make squash the range
			cout << -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i])) << " ";
		}
		else
		{
			// Hu Moments without log transform. 
			cout << huMoments[i] << " ";
		}

	}
	// One row per file
	cout << endl;

}
