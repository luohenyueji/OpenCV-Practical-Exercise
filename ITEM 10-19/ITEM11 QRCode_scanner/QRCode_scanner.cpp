// QRCode_scanner.cpp

#include "pch.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/**
 * @brief 用于显示检测到的QR码周围的框
 * 
 * @param im 
 * @param bbox 
 */
void display(Mat &im, Mat &bbox)
{
	int n = bbox.rows;
	for (int i = 0; i < n; i++)
	{
		line(im, Point2i(bbox.at<float>(i, 0), bbox.at<float>(i, 1)),
			 Point2i(bbox.at<float>((i + 1) % n, 0), bbox.at<float>((i + 1) % n, 1)), Scalar(255, 0, 0), 3);
	}
	imshow("Result", im);
}

int main()
{
	// Read image
	Mat inputImage = imread("./image/demo.jpg");

	//QR检测器
	QRCodeDetector qrDecoder = QRCodeDetector::QRCodeDetector();

	//二维码边框坐标，提取出来的二维码
	Mat bbox, rectifiedImage;

	//检测二维码
	std::string data = qrDecoder.detectAndDecode(inputImage, bbox, rectifiedImage);

	//获取二维码中的数据链接
	if (data.length() > 0)
	{
		cout << "Decoded Data : " << data << endl;
		display(inputImage, bbox);
		rectifiedImage.convertTo(rectifiedImage, CV_8UC3);
		//展示二维码
		imshow("Rectified QRCode", rectifiedImage);

		waitKey(0);
	}
	else
	{
		cout << "QR Code not detected" << endl;
	}
	return 0;
}
