#include <opencv2/opencv.hpp>
#include <opencv2/intensity_transform.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::intensity_transform;

// 计算对比度
double rmsContrast(Mat srcImg)
{
	Mat dstImg, dstImg_mean, dstImg_std;
	// 灰度化
	cvtColor(srcImg, dstImg, COLOR_BGR2GRAY);
	// 计算图像均值和方差
	meanStdDev(dstImg, dstImg_mean, dstImg_std);
	// 获得图像对比度
	double contrast = dstImg_std.at<double>(0, 0);
	return contrast;
}

// 保存图像
double saveImg(Mat srcImg, String saveType)
{
	String filename = "indicator";
	Mat saveImg = srcImg.clone();
	double contrast = rmsContrast(saveImg);

	putText(saveImg, format("contrast %.3f", contrast), Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 0, 255), 2, LINE_AA);
	imwrite("result/" + filename + "_" + saveType + "_result.jpg", saveImg);
}

// 设置命名空间避免污染用户变量
namespace
{
	// global variables
	Mat g_image;

	// gamma变换变量
	int g_gamma = 40;
	const int g_gammaMax = 500;
	Mat g_imgGamma;
	const std::string g_gammaWinName = "Gamma Correction";

	// 对比度拉伸
	Mat g_contrastStretch;
	int g_r1 = 70;
	int g_s1 = 15;
	int g_r2 = 120;
	int g_s2 = 240;
	const std::string g_contrastWinName = "Contrast Stretching";

	// 创建gamma变换滑动条
	static void onTrackbarGamma(int, void*)
	{
		float gamma = g_gamma / 100.0f;
		gammaCorrection(g_image, g_imgGamma, gamma);
		imshow(g_gammaWinName, g_imgGamma);
		cout << g_gammaWinName << ": " << rmsContrast(g_imgGamma) << endl;
		saveImg(g_imgGamma, "g_imgGamma");
	}

	// 创建对数变换滑动条
	static void onTrackbarContrastR1(int, void*)
	{
		contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);
		imshow("Contrast Stretching", g_contrastStretch);
		cout << g_contrastWinName << ": " << rmsContrast(g_contrastStretch) << endl;
		saveImg(g_contrastStretch, "g_contrastStretch");
	}

	static void onTrackbarContrastS1(int, void*)
	{
		contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);
		imshow("Contrast Stretching", g_contrastStretch);
		cout << g_contrastWinName << ": " << rmsContrast(g_contrastStretch) << endl;
		saveImg(g_contrastStretch, "g_contrastStretch");
	}

	static void onTrackbarContrastR2(int, void*)
	{
		contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);
		imshow("Contrast Stretching", g_contrastStretch);
		cout << g_contrastWinName << ": " << rmsContrast(g_contrastStretch) << endl;
		saveImg(g_contrastStretch, "g_contrastStretch");
	}

	static void onTrackbarContrastS2(int, void*)
	{
		contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);
		imshow("Contrast Stretching", g_contrastStretch);
		cout << g_contrastWinName << ": " << rmsContrast(g_contrastStretch) << endl;
		saveImg(g_contrastStretch, "g_contrastStretch");
	}
}

int main()
{
	// 图像路径
	const std::string inputFilename = "./image/car.png";

	// Read input image
	// 读图
	g_image = imread(inputFilename);

	if (g_image.empty())
	{
		printf("image is empty");
		return 0;
	}

	// Create trackbars
	// 创建滑动条
	namedWindow(g_gammaWinName);
	// 创建gamma变换筛选方法
	createTrackbar("Gamma value", g_gammaWinName, &g_gamma, g_gammaMax, onTrackbarGamma);

	// 对比度拉伸 Contrast Stretching
	namedWindow(g_contrastWinName);
	createTrackbar("Contrast R1", g_contrastWinName, &g_r1, 256, onTrackbarContrastR1);
	createTrackbar("Contrast S1", g_contrastWinName, &g_s1, 256, onTrackbarContrastS1);
	createTrackbar("Contrast R2", g_contrastWinName, &g_r2, 256, onTrackbarContrastR2);
	createTrackbar("Contrast S2", g_contrastWinName, &g_s2, 256, onTrackbarContrastS2);

	// Apply intensity transformations
	// 应用强度转换
	Mat imgAutoscaled, imgLog;
	// autoscaling
	autoscaling(g_image, imgAutoscaled);
	// gamma变换
	gammaCorrection(g_image, g_imgGamma, g_gamma / 100.0f);
	// 对数变换
	logTransform(g_image, imgLog);
	// 对比度拉伸
	contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);

	// Display intensity transformation results
	// 展示结果
	imshow("Original Image", g_image);
	cout << "Original Image: " << rmsContrast(g_image) << endl;
	saveImg(g_image, "g_image");

	imshow("Autoscale", imgAutoscaled);
	cout << "Autoscale: " << rmsContrast(imgAutoscaled) << endl;
	saveImg(imgAutoscaled, "imgAutoscaled");

	imshow(g_gammaWinName, g_imgGamma);
	cout << g_gammaWinName << ": " << rmsContrast(g_imgGamma) << endl;

	imshow("Log Transformation", imgLog);
	cout << "Log Transformation: " << rmsContrast(imgLog) << endl;
	saveImg(imgLog, "imgLog");

	imshow(g_contrastWinName, g_contrastStretch);
	cout << g_contrastWinName << ": " << rmsContrast(g_contrastStretch) << endl;

	waitKey(0);
	return 0;
}