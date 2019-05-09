#include "pch.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//孔洞填充
void fillHoles(Mat &mask)
{
	Mat maskFloodfill = mask.clone();
	//漫水填充
	floodFill(maskFloodfill, cv::Point(0, 0), Scalar(255));
	Mat mask2;
	//反色
	bitwise_not(maskFloodfill, mask2);
	//或运算
	mask = (mask2 | mask);
}

int main()
{
	// Read image 读彩色图像
	Mat img = imread("./image/red_eyes.jpg", CV_LOAD_IMAGE_COLOR);

	// Output image 输出图像
	Mat imgOut = img.clone();

	// Load HAAR cascade 读取haar分类器
	CascadeClassifier eyesCascade("./model/haarcascade_eye.xml");

	// Detect eyes 检测眼睛
	std::vector<Rect> eyes;
	//前四个参数：输入图像，眼睛结果，表示每次图像尺寸减小的比例，表示每一个目标至少要被检测到4次才算是真的
	//后两个参数：0 | CASCADE_SCALE_IMAGE表示不同的检测模式，最小检测尺寸
	eyesCascade.detectMultiScale(img, eyes, 1.3, 4, 0 | CASCADE_SCALE_IMAGE, Size(100, 100));

	// For every detected eye 每只眼睛都进行处理
	for (size_t i = 0; i < eyes.size(); i++)
	{
		// Extract eye from the image. 提取眼睛图像
		Mat eye = img(eyes[i]);

		// Split eye image into 3 channels. 颜色分离
		vector<Mat>bgr(3);
		split(eye, bgr);

		// Simple red eye detector 红眼检测器，获得结果掩模
		Mat mask = (bgr[2] > 150) & (bgr[2] > (bgr[1] + bgr[0]));

		// Clean mask 清理掩模
		//填充孔洞
		fillHoles(mask);
		//扩充孔洞
		dilate(mask, mask, Mat(), Point(-1, -1), 3, 1, 1);

		// Calculate the mean channel by averaging the green and blue channels
		//计算b通道和g通道的均值
		Mat mean = (bgr[0] + bgr[1]) / 2;
		//用该均值图像覆盖原图掩模部分图像
		mean.copyTo(bgr[2], mask);
		mean.copyTo(bgr[0], mask);
		mean.copyTo(bgr[1], mask);

		// Merge channels
		Mat eyeOut;
		//图像合并
		cv::merge(bgr, eyeOut);

		// Copy the fixed eye to the output image.
		// 眼部图像替换
		eyeOut.copyTo(imgOut(eyes[i]));
	}

	// Display Result
	imshow("Red Eyes", img);
	imshow("Red Eyes Removed", imgOut);
	waitKey(0);
	return 0;
} 