// 图像超分放大单输出

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn_superres.hpp>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace dnn_superres;

int main()
{
	string img_path = string("./image/image.png");
	// 可选择算法，bilinear, bicubic, edsr, espcn, fsrcnn or lapsrn
	string algorithm = string("lapsrn");
	// 放大比例，可输入值2，3，4
	int scale = 4;
	// 模型路径
	string path = "./model/lapsrn_x4.pb";

	// Load the image
	// 载入图像
	Mat img = cv::imread(img_path);
	// 如果输入的图像为空
	if (img.empty())
	{
		std::cerr << "Couldn't load image: " << img << "\n";
		return -2;
	}

	Mat original_img(img);
	// Make dnn super resolution instance
	// 创建dnn超分辨率对象
	DnnSuperResImpl sr;

	// 超分放大后的图像
	Mat img_new;

	// 双线性插值
	if (algorithm == "bilinear")
	{
		resize(img, img_new, Size(), scale, scale, cv::INTER_LINEAR);
	}
	// 双三次插值
	else if (algorithm == "bicubic")
	{
		resize(img, img_new, Size(), scale, scale, cv::INTER_CUBIC);
	}
	else if (algorithm == "edsr" || algorithm == "espcn" || algorithm == "fsrcnn" || algorithm == "lapsrn")
	{
		// 读取模型
		sr.readModel(path);
		// 设定算法和放大比例
		sr.setModel(algorithm, scale);
		// 放大图像
		sr.upsample(img, img_new);
	}
	else
	{
		std::cerr << "Algorithm not recognized. \n";
	}

	// 如果失败
	if (img_new.empty())
	{
		// 放大失败
		std::cerr << "Upsampling failed. \n";
		return -3;
	}
	cout << "Upsampling succeeded. \n";

	// Display image
	// 展示图片
	cv::namedWindow("Initial Image", WINDOW_AUTOSIZE);
	// 初始化图片
	cv::imshow("Initial Image", img_new);
	cv::imwrite("./saved.jpg", img_new);
	cv::waitKey(0);

	return 0;
}