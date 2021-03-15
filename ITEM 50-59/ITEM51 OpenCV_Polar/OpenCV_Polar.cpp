#include <opencv2/opencv.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>

using namespace std;
using namespace cv;

// ----- 全局参数
// PAI值
double PI = M_PI;
// 设置输入图像固定尺寸（必要）
double HEIGHT = 300;
double WIDTH = 300;
// 输入图像圆的半径，一般是宽高一半
int CIRCLE_RADIUS = int(HEIGHT / 2);
// 圆心坐标
cv::Point CIRCLE_CENTER = cv::Point(int(WIDTH / 2), int(HEIGHT / 2));
// 极坐标转换后图像的高，可自己设置
int LINE_HEIGHT = int(CIRCLE_RADIUS / 1.5);
// 极坐标转换后图像的宽，一般是原来圆形的周长
int LINE_WIDTH = int(2 * CIRCLE_RADIUS * PI);

// C++ OpenCV Mat读像素值用
typedef Point3_<uint8_t> Pixel;

// ----- 将圆环变为矩形
cv::Mat create_line_image(cv::Mat img)
{
	cv::Mat line_image = cv::Mat::zeros(Size(LINE_WIDTH, LINE_HEIGHT), CV_8UC3);
	// 角度
	double theta;
	// 半径
	double rho;

	// 按照圆的极坐标赋值
	for (int row = 0; row < line_image.rows; row++)
	{
		for (int col = 0; col < line_image.cols; col++)
		{
			// 最后的-0.2是用于优化结果，可以自行调整
			theta = PI * 2 / LINE_WIDTH * (col + 1) - 0.2;
			rho = CIRCLE_RADIUS - row - 1;

			// ----- 基础变换
			//int x = int(CIRCLE_CENTER.x + rho * std::cos(theta) + 0);
			//int y = int(CIRCLE_CENTER.y - rho * std::sin(theta) + 0);

			// ----- 任意起始位置变换
			//// 1 确定极坐标
			//double x0 = rho * std::cos(theta) + 0;
			//double y0 = rho * std::sin(theta) + 0;

			//// 2 确定旋转角度
			//double angle = PI * 2 * (-120.0) / 360;

			//// 3 确定直角坐标
			//double x1 = x0 * std::cos(angle) - y0 * std::sin(angle) + 0;
			//double y1 = x0 * std::sin(angle) + y0 * std::cos(angle) + 0;

			//// 4 切换为OpenCV图像坐标
			//int x = int(CIRCLE_CENTER.x + x1);
			//int y = int(CIRCLE_CENTER.y - y1);

			// ----- 任意起始位置顺时针变换
			// 1 确定极坐标
			double x0 = rho * std::sin(theta) + 0;
			double y0 = rho * std::cos(theta) + 0;

			// 2 确定旋转角度
			double angle = PI * 2 * (-150.0) / 360;

			// 3 确定直角坐标
			double x1 = x0 * std::cos(angle) - y0 * std::sin(angle) + 0;
			double y1 = x0 * std::sin(angle) + y0 * std::cos(angle) + 0;

			// 4 切换为opencv图像坐标
			int x = int(CIRCLE_CENTER.x + x1);
			int y = int(CIRCLE_CENTER.y - y1);

			// Obtain pixel at(y,x)直接访问像素数据(效率不高，可以修改）
			Pixel pixel = img.at<Pixel>(y, x);
			// 赋值
			line_image.at<Pixel>(row, col) = pixel;
		}
	}
	// 如果想改变输出图像方向，旋转就行了
	// cv::rotate(line_image, line_image, cv::ROTATE_90_CLOCKWISE);
	return line_image;
}

// ----- 主程序
int main()
{
	// 输入图像路径
	String imgpath = "./image/clock.jpg";
	// 读取图像
	cv::Mat img = cv::imread(imgpath);
	if (img.empty())
	{
		printf("please check image path");
		return -1;
	}
	// 图像重置为固定大小
	cv::resize(img, img, Size(WIDTH, HEIGHT));
	printf("shape is: %d,%d", img.rows, img.cols);
	// 展示原图
	cv::imshow("src", img);
	cv::Mat output = create_line_image(img);
	// 展示结果
	cv::imshow("dst", output);
	cv::waitKey();
	cv::destroyAllWindows();
	system("pause");
	return 0;
}