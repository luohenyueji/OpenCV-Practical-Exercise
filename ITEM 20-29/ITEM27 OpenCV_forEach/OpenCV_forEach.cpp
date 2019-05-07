#include "pch.h"
#include <opencv2/opencv.hpp>

// Use cv and std namespaces
using namespace cv;
using namespace std;

// Define a pixel 定义Pixel结构
typedef Point3_<uint8_t> Pixel;

/**
 * @brief tic is called to start timer 开始函数运行时间计算
 *
 * @param t
 */
void tic(double &t)
{
	t = (double)getTickCount();
}

/**
 * @brief toc is called to end timer 结束函数运行时间计算
 *
 * @param t
 * @return double 返回值运行时间ms
 */
double toc(double &t)
{
	return ((double)getTickCount() - t) / getTickFrequency() * 1000;
}

/**
 * @brief 阈值分割
 *
 * @param pixel
 */
void complicatedThreshold(Pixel &pixel)
{
	//x,y,z分别代表三个通道的值
	if (pow(double(pixel.x) / 10, 2.5) > 100)
	{
		pixel.x = 255;
		pixel.y = 255;
		pixel.z = 255;
	}
	else
	{
		pixel.x = 0;
		pixel.y = 0;
		pixel.z = 0;
	}
}

/**
 * @brief Parallel execution with function object. 并行处理函数结构体
 *
 */
struct Operator
{
	//处理函数
	void operator()(Pixel &pixel, const int *position) const
	{
		// Perform a simple threshold operation
		complicatedThreshold(pixel);
	}
};

int main()
{
	// Read image 读图
	Mat image = imread("./image/butterfly.jpg");

	// Scale image 30x 将图像扩大为30倍，长宽都变大30倍
	resize(image, image, Size(), 30, 30);

	// Print image size 打印图像尺寸
	cout << "Image size " << image.size() << endl;

	// Number of trials 测试次数
	int numTrials = 5;

	// Print number of trials 测试次数
	cout << "Number of trials : " << numTrials << endl;

	// Make two copies 图像复制
	Mat image1 = image.clone();
	Mat image2 = image.clone();
	Mat image3 = image.clone();

	// Start timer 时间函数,单位为ms
	double t;
	//开始计算时间
	tic(t);

	//循环测试numTrials次
	for (int n = 0; n < numTrials; n++)
	{
		// Naive pixel access at方法直接读取数据
		// Loop over all rows 遍历行
		for (int r = 0; r < image.rows; r++)
		{
			// Loop over all columns 遍历列
			for (int c = 0; c < image.cols; c++)
			{
				// Obtain pixel at (r, c) 直接访问像素数据
				Pixel pixel = image.at<Pixel>(r, c);
				// Apply complicatedTreshold 阈值分割
				complicatedThreshold(pixel);
				// Put result back 保存结果
				image.at<Pixel>(r, c) = pixel;
			}
		}
	}
	//计算函数执行时间
	cout << "Naive way: " << toc(t) << endl;

	// Start timer
	tic(t);

	// image1 is guaranteed to be continous, but
	// if you are curious uncomment the line below
	//需要判断图像连续存储，1表示图像连续，0不连续
	//cout << "Image 1 is continous : " << image1.isContinuous() << endl;

	//通过指针访问像素点，类似YUV图像处理，前提图像存储是连续的
	for (int n = 0; n < numTrials; n++)
	{
		// Get pointer to first pixel
		//初始指针
		Pixel *pixel = image1.ptr<Pixel>(0, 0);

		// Mat objects created using the create method are stored
		// in one continous memory block.
		// 访问像素点位置
		const Pixel *endPixel = pixel + image1.cols * image1.rows;

		// Loop over all pixels
		for (; pixel != endPixel; pixel++)
		{
			complicatedThreshold(*pixel);
		}
	}
	cout << "Pointer Arithmetic " << toc(t) << endl;

	tic(t);
	//forEach遍历像素
	for (int n = 0; n < numTrials; n++)
	{
		image2.forEach<Pixel>(Operator());
	}
	cout << "forEach : " << toc(t) << endl;

	//C++版本
	cout << __cplusplus << endl;

	//使用C++11 lambda特性
	tic(t);
	for (int n = 0; n < numTrials; n++)
	{
		// Parallel execution using C++11 lambda.
		image3.forEach<Pixel>([](Pixel &pixel, const int *position) -> void {
			complicatedThreshold(pixel);
		});
	}
	cout << "forEach C++11 : " << toc(t) << endl;

	return 0;
}