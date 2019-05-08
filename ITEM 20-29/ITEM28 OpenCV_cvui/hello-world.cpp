#include "pch.h"
#include <opencv2/opencv.hpp>
#define CVUI_IMPLEMENTATION
#include "cvui.h"

//cvui界面名字
#define WINDOW_NAME "CVUI Hello World!"

int main()
{
	cv::Mat frame = cv::Mat(200, 500, CV_8UC3);
	int count = 0;

	// Init a OpenCV window and tell cvui to use it.
	//创建cvui窗口
	cv::namedWindow(WINDOW_NAME);
	//初始化窗口
	cvui::init(WINDOW_NAME);

	//必须要用无限循环，每次变动cvui会生成新的一个图像，看起来界面变化了
	while (true)
	{
		// Fill the frame with a nice color 创建程序窗口背景图像
		frame = cv::Scalar(49, 52, 49);

		// Buttons will return true if they were clicked
		//在背景图像(110,80)点添加按钮(按钮的左上角顶点坐标，所有的cvui坐标都是左上角顶点)，按钮显示名字为“hello,world”
		//当按钮被点击时，会返回true
		if (cvui::button(frame, 110, 80, "Hello, world!"))
		{
			// The button was clicked, so let's increment our counter.
			//统计按钮被点击次数
			count++;
		}

		// Sometimes you want to show text that is not that simple, e.g. strings + numbers.
		// You can use cvui::printf for that. It accepts a variable number of parameter, pretty
		// much like printf does.
		// Let's show how many times the button has been clicked.
		//在frame(250,90)点添加一个文本框，文本框字体大小为0.5,颜色为0xff0000
		//显示的内容为"Button click count: %d", count
		cvui::printf(frame, 250, 90, 0.5, 0xff0000, "Button click count: %d", count);

		// This function must be called *AFTER* all UI components. It does
		// all the behind the scenes magic to handle mouse clicks, etc.
		//更新cvui界面
		cvui::update();

		// Show everything on the screen
		//把所有的东西显示出来
		cv::imshow(WINDOW_NAME, frame);
		// Check if ESC key was pressed
		//ESC退出循环
		if (cv::waitKey(20) == 27)
		{
			break;
		}
	}

	return 0;
}