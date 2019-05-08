
#include "pch.h"
#include <opencv2/opencv.hpp>
#define CVUI_IMPLEMENTATION
#include "cvui.h"

//cvui界面名字
#define WINDOW_NAME	"CVUI Canny Edge"

int main()
{
	//读图像
	cv::Mat lena = cv::imread("lena.jpg");
	//背景图像
	cv::Mat frame = lena.clone();
	//canny阈值
	int low_threshold = 50, high_threshold = 150;
	//是否使用边缘检测
	bool use_canny = false;

	// Init a OpenCV window and tell cvui to use it.
	// If cv::namedWindow() is not used, mouse events will
	// not be captured by cvui.
	//创建cvui窗口
	cv::namedWindow(WINDOW_NAME);
	//初始化窗口
	cvui::init(WINDOW_NAME);

	while (true)
	{
		// Should we apply Canny edge?
		//是否使用边缘检测
		if (use_canny) 
		{
			// Yes, we should apply it.
			cv::cvtColor(lena, frame, CV_BGR2GRAY);
			cv::Canny(frame, frame, low_threshold, high_threshold, 3);
			cv::cvtColor(frame, frame, CV_GRAY2BGR);
		} 
		else 
		{
			// No, so just copy the original image to the displaying frame.
			//直接显示图像
			lena.copyTo(frame);
		}

		// Render the settings window to house the checkbox
		// and the trackbars below.
		//debug下可能有bug
		//主要问题在于cvui.h，void window函数问题，解决办法aOverlay = theBlock.where.clone();
		//在frame(10,50)处设置一个长宽180，180的名为Settings窗口
		cvui::window(frame, 10, 50, 180, 180, "Settings");
		
		// Checkbox to enable/disable the use of Canny edge
		//在frame(15,80)点添加复选框，复选框文本名"Use Canny Edge"，调整参数use_canny
		cvui::checkbox(frame, 15, 80, "Use Canny Edge", &use_canny);

		// Two trackbars to control the low and high threshold values
		// for the Canny edge algorithm
		//滑动条控制最低分割阈值
		//在frame(15,110)点添加滑动条，滑动条宽165，控制值low_threshold，值变化范围5到150
		cvui::trackbar(frame, 15, 110, 165, &low_threshold, 5, 150);
		//滑动条控制最高分割阈值
		cvui::trackbar(frame, 15, 180, 165, &high_threshold, 80, 300);

		// This function must be called *AFTER* all UI components. It does
		// all the behind the scenes magic to handle mouse clicks, etc.
		//更新ui界面
		cvui::update();

		// Show everything on the screen
		//把所有的东西显示出来
		cv::imshow(WINDOW_NAME, frame);

		// Check if ESC was pressed
		//ESC退出
		if (cv::waitKey(30) == 27) 
		{
			break;
		}
	}

	return 0;
}