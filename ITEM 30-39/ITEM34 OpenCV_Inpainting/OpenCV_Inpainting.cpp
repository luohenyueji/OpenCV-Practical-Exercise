#include "pch.h"
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>

#include <iostream>

using namespace cv;
using namespace std;

// Declare Mat objects for original image and mask for inpainting
Mat img, inpaintMask;
// Mat object for result output
Mat res;
Point prevPt(-1, -1);

// onMouse function for Mouse Handling
// Used to draw regions required to inpaint
// 调用鼠标事件
static void onMouse(int event, int x, int y, int flags, void*)
{
	if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON))
		prevPt = Point(-1, -1);
	else if (event == EVENT_LBUTTONDOWN)
		prevPt = Point(x, y);
	else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
	{
		Point pt(x, y);
		if (prevPt.x < 0)
			prevPt = pt;
		line(inpaintMask, prevPt, pt, Scalar::all(255), 5, 8, 0);
		line(img, prevPt, pt, Scalar::all(255), 5, 8, 0);
		prevPt = pt;
		imshow("image", img);
		imshow("image: mask", inpaintMask);
	}
}

int main()
{
	string filename = "./image/flower-garden.jpg";
	// Read image in color mode 读图
	img = imread(filename);
	Mat img_mask;
	// Return error if image not read properly
	if (img.empty())
	{
		cout << "Failed to load image: " << filename << endl;
		return 0;
	}

	namedWindow("image");

	// Create a copy for the original image 复制原图像
	img_mask = img.clone();
	// Initialize mask (black image)
	inpaintMask = Mat::zeros(img_mask.size(), CV_8U);

	// Show the original image
	imshow("image", img);
	//调用鼠标在图像上画圈
	setMouseCallback("image", onMouse, NULL);

	for (;;)
	{
		char c = (char)waitKey();
		//按t选择INPAINT_TELEA处理
		if (c == 't')
		{
			// Use Algorithm proposed by Alexendra Telea
			inpaint(img, inpaintMask, res, 3, INPAINT_TELEA);
			imshow("Inpaint Output using FMM", res);
		}
		//按n选择INPAINT_NS处理
		if (c == 'n')
		{
			// Use Algorithm proposed by Bertalmio et. al.
			inpaint(img, inpaintMask, res, 3, INPAINT_NS);
			imshow("Inpaint Output using NS Technique", res);
		}
		//按r查看原图
		if (c == 'r')
		{
			inpaintMask = Scalar::all(0);
			img_mask.copyTo(img);
			imshow("image", inpaintMask);
		}
		//按ESC退出
		if (c == 27)
		{
			break;
		}
	}
	return 0;
}