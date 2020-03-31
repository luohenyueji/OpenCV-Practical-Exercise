// This code is written by Sunita Nayak at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html
// 虚拟现实

#include "pch.h"
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

using namespace cv;
using namespace aruco;
using namespace std;

int main()
{
	// Open a video file or an image file or a camera stream.
	string str, outputFile;
	VideoCapture cap;
	VideoWriter video;
	Mat frame, blob;

	// 新场景图像
	Mat im_src = imread("./image/new_scenery.jpg");
	// 检测类型
	String detectType = "video";
	String detectPath = "./video/test.mp4";

	try
	{
		// 输出文件名
		outputFile = "ar_out_cpp.avi";
		// 如果检测类型是图像
		if (detectType == "image")
		{
			// Open the image file
			str = detectPath;
			// 判断文件是否存在
			ifstream ifile(str);
			if (!ifile)
			{
				throw("error");
			}
			cap.open(str);
			// 重命名
			str.replace(str.end() - 4, str.end(), "_ar_out_cpp.jpg");
			// 输出文件
			outputFile = str;
		}
		// 如果检测类型是视频
		else if (detectType == "video")
		{
			// Open the video file
			// 打开视频
			str = detectPath;
			ifstream ifile(str);
			if (!ifile)
			{
				throw("error");
			}
			cap.open(str);
			str.replace(str.end() - 4, str.end(), "_ar_out_cpp.avi");
			outputFile = str;
		}
		// Open the webcaom
		// 打开网络摄像头
		else
			cap.open(0);
	}
	// 错误解决办法
	catch (...)
	{
		cout << "Could not open the input image/video stream" << endl;
		return 0;
	}

	// Get the video writer initialized to save the output video
	// 如果检测类别不是图像，则生成输出视频
	if (detectType != "image")
	{
		video.open(outputFile, VideoWriter::fourcc('M', 'J', 'P', 'G'), 28, Size(2 * cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
	}

	// Create a window
	// 创建显示窗口
	static const string kWinName = "Augmented Reality using Aruco markers in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);

	// Process frames.
	// 逐帧处理
	while (waitKey(1) < 0)
	{
		// get frame from the video
		cap >> frame;

		try
		{
			// Stop the program if reached end of video
			// 如果到了视频的结尾
			if (frame.empty())
			{
				cout << "Done processing !!!" << endl;
				cout << "Output file is stored as " << outputFile << endl;
				waitKey(3000);
				break;
			}

			vector<int> markerIds;

			// Load the dictionary that was used to generate the markers.
			// 加载用于标记的词典
			Ptr<Dictionary> dictionary = getPredefinedDictionary(DICT_6X6_250);

			// Declare the vectors that would contain the detected marker corners and the rejected marker candidates
			// 声明标记到的角点和没有被标记到的角点
			vector<vector<Point2f>> markerCorners, rejectedCandidates;

			// Initialize the detector parameters using default values
			// 使用默认值初始化检测器参数
			Ptr<DetectorParameters> parameters = DetectorParameters::create();

			// Detect the markers in the image
			// 检测标记
			/**
			*	frame 待检测marker的图像
			*	dictionary 字典对象
			*	markerCorners 检测出的图像的角的列表，从左下角顺时针开始，返回角的各个顶点的坐标
			*	markerIds markerCorners检测出的maker的id列表
			*	parameters 检测器参数
			*	rejectedCandidates 返回不是有效的角相关信息
			*/
			detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

			// Using the detected markers, locate the quadrilateral on the target frame where the new scene is going to be displayed.、
			// 使用检测到的标记，在目标帧上定位要显示新场景的四边形。
			vector<Point> pts_dst;
			// 0.015;
			// 计算缩减距离
			float scalingFac = 0.02;

			Point refPt1, refPt2, refPt3, refPt4;

			// finding top left corner point of the target quadrilateral
			// 寻找目标四边形的左上角点

			// 查找字典中id为25的标志，返回一个vector
			std::vector<int>::iterator it = std::find(markerIds.begin(), markerIds.end(), 25);
			// 返回markerIds中25的下标
			int index = std::distance(markerIds.begin(), it);
			// 返回markerIds中25的左上角坐标
			refPt1 = markerCorners.at(index).at(1);

			// finding top right corner point of the target quadrilateral
			// 求目标四边形的右上角点

			// 查找字典中id为33的标志，返回一个vector
			it = std::find(markerIds.begin(), markerIds.end(), 33);
			// 返回markerIds中33的下标
			index = std::distance(markerIds.begin(), it);
			// 返回markerIds中33的右上角坐标
			refPt2 = markerCorners.at(index).at(2);

			// 返回欧式距离
			float distance = norm(refPt1 - refPt2);
			// 将缩减后的坐标放入标记点容器
			pts_dst.push_back(Point(refPt1.x - round(scalingFac * distance), refPt1.y - round(scalingFac * distance)));

			pts_dst.push_back(Point(refPt2.x + round(scalingFac * distance), refPt2.y - round(scalingFac * distance)));

			// finding bottom right corner point of the target quadrilateral
			// 求目标四边形的右下角点
			it = std::find(markerIds.begin(), markerIds.end(), 30);
			index = std::distance(markerIds.begin(), it);
			refPt3 = markerCorners.at(index).at(0);
			pts_dst.push_back(Point(refPt3.x + round(scalingFac * distance), refPt3.y + round(scalingFac * distance)));

			// finding bottom left corner point of the target quadrilateral
			// 寻找目标四边形的左下角点
			it = std::find(markerIds.begin(), markerIds.end(), 23);
			index = std::distance(markerIds.begin(), it);
			refPt4 = markerCorners.at(index).at(0);
			pts_dst.push_back(Point(refPt4.x - round(scalingFac * distance), refPt4.y + round(scalingFac * distance)));

			// Get the corner points of the new scene image.
			// 全新图像的角点
			vector<Point> pts_src;
			// 从左上角开始顺时针存入pts_src中
			pts_src.push_back(Point(0, 0));
			pts_src.push_back(Point(im_src.cols, 0));
			pts_src.push_back(Point(im_src.cols, im_src.rows));
			pts_src.push_back(Point(0, im_src.rows));

			// Compute homography from source and destination points
			// 计算单应性矩阵
			Mat h = cv::findHomography(pts_src, pts_dst);

			// Warped image
			// 仿射变换后的图像
			Mat warpedImage;

			// Warp source image to destination based on homography
			// 基于单应性矩阵映射图像
			warpPerspective(im_src, warpedImage, h, frame.size(), INTER_CUBIC);

			// Prepare a mask representing region to copy from the warped image into the original frame.
			// 准备一个表示要从仿射图像图像复制到原始帧中的区域的遮罩。
			Mat mask = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
			// 计算单应性矩阵
			fillConvexPoly(mask, pts_dst, Scalar(255, 255, 255), LINE_AA);

			// Erode the mask to not copy the boundary effects from the warping
			// 侵蚀mask以不复制仿射图像的边界效果
			Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
			// Mat element = getStructuringElement( MORPH_RECT, Size(3,3));
			erode(mask, mask, element);

			// Copy the warped image into the original frame in the mask region.
			// 将仿射的图像复制到遮罩区域中的原始帧中。
			Mat imOut = frame.clone();
			warpedImage.copyTo(imOut, mask);

			// Showing the original image and the new output image side by side
			Mat concatenatedOutput;
			// 并排显示原始图像和新输出图像
			hconcat(frame, imOut, concatenatedOutput);

			// 保存图像
			if (detectType == "image")
			{
				imwrite(outputFile, concatenatedOutput);
			}
			// 写视频
			else
			{
				video.write(concatenatedOutput);
			}
			imshow(kWinName, concatenatedOutput);
		}
		// 输出错误
		catch (const std::exception &e)
		{
			cout << endl
				<< " e : " << e.what() << endl;
			cout << "Could not do homography !! " << endl;
			// return 0;
		}
	}

	cap.release();
	video.release();

	return 0;
}