// Mask R-CNN in OpenCV.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>

#include <fstream>
#include <sstream>
#include <iostream>
#include <string.h>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

// Initialize the parameters
// Confidence threshold 置信度阈值
float confThreshold = 0.5;
// Mask threshold 掩模阈值
float maskThreshold = 0.3;

vector<string> classes;
vector<Scalar> colors;

// Draw the predicted bounding box
void drawBox(Mat &frame, int classId, float conf, Rect box, Mat &objectMask);

// Postprocess the neural network's output for each frame
void postprocess(Mat &frame, const vector<Mat> &outs);

int main()
{
	//0-image,1-video,2-camera
	int read_file = 0;
	// Load names of classes 导入分类名文件
	string classesFile = "./model/mscoco_labels.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line))
	{
		classes.push_back(line);
	}

	// Load the colors 导入颜色类文件
	string colorsFile = "./model/colors.txt";
	ifstream colorFptr(colorsFile.c_str());
	while (getline(colorFptr, line))
	{
		char *pEnd;
		double r, g, b;
		//字符串转换成浮点数
		r = strtod(line.c_str(), &pEnd);
		g = strtod(pEnd, NULL);
		b = strtod(pEnd, NULL);
		Scalar color = Scalar(r, g, b, 255.0);
		colors.push_back(Scalar(r, g, b, 255.0));
	}

	// Give the configuration and weight files for the model
	String textGraph = "./model/mask_rcnn_inception_v2_coco.pbtxt";
	String modelWeights = "./model/mask_rcnn_inception_v2_coco.pb";

	// Load the network 导入网络
	Net net = readNetFromTensorflow(modelWeights, textGraph);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	//只使用CPU
	net.setPreferableTarget(DNN_TARGET_CPU);

	// Open a video file or an image file or a camera stream.
	string str, outputFile;
	VideoCapture cap;
	VideoWriter video;
	Mat frame, blob;

	try
	{
		//输出文件，默认是视频
		outputFile = "mask_rcnn_out_cpp.avi";
		if (read_file == 0)
		{
			// Open the image file 打开图像文件
			str = "image/cars.jpg";
			//cout << "Image file input : " << str << endl;
			ifstream ifile(str);
			if (!ifile)
			{
				throw("error");
			}
			frame = imread(str);
			str.replace(str.end() - 4, str.end(), "_mask_rcnn_out.jpg");
			outputFile = str;
		}
		else if (read_file == 1)
		{
			// Open the video file 打开视频文件
			str = "./image/cars.mp4";
			ifstream ifile(str);
			if (!ifile)
			{
				throw("error");
			}
			cap.open(str);
			str.replace(str.end() - 4, str.end(), "_mask_rcnn_out.avi");
			outputFile = str;
		}
		// Open the webcam 打开摄像头
		else
		{
			cap.open(0);
		}
	}
	catch (...)
	{
		cout << "Could not open the input image/video stream" << endl;
		return 0;
	}

	// Get the video writer initialized to save the output video 如果读入的不是图像，生成输出视频
	if (read_file != 0)
	{
		video.open(outputFile, VideoWriter::fourcc('M', 'J', 'P', 'G'), 28,
				   Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
	}

	// Create a window 显示窗口
	static const string kWinName = "Deep learning object detection in OpenCV";

	//Process frames 处理图像
	while (waitKey(1) < 0)
	{
		//如果是视频
		if (read_file != 0)
		{
			// get frame from the video 获取单帧图像
			cap >> frame;
		}

		// Stop the program if reached end of video 如果图像不存在
		if (frame.empty())
		{
			cout << "Done processing !!!" << endl;
			cout << "Output file is stored as " << outputFile << endl;
			waitKey(0);
			break;
		}

		// Create a 4D blob from a frame 获得深度学习的输入图像
		blobFromImage(frame, blob, 1.0, Size(frame.cols, frame.rows), Scalar(), true, false);
		//blobFromImage(frame, blob);

		//Sets the input to the network 设置输入
		net.setInput(blob);

		// Runs the forward pass to get output from the output layers 获得输出层
		std::vector<String> outNames(2);
		outNames[0] = "detection_out_final";
		outNames[1] = "detection_masks";
		vector<Mat> outs;
		net.forward(outs, outNames);

		// Extract the bounding box and mask for each of the detected objects 提取预测框和掩模
		postprocess(frame, outs);

		// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		string label = format("Mask-RCNN Inference time for a frame : %0.0f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

		// Write the frame with the detection boxes 保存结果
		Mat detectedFrame;
		frame.convertTo(detectedFrame, CV_8U);
		namedWindow(kWinName, WINDOW_NORMAL);
		imshow(kWinName, frame);
		//enter退出
		if (waitKey(1000) == 27)
		{
			break;
		}
		if (read_file == 0)
		{
			imwrite(outputFile, detectedFrame);
			break;
		}
		else
		{
			video.write(detectedFrame);
		}
	}

	cap.release();
	//释放生成的视频
	if (read_file != 0)
	{
		video.release();
	}

	return 0;
}


/**
 * @brief For each frame, extract the bounding box and mask for each detected object 提取每张图像的预测框和掩模
 * 
 * @param frame 
 * @param outs 
 */
void postprocess(Mat &frame, const vector<Mat> &outs)
{
	//预测框结果
	Mat outDetections = outs[0];
	//掩模结果
	Mat outMasks = outs[1];

	// Output size of masks is NxCxHxW where
	// N - number of detected boxes
	// C - number of classes (excluding background)
	// HxW - segmentation shape
	//预测的框个数
	const int numDetections = outDetections.size[2];
	//类别数
	const int numClasses = outMasks.size[1];

	outDetections = outDetections.reshape(1, outDetections.total() / 7);
	//筛选预测框数
	for (int i = 0; i < numDetections; ++i)
	{
		//提取预测框置信度
		float score = outDetections.at<float>(i, 2);
		//超过阈值
		if (score > confThreshold)
		{
			// Extract the bounding box
			//类别
			int classId = static_cast<int>(outDetections.at<float>(i, 1));
			int left = static_cast<int>(frame.cols * outDetections.at<float>(i, 3));
			int top = static_cast<int>(frame.rows * outDetections.at<float>(i, 4));
			int right = static_cast<int>(frame.cols * outDetections.at<float>(i, 5));
			int bottom = static_cast<int>(frame.rows * outDetections.at<float>(i, 6));

			//防止框画在外面
			left = max(0, min(left, frame.cols - 1));
			top = max(0, min(top, frame.rows - 1));
			right = max(0, min(right, frame.cols - 1));
			bottom = max(0, min(bottom, frame.rows - 1));
			Rect box = Rect(left, top, right - left + 1, bottom - top + 1);

			// Extract the mask for the object 提取掩模
			Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i, classId));

			// Draw bounding box, colorize and show the mask on the image
			drawBox(frame, classId, score, box, objectMask);
		}
	}
}


/**
 * @brief  Draw the predicted bounding box, colorize and show the mask on the image 画图
 * 
 * @param frame 
 * @param classId 
 * @param conf 
 * @param box 
 * @param objectMask 
 */
void drawBox(Mat &frame, int classId, float conf, Rect box, Mat &objectMask)
{
	//Draw a rectangle displaying the bounding box 画预测框
	rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(255, 178, 50), 3);

	//Get the label for the class name and its confidence
	//置信度获取
	string label = format("%.2f", conf);
	//获取标签
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	//获取字符串的高度和宽度
	//标签，字体，文本大小的倍数，文本粗细，文本最低点对应的纵坐标
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	box.y = max(box.y, labelSize.height);
	//画框打标签
	rectangle(frame, Point(box.x, box.y - round(1.5 * labelSize.height)), Point(box.x + round(1.5 * labelSize.width), box.y + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
	//填充颜色
	Scalar color = colors[classId % colors.size()];

	// Resize the mask, threshold, color and apply it on the image 重置大小
	resize(objectMask, objectMask, Size(box.width, box.height));
	Mat mask = (objectMask > maskThreshold);
	//叠加获得颜色掩模
	Mat coloredRoi = (0.3 * color + 0.7 * frame(box));
	coloredRoi.convertTo(coloredRoi, CV_8UC3);

	// Draw the contours on the image 画轮廓
	vector<Mat> contours;
	Mat hierarchy;
	mask.convertTo(mask, CV_8U);
	findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
	drawContours(coloredRoi, contours, -1, color, 5, LINE_8, hierarchy, 100);
	coloredRoi.copyTo(frame(box), mask);
}