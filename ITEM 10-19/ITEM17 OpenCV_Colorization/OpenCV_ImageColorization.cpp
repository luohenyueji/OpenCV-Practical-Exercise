// OpenCV_Colorization.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// the 313 ab cluster centers from pts_in_hull.npy (already transposed)
static float hull_pts[] = 
{
	-90., -90., -90., -90., -90., -80., -80., -80., -80., -80., -80., -80., -80., -70., -70., -70., -70., -70., -70., -70., -70.,
	-70., -70., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -50., -50., -50., -50., -50., -50., -50., -50.,
	-50., -50., -50., -50., -50., -50., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -30.,
	-30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -20., -20., -20., -20., -20., -20., -20.,
	-20., -20., -20., -20., -20., -20., -20., -20., -20., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10.,
	-10., -10., -10., -10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 10., 10., 10., 10., 10., 10.,
	10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20.,
	20., 20., 20., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 40., 40., 40., 40.,
	40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50.,
	50., 50., 50., 50., 50., 50., 50., 50., 50., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60.,
	60., 60., 60., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 80., 80., 80.,
	80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 90., 90., 90., 90., 90., 90., 90., 90., 90., 90.,
	90., 90., 90., 90., 90., 90., 90., 90., 90., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 50., 60., 70., 80., 90.,
	20., 30., 40., 50., 60., 70., 80., 90., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -20., -10., 0., 10., 20., 30., 40., 50.,
	60., 70., 80., 90., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -40., -30., -20., -10., 0., 10., 20.,
	30., 40., 50., 60., 70., 80., 90., 100., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -50.,
	-40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -60., -50., -40., -30., -20., -10., 0., 10., 20.,
	30., 40., 50., 60., 70., 80., 90., 100., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90.,
	100., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -80., -70., -60., -50.,
	-40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -90., -80., -70., -60., -50., -40., -30., -20., -10.,
	0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30.,
	40., 50., 60., 70., 80., 90., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70.,
	80., -110., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., -110., -100.,
	-90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., -110., -100., -90., -80., -70.,
	-60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., -110., -100., -90., -80., -70., -60., -50., -40., -30.,
	-20., -10., 0., 10., 20., 30., 40., 50., 60., 70., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0.
};

int main()
{

	string imageFileName = "./image/greyscaleImage.png";
	Mat img = imread(imageFileName);
	if (img.empty())
	{
		cout << "Can't read image from file: " << imageFileName << endl;
		return 2;
	}

	string protoFile = "./model/colorization_deploy_v2.prototxt";
	string weightsFile = "./model/colorization_release_v2.caffemodel";
	//string weightsFile = "./model/colorization_release_v2_norebal.caffemodel";

	double t = (double)cv::getTickCount();

	// fixed input size for the pretrained network 加载模型
	const int W_in = 224;
	const int H_in = 224;
	Net net = dnn::readNetFromCaffe(protoFile, weightsFile);

	// setup additional layers 在网络里额外添加两层
	int sz[] = { 2, 313, 1, 1 };
	//添加一个ab转换层
	const Mat pts_in_hull(4, sz, CV_32F, hull_pts);
	Ptr<dnn::Layer> class8_ab = net.getLayer("class8_ab");
	class8_ab->blobs.push_back(pts_in_hull);
	//一个防止为输出为0的层
	Ptr<dnn::Layer> conv8_313_rh = net.getLayer("conv8_313_rh");
	conv8_313_rh->blobs.push_back(Mat(1, 313, CV_32F, Scalar(2.606)));

	// extract L channel and subtract mean 将原图转换为Lab空间图，并归一化
	Mat lab, L, input;
	img.convertTo(img, CV_32F, 1.0 / 255);
	cvtColor(img, lab, COLOR_BGR2Lab);
	//提取亮度信息
	extractChannel(lab, L, 0);
	//重置大小
	resize(L, input, Size(W_in, H_in));
	input -= 50;

	// run the L channel through the network 获得网络输出结果
	Mat inputBlob = blobFromImage(input);
	net.setInput(inputBlob);
	Mat result = net.forward();

	// retrieve the calculated a,b channels from the network output
	Size siz(result.size[2], result.size[3]);
	//输出为56X56
	Mat a = Mat(siz, CV_32F, result.ptr(0, 0));
	Mat b = Mat(siz, CV_32F, result.ptr(0, 1));
	//重置大小
	resize(a, a, img.size());
	resize(b, b, img.size());

	// merge, and convert back to BGR 合并lab，转换为RGB图像
	Mat color, chn[] = { L, a, b };
	merge(chn, 3, lab);
	cvtColor(lab, color, COLOR_Lab2BGR);

	//计算时间
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	cout << "Time taken : " << t << " secs" << endl;

	//保存图像
	string str = imageFileName;
	str.replace(str.end() - 4, str.end(), "");
	str = str + "_colorized.png";

	//反归一化
	color = color * 255;
	color.convertTo(color, CV_8U);
	imwrite(str, color);

	cout << "Colorized image saved as " << str << endl;

	return 0;
}
