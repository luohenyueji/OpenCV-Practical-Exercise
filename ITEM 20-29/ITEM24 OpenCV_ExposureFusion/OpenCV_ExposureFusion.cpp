#include "pch.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
using namespace cv;
using namespace std;

// Read Images
void readImages(vector<Mat> &images)
{
	int numImages = 16;
	static const char* filenames[] =
	{
	  "image/memorial0061.jpg",
	  "image/memorial0062.jpg",
	  "image/memorial0063.jpg",
	  "image/memorial0064.jpg",
	  "image/memorial0065.jpg",
	  "image/memorial0066.jpg",
	  "image/memorial0067.jpg",
	  "image/memorial0068.jpg",
	  "image/memorial0069.jpg",
	  "image/memorial0070.jpg",
	  "image/memorial0071.jpg",
	  "image/memorial0072.jpg",
	  "image/memorial0073.jpg",
	  "image/memorial0074.jpg",
	  "image/memorial0075.jpg",
	  "image/memorial0076.jpg"
	};
	//读图
	for (int i = 0; i < numImages; i++)
	{
		Mat im = imread(filenames[i]);
		images.push_back(im);
	}
}

int main()
{
	// Read images 读取图像
	cout << "Reading images ... " << endl;
	vector<Mat> images;

	//是否图像映射
	bool needsAlignment = true;

	// Read example images 读取例子图像
	readImages(images);
	//needsAlignment = false;

	// Align input images
	if (needsAlignment)
	{
		cout << "Aligning images ... " << endl;
		Ptr<AlignMTB> alignMTB = createAlignMTB();
		alignMTB->process(images, images);
	}
	else
	{
		cout << "Skipping alignment ... " << endl;
	}

	// Merge using Exposure Fusion 图像融合
	cout << "Merging using Exposure Fusion ... " << endl;
	Mat exposureFusion;
	Ptr<MergeMertens> mergeMertens = createMergeMertens();
	mergeMertens->process(images, exposureFusion);

	// Save output image 图像保存
	cout << "Saving output ... exposure-fusion.jpg" << endl;
	imwrite("exposure-fusion.jpg", exposureFusion * 255);

	return 0;
}