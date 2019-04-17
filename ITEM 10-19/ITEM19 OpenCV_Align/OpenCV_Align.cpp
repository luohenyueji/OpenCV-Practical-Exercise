// OpenCV_Align.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>

#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

//最大特征点数
const int MAX_FEATURES = 500;
//好的特征点数
const float GOOD_MATCH_PERCENT = 0.15f;

/**
 * @brief 图像对齐
 *
 * @param im1 对齐图像
 * @param im2 模板图像
 * @param im1Reg 输出图像
 * @param h
 */
void alignImages(Mat &im1, Mat &im2, Mat &im1Reg, Mat &h)
{
	// Convert images to grayscale
	Mat im1Gray, im2Gray;
	//转换为灰度图
	cvtColor(im1, im1Gray, CV_BGR2GRAY);
	cvtColor(im2, im2Gray, CV_BGR2GRAY);

	// Variables to store keypoints and descriptors
	//关键点
	std::vector<KeyPoint> keypoints1, keypoints2;
	//特征描述符
	Mat descriptors1, descriptors2;

	// Detect ORB features and compute descriptors. 计算ORB特征和描述子
	Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
	orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
	orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);

	// Match features. 特征点匹配
	std::vector<DMatch> matches;
	//汉明距离进行特征点匹配
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptors1, descriptors2, matches, Mat());

	// Sort matches by score 按照特征点匹配结果从优到差排列
	std::sort(matches.begin(), matches.end());

	// Remove not so good matches 移除不好的特征点
	const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
	matches.erase(matches.begin() + numGoodMatches, matches.end());

	// Draw top matches
	Mat imMatches;
	//画出特征点匹配图
	drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
	imwrite("matches.jpg", imMatches);

	// Extract location of good matches
	std::vector<Point2f> points1, points2;

	//保存对应点
	for (size_t i = 0; i < matches.size(); i++)
	{
		//queryIdx是对齐图像的描述子和特征点的下标。
		points1.push_back(keypoints1[matches[i].queryIdx].pt);
		//queryIdx是是样本图像的描述子和特征点的下标。
		points2.push_back(keypoints2[matches[i].trainIdx].pt);
	}

	// Find homography 计算Homography，RANSAC随机抽样一致性算法
	h = findHomography(points1, points2, RANSAC);

	// Use homography to warp image 映射
	warpPerspective(im1, im1Reg, h, im2.size());
}

int main()
{
	// Read reference image 读取参考图像
	string refFilename("./image/form.jpg");
	cout << "Reading reference image : " << refFilename << endl;
	Mat imReference = imread(refFilename);

	// Read image to be aligned 读取对准图像
	string imFilename("./image/scanned-form.jpg");
	cout << "Reading image to align : " << imFilename << endl;
	Mat im = imread(imFilename);

	// Registered image will be resotred in imReg.
	// The estimated homography will be stored in h.
	//结果图像，单应性矩阵
	Mat imReg, h;

	// Align images
	cout << "Aligning images ..." << endl;
	alignImages(im, imReference, imReg, h);

	// Write aligned image to disk.
	string outFilename("aligned.jpg");
	cout << "Saving aligned image : " << outFilename << endl;
	imwrite(outFilename, imReg);

	// Print estimated homography
	cout << "Estimated homography : \n" << h << endl;
	return 0;
}