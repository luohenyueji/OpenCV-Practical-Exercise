#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>

// Defining the dimensions of checkerboard
// 定义棋盘格的尺寸，w=6,h=9
int CHECKERBOARD[2]{ 6,9 };

int main()
{
	// Creating vector to store vectors of 3D points for each checkerboard image
	// 创建vector以存储每个棋盘图像的3D点矢量
	std::vector<std::vector<cv::Point3f> > objpoints;

	// Creating vector to store vectors of 2D points for each checkerboard image
	// 创建vector以存储每个棋盘图像的2D点矢量
	std::vector<std::vector<cv::Point2f> > imgpointsL, imgpointsR;

	// Defining the world coordinates for 3D points
	// 定义三维点的世界坐标
	std::vector<cv::Point3f> objp;
	// 初始化点，cv::Point3f(j, i, 0)保存的是x,y,z坐标。
	for (int i{ 0 }; i < CHECKERBOARD[1]; i++)
	{
		for (int j{ 0 }; j < CHECKERBOARD[0]; j++)
		{
			objp.push_back(cv::Point3f(j, i, 0));
		}
	}

	// Extracting path of individual image stored in a given directory
	// 提取存储在给定目录中的单个图像的路径
	std::vector<cv::String> imagesL, imagesR;
	// Path of the folder containing checkerboard images
	// 包含棋盘图像的文件夹的路径
	// pathL和pathR分别为两个摄像头在多个时刻拍摄的图像
	std::string pathL = "./data/stereoL/*.png";
	std::string pathR = "./data/stereoR/*.png";

	// 提取两个文件夹 所有的图像
	cv::glob(pathL, imagesL);
	cv::glob(pathR, imagesR);

	cv::Mat frameL, frameR, grayL, grayR;
	// vector to store the pixel coordinates of detected checker board corners
	// 用于存储检测到的棋盘角点的像素坐标的向量
	std::vector<cv::Point2f> corner_ptsL, corner_ptsR;
	bool successL, successR;

	// Looping over all the images in the directory
	// 遍历目录中的所有图像
	for (int i{ 0 }; i < imagesL.size(); i++)
	{
		// 提取同一时刻分别用两个摄像头拍到的照片
		frameL = cv::imread(imagesL[i]);
		cv::cvtColor(frameL, grayL, cv::COLOR_BGR2GRAY);

		frameR = cv::imread(imagesR[i]);
		cv::cvtColor(frameR, grayR, cv::COLOR_BGR2GRAY);

		// Finding checker board corners
		// 寻找棋盘图的内角点位置
		// If desired number of corners are found in the image then success = true
		// 如果在图像中找到所需的角数，则success=true
		// 具体函数使用介绍见https://blog.csdn.net/h532600610/article/details/51800488
		successL = cv::findChessboardCorners(
			grayL,
			cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]),
			corner_ptsL);
		// cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

		successR = cv::findChessboardCorners(
			grayR,
			cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]),
			corner_ptsR);
		// cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

		// 如果检测到所需的角点个数，则细化像素坐标并将其显示在棋盘格图像上
		if ((successL) && (successR))
		{
			// TermCriteria定义迭代算法终止条件的类
			// 具体使用见 https://www.jianshu.com/p/548868c4d34e
			cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

			// refining pixel coordinates for given 2d points.
			// 细化给定2D点的像素坐标，cornerSubPix用于亚像素角点检测
			// 具体参数见https://blog.csdn.net/guduruyu/article/details/69537083
			cv::cornerSubPix(grayL, corner_ptsL, cv::Size(11, 11), cv::Size(-1, -1), criteria);
			cv::cornerSubPix(grayR, corner_ptsR, cv::Size(11, 11), cv::Size(-1, -1), criteria);

			// Displaying the detected corner points on the checker board
			// drawChessboardCorners用于绘制棋盘格角点的函数
			cv::drawChessboardCorners(frameL, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_ptsL, successL);
			cv::drawChessboardCorners(frameR, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_ptsR, successR);

			// 保存数据以供后续使用
			// 保存三维点数据
			objpoints.push_back(objp);
			// 保存角点信息
			imgpointsL.push_back(corner_ptsL);
			imgpointsR.push_back(corner_ptsR);
		}

		//cv::imshow("ImageL", frameL);
		//cv::imshow("ImageR", frameR);
		//cv::waitKey(0);
	}

	// 关闭所有窗口
	cv::destroyAllWindows();

	// 通过传递已知三维点（objpoints）的值和检测到的角点（imgpoints）的相应像素坐标来执行相机校准
	// mtxL,mtxR为内参数矩阵， distL和distR为畸变矩阵
	// R_L和R_R为旋转向量，T_L和T_R为位移向量
	cv::Mat mtxL, distL, R_L, T_L;
	cv::Mat mtxR, distR, R_R, T_R;

	cv::Mat new_mtxL, new_mtxR;

	// Calibrating left camera
	// 校正左边相机

	// 相机标定函数
	// 函数使用见https://blog.csdn.net/u011574296/article/details/73823569
	cv::calibrateCamera(objpoints,
		imgpointsL,
		grayL.size(),
		mtxL,
		distL,
		R_L,
		T_L);

	// 去畸变，优化相机内参，这一步可选
	// getOptimalNewCameraMatrix函数使用见https://www.jianshu.com/p/df78749b4318
	new_mtxL = cv::getOptimalNewCameraMatrix(mtxL,
		distL,
		grayL.size(),
		1,
		grayL.size(),
		0);

	// Calibrating right camera
	// 校正右边相机
	cv::calibrateCamera(objpoints,
		imgpointsR,
		grayR.size(),
		mtxR,
		distR,
		R_R,
		T_R);

	new_mtxR = cv::getOptimalNewCameraMatrix(mtxR,
		distR,
		grayR.size(),
		1,
		grayR.size(),
		0);

	// Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat
	// are calculated. Hence intrinsic parameters are the same.
	// 在这里，我们修正了固有的camara矩阵，以便只计算Rot、Trns、Emat和Fmat。因此内在参数是相同的。
	cv::Mat Rot, Trns, Emat, Fmat;

	int flag = 0;
	flag |= cv::CALIB_FIX_INTRINSIC;

	// This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
	// 同时标定两个摄像头，函数介绍见https://www.cnblogs.com/zyly/p/9373991.html
	cv::stereoCalibrate(objpoints,
		imgpointsL,
		imgpointsR,
		new_mtxL,
		distL,
		new_mtxR,
		distR,
		grayR.size(),
		Rot,
		Trns,
		Emat,
		Fmat,
		flag,
		cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 1e-6));

	cv::Mat rect_l, rect_r, proj_mat_l, proj_mat_r, Q;

	// Once we know the transformation between the two cameras we can perform stereo rectification
	// 一旦我们知道两个摄像机之间的变换，我们就可以进行立体校正
	// stereoRectify同时校正两个摄像机，函数介绍见https://www.cnblogs.com/zyly/p/9373991.html
	cv::stereoRectify(new_mtxL,
		distL,
		new_mtxR,
		distR,
		grayR.size(),
		Rot,
		Trns,
		rect_l,
		rect_r,
		proj_mat_l,
		proj_mat_r,
		Q,
		1);

	// Use the rotation matrixes for stereo rectification and camera intrinsics for undistorting the image
	// Compute the rectification map (mapping between the original image pixels and
	// their transformed values after applying rectification and undistortion) for left and right camera frames
	// 根据相机单目标定得到的内参以及stereoRectify计算出来的值来计算畸变矫正和立体校正的映射变换矩阵
	cv::Mat Left_Stereo_Map1, Left_Stereo_Map2;
	cv::Mat Right_Stereo_Map1, Right_Stereo_Map2;

	// 函数介绍见https://www.cnblogs.com/zyly/p/9373991.html
	cv::initUndistortRectifyMap(new_mtxL,
		distL,
		rect_l,
		proj_mat_l,
		grayR.size(),
		CV_16SC2,
		Left_Stereo_Map1,
		Left_Stereo_Map2);

	cv::initUndistortRectifyMap(new_mtxR,
		distR,
		rect_r,
		proj_mat_r,
		grayR.size(),
		CV_16SC2,
		Right_Stereo_Map1,
		Right_Stereo_Map2);

	// 保存校正信息
	cv::FileStorage cv_file = cv::FileStorage("data/params_cpp.xml", cv::FileStorage::WRITE);
	cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map1);
	cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map2);
	cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map1);
	cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map2);
	cv_file.release();
	return 0;
}