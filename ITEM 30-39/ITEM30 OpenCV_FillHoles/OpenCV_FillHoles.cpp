#include "pch.h"
#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    // Read image 读取图像
    Mat im_in = imread("./image/nickel.jpg", IMREAD_GRAYSCALE);

  
    // Threshold.
    // Set values equal to or above 220 to 0.
    // Set values below 220 to 255.
	//阈值分割
    Mat im_th;
    threshold(im_in, im_th, 220, 255, THRESH_BINARY_INV);
    
    // Floodfill from point (0, 0) 以点(0,0)为种子点，进行漫水填充
    Mat im_floodfill = im_th.clone();
    floodFill(im_floodfill, cv::Point(0,0), Scalar(255));
    
    // Invert floodfilled image 反转图像
    Mat im_floodfill_inv;
    bitwise_not(im_floodfill, im_floodfill_inv);
    
    // Combine the two images to get the foreground. 获得前景
    Mat im_out = (im_th | im_floodfill_inv);

    // Display images 图像展示
    imshow("Thresholded Image", im_th);
    imshow("Floodfilled Image", im_floodfill);
    imshow("Inverted Floodfilled Image", im_floodfill_inv);
    imshow("Foreground", im_out);
    waitKey(0);
	return 0;
}