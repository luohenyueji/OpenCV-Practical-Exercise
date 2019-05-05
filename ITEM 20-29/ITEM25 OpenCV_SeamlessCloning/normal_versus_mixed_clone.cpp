#include "pch.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
    // Read images : src image will be cloned into dst
	//Ä¿±êÍ¼Ïñ
    Mat src = imread("image/iloveyouticket.jpg");
	//±³¾°Í¼Ïñ
    Mat dst = imread("image/wood-texture.jpg");
    
    // Create an all white mask °×É«ÑÚÄ£
    Mat src_mask = 255 * Mat::ones(src.rows, src.cols, src.depth());
    
    // The location of the center of the src in the dst Í¼ÏñÖÐÐÄ
    Point center(dst.cols/2,dst.rows/2);
    
    // Seamlessly clone src into dst and put the results in output
    Mat normal_clone;
    Mat mixed_clone;
	Mat nonochrome_clone;

    seamlessClone(src, dst, src_mask, center, normal_clone, NORMAL_CLONE);
    seamlessClone(src, dst, src_mask, center, mixed_clone, MIXED_CLONE);
	seamlessClone(src, dst, src_mask, center, nonochrome_clone, MONOCHROME_TRANSFER);

    // Write results
    imwrite("opencv-normal-clone-example.jpg", normal_clone);
    imwrite("opencv-mixed-clone-example.jpg", mixed_clone);
	imwrite("opencv-nonochrome-clone-example.jpg", nonochrome_clone);

	return 0;
}
