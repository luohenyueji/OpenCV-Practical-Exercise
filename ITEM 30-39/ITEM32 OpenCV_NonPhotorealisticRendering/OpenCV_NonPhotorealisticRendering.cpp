#include "pch.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
	// Read image ¶ÁÈ¡Í¼Ïñ
	Mat im = imread("./image/cow.jpg");
	Mat imout, imout_gray;

	// Edge preserving filter with two different flags. ±£±ßÂË²¨Æ÷
	edgePreservingFilter(im, imout, RECURS_FILTER);
	imwrite("edge-preserving-recursive-filter.jpg", imout);

	edgePreservingFilter(im, imout, NORMCONV_FILTER);
	imwrite("edge-preserving-normalized-convolution-filter.jpg", imout);

	// Detail enhance filter ±ßÔµÔöÇ¿ÂË²¨Æ÷
	detailEnhance(im, imout);
	imwrite("detail-enhance.jpg", imout);

	// Pencil sketch filter ËØÃèÂË²¨Æ÷
	pencilSketch(im, imout_gray, imout);
	imwrite("pencil-sketch.jpg", imout_gray);
	imwrite("pencil-sketch-color.jpg", imout_gray);

	// Stylization filter ·ç¸ñ»¯ÂË²¨Æ÷
	stylization(im, imout);
	imwrite("stylization.jpg", imout);

	return 0;
}