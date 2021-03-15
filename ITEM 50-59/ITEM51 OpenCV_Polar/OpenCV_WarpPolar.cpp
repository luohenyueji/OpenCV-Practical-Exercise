#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	// log_polar_img 半对数极坐标变换结果
	// lin_polar_img 极坐标变换结果
	// recovered_log_polar 半对数极坐标逆变换结果
	// recovered_lin_polar_img 极坐标逆变换结果
	Mat log_polar_img, lin_polar_img, recovered_log_polar, recovered_lin_polar_img;
	// INTER_LINEAR 双线性插值，WARP_FILL_OUTLIERS填充所有目标图像像素
	int flags = INTER_LINEAR + WARP_FILL_OUTLIERS;

	// 读图
	String imagepath = "image/clock.jpg";
	Mat src = imread(imagepath);
	if (src.empty())
	{
		fprintf(stderr, "Could not initialize capturing...\n");
		return -1;
	}

	// 圆心坐标
	Point2f center((float)src.cols / 2, (float)src.rows / 2);
	// 圆的半径
	double maxRadius = min(center.y, center.x);

	// direct transform
	// linear Polar 极坐标变换, Size()表示OpenCV根据输入自行决定输出图像尺寸
	warpPolar(src, lin_polar_img, Size(), center, maxRadius, flags);
	// semilog Polar 半对数极坐标变换, Size()表示OpenCV根据输入自行决定输出图像尺寸
	warpPolar(src, log_polar_img, Size(), center, maxRadius, flags + WARP_POLAR_LOG);
	// inverse transform 逆变换
	warpPolar(lin_polar_img, recovered_lin_polar_img, src.size(), center, maxRadius, flags + WARP_INVERSE_MAP);
	warpPolar(log_polar_img, recovered_log_polar, src.size(), center, maxRadius, flags + WARP_POLAR_LOG + WARP_INVERSE_MAP);

	// 改变结果方向
	// rotate(lin_polar_img, lin_polar_img, ROTATE_90_CLOCKWISE);

	// 展示图片
	imshow("Src frame", src);
	imshow("Log-Polar", log_polar_img);
	imshow("Linear-Polar", lin_polar_img);
	imshow("Recovered Linear-Polar", recovered_lin_polar_img);
	imshow("Recovered Log-Polar", recovered_log_polar);
	waitKey(0);
	system("pause");
	return 0;
}