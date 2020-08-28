#include <opencv2/opencv.hpp>
#include <opencv2/img_hash.hpp>

#include <iostream>

using namespace cv;
using namespace cv::img_hash;
using namespace std;

template <typename T>
inline void test_one(const std::string &title, const Mat &a, const Mat &b)
{
	cout << "=== " << title << " ===" << endl;
	TickMeter tick;
	Mat hashA, hashB;
	// 模板方便重复利用
	Ptr<ImgHashBase> func;
	func = T::create();

	tick.reset();
	tick.start();
	// 计算图a的哈希值
	func->compute(a, hashA);
	tick.stop();
	cout << "compute1: " << tick.getTimeMilli() << " ms" << endl;

	tick.reset();
	tick.start();
	// 计算图b的哈希值
	func->compute(b, hashB);
	tick.stop();
	cout << "compute2: " << tick.getTimeMilli() << " ms" << endl;

	// 比较两张图像哈希值的距离
	cout << "compare: " << func->compare(hashA, hashB) << endl << endl;
}

int main()
{
	// 打开两张图像进行相似度比较
	Mat input = imread("./image/img1.jpg");
	Mat target = imread("./image/img4.jpg");

	// 通过不同方法比较图像相似性
	test_one<AverageHash>("AverageHash", input, target);
	test_one<PHash>("PHash", input, target);
	test_one<MarrHildrethHash>("MarrHildrethHash", input, target);
	test_one<RadialVarianceHash>("RadialVarianceHash", input, target);
	test_one<BlockMeanHash>("BlockMeanHash", input, target);
	test_one<ColorMomentHash>("ColorMomentHash", input, target);

	system("pause");
	return 0;
}