// 不同图像超分算法速度评估

#include <iostream>

#include <opencv2/dnn_superres.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dnn_superres;

static void showBenchmark(vector<Mat> images, string title, Size imageSize,
	const vector<String> imageTitles,
	const vector<double> perfValues)
{
	int fontFace = FONT_HERSHEY_COMPLEX_SMALL;
	int fontScale = 1;
	Scalar fontColor = Scalar(255, 255, 255);

	int len = static_cast<int>(images.size());

	int cols = 2, rows = 2;

	Mat fullImage = Mat::zeros(Size((cols * 10) + imageSize.width * cols, (rows * 10) + imageSize.height * rows),
		images[0].type());

	stringstream ss;
	int h_ = -1;
	for (int i = 0; i < len; i++)
	{
		int fontStart = 15;
		int w_ = i % cols;
		if (i % cols == 0)
			h_++;

		Rect ROI((w_ * (10 + imageSize.width)), (h_ * (10 + imageSize.height)), imageSize.width, imageSize.height);
		Mat tmp;
		resize(images[i], tmp, Size(ROI.width, ROI.height));

		ss << imageTitles[i];
		putText(tmp,
			ss.str(),
			Point(5, fontStart),
			fontFace,
			fontScale,
			fontColor,
			1,
			16);

		ss.str("");
		fontStart += 20;

		ss << perfValues[i];
		putText(tmp,
			ss.str(),
			Point(5, fontStart),
			fontFace,
			fontScale,
			fontColor,
			1,
			16);
		ss.str("");
		tmp.copyTo(fullImage(ROI));
	}

	namedWindow(title, 1);
	imshow(title, fullImage);
	imwrite("save.jpg", fullImage);
	waitKey();
}

int main()
{
	// 图片路径
	string img_path = string("./image/image.png");
	// 算法名称 edsr, espcn, fsrcnn or lapsrn
	string algorithm = string("lapsrn");

	// 模型路径，根据算法确定
	string model = string("./model/LapSRN_x2.pb");
	// 放大系数
	int scale = 2;

	Mat img = imread(img_path);
	if (img.empty())
	{
		cerr << "Couldn't load image: " << img << "\n";
		return -2;
	}

	// Crop the image so the images will be aligned
	// 对齐图像
	int width = img.cols - (img.cols % scale);
	int height = img.rows - (img.rows % scale);
	Mat cropped = img(Rect(0, 0, width, height));

	// Downscale the image for benchmarking
	// 缩小图像，以实现基准测试
	Mat img_downscaled;
	resize(cropped, img_downscaled, Size(), 1.0 / scale, 1.0 / scale);

	// Make dnn super resolution instance
	DnnSuperResImpl sr;
	Mat img_new;

	// Read and set the dnn model
	// 读取模型
	sr.readModel(model);
	sr.setModel(algorithm, scale);

	double elapsed = 0.0;
	vector<double> perf;

	TickMeter tm;

	// DL MODEL
	// 计算时间
	tm.start();
	sr.upsample(img_downscaled, img_new);
	tm.stop();
	// 运行时间s
	elapsed = tm.getTimeSec() / tm.getCounter();
	perf.push_back(elapsed);

	cout << sr.getAlgorithm() << " : " << elapsed << endl;

	// BICUBIC
	Mat bicubic;
	tm.start();
	resize(img_downscaled, bicubic, Size(), scale, scale, INTER_CUBIC);
	tm.stop();
	elapsed = tm.getTimeSec() / tm.getCounter();
	perf.push_back(elapsed);

	cout << "Bicubic" << " : " << elapsed << endl;

	// NEAREST NEIGHBOR
	Mat nearest;
	tm.start();
	resize(img_downscaled, nearest, Size(), scale, scale, INTER_NEAREST);
	tm.stop();
	elapsed = tm.getTimeSec() / tm.getCounter();
	perf.push_back(elapsed);

	cout << "Nearest" << " : " << elapsed << endl;

	// LANCZOS
	Mat lanczos;
	tm.start();
	resize(img_downscaled, lanczos, Size(), scale, scale, INTER_LANCZOS4);
	tm.stop();
	elapsed = tm.getTimeSec() / tm.getCounter();
	perf.push_back(elapsed);

	cout << "Lanczos" << " : " << elapsed << endl;

	vector <Mat> imgs{ img_new, bicubic, nearest, lanczos };
	vector <String> titles{ sr.getAlgorithm(), "Bicubic", "Nearest neighbor", "Lanczos" };
	showBenchmark(imgs, "Time benchmark", Size(bicubic.cols, bicubic.rows), titles, perf);

	waitKey(0);

	return 0;
}