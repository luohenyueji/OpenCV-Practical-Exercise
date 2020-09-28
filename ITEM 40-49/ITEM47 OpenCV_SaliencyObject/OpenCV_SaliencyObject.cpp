#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace saliency;

int main()
{
	// 显著性检测算法
	// 可选：SPECTRAL_RESIDUAL，FINE_GRAINED，BING，BinWangApr2014
	String saliency_algorithm = "BING";
	// 检测视频或者图像
	String video_name = "video/vtest.avi";
	// String video_name = "video/dog.jpg";
	// 起始帧
	int start_frame = 0;
	// 模型路径
	String training_path = "ObjectnessTrainedModel";

	// 如果算法名和视频名为空，停止检测
	if (saliency_algorithm.empty() || video_name.empty())
	{
		cout << "Please set saliency_algorithm and video_name";
		return -1;
	}

	// open the capture
	VideoCapture cap;
	// 打开视频
	cap.open(video_name);
	// 设置视频起始帧
	cap.set(CAP_PROP_POS_FRAMES, start_frame);

	// 输入图像
	Mat frame;

	// instantiates the specific Saliency
	// 实例化saliencyAlgorithm结构
	Ptr<Saliency> saliencyAlgorithm;

	// 二值化检测结果
	Mat binaryMap;
	// 检测图像
	Mat image;

	// 读图
	cap >> frame;
	if (frame.empty())
	{
		return 0;
	}

	frame.copyTo(image);

	// 根据输入的方法确定检测类型
	// StaticSaliencySpectralResidual
	if (saliency_algorithm.find("SPECTRAL_RESIDUAL") == 0)
	{
		// 检测结果，白色区域表示显著区域
		Mat saliencyMap;
		saliencyAlgorithm = StaticSaliencySpectralResidual::create();
		// 计算显著性
		double start = static_cast<double>(getTickCount());
		bool success = saliencyAlgorithm->computeSaliency(image, saliencyMap);
		double duration = ((double)getTickCount() - start) / getTickFrequency();
		cout << "computeSaliency cost time is: " << duration * 1000 << "ms" << endl;

		if (success)
		{
			StaticSaliencySpectralResidual spec;
			// 二值化图像
			double start = static_cast<double>(getTickCount());
			spec.computeBinaryMap(saliencyMap, binaryMap);
			double duration = ((double)getTickCount() - start) / getTickFrequency();
			cout << "computeBinaryMap cost time is: " << duration * 1000 << "ms" << endl;

			imshow("Original Image", image);
			imshow("Saliency Map", saliencyMap);
			imshow("Binary Map", binaryMap);

			// 转换格式才能保存图片
			saliencyMap.convertTo(saliencyMap, CV_8UC3, 256);
			imwrite("Results/SPECTRAL_RESIDUAL_saliencyMap.jpg", saliencyMap);
			imwrite("Results/SPECTRAL_RESIDUAL_binaryMap.jpg", binaryMap);
			waitKey(0);
		}
	}

	// StaticSaliencyFineGrained
	else if (saliency_algorithm.find("FINE_GRAINED") == 0)
	{
		Mat saliencyMap;
		saliencyAlgorithm = StaticSaliencyFineGrained::create();
		// 计算显著性
		double start = static_cast<double>(getTickCount());
		bool success = saliencyAlgorithm->computeSaliency(image, saliencyMap);
		double duration = ((double)getTickCount() - start) / getTickFrequency();
		cout << "computeSaliency cost time is: " << duration * 1000 << "ms" << endl;

		if (success)
		{
			StaticSaliencyFineGrained spec;
			// 二值化图像
			double start = static_cast<double>(getTickCount());
			spec.computeBinaryMap(saliencyMap, binaryMap);
			double duration = ((double)getTickCount() - start) / getTickFrequency();
			cout << "computeBinaryMap cost time is: " << duration * 1000 << "ms" << endl;

			imshow("Saliency Map", saliencyMap);
			imshow("Original Image", image);
			imshow("Binary Map", binaryMap);

			// 转换格式才能保存图片
			saliencyMap.convertTo(saliencyMap, CV_8UC3, 256);
			imwrite("Results/FINE_GRAINED_saliencyMap.jpg", saliencyMap);
			imwrite("Results/FINE_GRAINED_binaryMap.jpg", binaryMap);
			waitKey(0);
		}
	}

	// ObjectnessBING
	else if (saliency_algorithm.find("BING") == 0)
	{
		// 判断模型是否存在
		if (training_path.empty())
		{
			cout << "Path of trained files missing! " << endl;
			return -1;
		}

		else
		{
			saliencyAlgorithm = ObjectnessBING::create();
			vector<Vec4i> saliencyMap;
			// 提取模型文件参数
			saliencyAlgorithm.dynamicCast<ObjectnessBING>()->setTrainingPath(training_path);
			// 将算法检测结果保存在Results文件夹内
			saliencyAlgorithm.dynamicCast<ObjectnessBING>()->setBBResDir("Results");
			// 设置非极大值抑制，值越大检测到的目标越少，检测速度越快
			saliencyAlgorithm.dynamicCast<ObjectnessBING>()->setNSS(50);

			// 计算显著性
			double start = static_cast<double>(getTickCount());
			// 基于三个颜色空间进行检测，可以只检测一个空间，把training_path下其他空间模型删除即可
			// 如只保留ObjNessB2W8MAXBGR前缀的文件，算法耗时只有原来的一半
			bool success = saliencyAlgorithm->computeSaliency(image, saliencyMap);
			double duration = ((double)getTickCount() - start) / getTickFrequency();
			cout << "computeSaliency cost time is: " << duration * 1000 << "ms" << endl;

			if (success)
			{
				// saliencyMap获取检测到的目标个数
				int ndet = int(saliencyMap.size());
				std::cout << "Objectness done " << ndet << std::endl;
				// The result are sorted by objectness. We only use the first maxd boxes here.
				// 目标按可能性从大到小排列，maxd为显示前5个目标，step设置颜色，jitter设置矩形框微调
				int maxd = 5, step = 255 / maxd, jitter = 9;
				Mat draw = image.clone();
				for (int i = 0; i < std::min(maxd, ndet); i++)
				{
					// 获得矩形框坐标点
					Vec4i bb = saliencyMap[i];
					// 设定颜色
					Scalar col = Scalar(((i*step) % 255), 50, 255 - ((i*step) % 255));
					// 矩形框微调
					Point off(theRNG().uniform(-jitter, jitter), theRNG().uniform(-jitter, jitter));
					// 画矩形
					rectangle(draw, Point(bb[0] + off.x, bb[1] + off.y), Point(bb[2] + off.x, bb[3] + off.y), col, 2);
					// mini temperature scale
					// 颜色标注
					rectangle(draw, Rect(20, 20 + i * 10, 10, 10), col, -1);
				}
				imshow("BING", draw);

				// 保存图片
				imwrite("Results/BING_draw.jpg", draw);
				waitKey();
			}
			else
			{
				std::cout << "No saliency found for " << video_name << std::endl;
			}
		}
	}

	// BinWangApr2014
	else if (saliency_algorithm.find("BinWangApr2014") == 0)
	{
		saliencyAlgorithm = MotionSaliencyBinWangApr2014::create();
		// 设置数据结构大小
		saliencyAlgorithm.dynamicCast<MotionSaliencyBinWangApr2014>()->setImagesize(image.cols, image.rows);
		// 初始化
		saliencyAlgorithm.dynamicCast<MotionSaliencyBinWangApr2014>()->init();

		bool paused = false;
		int i = 0;
		for (;; )
		{
			if (!paused)
			{
				cap >> frame;
				if (frame.empty())
				{
					return 0;
				}
				Mat srcImg = frame.clone();
				cvtColor(frame, frame, COLOR_BGR2GRAY);

				Mat saliencyMap;
				// 计算
				double start = static_cast<double>(getTickCount());
				saliencyAlgorithm->computeSaliency(frame, saliencyMap);
				double duration = ((double)getTickCount() - start) / getTickFrequency();
				cout << "computeSaliency cost time is: " << duration * 1000 << "ms" << endl;

				imshow("image", frame);
				// 显示
				imshow("saliencyMap", saliencyMap * 255);

				i++;
				if (i == 100)
				{
					imwrite("Results/origin.jpg", srcImg);
					// 转换格式才能保存图片
					saliencyMap.convertTo(saliencyMap, CV_8UC3, 256);
					imwrite("Results/BinWangApr2014_saliencyMap.jpg", saliencyMap);
				}
			}

			char c = (char)waitKey(2);
			if (c == 'q')
				break;
			if (c == 'p')
				paused = !paused;
		}
	}

	destroyAllWindows();
	return 0;
}