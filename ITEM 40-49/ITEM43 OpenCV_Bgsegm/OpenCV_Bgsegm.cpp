#include <opencv2/opencv.hpp>
#include <opencv2/bgsegm.hpp>
#include <iostream>

using namespace cv;
using namespace cv::bgsegm;

const String algos[7] = { "GMG", "CNT", "KNN", "MOG", "MOG2", "GSOC", "LSBP" };

// 创建不同的背景分割识别器
static Ptr<BackgroundSubtractor> createBGSubtractorByName(const String& algoName)
{
	Ptr<BackgroundSubtractor> algo;
	if (algoName == String("GMG"))
		algo = createBackgroundSubtractorGMG(20, 0.7);
	else if (algoName == String("CNT"))
		algo = createBackgroundSubtractorCNT();
	else if (algoName == String("KNN"))
		algo = createBackgroundSubtractorKNN();
	else if (algoName == String("MOG"))
		algo = createBackgroundSubtractorMOG();
	else if (algoName == String("MOG2"))
		algo = createBackgroundSubtractorMOG2();
	else if (algoName == String("GSOC"))
		algo = createBackgroundSubtractorGSOC();
	else if (algoName == String("LSBP"))
		algo = createBackgroundSubtractorLSBP();

	return algo;
}

int main()
{
	// 视频路径
	String videoPath = "./video/vtest.avi";

	// 背景分割识别器序号
	int algo_index = 0;
	// 创建背景分割识别器
	Ptr<BackgroundSubtractor> bgfs = createBGSubtractorByName(algos[algo_index]);

	// 打开视频
	VideoCapture cap;
	cap.open(videoPath);

	// 如果视频没有打开
	if (!cap.isOpened())
	{
		std::cerr << "Cannot read video. Try moving video file to sample directory." << std::endl;
		return -1;
	}

	// 输入图像
	Mat frame;
	// 运动前景
	Mat fgmask;
	// 最后显示的图像
	Mat segm;

	// 延迟等待时间
	int delay = 30;
	// 获得运行环境CPU的核心数
	int nthreads = getNumberOfCPUs();
	// 设置线程数
	setNumThreads(nthreads);

	// 是否显示运动前景
	bool show_fgmask = false;

	// 平均执行时间
	float average_Time = 0.0;
	// 当前帧数
	int frame_num = 0;
	// 总执行时间
	float sum_Time = 0.0;

	for (;;)
	{
		// 提取帧
		cap >> frame;

		// 如果图片为空
		if (frame.empty())
		{
			// CAP_PROP_POS_FRAMES表示当前帧
			// 本句话表示将当前帧设定为第0帧
			cap.set(CAP_PROP_POS_FRAMES, 0);
			cap >> frame;
		}

		double time0 = static_cast<double>(getTickCount());

		// 背景建模
		bgfs->apply(frame, fgmask);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		// 总执行时间
		sum_Time += time0;
		// 平均每帧执行时间
		average_Time = sum_Time / (frame_num + 1);

		if (show_fgmask)
		{
			segm = fgmask;
		}
		else
		{
			// 根据segm = alpha * frame + beta改变图片
			// 参数分别为，输出图像，输出图像格式，alpha值，beta值
			frame.convertTo(segm, CV_8U, 0.5);
			// 图像叠加
			// 参数分别为，输入图像/颜色1，输入图像/颜色2，输出图像，掩膜
			// 掩膜表示叠加范围
			add(frame, Scalar(100, 100, 0), segm, fgmask);
		}

		// 显示当前方法
		cv::putText(segm, algos[algo_index], Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 0, 255), 2, LINE_AA);
		// 显示当前线程数
		cv::putText(segm, format("%d threads", nthreads), Point(10, 60), FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 0, 255), 2, LINE_AA);
		// 显示当前每帧执行时间
		cv::putText(segm, format("averageTime %f s", average_Time), Point(10, 90), FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 0, 255), 2, LINE_AA);

		cv::imshow("FG Segmentation", segm);

		int c = waitKey(delay);

		// 修改等待时间
		if (c == ' ')
		{
			delay = delay == 30 ? 1 : 30;
		}

		// 按C背景分割识别器
		if (c == 'c' || c == 'C')
		{
			algo_index++;
			if (algo_index > 6)
				algo_index = 0;

			bgfs = createBGSubtractorByName(algos[algo_index]);
		}

		// 设置线程数
		if (c == 'n' || c == 'N')
		{
			nthreads++;
			if (nthreads > 8)
				nthreads = 1;

			setNumThreads(nthreads);
		}

		// 是否显示背景
		if (c == 'm' || c == 'M')
		{
			show_fgmask = !show_fgmask;
		}

		// 退出
		if (c == 'q' || c == 'Q' || c == 27)
		{
			break;
		}

		// 当前帧数增加
		frame_num++;
		if (100 == frame_num)
		{
			String strSave = "out_" + algos[algo_index] + ".jpg";
			imwrite(strSave, segm);
		}
	}

	return 0;
}