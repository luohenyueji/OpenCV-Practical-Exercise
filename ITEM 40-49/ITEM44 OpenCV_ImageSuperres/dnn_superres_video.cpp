// 视频超分放大多输出

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn_superres.hpp>

using namespace std;
using namespace cv;
using namespace dnn_superres;

int main()
{
	string input_path = string("./video/chaplin.mp4");
	string output_path = string("./video/out_chaplin.mp4");
	// 选择模型 edsr, espcn, fsrcnn or lapsrn
	string algorithm = string("lapsrn");
	// 放大比例，2，3，4，8，根据模型结构选择
	int scale = 2;
	// 模型路径
	string path = string("./model/LapSRN_x2.pb");

	// 打开视频
	VideoCapture input_video(input_path);
	// 输入图像编码尺寸
	int ex = static_cast<int>(input_video.get(CAP_PROP_FOURCC));
	// 获得输出视频图像尺寸
	Size S = Size((int)input_video.get(CAP_PROP_FRAME_WIDTH) * scale,
		(int)input_video.get(CAP_PROP_FRAME_HEIGHT) * scale);

	VideoWriter output_video;
	output_video.open(output_path, ex, input_video.get(CAP_PROP_FPS), S, true);

	// 如果视频没有打开
	if (!input_video.isOpened())
	{
		std::cerr << "Could not open the video." << std::endl;
		return -1;
	}

	// 读取超分放大模型
	DnnSuperResImpl sr;
	sr.readModel(path);
	sr.setModel(algorithm, scale);

	for (;;)
	{
		Mat frame, output_frame;
		input_video >> frame;

		if (frame.empty())
			break;

		// 上采样图像
		sr.upsample(frame, output_frame);
		output_video << output_frame;

		namedWindow("Upsampled video", WINDOW_AUTOSIZE);
		imshow("Upsampled video", output_frame);

		namedWindow("Original video", WINDOW_AUTOSIZE);
		imshow("Original video", frame);

		char c = (char)waitKey(1);
		// esc退出
		if (c == 27)
		{
			break;
		}
	}

	input_video.release();
	output_video.release();

	return 0;
}