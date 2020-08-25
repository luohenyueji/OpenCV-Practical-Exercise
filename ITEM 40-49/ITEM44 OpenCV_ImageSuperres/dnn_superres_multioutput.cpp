// 图像超分放大多输出
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn_superres.hpp>

using namespace std;
using namespace cv;
using namespace dnn_superres;

int main()
{
	// 图像路径
	string img_path = string("./image/image.png");
	if (img_path.empty())
	{
		printf("image is empty!");
	}
	// 可选多输入放大比例2，4，8。','分隔放大比例
	string scales_str = string("2,4,8");
	// 可选模型输出放大层比例名，NCHW_output_2x,NCHW_output_4x，NCHW_output_8x
	// 需要根据模型和输入放大比例共同确定确定
	string output_names_str = string("NCHW_output_2x,NCHW_output_4x,NCHW_output_8x");
	// 模型路径
	std::string path = string("./model/LapSRN_x8.pb");

	// Parse the scaling factors
	// 解析放大比例因子
	std::vector<int> scales;
	char delim = ',';
	{
		std::stringstream ss(scales_str);
		std::string token;
		while (std::getline(ss, token, delim))
		{
			scales.push_back(atoi(token.c_str()));
		}
	}

	// Parse the output node names
	// 解析模型放大层参数
	std::vector<String> node_names;
	{
		std::stringstream ss(output_names_str);
		std::string token;
		while (std::getline(ss, token, delim))
		{
			node_names.push_back(token);
		}
	}

	// Load the image
	// 导入图片
	Mat img = cv::imread(img_path);
	Mat original_img(img);
	if (img.empty())
	{
		std::cerr << "Couldn't load image: " << img << "\n";
		return -2;
	}

	// Make dnn super resolution instance
	// 创建Dnn Superres对象
	DnnSuperResImpl sr;
	// 获得最大放大比例
	int scale = *max_element(scales.begin(), scales.end());
	std::vector<Mat> outputs;
	// 读取模型
	sr.readModel(path);
	// 设定模型输出
	sr.setModel("lapsrn", scale);
	// 多输出超分放大图像
	sr.upsampleMultioutput(img, outputs, scales, node_names);

	for (unsigned int i = 0; i < outputs.size(); i++)
	{
		cv::namedWindow("Upsampled image", WINDOW_AUTOSIZE);
		// 在图上显示当前放大比例
		cv::putText(outputs[i], format("Scale %d", scales[i]), Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 0, 255), 2, LINE_AA);
		cv::imshow("Upsampled image", outputs[i]);
		cv::imwrite(to_string(i) + ".jpg", outputs[i]);
		cv::waitKey(-1);
	}

	return 0;
}