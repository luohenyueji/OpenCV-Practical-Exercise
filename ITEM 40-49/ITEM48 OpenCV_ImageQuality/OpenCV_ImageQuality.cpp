#include <opencv2/opencv.hpp>
#include <opencv2/quality.hpp>

using namespace std;
using namespace cv;

// 计算结果均值
double calMEAN(Scalar result)
{
	int i = 0;
	double sum = 0;
	// 计算总和
	for (auto val : result.val)
	{
		if (0 == val || isinf(val))
		{
			break;
		}
		sum += val;
		i++;
	}
	return sum / i;
}

// 均方误差 MSE
double MSE(Mat img1, Mat img2)
{
	// output quality map
	// 质量结果图
	// 质量结果图quality_map就是检测图像和基准图像各个像素点差值图像
	cv::Mat quality_map;
	// compute MSE via static method
	// cv::noArray() if not interested in output quality maps
	// 静态方法，一步到位
	// 如果不想获得质量结果图，将quality_map替换为noArray()
	cv::Scalar result_static = quality::QualityMSE::compute(img1, img2, quality_map);

	/* 另外一种动态计算的方法
	// alternatively, compute MSE via instance
	cv::Ptr<quality::QualityBase> ptr = quality::QualityMSE::create(img1);
	// compute MSE, compare img1 vs img2
	cv::Scalar result = ptr->compute(img2);
	ptr->getQualityMap(quality_map);
	*/

	return calMEAN(result_static);
}

// 峰值信噪比 PSNR
double PSNR(Mat img1, Mat img2)
{
	// 质量结果图
	// 质量结果图quality_map就是检测图像和基准图像各个像素点差值图像
	cv::Mat quality_map;
	// 静态方法，一步到位
	// 如果不想获得质量结果图，将quality_map替换为noArray()
	// 第四个参数为PSNR计算公式中的MAX，即图片可能的最大像素值，通常为255
	cv::Scalar result_static = quality::QualityPSNR::compute(img1, img2, quality_map, 255.0);

	/* 另外一种动态计算的方法
	cv::Ptr<quality::QualityBase> ptr = quality::QualityPSNR::create(img1, 255.0);
	cv::Scalar result = ptr->compute(img2);
	ptr->getQualityMap(quality_map);*/

	return calMEAN(result_static);
}

// 梯度幅度相似性偏差 GMSD
double GMSD(Mat img1, Mat img2)
{
	// 质量结果图
	// 质量结果图quality_map就是检测图像和基准图像各个像素点差值图像
	cv::Mat quality_map;
	// 静态方法，一步到位
	// 如果不想获得质量结果图，将quality_map替换为noArray()
	cv::Scalar result_static = quality::QualityGMSD::compute(img1, img2, quality_map);
	/* 另外一种动态计算的方法
	cv::Ptr<quality::QualityBase> ptr = quality::QualityGMSD::create(img1);
	cv::Scalar result = ptr->compute(img2);
	ptr->getQualityMap(quality_map);*/
	return calMEAN(result_static);
}

// 结构相似性 SSIM
double SSIM(Mat img1, Mat img2)
{
	// 质量结果图
	// 质量结果图quality_map就是检测图像和基准图像各个像素点差值图像
	cv::Mat quality_map;
	// 静态方法，一步到位
	// 如果不想获得质量结果图，将quality_map替换为noArray()
	cv::Scalar result_static = quality::QualitySSIM::compute(img1, img2, quality_map);
	/* 另外一种动态计算的方法
	cv::Ptr<quality::QualityBase> ptr = quality::QualitySSIM::create(img1);
	cv::Scalar result = ptr->compute(img2);
	ptr->getQualityMap(quality_map);*/
	return calMEAN(result_static);
}

// 盲/无参考图像空间质量评估器 BRISQUE
double BRISQUE(Mat img)
{
	// path to the trained model
	cv::String model_path = "./model/brisque_model_live.yml";
	// path to range file
	cv::String range_path = "./model/brisque_range_live.yml";
	// 静态计算方法
	cv::Scalar result_static = quality::QualityBRISQUE::compute(img, model_path, range_path);
	/* 另外一种动态计算的方法
	cv::Ptr<quality::QualityBase> ptr = quality::QualityBRISQUE::create(model_path, range_path);
	// computes BRISQUE score for img
	cv::Scalar result = ptr->compute(img);*/
	return calMEAN(result_static);
}

void qualityCompute(String methodType, Mat img1, Mat img2)
{
	// 算法结果和算法耗时
	double result;
	TickMeter costTime;

	costTime.start();
	if ("MSE" == methodType)
		result = MSE(img1, img2);
	else if ("PSNR" == methodType)
		result = PSNR(img1, img2);
	else if ("PSNR" == methodType)
		result = PSNR(img1, img2);
	else if ("GMSD" == methodType)
		result = GMSD(img1, img2);
	else if ("SSIM" == methodType)
		result = SSIM(img1, img2);
	else if ("BRISQUE" == methodType)
		result = BRISQUE(img2);
	costTime.stop();
	cout << methodType << "_result is: " << result << endl;
	cout << methodType << "_cost time is: " << costTime.getTimeSec() / costTime.getCounter() << " s" << endl;
}

int main()
{
	// img1为基准图像，img2为检测图像
	cv::Mat img1, img2;
	img1 = cv::imread("image/cut-original-rotated-image.jpg");
	img2 = cv::imread("image/cut-original-rotated-image.jpg");

	if (img1.empty() || img2.empty())
	{
		cout << "img empty" << endl;
		return 0;
	}

	// 结果越小，检测图像和基准图像的差距越小
	qualityCompute("MSE", img1, img2);
	// 结果越小，检测图像和基准图像的差距越小
	qualityCompute("PSNR", img1, img2);
	// 结果为一个0到1之间的数，越大表示检测图像和基准图像的差距越小
	qualityCompute("GMSD", img1, img2);
	// 结果为一个0到1之间的数，越大表示检测图像和基准图像的差距越小
	qualityCompute("SSIM", img1, img2);
	// BRISQUE不需要基准图像
	// 结果为一个0到100之间的数，越小表示检测图像质量越好
	qualityCompute("BRISQUE", cv::Mat{}, img2);
	system("pause");
	return 0;
}