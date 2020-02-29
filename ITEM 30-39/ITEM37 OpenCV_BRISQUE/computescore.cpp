#include "brisque.h"
#include "libsvm/svm.h"

// 计算质量评价得分，分数越高图像质量越差
float computescore(string imagename, string modelfile)
{
	// pre-loaded vectors from allrange file
	// 从allrange文件加载预向量
	// 就是main.cpp中的rescale_vector，只不过作者直接写出来了，用于svm预测时归一化
	float min_[36] = { 0.336999 ,0.019667 ,0.230000 ,-0.125959 ,0.000167 ,0.000616 ,0.231000 ,-0.125873 ,0.000165 ,0.000600 ,0.241000 ,-0.128814 ,0.000179 ,0.000386 ,0.243000 ,-0.133080 ,0.000182 ,0.000421 ,0.436998 ,0.016929 ,0.247000 ,-0.200231 ,0.000104 ,0.000834 ,0.257000 ,-0.200017 ,0.000112 ,0.000876 ,0.257000 ,-0.155072 ,0.000112 ,0.000356 ,0.258000 ,-0.154374 ,0.000117 ,0.000351 };
	float max_[36] = { 9.999411, 0.807472, 1.644021, 0.202917, 0.712384, 0.468672, 1.644021, 0.169548, 0.713132, 0.467896, 1.553016, 0.101368, 0.687324, 0.533087, 1.554016, 0.101000, 0.689177, 0.533133, 3.639918, 0.800955, 1.096995, 0.175286, 0.755547, 0.399270, 1.095995, 0.155928, 0.751488, 0.402398, 1.041992, 0.093209, 0.623516, 0.532925, 1.042992, 0.093714, 0.621958, 0.534484 };

	// 图像得分
	double qualityscore;
	int i;

	// create svm model object
	// 创建svm结构图
	struct svm_model* model;

	// read image (color mode)
	// 读图
	Mat orig = imread(imagename, 1);

	// feature vector initialization
	// 特征向量初始化
	vector<double> brisqueFeatures;

	// compute brisque features
	//  计算特征向量
	ComputeBrisqueFeature(orig, brisqueFeatures);

	// 打开svm模型
	if ((model = svm_load_model(modelfile.c_str())) == 0)
	{
		fprintf(stderr, "can't open model file allmodel\n");
		exit(1);
	}

	// float min_[37];
	// float max_[37];

	struct svm_node x[37];
	// rescale the brisqueFeatures vector from -1 to 1
	// 将brisqueFeatures向量从-1重新缩放到1
	// also convert vector to svm node array object
	// 将向量转换为svm节点数组对象
	for (i = 0; i < 36; ++i)
	{
		// 归一化
		float min = min_[i];
		float max = max_[i];

		x[i].value = -1 + (2.0 / (max - min) * (brisqueFeatures[i] - min));
		x[i].index = i + 1;
	}
	x[36].index = -1;

	// 计算内存消耗
	int nr_class = svm_get_nr_class(model);
	double* prob_estimates = (double*)malloc(nr_class * sizeof(double));

	// predict quality score using libsvm class
	// SVM预测
	qualityscore = svm_predict_probability(model, x, prob_estimates);

	free(prob_estimates);
	svm_free_and_destroy_model(&model);

	return qualityscore;
}