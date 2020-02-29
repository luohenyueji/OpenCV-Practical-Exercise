#include "brisque.h"
#include <iostream>
#include <fstream>
#include <iostream>
#include <fstream>

// rescaling based on training data i libsvm
// 训练数据获得，图像各个维度最大值和最小值
float rescale_vector[36][2];

using namespace std;

int read_range_file(string range_fname)
{
	//check if file exists
	char buff[100];
	int i;
	FILE* range_file = fopen(range_fname.c_str(), "r");
	if (range_file == NULL) return 1;
	//assume standard file format for this program
	fgets(buff, 100, range_file);
	fgets(buff, 100, range_file);
	//now we can fill the array
	for (i = 0; i < 36; ++i)
	{
		float a, b, c;
		fscanf(range_file, "%f %f %f", &a, &b, &c);
		rescale_vector[i][0] = b;
		rescale_vector[i][1] = c;
	}
	return 0;
}

int main()
{
	// use the pre-trained allmodel file
	// 使用预训练模型
	string modelfile = "allmodel";

	// 训练图像的
	string range_fname = "allrange";
	// 待检测图像路径
	string imagename = "./images/original-scaled-image.jpg";

	//read in the allrange file to setup internal scaling array
	if (read_range_file(range_fname))
	{
		cerr << "unable to open allrange file" << endl;
		return -1;
	}

	// 计算质量评价得分，分数越高图像质量越差
	float qualityscore = computescore(imagename, modelfile);
	cout << "Quality Score: " << qualityscore << endl;

	return 0;
}