#include "brisque.h"
#include <iostream>

using namespace std;

// function to compute brisque features
// feature dimensions : 1 x 36
// 提取特征
void ComputeBrisqueFeature(Mat& orig, vector<double>& featurevector)
{
	// 归一化的图像
	Mat orig_bw_int(orig.size(), CV_64F, 1);

	// convert to grayscale
	// 转换为灰度图像
	cvtColor(orig, orig_bw_int, COLOR_BGR2GRAY);

	// create a copy of original image
	// 原图的副本
	Mat orig_bw(orig_bw_int.size(), CV_64FC1, 1);

	// 归一化
	orig_bw_int.convertTo(orig_bw, 1.0 / 255);
	orig_bw_int.release();

	// orig_bw now contains the grayscale image normalized to the range 0,1
	// orig_bw 现在为像素为0到1的灰度图像

	// number of times to scale the image
	int scalenum = 2;
	for (int itr_scale = 1; itr_scale <= scalenum; itr_scale++)
	{
		// resize image
		// 重置图像大小
		Size dst_size(orig_bw.cols / cv::pow((double)2, itr_scale - 1), orig_bw.rows / pow((double)2, itr_scale - 1));
		Mat imdist_scaled;
		// INTER_CUBIC
		resize(orig_bw, imdist_scaled, dst_size, 0, 0, INTER_CUBIC);
		imdist_scaled.convertTo(imdist_scaled, CV_64FC1, 1.0 / 255.0);

		// calculating MSCN coefficients
		// 计算MSCN系数

		// compute mu (local mean)
		// 计算局部均值
		Mat mu(imdist_scaled.size(), CV_64FC1, 1);
		GaussianBlur(imdist_scaled, mu, Size(7, 7), 1.166);

		Mat mu_sq;
		cv::pow(mu, double(2.0), mu_sq);

		// compute sigma (local sigma)
		// 计算局部方差
		Mat sigma(imdist_scaled.size(), CV_64FC1, 1);
		cv::multiply(imdist_scaled, imdist_scaled, sigma);
		GaussianBlur(sigma, sigma, Size(7, 7), 1.166);
		cv::subtract(sigma, mu_sq, sigma);
		cv::pow(sigma, double(0.5), sigma);
		// to avoid DivideByZero Error
		// 避免局部方差为0，因为后面计算MSCN系数要除以局部方差
		add(sigma, Scalar(1.0 / 255), sigma);

		// structdis is MSCN image
		// 计算MSCN
		Mat structdis(imdist_scaled.size(), CV_64FC1, 1);
		subtract(imdist_scaled, mu, structdis);
		divide(structdis, sigma, structdis);

		// Compute AGGD fit to MSCN image
		// lsgima_best 左方差，rsigma_best右方差，gamma均值
		double lsigma_best, rsigma_best, gamma_best;

		// 非对称广义高斯分布拟合
		structdis = AGGDfit(structdis, lsigma_best, rsigma_best, gamma_best);

		// 形状参数
		featurevector.push_back(gamma_best);
		// 方差参数
		featurevector.push_back((lsigma_best * lsigma_best + rsigma_best * rsigma_best) / 2);

		// Compute paired product images
		// indices for orientations (H, V, D1, D2)
		// 计算两两对称参数
		int shifts[4][2] = { {0,1},{1,0},{1,1},{-1,1} };

		for (int itr_shift = 1; itr_shift <= 4; itr_shift++)
		{
			// select the shifting index from the 2D array
			int* reqshift = shifts[itr_shift - 1];

			// declare shifted_structdis as pairwise image
			Mat shifted_structdis(imdist_scaled.size(), CV_64F, 1);

			// create copies of the images using BwImage constructor
			// utility constructor for better subscript access (for pixels)
			BwImage OrigArr(structdis);
			BwImage ShiftArr(shifted_structdis);

			// create pair-wise product for the given orientation (reqshift)
			for (int i = 0; i < structdis.rows; i++)
			{
				for (int j = 0; j < structdis.cols; j++)
				{
					if (i + reqshift[0] >= 0 && i + reqshift[0] < structdis.rows && j + reqshift[1] >= 0 && j + reqshift[1] < structdis.cols)
					{
						ShiftArr[i][j] = OrigArr[i + reqshift[0]][j + reqshift[1]];
					}
					else
					{
						ShiftArr[i][j] = 0;
					}
				}
			}

			// Mat structdis_pairwise;
			shifted_structdis = ShiftArr.equate(shifted_structdis);

			// calculate the products of the pairs
			multiply(structdis, shifted_structdis, shifted_structdis);

			// fit the pairwise product to AGGD
			shifted_structdis = AGGDfit(shifted_structdis, lsigma_best, rsigma_best, gamma_best);

			double constant = sqrt(tgamma(1 / gamma_best)) / sqrt(tgamma(3 / gamma_best));
			double meanparam = (rsigma_best - lsigma_best) * (tgamma(2 / gamma_best) / tgamma(1 / gamma_best)) * constant;

			// push the calculated parameters from AGGD fit to pair-wise products
			featurevector.push_back(gamma_best);
			featurevector.push_back(meanparam);
			featurevector.push_back(cv::pow(lsigma_best, 2));
			featurevector.push_back(cv::pow(rsigma_best, 2));
		}
	}
}

// function to compute best fit parameters from AGGDfit
// 拟合非广义分布
Mat AGGDfit(Mat structdis, double& lsigma_best, double& rsigma_best, double& gamma_best)
{
	// create a copy of an image using BwImage constructor (brisque.h - more info)
	BwImage ImArr(structdis);

	long int poscount = 0, negcount = 0;
	double possqsum = 0, negsqsum = 0, abssum = 0;
	for (int i = 0; i < structdis.rows; i++)
	{
		for (int j = 0; j < structdis.cols; j++)
		{
			// BwImage provides [][] access
			double pt = ImArr[i][j];
			if (pt > 0)
			{
				poscount++;
				possqsum += pt * pt;
				abssum += pt;
			}
			else if (pt < 0)
			{
				negcount++;
				negsqsum += pt * pt;
				abssum -= pt;
			}
		}
	}

	lsigma_best = cv::pow(negsqsum / negcount, 0.5);
	rsigma_best = cv::pow(possqsum / poscount, 0.5);

	double gammahat = lsigma_best / rsigma_best;
	long int totalcount = (structdis.cols) * (structdis.rows);
	double rhat = cv::pow(abssum / totalcount, static_cast<double>(2)) / ((negsqsum + possqsum) / totalcount);
	double rhatnorm = rhat * (cv::pow(gammahat, 3) + 1) * (gammahat + 1) / pow(pow(gammahat, 2) + 1, 2);

	double prevgamma = 0;
	double prevdiff = 1e10;
	float sampling = 0.001;

	// possible to coarsen sampling to quicken the code, with some loss of accuracy
	for (float gam = 0.2; gam < 10; gam += sampling)
	{
		double r_gam = tgamma(2 / gam) * tgamma(2 / gam) / (tgamma(1 / gam) * tgamma(3 / gam));
		double diff = abs(r_gam - rhatnorm);
		if (diff > prevdiff) break;
		prevdiff = diff;
		prevgamma = gam;
	}
	gamma_best = prevgamma;

	return structdis.clone();
}