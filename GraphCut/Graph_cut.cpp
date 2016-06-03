#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>

#include "GraphCut.hpp"
#include "GrabCut.hpp"

using namespace std;
using namespace cv;

/*----------------------------
* 功能 : 从 hotmap 文件中读入数据，保存到 cv::Mat 矩阵
*		- 默认按 float 格式读入数据，
*		- 如果没有指定矩阵的行、列和通道数，则输出的矩阵是单通道、N 行 1 列的
*----------------------------
* 函数 : LoadData
* 访问 : public
* 返回 : -1：打开文件失败；0：按设定的矩阵参数读取数据成功；1：按默认的矩阵参数读取数据
*
* 参数 : fileName	[in]	文件名
* 参数 : matData	[out]	矩阵数据
* 参数 : matRows	[in]	矩阵行数，默认为 0
* 参数 : matCols	[in]	矩阵列数，默认为 0
* 参数 : matChns	[in]	矩阵通道数，默认为 0
*/
int LoadData(string fileName, cv::Mat& matData, size_t matRows = 0, size_t matCols = 0, size_t matChns = 0)
{
	int retVal = 0;

	// 打开文件
	ifstream inFile(fileName.c_str(), ios_base::in);
	if (!inFile.is_open())
	{
		cout << "读取文件失败" << endl;
		retVal = -1;
		return (retVal);
	}

	// 载入数据
	istream_iterator<float> begin(inFile);	//按 float 格式取文件数据流的起始指针
	istream_iterator<float> end;			//取文件流的终止位置
	vector<float> inData(begin, end);		//将文件数据保存至 std::vector 中
	cv::Mat tmpMat = cv::Mat( inData);		//将数据由 std::vector 转换为 cv::Mat

	// 检查设定的矩阵尺寸和通道数
	size_t dataLength = inData.size();
	//1.通道数
	if (matChns == 0)
	{
		matChns = 1;
	}
	//2.行列数
	if (matRows != 0 && matCols == 0)
	{
		matCols = dataLength / matChns / matRows;
	}
	else if (matCols != 0 && matRows == 0)
	{
		matRows = dataLength / matChns / matCols;
	}
	else if (matCols == 0 && matRows == 0)
	{
		matRows = dataLength / matChns;
		matCols = 1;
	}
	//3.数据总长度
	if (dataLength != (matRows * matCols * matChns))
	{
		cout << "读入的数据长度 不满足 设定的矩阵尺寸与通道数要求，将按默认方式输出矩阵！" << endl;
		retVal = 1;
		matChns = 1;
		matRows = dataLength;
	}

	// 将文件数据保存至输出矩阵
	matData = tmpMat.reshape(matChns, matRows).clone();

	return (retVal);
}


int main(int argc, char* argv[])
{
	if (argc != 3) {
		std::cerr << "lack of parameters " << argv[0] << std::endl;
		return 1;
	}

	string TheImage_Path = argv[1];
	string Uitary_Path = argv[2];

	Mat TheImage = imread(TheImage_Path);
	Mat mask;

	mask.create(TheImage.size(), CV_8UC1 );
	mask.setTo( GC_PR_FGD );
	int MARGIN = 5;
	for (int i = 0; i < mask.rows; i ++) {
		for (int j = 0; j < MARGIN; j ++) {
			mask.at<uchar>(i, j) = GC_BGD;
			mask.at<uchar>(i, mask.cols - 1 - j) = GC_BGD;
		}
	}
	for (int i = 0; i < MARGIN; i ++) {
		for (int j = 0; j < mask.cols; j ++) {
			mask.at<uchar>(i, j) = GC_BGD;
			mask.at<uchar>(mask.rows - 1 - i, j) = GC_BGD;
		}
	}

	//  GC_BGD = 0,  //!< background
	//	GC_FGD = 1,  //!< foreground
	//	GC_PR_BGD = 2,  //!< most probably background
	//	GC_PR_FGD = 3   //!< most probably foreground
	Mat Uitary;
	LoadData(Uitary_Path, Uitary, TheImage.rows, TheImage.cols);

	Mat Uitary_show(TheImage.rows, TheImage.cols, CV_32FC1);
	for (int i = 0; i < Uitary_show.rows; i++)
	{
		for (int j = 0; j < Uitary_show.cols; j++)
		{
			Uitary_show.at<float>(i, j) = Uitary.at<float>(i, j);
		}
	}

	const double gamma = 50;
	const double lambda = 9 * gamma;
	const double beta = MycalcBeta(TheImage);
	Mat leftW, upleftW, upW, uprightW;
	MycalcNWeights(TheImage, leftW, upleftW, upW, uprightW, beta, gamma);

	GCGraph<double> graph;
	MyconstructGCGraph(TheImage, mask, Uitary_show, lambda, leftW, upleftW, upW, uprightW, graph);
	MyestimateSegmentation(graph, mask);

	Mat bgd, fgd;
	MyGMM bgdGMM(bgd), fgdGMM(fgd);
	Mat compIdxs( TheImage.size(), CV_32SC1 );
	MyinitGMMs( TheImage, mask, bgdGMM, fgdGMM );
	for ( int i = 0; i < 3; i++ )
	{
		GCGraph<double> graph;
		MyassignGMMsComponents( TheImage, mask, bgdGMM, fgdGMM, compIdxs );
		MylearnGMMs( TheImage, mask, compIdxs, bgdGMM, fgdGMM );
		MyconstructGCGraph_grab(TheImage, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph );
		MyestimateSegmentation( graph, mask );
	}

	Mat foreground(TheImage.size(), CV_8UC3, Scalar(255, 255, 255));//BGR
	compare(mask, cv::GC_PR_FGD, mask, CMP_EQ);
	TheImage.copyTo(foreground, mask);
	// imshow("【结果】" , foreground);
	imwrite("result.jpg", foreground);
	// waitKey(0);

	return 0;
}