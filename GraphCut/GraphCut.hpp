#ifndef GRAPHCUT_H_
#define GRAPHCUT_H_


#include <opencv2/opencv.hpp>
#include "gcgraph.hpp"

using namespace cv;

extern double MycalcBeta(const Mat& img);

extern void MycalcNWeights(const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma);

extern void MyconstructGCGraph(const Mat& img, const Mat& mask, const CvMat* Pij, double lambda,
                               const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
                               GCGraph<double>& graph);

extern void MyconstructGCGraph(const Mat& img, const Mat& mask, const Mat& Uitary, double lambda,
                               const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
                               GCGraph<double>& graph);

extern void MyestimateSegmentation(GCGraph<double>& graph, Mat& mask);

#endif
