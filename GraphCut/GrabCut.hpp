#ifndef __GRAB_CUT__
#define __GRAB_CUT__

#include <opencv2/opencv.hpp>
#include <iostream>
#include <limits>

#include "GraphCut.hpp"

using namespace cv;

class MyGMM
{
public:
    static const int componentsCount = 2;

    MyGMM( Mat& _model );
    double operator()( const Vec3d color ) const;
    double operator()( int ci, const Vec3d color ) const;
    int MywhichComponent( const Vec3d color ) const;

    void MyinitLearning();
    void MyaddSample( int ci, const Vec3d color );
    void MyendLearning();

private:
    void MycalcInverseCovAndDeterm( int ci );
    Mat model;
    double* coefs;
    double* mean;
    double* cov;

    double inverseCovs[componentsCount][3][3];
    double covDeterms[componentsCount];

    double sums[componentsCount][3];
    double prods[componentsCount][3][3];
    int sampleCounts[componentsCount];
    int totalSampleCount;
};

void MyinitGMMs( const Mat& img, const Mat& mask, MyGMM& bgdGMM, MyGMM& fgdGMM );

void MyassignGMMsComponents( const Mat& img, const Mat& mask, const MyGMM& bgdGMM, const MyGMM& fgdGMM, Mat& compIdxs );

void MylearnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs, MyGMM& bgdGMM, MyGMM& fgdGMM );

void MyconstructGCGraph_grab( const Mat& img, const Mat& mask, const MyGMM& bgdGMM, const MyGMM& fgdGMM, double lambda,
                              const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
                              GCGraph<double>& graph );

#endif