#include "GrabCut.hpp"

MyGMM::MyGMM( Mat& _model )
{
    const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
    if ( _model.empty() )
    {
        _model.create( 1, modelSize * componentsCount, CV_64FC1 );
        _model.setTo(Scalar(0));
    }
    else if ( (_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize * componentsCount) )
        CV_Error( CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount" );

    model = _model;

    coefs = model.ptr<double>(0);
    mean = coefs + componentsCount;
    cov = mean + 3 * componentsCount;

    for ( int ci = 0; ci < componentsCount; ci++ )
        if ( coefs[ci] > 0 )
            MycalcInverseCovAndDeterm( ci );
}

double MyGMM::operator()( const Vec3d color ) const
{
    double res = 0;
    for ( int ci = 0; ci < componentsCount; ci++ )
        res += coefs[ci] * (*this)(ci, color );
    return res;
}

double MyGMM::operator()( int ci, const Vec3d color ) const
{
    double res = 0;
    if ( coefs[ci] > 0 )
    {
        CV_Assert( covDeterms[ci] > std::numeric_limits<double>::epsilon() );
        Vec3d diff = color;
        double* m = mean + 3 * ci;
        diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
        double mult = diff[0] * (diff[0] * inverseCovs[ci][0][0] + diff[1] * inverseCovs[ci][1][0] + diff[2] * inverseCovs[ci][2][0])
                      + diff[1] * (diff[0] * inverseCovs[ci][0][1] + diff[1] * inverseCovs[ci][1][1] + diff[2] * inverseCovs[ci][2][1])
                      + diff[2] * (diff[0] * inverseCovs[ci][0][2] + diff[1] * inverseCovs[ci][1][2] + diff[2] * inverseCovs[ci][2][2]);
        res = 1.0f / sqrt(covDeterms[ci]) * exp(-0.5f * mult);
    }
    return res;
}

int MyGMM::MywhichComponent( const Vec3d color ) const
{
    int k = 0;
    double max = 0;

    for ( int ci = 0; ci < componentsCount; ci++ )
    {
        double p = (*this)( ci, color );
        if ( p > max )
        {
            k = ci;
            max = p;
        }
    }
    return k;
}

void MyGMM::MyinitLearning()
{
    for ( int ci = 0; ci < componentsCount; ci++)
    {
        sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
        prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
        prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
        prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
        sampleCounts[ci] = 0;
    }
    totalSampleCount = 0;
}

void MyGMM::MyaddSample( int ci, const Vec3d color )
{
    sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
    prods[ci][0][0] += color[0] * color[0]; prods[ci][0][1] += color[0] * color[1]; prods[ci][0][2] += color[0] * color[2];
    prods[ci][1][0] += color[1] * color[0]; prods[ci][1][1] += color[1] * color[1]; prods[ci][1][2] += color[1] * color[2];
    prods[ci][2][0] += color[2] * color[0]; prods[ci][2][1] += color[2] * color[1]; prods[ci][2][2] += color[2] * color[2];
    sampleCounts[ci]++;
    totalSampleCount++;
}

void MyGMM::MyendLearning()
{
    const double variance = 0.01;
    for ( int ci = 0; ci < componentsCount; ci++ )
    {
        int n = sampleCounts[ci];
        if ( n == 0 )
            coefs[ci] = 0;
        else
        {
            coefs[ci] = (double)n / totalSampleCount;

            double* m = mean + 3 * ci;
            m[0] = sums[ci][0] / n; m[1] = sums[ci][1] / n; m[2] = sums[ci][2] / n;

            double* c = cov + 9 * ci;
            c[0] = prods[ci][0][0] / n - m[0] * m[0]; c[1] = prods[ci][0][1] / n - m[0] * m[1]; c[2] = prods[ci][0][2] / n - m[0] * m[2];
            c[3] = prods[ci][1][0] / n - m[1] * m[0]; c[4] = prods[ci][1][1] / n - m[1] * m[1]; c[5] = prods[ci][1][2] / n - m[1] * m[2];
            c[6] = prods[ci][2][0] / n - m[2] * m[0]; c[7] = prods[ci][2][1] / n - m[2] * m[1]; c[8] = prods[ci][2][2] / n - m[2] * m[2];

            double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
            if ( dtrm <= std::numeric_limits<double>::epsilon() )
            {
                // Adds the white noise to avoid singular covariance matrix.
                c[0] += variance;
                c[4] += variance;
                c[8] += variance;
            }

            MycalcInverseCovAndDeterm(ci);
        }
    }
}

void MyGMM::MycalcInverseCovAndDeterm( int ci )
{
    if ( coefs[ci] > 0 )
    {
        double *c = cov + 9 * ci;
        double dtrm =
            covDeterms[ci] = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);

        CV_Assert( dtrm > std::numeric_limits<double>::epsilon() );
        inverseCovs[ci][0][0] =  (c[4] * c[8] - c[5] * c[7]) / dtrm;
        inverseCovs[ci][1][0] = -(c[3] * c[8] - c[5] * c[6]) / dtrm;
        inverseCovs[ci][2][0] =  (c[3] * c[7] - c[4] * c[6]) / dtrm;
        inverseCovs[ci][0][1] = -(c[1] * c[8] - c[2] * c[7]) / dtrm;
        inverseCovs[ci][1][1] =  (c[0] * c[8] - c[2] * c[6]) / dtrm;
        inverseCovs[ci][2][1] = -(c[0] * c[7] - c[1] * c[6]) / dtrm;
        inverseCovs[ci][0][2] =  (c[1] * c[5] - c[2] * c[4]) / dtrm;
        inverseCovs[ci][1][2] = -(c[0] * c[5] - c[2] * c[3]) / dtrm;
        inverseCovs[ci][2][2] =  (c[0] * c[4] - c[1] * c[3]) / dtrm;
    }
}

void MyinitGMMs( const Mat& img, const Mat& mask, MyGMM& bgdGMM, MyGMM& fgdGMM )
{
    const int kMeansItCount = 10;
    const int kMeansType = KMEANS_PP_CENTERS;

    Mat bgdLabels, fgdLabels;
    vector<Vec3f> bgdSamples, fgdSamples;
    Point p;
    for ( p.y = 0; p.y < img.rows; p.y++ )
    {
        for ( p.x = 0; p.x < img.cols; p.x++ )
        {
            if ( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                bgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
            else // GC_FGD | GC_PR_FGD
                fgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
        }
    }
    CV_Assert( !bgdSamples.empty() && !fgdSamples.empty() );
    Mat _bgdSamples( (int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0] );
    kmeans( _bgdSamples, MyGMM::componentsCount, bgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    Mat _fgdSamples( (int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0] );
    kmeans( _fgdSamples, MyGMM::componentsCount, fgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );

    bgdGMM.MyinitLearning();
    for ( int i = 0; i < (int)bgdSamples.size(); i++ )
        bgdGMM.MyaddSample( bgdLabels.at<int>(i, 0), bgdSamples[i] );
    bgdGMM.MyendLearning();

    fgdGMM.MyinitLearning();
    for ( int i = 0; i < (int)fgdSamples.size(); i++ )
        fgdGMM.MyaddSample( fgdLabels.at<int>(i, 0), fgdSamples[i] );
    fgdGMM.MyendLearning();
}

void MyassignGMMsComponents( const Mat& img, const Mat& mask, const MyGMM& bgdGMM, const MyGMM& fgdGMM, Mat& compIdxs )
{
    Point p;
    for ( p.y = 0; p.y < img.rows; p.y++ )
    {
        for ( p.x = 0; p.x < img.cols; p.x++ )
        {
            Vec3d color = img.at<Vec3b>(p);
            compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ?
                                  bgdGMM.MywhichComponent(color) : fgdGMM.MywhichComponent(color);
        }
    }
}

void MylearnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs, MyGMM& bgdGMM, MyGMM& fgdGMM )
{
    bgdGMM.MyinitLearning();
    fgdGMM.MyinitLearning();
    Point p;
    for ( int ci = 0; ci < MyGMM::componentsCount; ci++ )
    {
        for ( p.y = 0; p.y < img.rows; p.y++ )
        {
            for ( p.x = 0; p.x < img.cols; p.x++ )
            {
                if ( compIdxs.at<int>(p) == ci )
                {
                    if ( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                        bgdGMM.MyaddSample( ci, img.at<Vec3b>(p) );
                    else
                        fgdGMM.MyaddSample( ci, img.at<Vec3b>(p) );
                }
            }
        }
    }
    bgdGMM.MyendLearning();
    fgdGMM.MyendLearning();
}


void MyconstructGCGraph_grab( const Mat& img, const Mat& mask, const MyGMM& bgdGMM, const MyGMM& fgdGMM, double lambda,
                              const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
                              GCGraph<double>& graph )
{
    int vtxCount = img.cols * img.rows,
        edgeCount = 2 * (4 * img.cols * img.rows - 3 * (img.cols + img.rows) + 2);
    graph.create(vtxCount, edgeCount);
    Point p;
    Mat Mat_bgdGMMImage(img.rows, img.cols, CV_8UC1);
    Mat Mat_fgdGMMImage(img.rows, img.cols, CV_8UC1);

    for ( p.y = 0; p.y < img.rows; p.y++ )
    {
        for ( p.x = 0; p.x < img.cols; p.x++)
        {
            // add node
            int vtxIdx = graph.addVtx();
            Vec3b color = img.at<Vec3b>(p);

            // set t-weights
            double fromSource, toSink;
            if ( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                fromSource = -log( bgdGMM(color) );
                toSink = -log( fgdGMM(color) );

                //显示计算的前景背景概率
                Mat_bgdGMMImage.at<unsigned char>(p.y , p.x) = (unsigned char)( fromSource * 10);
                Mat_fgdGMMImage.at<unsigned char>(p.y, p.x) = (int)( toSink * 10);


            }
            else if ( mask.at<uchar>(p) == GC_BGD )
            {
                fromSource = 0;
                toSink = lambda;

                Mat_bgdGMMImage.at<unsigned char>(p.y , p.x) = (unsigned char)( fromSource * 10);
                Mat_fgdGMMImage.at<unsigned char>(p.y, p.x) = (int)( toSink * 10);
            }
            else // GC_FGD
            {
                fromSource = lambda;
                toSink = 0;
                Mat_bgdGMMImage.at<unsigned char>(p.y , p.x) = (unsigned char)( fromSource * 10);
                Mat_fgdGMMImage.at<unsigned char>(p.y, p.x) = (int)( toSink * 10);
            }
            graph.addTermWeights( vtxIdx, fromSource, toSink );

            // set n-weights
            if ( p.x > 0 )
            {
                double w = leftW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx - 1, w, w );
            }
            if ( p.x > 0 && p.y > 0 )
            {
                double w = upleftW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx - img.cols - 1, w, w );
            }
            if ( p.y > 0 )
            {
                double w = upW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx - img.cols, w, w );
            }
            if ( p.x < img.cols - 1 && p.y > 0 )
            {
                double w = uprightW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx - img.cols + 1, w, w );
            }
        }
    }
}