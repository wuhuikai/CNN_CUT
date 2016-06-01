#include "GraphCut.hpp"

using namespace cv;

double MycalcBeta(const Mat& img)
{
    double beta = 0;
    for (int y = 0; y < img.rows; y++)
    {
        for (int x = 0; x < img.cols; x++)
        {
            Vec3d color = img.at<Vec3b>(y, x);
            if (x > 0) // left
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y, x - 1);
                beta += diff.dot(diff);
            }
            if (y > 0 && x > 0) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x - 1);
                beta += diff.dot(diff);
            }
            if (y > 0) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x);
                beta += diff.dot(diff);
            }
            if (y > 0 && x < img.cols - 1) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x + 1);
                beta += diff.dot(diff);
            }
        }
    }
    if (beta <= std::numeric_limits<double>::epsilon())
        beta = 0;
    else
        beta = 1.f / (2 * beta / (4 * img.cols * img.rows - 3 * img.cols - 3 * img.rows + 2) );

    return beta;
}

void MycalcNWeights(const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma)
{
    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
    leftW.create( img.rows, img.cols, CV_64FC1 );
    upleftW.create( img.rows, img.cols, CV_64FC1 );
    upW.create( img.rows, img.cols, CV_64FC1 );
    uprightW.create( img.rows, img.cols, CV_64FC1 );
    for (int y = 0; y < img.rows; y++)
    {
        for (int x = 0; x < img.cols; x++)
        {
            Vec3d color = img.at<Vec3b>(y, x);
            if (x - 1 >= 0) // left
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y, x - 1);
                leftW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
            }
            else
                leftW.at<double>(y, x) = 0;
            if (x - 1 >= 0 && y - 1 >= 0) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x - 1);
                upleftW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta * diff.dot(diff));
            }
            else
                upleftW.at<double>(y, x) = 0;
            if (y - 1 >= 0) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x);
                upW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
            }
            else
                upW.at<double>(y, x) = 0;
            if (x + 1 < img.cols && y - 1 >= 0) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x + 1);
                uprightW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta * diff.dot(diff));
            }
            else
                uprightW.at<double>(y, x) = 0;
        }
    }
}

void MyconstructGCGraph(const Mat& img, const Mat& mask, const Mat& Uitary, double lambda,
                        const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
                        GCGraph<double>& graph)
{
    int vtxCount = img.cols * img.rows,
        edgeCount = 2 * (4 * img.cols * img.rows - 3 * (img.cols + img.rows) + 2);
    graph.create(vtxCount, edgeCount);
    Point p;
    Mat Mat_bgdGMMImage(img.rows, img.cols, CV_8UC1);
    Mat Mat_fgdGMMImage(img.rows, img.cols, CV_8UC1);

    for (p.y = 0; p.y < img.rows; p.y++)
    {
        for (p.x = 0; p.x < img.cols; p.x++)
        {
            // add node
            int vtxIdx = graph.addVtx();
            Vec3b color = img.at<Vec3b>(p);

            // set t-weights
            double fromSource, toSink;
            if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD)
            {
                fromSource = 450 * Uitary.at<float>(p);
                toSink = 20 * (1 - Uitary.at<float>(p));

                //显示计算的前景背景概率
                Mat_bgdGMMImage.at<unsigned char>(p.y , p.x) = (unsigned char)( fromSource * 10);
                Mat_fgdGMMImage.at<unsigned char>(p.y, p.x) = (int)( toSink * 10);

            }
            else if (mask.at<uchar>(p) == GC_BGD)
            {
                fromSource = 0;
                toSink = lambda;
                //显示计算的前景背景概率
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
            graph.addTermWeights(vtxIdx, fromSource, toSink);

            // set n-weights
            if (p.x > 0)
            {
                double w = leftW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx - 1, w, w );
            }
            if (p.x > 0 && p.y > 0)
            {
                double w = upleftW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx - img.cols - 1, w, w );
            }
            if (p.y > 0)
            {
                double w = upW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx - img.cols, w, w );
            }
            if (p.x < img.cols - 1 && p.y > 0)
            {
                double w = uprightW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx - img.cols + 1, w, w );
            }
        }
    }
}

void MyestimateSegmentation( GCGraph<double>& graph, Mat& mask )
{
    graph.maxFlow();  //gcgraph.hpp
    Point p;
    for ( p.y = 0; p.y < mask.rows; p.y++ )
    {
        for ( p.x = 0; p.x < mask.cols; p.x++ )
        {
            if ( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                if ( graph.inSourceSegment( p.y * mask.cols + p.x /*vertex index*/ ) )
                    mask.at<uchar>(p) = GC_PR_FGD;
                else
                    mask.at<uchar>(p) = GC_PR_BGD;
            }
        }
    }
}