#define WINDOW_SIZE 11
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void dftByAxis(Mat img, Mat dst, Size dftSize, int byRow=1);

void idftByAxis(Mat spectrum, Mat dst, int axisSize, int byRow=1);

void SepConvFourier2D(Mat img, Mat kernelRowDFT, Mat kernelColDFT, Mat dst, Size dftSize);

double Gaussian2DPdf(vector <double>, vector <double>, double);

double Gaussian1DPdf(double, double, double);

Mat GaussianWindow(string version="2D", double sigma=1.5);

Mat getGradientImage(Mat);
