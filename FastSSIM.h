#include <iostream>
#include <chrono> 
#include "FilterUtils.h"

Mat SepConvSSIM(Mat img, Mat imgRef, double radius=1.5, int dynamicRange=0, vector<double> exponents={1.0, 1.0, 1.0});
Mat SepConvSSIMv2(Mat img, Mat imgRef, double radius=1.5, int dynamicRange=0, vector<double> exponents={1.0, 1.0, 1.0});
Mat FourierSSIM(Mat img, Mat imgRef, double radius=1.5, int dynamicRange=0, vector<double> exponents={1.0, 1.0, 1.0});
Mat BovikFastSSIM(Mat img, Mat imgRef, double radius=1.5, int dynamicRange=0, vector<double> exponents={1.0, 1.0, 1.0});
Mat BovikFastSSIMv2(Mat img, Mat imgRef, double radius=1.5, int dynamicRange=0, vector<double> exponents={1.0, 1.0, 1.0});

