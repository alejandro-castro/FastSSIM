#include "FastSSIM.h"

void getCummulativeImage(Mat img, Mat dst){
	double leftCorner, upperCorner, upperLeftCorner;
	for(int i=0; i<img.rows; i++){
		for(int j=0; j<img.cols; j++){
			leftCorner = (j>0)? dst.at<double>(i, j-1) : 0.0;
			upperCorner = (i>0)? dst.at<double>(i-1, j) : 0.0;
			upperLeftCorner = (i>0 && j>0)? dst.at<double>(i-1, j-1) : 0.0;
			dst.at<double>(i, j) =  (img.at<double>(i, j) + leftCorner) +  (upperCorner - upperLeftCorner);
		}
	}

}
 
void getMeanByCummulativeImage(Mat accumImg, Mat dst){
	// img and dst cannot be the same image, because it will corrupt the operation
	int xLeft, xRight, yUp, yDown;
	/*for(int i=0; i<accumImg.rows; i++){
		for(int j=0; j<accumImg.cols; j++){
			xLeft = max(j - (WINDOW_SIZE-1)/2, 0);
			xRight = min(j + (WINDOW_SIZE-1)/2, accumImg.cols-1);
			yUp = max(i - (WINDOW_SIZE-1)/2, 0);
			yDown = min(i + (WINDOW_SIZE-1)/2, accumImg.rows-1);

			dst.at<double>(i, j) = (accumImg.at<double>(yDown, xRight) + accumImg.at<double>(yUp, xLeft)) - (accumImg.at<double>(yUp, xRight) + accumImg.at<double>(yDown, xLeft));
		}
	}*/
	
	Mat temp(accumImg.rows+WINDOW_SIZE-1, accumImg.cols+WINDOW_SIZE-1, accumImg.type());
	copyMakeBorder(accumImg, temp, (WINDOW_SIZE-1)/2, (WINDOW_SIZE-1)/2,
               (WINDOW_SIZE-1)/2, (WINDOW_SIZE-1)/2, BORDER_REPLICATE);
	

	Mat leftUpperImg(temp, Rect(0, 0, accumImg.cols, accumImg.rows));
	Mat leftImg(temp, Rect(0, (WINDOW_SIZE-1)/2, accumImg.cols, accumImg.rows));
	Mat upperImg(temp, Rect((WINDOW_SIZE-1)/2, 0, accumImg.cols, accumImg.rows));
	//Mat img(accumImg, Rect((WINDOW_SIZE-1)/2, (WINDOW_SIZE-1)/2, accumImg.cols, accumImg.rows));
	
	dst = (accumImg + leftUpperImg) - (leftImg + upperImg);
}


void getMeanImage(Mat img, Mat dst){
	Mat tmp(img.size()+Size(1, 1), img.type());
	integral(img, tmp, -1);
	Mat accumImg(tmp, Rect(1, 1, img.cols, img.rows));
	
	//getCummulativeImage(img, tmp);
	getMeanByCummulativeImage(accumImg, dst);
}


Mat FourierSSIM(Mat img, Mat imgRef, double radius, int dynamicRange, vector<double> exponents){
	Size imgSize = img.size();
	
    //We are doing this because of casting problems. This change the image only here, because C++ copies the mat object, but including the pointer to the data, therefore it is inexpensive
	img.convertTo(img, CV_64F); imgRef.convertTo(imgRef, CV_64F);

	//Initialization of parameters for the SSIM
    if (!dynamicRange){
        int nbits = 8*img.elemSize();
        dynamicRange = pow(2, nbits) - 1;
    }

	double alpha = exponents[0], beta = exponents[1], gamma = exponents[2];
    double C1 = pow(0.01*dynamicRange, 2);
    double C2 = pow(0.03*dynamicRange, 2);
    double C3 = C2/2;


	//Fourier initializations
	Size dftSize;
    //calculate the size of DFT transform
    dftSize.width = getOptimalDFTSize(img.cols + WINDOW_SIZE - 1);//For the conv in freq by rows
    dftSize.height = getOptimalDFTSize(img.rows + WINDOW_SIZE - 1);//For the conv in freq by cols
    

    //Definition of the circular-symetric Gaussian window of length WINDOW_SIZE
	Mat weights = GaussianWindow("1D", radius);
	Mat weightsDFTbyRow(Size(dftSize.width, img.rows), img.type(), Scalar::all(0)); 
	Mat weightsDFTbyCol(Size(dftSize.height, img.cols), img.type(), Scalar::all(0)); 
	
	dftByAxis(weights, weightsDFTbyRow, Size(dftSize.width, img.rows), 1);
	dftByAxis(weights, weightsDFTbyCol, Size(dftSize.height, img.cols), 0);
	
	//Calculation of some statistics
    Mat mu_x(imgSize, CV_64F);
    Mat autocorr_x(imgSize, CV_64F);
	Mat var_x(imgSize, CV_64F);
	
	SepConvFourier2D(img, weightsDFTbyRow, weightsDFTbyCol, mu_x, dftSize);
   	SepConvFourier2D(img.mul(img), weightsDFTbyRow, weightsDFTbyCol, autocorr_x, dftSize);
    var_x = autocorr_x - mu_x.mul(mu_x);



    Mat mu_y(imgSize, CV_64F);
	Mat autocorr_y(imgSize, CV_64F);
	Mat var_y(imgSize, CV_64F);
	
	SepConvFourier2D(imgRef, weightsDFTbyRow, weightsDFTbyCol, mu_y, dftSize);
   	SepConvFourier2D(imgRef.mul(imgRef), weightsDFTbyRow, weightsDFTbyCol, autocorr_y, dftSize);
    var_y = autocorr_y - mu_y.mul(mu_y);
    
	
	
	Mat corr_x_y(imgSize, CV_64F);
	Mat covar(imgSize, CV_64F);
	Mat std_xstd_y(imgSize, CV_64F);
	
	SepConvFourier2D(img.mul(imgRef), weightsDFTbyRow, weightsDFTbyCol, corr_x_y, dftSize);
    covar = corr_x_y - mu_y.mul(mu_x);

    sqrt(var_x.mul(var_y), std_xstd_y); 
    
    
    
    //Allocation of memory space for the luminance, contrast, structure and SSIM images. This is a need to make the SSIM the most generic possible using alpha, beta and gamma exponents as parameters
    Mat l(imgSize, CV_64F);
    Mat c(imgSize, CV_64F);
    Mat s(imgSize, CV_64F);
    Mat SSIM_map(imgSize, CV_64F);
        
        
    //Luminance metric
    l = (2*mu_x.mul(mu_y) + C1).mul(1/(mu_x.mul(mu_x) + mu_y.mul(mu_y) + C1));
    pow(l, alpha, l);
    
    //Contrast metric
    c = (2*std_xstd_y + C2)/(var_x + var_y + C2);
	pow(c, beta, c);
    
    //Structure metric
    s = (covar + C3)/(std_xstd_y+ C3);
    pow(s, gamma, s);
    
    SSIM_map = l.mul(c).mul(s);
    return SSIM_map;  
}


Mat SepConvSSIM(Mat img, Mat imgRef, double radius, int dynamicRange, vector<double> exponents){
	Size imgSize = img.size();
	
    //We are doing this because of casting problems. This change the image only here, because C++ copies the mat object, but including the pointer to the data, therefore it is inexpensive
	img.convertTo(img, CV_64F); imgRef.convertTo(imgRef, CV_64F);

	//Initialization of parameters for the SSIM
    if (!dynamicRange){
        int nbits = 8*img.elemSize();
        dynamicRange = pow(2, nbits) - 1;
    }

	double alpha = exponents[0], beta = exponents[1], gamma = exponents[2];
    double C1 = pow(0.01*dynamicRange, 2);
    double C2 = pow(0.03*dynamicRange, 2);
    double C3 = C2/2;

	//Definition of the circular-symetric Gaussian window of length WINDOW_SIZE
	Mat weights = GaussianWindow("1D", radius);
	
	
	
	//Allocation of memory space for the img that will be used
    Mat mu_x(imgSize, CV_64F);
	Mat autocorr_x(imgSize, CV_64F);
	Mat var_x(imgSize, CV_64F);

    Mat mu_y(imgSize, CV_64F);
	Mat autocorr_y(imgSize, CV_64F);
	Mat var_y(imgSize, CV_64F);
	
	
	Mat corr_x_y(imgSize, CV_64F);
	Mat covar(imgSize, CV_64F);
	Mat std_xstd_y(imgSize, CV_64F);
	
	
	
	//Calculation of all the statistics mentioned before
	//Don't care if we are doing cross-correlation or convolution because the filter is symmetric    
    sepFilter2D(img, mu_x, -1, weights, weights, Point(-1, -1), 0, BORDER_CONSTANT);
    sepFilter2D(img.mul(img), autocorr_x, -1, weights, weights, Point(-1, -1), 0, BORDER_CONSTANT);
    var_x = autocorr_x - mu_x.mul(mu_x);
    
    sepFilter2D(imgRef, mu_y, -1, weights, weights, Point(-1, -1), 0, BORDER_CONSTANT);
    sepFilter2D(imgRef.mul(imgRef), autocorr_y, -1, weights, weights, Point(-1, -1), 0, BORDER_CONSTANT);
    var_y = autocorr_y - mu_y.mul(mu_y);
    
	
    sepFilter2D(img.mul(imgRef), corr_x_y, -1, weights, weights, Point(-1, -1), 0, BORDER_CONSTANT);
    covar = corr_x_y - mu_y.mul(mu_x);

    sqrt(var_x.mul(var_y), std_xstd_y); 
    
    
    
    //Allocation of memory space for the luminance, contrast, structure and SSIM images. This is a need to make the SSIM the most generic possible using alpha, beta and gamma exponents as parameters
    Mat l(imgSize, CV_64F);
    Mat c(imgSize, CV_64F);
    Mat s(imgSize, CV_64F);
    Mat SSIM_map(imgSize, CV_64F);
        
        
    //Luminance metric
    l = (2*mu_x.mul(mu_y) + C1).mul(1/(mu_x.mul(mu_x) + mu_y.mul(mu_y) + C1));
    pow(l, alpha, l);
    
    //Contrast metric
    c = (2*std_xstd_y + C2)/(var_x + var_y + C2);
	pow(c, beta, c);
    
    //Structure metric
    s = (covar + C3)/(std_xstd_y+ C3);
    pow(s, gamma, s);
    
    SSIM_map = l.mul(c).mul(s);
    return SSIM_map;  
}


Mat SepConvSSIMv2(Mat img, Mat imgRef, double radius, int dynamicRange, vector<double> exponents){
	Size imgSize = img.size();
	
    //We are doing this because of casting problems. This change the image only here, because C++ copies the mat object, but including the pointer to the data, therefore it is inexpensive
	img.convertTo(img, CV_64F); imgRef.convertTo(imgRef, CV_64F);

	//Initialization of parameters for the SSIM
    if (!dynamicRange){
        int nbits = 8*img.elemSize();
        dynamicRange = pow(2, nbits) - 1;
    }

	double alpha = exponents[0], beta = exponents[1], gamma = exponents[2];
    double C1 = pow(0.01*dynamicRange, 2);
    double C2 = pow(0.03*dynamicRange, 2);
    double C3 = C2/2;

	//Definition of the circular-symetric Gaussian window of length WINDOW_SIZE
	Mat weights = GaussianWindow("1D", radius);
	
	
	
	//Allocation of memory space for the img that will be used
    Mat mu_x(imgSize, CV_64F);
	Mat temp_x(imgSize, CV_64F);
	Mat var_x(imgSize, CV_64F);

    Mat mu_y(imgSize, CV_64F);
	Mat temp_y(imgSize, CV_64F);
	Mat var_y(imgSize, CV_64F);
	
	
	Mat corr_x_y(imgSize, CV_64F);
	Mat covar(imgSize, CV_64F);
	Mat std_xstd_y(imgSize, CV_64F);
	
	
	
	//Calculation of all the statistics mentioned before
	//Don't care if we are doing cross-correlation or convolution because the filter is symmetric    
    sepFilter2D(img, mu_x, -1, weights, weights, Point(-1, -1), 0, BORDER_CONSTANT);
    temp_x = img - mu_x;
    sepFilter2D(temp_x.mul(temp_x), var_x, -1, weights, weights, Point(-1, -1), 0, BORDER_CONSTANT);
    
    
    sepFilter2D(imgRef, mu_y, -1, weights, weights, Point(-1, -1), 0, BORDER_CONSTANT);
	temp_y = imgRef - mu_y;
    sepFilter2D(temp_y.mul(temp_y), var_y, -1, weights, weights, Point(-1, -1), 0, BORDER_CONSTANT);
    
	
    sepFilter2D(img.mul(imgRef), corr_x_y, -1, weights, weights, Point(-1, -1), 0, BORDER_CONSTANT);
    covar = corr_x_y - mu_y.mul(mu_x);

    sqrt(var_x.mul(var_y), std_xstd_y); 
    
    
    
    //Allocation of memory space for the luminance, contrast, structure and SSIM images. This is a need to make the SSIM the most generic possible using alpha, beta and gamma exponents as parameters
    Mat l(imgSize, CV_64F);
    Mat c(imgSize, CV_64F);
    Mat s(imgSize, CV_64F);
    Mat SSIM_map(imgSize, CV_64F);
        
        
    //Luminance metric
    l = (2*mu_x.mul(mu_y) + C1).mul(1/(mu_x.mul(mu_x) + mu_y.mul(mu_y) + C1));
    pow(l, alpha, l);
    
    //Contrast metric
    c = (2*std_xstd_y + C2)/(var_x + var_y + C2);
	pow(c, beta, c);
    
    //Structure metric
    s = (covar + C3)/(std_xstd_y+ C3);
    pow(s, gamma, s);
    
    SSIM_map = l.mul(c).mul(s);
    return SSIM_map;  
}

Mat BovikFastSSIM(Mat img, Mat imgRef, double radius, int dynamicRange, vector<double> exponents){
	Size imgSize = img.size();
	
    //We are doing this because of casting problems. This change the image only here, because C++ copies the mat object, but including the pointer to the data, therefore it is inexpensive
	img.convertTo(img, CV_64F); imgRef.convertTo(imgRef, CV_64F);

	//Initialization of parameters for the SSIM
    if (!dynamicRange){
        int nbits = 8*img.elemSize();
        dynamicRange = pow(2, nbits) - 1;
    }

	double alpha = exponents[0], beta = exponents[1], gamma = exponents[2];
    double C1 = pow(0.01*dynamicRange, 2);
    double C2 = pow(0.03*dynamicRange, 2);
    double C3 = C2/2;

	//Definition of the circular-symetric Gaussian window of length WINDOW_SIZE
	Mat weights = GaussianWindow("2D", radius);
	
	Mat gradientImg = getGradientImage(img);
	Mat gradientImgRef = getGradientImage(imgRef);
	
	
	//Allocation of memory space for the img that will be used
    Mat mu_x(imgSize, CV_64F);
	Mat std_x(imgSize, CV_64F);

    Mat mu_y(imgSize, CV_64F);
	Mat std_y(imgSize, CV_64F);
		
	
	Mat covar(imgSize, CV_64F);
	Mat std_xstd_y(imgSize, CV_64F);
	
	
	
	//Calculation of all the statistics mentioned before
	//Don't care if we are doing cross-correlation or convolution because the filter is symmetric    
    filter2D(img, mu_x, -1, weights, Point(-1, -1), 0, BORDER_CONSTANT);//getMeanImage(img, mu_x);//
    filter2D(gradientImg, std_x, -1, weights, Point(-1, -1), 0, BORDER_CONSTANT);
    
    filter2D(imgRef, mu_y, -1, weights, Point(-1, -1), 0, BORDER_CONSTANT);//getMeanImage(imgRef, mu_y);//
    filter2D(gradientImgRef, std_y, -1, weights, Point(-1, -1), 0, BORDER_CONSTANT);
    
	
    filter2D(gradientImg.mul(gradientImgRef), covar, -1, weights, Point(-1, -1), 0, BORDER_CONSTANT);

    std_xstd_y = std_x.mul(std_y);
    
    
    
    //Allocation of memory space for the luminance, contrast, structure and SSIM images. This is a need to make the SSIM the most generic possible using alpha, beta and gamma exponents as parameters
    Mat l(imgSize, CV_64F);
    Mat c(imgSize, CV_64F);
    Mat s(imgSize, CV_64F);
    Mat SSIM_map(imgSize, CV_64F);
        
        
    //Luminance metric
    l = (2*mu_x.mul(mu_y) + C1).mul(1/(mu_x.mul(mu_x) + mu_y.mul(mu_y) + C1));
    pow(l, alpha, l);
    
    //Contrast metric
    c = (2*std_xstd_y + C2)/(std_x.mul(std_x) + std_y.mul(std_y) + C2);
	pow(c, beta, c);
    
    //Structure metric
    s = (covar + C3)/(std_xstd_y+ C3);
    pow(s, gamma, s);
    
    SSIM_map = l.mul(c).mul(s);

    return SSIM_map;  
}


Mat BovikFastSSIMv2(Mat img, Mat imgRef, double radius, int dynamicRange, vector<double> exponents){
	Size imgSize = img.size();
	
    //We are doing this because of casting problems. This change the image only here, because C++ copies the mat object, but including the pointer to the data, therefore it is inexpensive
	img.convertTo(img, CV_64F); imgRef.convertTo(imgRef, CV_64F);

	//Initialization of parameters for the SSIM
    if (!dynamicRange){
        int nbits = 8*img.elemSize();
        dynamicRange = pow(2, nbits) - 1;
    }

	double alpha = exponents[0], beta = exponents[1], gamma = exponents[2];
    double C1 = pow(0.01*dynamicRange, 2);
    double C2 = pow(0.03*dynamicRange, 2);
    double C3 = C2/2;

	//Definition of the circular-symetric Gaussian window of length WINDOW_SIZE
	Mat weights = GaussianWindow("1D", radius);
	
	Mat gradientImg = getGradientImage(img);
	Mat gradientImgRef = getGradientImage(imgRef);
	
	
	//Allocation of memory space for the img that will be used
    Mat mu_x(imgSize, CV_64F);
	Mat std_x(imgSize, CV_64F);

    Mat mu_y(imgSize, CV_64F);
	Mat std_y(imgSize, CV_64F);
		
	
	Mat covar(imgSize, CV_64F);
	Mat std_xstd_y(imgSize, CV_64F);
	
	
	
	//Calculation of all the statistics mentioned before
	//Don't care if we are doing cross-correlation or convolution because the filter is symmetric    
    sepFilter2D(img, mu_x, -1, weights, weights, Point(-1, -1), 0, BORDER_CONSTANT);
    sepFilter2D(gradientImg, std_x, -1, weights, weights, Point(-1, -1), 0, BORDER_CONSTANT);
    
    sepFilter2D(imgRef, mu_y, -1, weights, weights, Point(-1, -1), 0, BORDER_CONSTANT);
    sepFilter2D(gradientImgRef, std_y, -1, weights, weights, Point(-1, -1), 0, BORDER_CONSTANT);
    
	
    sepFilter2D(gradientImg.mul(gradientImgRef), covar, -1, weights, weights, Point(-1, -1), 0, BORDER_CONSTANT);

    std_xstd_y = std_x.mul(std_y);
    
    
    
    //Allocation of memory space for the luminance, contrast, structure and SSIM images. This is a need to make the SSIM the most generic possible using alpha, beta and gamma exponents as parameters
    Mat l(imgSize, CV_64F);
    Mat c(imgSize, CV_64F);
    Mat s(imgSize, CV_64F);
    Mat SSIM_map(imgSize, CV_64F);
        
        
    //Luminance metric
    l = (2*mu_x.mul(mu_y) + C1).mul(1/(mu_x.mul(mu_x) + mu_y.mul(mu_y) + C1));
    pow(l, alpha, l);
    
    //Contrast metric
    c = (2*std_xstd_y + C2)/(std_x.mul(std_x) + std_y.mul(std_y) + C2);
	pow(c, beta, c);
    
    //Structure metric
    s = (covar + C3)/(std_xstd_y+ C3);
    pow(s, gamma, s);
    
    SSIM_map = l.mul(c).mul(s);

    return SSIM_map;  
}

