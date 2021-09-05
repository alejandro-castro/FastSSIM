#define _USE_MATH_DEFINES
#include <cmath>
#include "FilterUtils.h"


void idftByAxis(Mat spectrum,  Mat dst, int axisSize, int byRow){
	//Modifies the spectrum
	dft(spectrum, spectrum, DFT_REAL_OUTPUT + DFT_ROWS + DFT_INVERSE + DFT_SCALE, 0);//it is redundant to put nonzeroRows = spectrum.rows
	
	if (byRow==1){
		spectrum(Rect(0, 0, dst.cols, dst.rows)).copyTo(dst);
	}
	else{
		//We consider that the spectrum was calculated using the transposed version of image
		Mat roiDst(spectrum, Rect(0, 0, dst.rows, dst.cols));
		transpose(roiDst, dst);
	}
}


void dftByAxis(Mat img, Mat dst, Size dftSize, int byRow){
	//Mat temp(dftSize, img.type(), Scalar::all(0));
		
	if (img.cols==1 || img.rows==1){
		//Returns a spectrum where each row is the DFT of the vector
		Mat temp2(Size(dftSize.width, 1), img.type(), Scalar::all(0));
		Mat roiImg(temp2, Rect(0, 0, img.total(), 1));
		
		if (img.cols == 1)	img = img.t(); 
		img.copyTo(roiImg);
		
		dft(temp2, temp2, DFT_ROWS, 0);//it is redundant to put nonzeroRows = temp2.height or 1
		/*for(int i=0; i<dftSize.height ;i++){
			dst.row(i) = temp2;
		}*/
		repeat(temp2, dftSize.height, 1, dst);

	}
	else if (byRow==1){
		// copy Img to the top-left corners of temp, we suppose dftSize is greater or equal than img.size()
		Mat roiImg(dst, Rect(0, 0, img.cols, img.rows));
		img.copyTo(roiImg);
		dft(dst, dst, DFT_ROWS, img.rows);
	}
	else{
		// copy the transpose of Img to the top-left corners of temp, we suppose dftSize is greater or equal than img.t().size()
		Mat roiImg(dst, Rect(0, 0, img.rows, img.cols));
		transpose(img, roiImg);
		dft(dst, dst, DFT_ROWS, img.cols);
		//Return the transpose of the DFT, because of the CCS implementation we shouldn't apply transpose to the spectrum carelessly
	}
}



void SepConvFourier2D(Mat img, Mat kernelRowDFT, Mat kernelColDFT, Mat dst, Size dftSize){
	//Never use the DFT of a 1D kernel as dst  because all the rows are pointers to the same data
	//Convolution by row
	Mat imgDFT(Size(dftSize.width, img.rows), img.type(), Scalar::all(0));
	dftByAxis(img, imgDFT, Size(dftSize.width, img.rows), 1);
	mulSpectrums(imgDFT, kernelRowDFT, imgDFT, DFT_ROWS);
	idftByAxis(imgDFT, dst, img.cols, 1); 
	
	//Convolution by col
	Mat temp2(Size(dftSize.height, img.cols), img.type(), Scalar::all(0));
	dftByAxis(dst, temp2, Size(dftSize.height, img.cols), 0);
	mulSpectrums(temp2, kernelColDFT, temp2, DFT_ROWS);
    idftByAxis(temp2, dst, img.rows, 0);
} 


double Gaussian2DPdf(vector <double> x, vector <double> mean, double sigma){
	double norm_sq = pow(x[0] - mean[0], 2) + pow(x[1] - mean[1], 2);
    return exp(-norm_sq/(2*pow(sigma, 2)))/(2*M_PI*pow(sigma, 2));
}


double Gaussian1DPdf(double x, double mean, double sigma){
	double norm_sq = pow(x - mean, 2);
    return exp(-norm_sq/(2*pow(sigma, 2)))/sqrt(2*M_PI*pow(sigma, 2));
}

Mat GaussianWindow(string version, double sigma){
    /*
    Implements a 11 x 11 weighting function using the pdf of a 2-D Gaussian rv and normalizing it to one.
    */
    
    vector<double> mean{0.0, 0.0};
    if (!version.compare("2D")){
    	Mat window(WINDOW_SIZE, WINDOW_SIZE, CV_64F);
    	double sum = 0.0;
    	for(int i=-(WINDOW_SIZE-1)/2; i<(WINDOW_SIZE+1)/2;i++){
    		for(int j=-(WINDOW_SIZE-1)/2; j<(WINDOW_SIZE+1)/2; j++){
				vector <double> pos{i, j};
				double pdf = Gaussian2DPdf(pos, mean, sigma);
				window.at<double>(Point(i+(WINDOW_SIZE-1)/2, j+(WINDOW_SIZE-1)/2)) = pdf;
				sum = sum + pdf;
    		}
    	}      
        //Normalizing the window to 1
        window = window/sum;
        return window;
    }

    else{
    	double sum = 0.0;
    	Mat window(WINDOW_SIZE, 1, CV_64F);
    	for (int i=-(WINDOW_SIZE-1)/2; i<(WINDOW_SIZE+1)/2; i++){
    		double pdf = Gaussian1DPdf(i, 0.0, sigma);
    		window.at<double>(Point(i+(WINDOW_SIZE-1)/2, 1)) = pdf;
    		sum = sum + pdf;
    	}
        //Normalizing the window to 1
        window = window/sum;
    	return window;
    }

}

Mat getGradientImage(Mat img){
	Size imgSize = img.size();
	//Getting the Roberts gradient approximation
	Mat h1 = (Mat_<double>(2,2) << -1, 0, 0, 1);
	Mat h2 = (Mat_<double>(2,2) << 0, -1, 1, 0);
	Mat abs_delta_i(imgSize, CV_64F);
	Mat abs_delta_j(imgSize, CV_64F);
	
	
	filter2D(img, abs_delta_i, -1, h1, Point(-1, -1), 0, BORDER_CONSTANT);
	filter2D(img, abs_delta_j, -1, h2, Point(-1, -1), 0, BORDER_CONSTANT);
	abs_delta_i = abs(abs_delta_i);	abs_delta_j = abs(abs_delta_j);
	
	
	Mat maxImg(imgSize, CV_64F);
	Mat minImg(imgSize, CV_64F);
	max(abs_delta_i, abs_delta_j, maxImg);
	min(abs_delta_i, abs_delta_j, minImg);


	Mat gradientMagnitude(imgSize, CV_64F); 
	gradientMagnitude = maxImg + 0.25*minImg;
	return gradientMagnitude;
}

