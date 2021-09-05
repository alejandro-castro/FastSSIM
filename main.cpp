#include "FastSSIM.h"
#define CURRENT_CPU_CLOCK 3.6
#define BOVIK_CPU_CLOCK 2.2
#define FACTOR CURRENT_CPU_CLOCK/BOVIK_CPU_CLOCK
#define NSIM 1


int main(){
	int i;
	chrono::time_point<chrono::high_resolution_clock>  start, end;
	chrono::duration<double>  diffTime;
	
	Mat img = imread("./img.png", IMREAD_UNCHANGED);
	Mat imgRef = imread("./imgRef.png", IMREAD_UNCHANGED);
	
	start = chrono::high_resolution_clock::now();
	for(i=0; i<NSIM; i++){
		SepConvSSIM(img, imgRef, 1.5);
	}
    end = chrono::high_resolution_clock::now();
    diffTime = end - start;
    cout << "The processing speed using the SSIM calculated by convolution with separable filters v1 is: " << NSIM/(diffTime.count()*FACTOR) <<" fps."<< endl;
    
    
    start = chrono::high_resolution_clock::now();
	for(i=0; i<NSIM; i++){
		SepConvSSIMv2(img, imgRef, 1.5);
	}
    end = chrono::high_resolution_clock::now();
    diffTime = end - start;
    cout << "The processing speed using the SSIM calculated by convolution with separable filters v2 is: " << NSIM/(diffTime.count()*FACTOR) <<" fps."<< endl;
    
    
    start = chrono::high_resolution_clock::now();
    for(i=0; i<NSIM; i++){
    	FourierSSIM(img, imgRef, 1.5);
    }
    end = chrono::high_resolution_clock::now();
    diffTime = end - start;
    cout << "The processing speed using the SSIM calculated by DFTs is: " << NSIM/(diffTime.count()*FACTOR) <<" fps."<< endl;

	
    start = chrono::high_resolution_clock::now();
    for(i=0; i<NSIM; i++){
    	BovikFastSSIM(img, imgRef, 1.5);
    }
    end = chrono::high_resolution_clock::now();
    diffTime = end - start;
    cout << "The processing speed using the SSIM calculated by FastSSIM algorithm proposed by Bovik without separable filters is: " << NSIM/(diffTime.count()*FACTOR) <<" fps."<< endl;
    
    
    start = chrono::high_resolution_clock::now();
    for(i=0; i<NSIM; i++){
    	BovikFastSSIMv2(img, imgRef, 1.5);
    }
    end = chrono::high_resolution_clock::now();
    diffTime = end - start;
    cout << "The processing speed using the SSIM calculated by FastSSIM algorithm proposed by Bovik with separable filters is: " << NSIM/(diffTime.count()*FACTOR) <<" fps."<< endl;
    
    
	return 0;
}


