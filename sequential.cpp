#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>

using namespace std;
using namespace cv;
using namespace chrono;


int main() {

    // ask user for image path
    string imagePath;
    cout << "Enter the image name: ";
    cin >> imagePath;

    // Load the image
    Mat image = imread(imagePath);

    // check the entered image path
    if (image.empty()) {
        cerr << "Error: Could not read image file." << endl;
        return 1;
    }

    // ask user for kernel size 
    int Ksize;
    cout << "Enter the kernel size: ";
    cin >> Ksize;

    // check that the kernel number entered is odd
    while(true)
    {
        if(Ksize < 3){
            cout << "enter number more than or equal 3";
        }else if (Ksize % 2 == 1) {
            break;
        } else {
            cout << "the kernel size must be odd number, Enter the kernel size: ";
            cin >> Ksize;
        }
    }

    // Create a kernel for blurring
    int k = Ksize / 2;
    Mat kernel = Mat::ones(Ksize, Ksize, CV_32F) / (float)(Ksize * Ksize);

    // border replication
    Mat paddedImage;
    copyMakeBorder(image, paddedImage, k, k, k, k, BORDER_REPLICATE);
    
    // Create a copy of the image to hold the blurred image
    Mat blurred_image = image.clone();
    
    // start timer
    auto start_time = high_resolution_clock::now();

    // Apply the kernel to each pixel in the image
    for (int i = k; i <= paddedImage.rows - k; i++) {
        for (int j = k; j <= paddedImage.cols - k; j++) {
            float sum_r = 0;
            float sum_g = 0;
            float sum_b = 0;
            for (int x = -k; x <= k; x++) {
                for (int y = -k; y <= k; y++) {
                    Vec3b pixel = paddedImage.at<Vec3b>(i + x, j + y);
                    sum_b += pixel[0] * kernel.at<float>(k + x, k + y);
                    sum_g += pixel[1] * kernel.at<float>(k + x, k + y);
                    sum_r += pixel[2] * kernel.at<float>(k + x, k + y);
                }
            }
            Vec3b new_pixel(sum_b, sum_g, sum_r);
            blurred_image.at<Vec3b>(i-k, j-k) = new_pixel;
        }
    }

    // end timer
    auto end_time = high_resolution_clock::now();

    // show both clear and blurred image
    namedWindow("Original image", WINDOW_NORMAL);
    namedWindow("Filtered image", WINDOW_NORMAL);
    imshow("Original image", image);
    imshow("Filtered image", blurred_image);

    // show the execution time in seconds
    cout << "Execution time: " << duration_cast<microseconds>(end_time - start_time).count() / 1000000.0 << " seconds\n";

    waitKey(0);
    return 0;
}
