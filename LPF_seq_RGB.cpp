#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>

using namespace std;
using namespace cv;
using namespace chrono;

int main() {
    string imagePath;
    cout << "Enter the image name: ";
    cin >> imagePath;

    Mat img = imread(imagePath,  IMREAD_UNCHANGED);
    if (img.empty()) {
        cerr << "Error: Could not read image file." << endl;
        return 1;
    }

    if (img.channels() == 4) {
        cvtColor(img, img, COLOR_RGBA2RGB);
    }

    int Ksize;
    cout << "Enter the kernel size: ";
    cin >> Ksize;

    while(true)
    {
        if (Ksize % 2 == 1) {
            break;
        }else{
            cout << "the kernel size must be odd number, Enter the kernel size: ";
            cin >> Ksize;
        }
    }

    auto start_time = high_resolution_clock::now();

    Mat filtered_img;
    bilateralFilter(img, filtered_img, Ksize, 75, 75);

    auto end_time = high_resolution_clock::now();

    namedWindow("Original image", WINDOW_NORMAL);
    namedWindow("Filtered image", WINDOW_NORMAL);
    imshow("Original image", img);
    imshow("Filtered image", filtered_img);

    cout << "Execution time: " << duration_cast<microseconds>(end_time - start_time).count() / 1000000.0 << " seconds\n";
    
    waitKey(0);
    return 0;
}
