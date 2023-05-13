// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <string>
// #include <omp.h>


// using namespace std;
// using namespace cv;

// int main() {
//     string imagePath;
//     cout << "Enter the image name: ";
//     cin >> imagePath;

//     Mat img = imread(imagePath, IMREAD_UNCHANGED);
//     if (img.empty()) {
//         cerr << "Error: Could not read image file." << endl;
//         return 1;
//     }

//     if (img.channels() == 4) {
//         cvtColor(img, img, COLOR_RGBA2RGB);
//     }

//     Mat filtered_img(img.size(), img.type());


//     int Ksize;
//     cout << "Enter the kernel size: ";
//     cin >> Ksize;

//     while(true)
//     {
//         if (Ksize % 2 == 1) {
//             break;
//         }else{
//             cout << "the kernel size must be odd number, Enter the kernel size: ";
//             cin >> Ksize;
//         }
//     }
//     double start_time = omp_get_wtime();
//     #pragma omp parallel num_threads(num_threads)
//     {
//         #pragma omp for
//         for (int i = 0; i < img.rows; i++) {
//             Mat filtered_row;
//             bilateralFilter(img.row(i), filtered_row, Ksize, 75, 75);
//             #pragma omp critical
//             {
//                 filtered_row.copyTo(filtered_img.row(i));
//             }
//         }
//     }
//     double end_time = omp_get_wtime();

//     namedWindow("Original image", WINDOW_NORMAL);
//     namedWindow("Filtered image", WINDOW_NORMAL);
//     imshow("Original image", img);
//     imshow("Filtered image", filtered_img);

//     cout << "Execution time: " << end_time - start_time << " seconds\n";
    
//     waitKey(0);
//     return 0;
// }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <omp.h>

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
    while (true) {
        if(Ksize < 3){
            cout << "enter number more than or equal 3";
        }else if (Ksize % 2 == 1) {
            break;
        } else {
            cout << "the kernel size must be odd number, Enter the kernel size: ";
            cin >> Ksize;
        }
    }

    // ask user for number of threads to use
    int num_threads;
    cout << "Enter the number of threads to use: ";
    cin >> num_threads;

    // Create a kernel for blurring
    int k = Ksize / 2;
    Mat kernel = Mat::ones(Ksize, Ksize, CV_32F) / (float)(Ksize * Ksize);

    // border replication
    Mat paddedImage;
    copyMakeBorder(image, paddedImage, k, k, k, k, BORDER_REPLICATE);

    // Create a copy of the image to hold the blurred image
    Mat blurred_image = image.clone();

    // start timer
    double start_time = omp_get_wtime();

    // Apply the kernel to each pixel in the image
    #pragma omp parallel for num_threads(num_threads)
    for (int i = k; i < paddedImage.rows + k; i++) {
        for (int j = k; j < paddedImage.cols + k; j++) {
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
    double end_time = omp_get_wtime();

    // show both clear and blurred image
    namedWindow("Original image", WINDOW_NORMAL);
    namedWindow("Filtered image", WINDOW_NORMAL);
    imshow("Original image", image);
    imshow("Filtered image", blurred_image);

    // show the execution time in seconds
        cout << "Execution time: " << end_time - start_time << " seconds\n";


    waitKey(0);
    return 0;
}

//g++ -I/usr/local/include/opencv4 -fopenmp -o LPF_openMP_RGB LPF_openMP_RGB.cpp `pkg-config --cflags --libs opencv4`