#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <omp.h>


using namespace std;
using namespace cv;

int main() {
    string imagePath;
    cout << "Enter the image name: ";
    cin >> imagePath;

    Mat img = imread(imagePath, IMREAD_UNCHANGED);
    if (img.empty()) {
        cerr << "Error: Could not read image file." << endl;
        return 1;
    }

    if (img.channels() == 4) {
        cvtColor(img, img, COLOR_RGBA2RGB);
    }

    Mat filtered_img(img.size(), img.type());

    int num_threads = 1;
    cout << "Enter the number of threads to use: ";
    cin >> num_threads;
    if (num_threads < 1) {
        cerr << "Error: Invalid number of threads." << endl;
        return 1;
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
    
    double start_time = omp_get_wtime();
    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp for
        for (int i = 0; i < img.rows; i++) {
            Mat filtered_row;
            bilateralFilter(img.row(i), filtered_row, Ksize, 75, 75);
            #pragma omp critical
            {
                filtered_row.copyTo(filtered_img.row(i));
            }
        }
    }
    double end_time = omp_get_wtime();

    namedWindow("Original image", WINDOW_NORMAL);
    namedWindow("Filtered image", WINDOW_NORMAL);
    imshow("Original image", img);
    imshow("Filtered image", filtered_img);

    cout << "Execution time: " << end_time - start_time << " seconds\n";
    
    waitKey(0);
    return 0;
}
