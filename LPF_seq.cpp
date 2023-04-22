#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    Mat image = imread("milky way.png", IMREAD_GRAYSCALE); // Read input image in grayscale
    if (image.empty()) // Check if the image was successfully loaded
    {
        cout << "Error: Could not read image file." << endl;
        return -1;
    }

    // Set up the kernel for the low pass filter
    float kernel_data[3][3] = { {1.0/9, 1.0/9, 1.0/9},
                                {1.0/9, 1.0/9, 1.0/9},
                                {1.0/9, 1.0/9, 1.0/9} };
    Mat kernel = Mat(3, 3, CV_32F, kernel_data);

    // Apply the low pass filter to the image
    Mat filtered_image;
    filter2D(image, filtered_image, -1, kernel);

    // Display the original and filtered images
    namedWindow("Original Image", WINDOW_NORMAL);
    namedWindow("Filtered Image", WINDOW_NORMAL);
    imshow("Original Image", image);
    imshow("Filtered Image", filtered_image);

    waitKey(0); // Wait for user to press a key

    return 0;
}
