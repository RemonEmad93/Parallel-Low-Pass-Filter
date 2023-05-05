// #include <iostream>
// #include <stdlib.h>
// #include <math.h>
// #include <stb_image.h>
// #include <stb_image_write.h>

// using namespace std;

// int main()
// {
//   // Read the image.
//   int width, height, channels;
//   unsigned char *imageData = stbi_load("lena.jpg", &width, &height, &channels, 0);

//   // Create a low-pass filter.
//   float kernel[3][3] = {
//     {1 / 9, 1 / 9, 1 / 9},
//     {1 / 9, 1 / 9, 1 / 9},
//     {1 / 9, 1 / 9, 1 / 9}
//   };

//   // Apply the filter to the image.
//   unsigned char *filteredImageData = new unsigned char[width * height * channels];
//   for (int i = 0; i < height; i++) {
//     for (int j = 0; j < width; j++) {
//       float sum = 0.0f;
//       for (int k = -1; k <= 1; k++) {
//         for (int l = -1; l <= 1; l++) {
//           if (i + k >= 0 && i + k < height && j + l >= 0 && j + l < width) {
//             sum += kernel[k + 1][l + 1] * imageData[(i + k) * width * channels + (j + l) * channels];
//           }
//         }
//       }
//       filteredImageData[(i) * width * channels + (j) * channels] = sum;
//     }
//   }

//   // Save the filtered image.
//   stbi_write_jpg("filtered_image.jpg", width, height, channels, filteredImageData, 100);

//   // Free memory.
//   delete[] filteredImageData;

//   return 0;
// }


#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    // Load input image
    Mat img = imread("lena.png");

    // Convert image to grayscale
    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);

    // Define kernel for low-pass filter
    Mat kernel = Mat::ones(3, 3, CV_32F) / 9.0;

    // Apply filter using convolution
    Mat img_lp;
    filter2D(img_gray, img_lp, -1, kernel, Point(-1,-1), 0, BORDER_DEFAULT);

    // Display output image
    imshow("Low-pass filtered image", img_lp);
    waitKey(0);

    return 0;
}
