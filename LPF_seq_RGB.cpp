// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <string>
// #include <chrono>

// using namespace std;
// using namespace cv;
// using namespace chrono;




// void bf(const Mat& src, Mat& dst, int d, double sigmaColor, double sigmaSpace)
// {
//     CV_Assert(d > 0 && sigmaColor > 0 && sigmaSpace > 0);
//     dst.create(src.size(), src.type());

//     int r = d/2;

//     Mat temp;
//     copyMakeBorder(src, temp, r, r, r, r, BORDER_REPLICATE);

//     vector<double> color_weight(256, 0);
//     for(int i = 0; i < 256; i++)
//         color_weight[i] = exp(-0.5 * pow((double)i / sigmaColor, 2));

//     double space_weight;
//     Vec3d sum;
//     double weight;

//     for(int y = r; y < src.rows + r; y++)
//     {
//         for(int x = r; x < src.cols + r; x++)
//         {
//             sum = Vec3d(0, 0, 0);
//             space_weight = 0;
//             for(int j = -r; j <= r; j++)
//             {
//                 for(int i = -r; i <= r; i++)
//                 {
//                     double dist = (double)(i * i + j * j);
//                     weight = color_weight[abs(temp.at<Vec3b>(y + j, x + i)[0] - temp.at<Vec3b>(y, x)[0])] *
//                         exp(-0.5 * dist / (sigmaSpace * sigmaSpace));
//                     sum += static_cast<Vec3d>(temp.at<Vec3b>(y + j, x + i)) * weight;
//                     space_weight += weight;
//                 }
//             }
//             dst.at<Vec3b>(y - r, x - r) = static_cast<Vec3b>(sum / space_weight);
//         }
//     }
// }





// int main() {
//     string imagePath;
//     cout << "Enter the image name: ";
//     cin >> imagePath;

//     Mat img = imread(imagePath,  IMREAD_UNCHANGED);
//     if (img.empty()) {
//         cerr << "Error: Could not read image file." << endl;
//         return 1;
//     }

//     if (img.channels() == 4) {
//         cvtColor(img, img, COLOR_RGBA2RGB);
//     }

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

//     auto start_time = high_resolution_clock::now();

//     Mat filtered_img;
//     bilateralFilter(img, filtered_img, Ksize, 75, 75);

//     auto end_time = high_resolution_clock::now();

//     namedWindow("Original image", WINDOW_NORMAL);
//     namedWindow("Filtered image", WINDOW_NORMAL);
//     imshow("Original image", img);
//     imshow("Filtered image", filtered_img);

//     cout << "Execution time: " << duration_cast<microseconds>(end_time - start_time).count() / 1000000.0 << " seconds\n";
    
//     waitKey(0);
//     return 0;
// }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// #include <iostream>
// #include <opencv2/opencv.hpp>

// using namespace std;
// using namespace cv;

// int main() {
//     // Load color image
//     Mat img = imread("lena.png", IMREAD_COLOR);

//     // Check if image is loaded successfully
//     if(img.empty()) {
//         cout << "Could not read the image" << endl;
//         return -1;
//     }

//     // Convert image to grayscale
//     Mat gray;
//     cvtColor(img, gray, COLOR_BGR2GRAY);

//     // Apply 3x3 low-pass filter
//     Mat filtered;
//     blur(gray, filtered, Size(3,3));

//     // Display original and filtered images
//     imshow("Original", img);
//     imshow("Filtered", filtered);
//     waitKey();

//     return 0;
// }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// #include <opencv2/core.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>
// #include <iostream>

// using namespace cv;
// using namespace std;

// int main(int argc, char** argv) {
//   // Check the number of arguments
//   if (argc != 3) {
//     cout << "Usage: blur <image> <kernel size>\n";
//     return -1;
//   }

//   // Load the image
//   Mat image = imread(argv[1]);
//   if (image.empty()) {
//     cout << "Could not open or find the image.\n";
//     return -1;
//   }

//   // Get the kernel size
//   int kernelSize = atoi(argv[2]);

//   // Create the Gaussian kernel
//   Mat kernel = getGaussianKernel(kernelSize, 0);

//   // Blur the image
//   Mat blurredImage;
//   for (int i = 0; i < image.rows; i++) {
//     for (int j = 0; j < image.cols; j++) {
//       blurredImage.at<uchar>(i, j) = 0;
//       for (int k = 0; k < kernel.rows; k++) {
//         for (int l = 0; l < kernel.cols; l++) {
//           blurredImage.at<uchar>(i, j) += image.at<uchar>(i + k, j + l) * kernel.at<float>(k, l);
//         }
//       }
//     }
//   }

//   // Save the blurred image
//   imwrite("blurred_image.png", blurredImage);

//   // Display the blurred image
//   namedWindow("Blurred Image", WINDOW_AUTOSIZE);
//   imshow("Blurred Image", blurredImage);
//   waitKey(0);

//   return 0;
// }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// #include <opencv2/opencv.hpp>
// #include <iostream>

// using namespace cv;
// using namespace std;

// int main()
// {
//         string imagePath;
//         cout << "Enter the image name: ";
//         cin >> imagePath;

//         Mat img = imread(imagePath,  IMREAD_UNCHANGED);
//         if (img.empty()) {
//             cerr << "Error: Could not read image file." << endl;
//             return 1;
//         }

//     //     if (img.channels() == 4) {
//     //         cvtColor(img, img, COLOR_RGBA2RGB);
//     //     }

//         int Ksize;
//         cout << "Enter the kernel size: ";
//         cin >> Ksize;

//     // Check if the user has provided an input image file path and kernel size
//     // if (argc != 3)
//     // {
//     //     cout << "Usage: ./blur_image <input_image_file_path> <kernel_size>" << endl;
//     //     return -1;
//     // }

//     // Read the input image file
//     // Mat image = imread(argv[1]);

//     // Check if the input image was read successfully
//     // if (image.empty())
//     // {
//     //     cout << "Could not read the input image file: " << argv[1] << endl;
//     //     return -1;
//     // }

//     // Convert the input image to grayscale
//     Mat gray_image;
//     cvtColor(img, gray_image, COLOR_BGR2GRAY);

//     // Convert the kernel size from string to integer
//     // int kernel_size = atoi(argv[2]);

//     // Create the Gaussian blur kernel
//     Mat kernel = getGaussianKernel(Ksize, 0);

//     // Normalize the kernel so that it sums up to 1
//     kernel = kernel * kernel.t();

//     // Apply the Gaussian blur to the grayscale image
//     Mat blurred_image;
//     filter2D(gray_image, blurred_image, -1, kernel);

//     // Write the blurred image to an output file
//     imwrite("blurred_image.png", blurred_image);

//     return 0;
// }

///////////////////////////////////////////////////////////working code not clean////////////////////////////////////////////////////////////////////////

// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <string>
// #include <chrono>

// using namespace std;
// using namespace cv;
// using namespace chrono;



// int main() {
//     string imagePath;
//     cout << "Enter the image name: ";
//     cin >> imagePath;

//     // Mat img = imread(imagePath,  IMREAD_UNCHANGED);
//     // if (img.empty()) {
//     //     cerr << "Error: Could not read image file." << endl;
//     //     return 1;
//     // }

//     // if (img.channels() == 4) {
//     //     cvtColor(img, img, COLOR_RGBA2RGB);
//     // }

//     int Ksize;
//     cout << "Enter the kernel size: ";
//     cin >> Ksize;

//     // Load the image
//     Mat image = imread(imagePath);
    
//     // Create a kernel for blurring
//     int k = Ksize / 2;
//     Mat kernel = Mat::ones(Ksize, Ksize, CV_32F) / (float)(Ksize * Ksize);
    
//     // Create a copy of the image to hold the blurred image
//     Mat blurred_image = image.clone();
    
//     auto start_time = high_resolution_clock::now();
//     // Apply the kernel to each pixel in the image
//     for (int i = k; i < image.rows - k; i++) {
//         for (int j = k; j < image.cols - k; j++) {
//             float sum_r = 0;
//             float sum_g = 0;
//             float sum_b = 0;
//             for (int x = -k; x <= k; x++) {
//                 for (int y = -k; y <= k; y++) {
//                     Vec3b pixel = image.at<Vec3b>(i + x, j + y);
//                     sum_b += pixel[0] * kernel.at<float>(k + x, k + y);
//                     sum_g += pixel[1] * kernel.at<float>(k + x, k + y);
//                     sum_r += pixel[2] * kernel.at<float>(k + x, k + y);
//                 }
//             }
//             Vec3b new_pixel(sum_b, sum_g, sum_r);
//             blurred_image.at<Vec3b>(i, j) = new_pixel;
//         }
//     }

//     auto end_time = high_resolution_clock::now();
//     // imshow("Blurred Image", blurred_image);
    
//     // waitKey(0);
//     // destroyAllWindows();
    
//     namedWindow("Original image", WINDOW_NORMAL);
//     namedWindow("Filtered image", WINDOW_NORMAL);
//     imshow("Original image", image);
//     imshow("Filtered image", blurred_image);

//     cout << "Execution time: " << duration_cast<microseconds>(end_time - start_time).count() / 1000000.0 << " seconds\n";
//     // Save the blurred image
//     // imwrite("blurred_image.jpg", blurred_image);
//     waitKey(0);
//     return 0;
// }

////////////////////////////////////////////////////////cleaned working code////////////////////////////////////////////////////////////////////////


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
        if (Ksize % 2 == 1) {
            break;
        }else{
            cout << "the kernel size must be odd number, Enter the kernel size: ";
            cin >> Ksize;
        }
    }

    // Create a kernel for blurring
    int k = Ksize / 2;
    Mat kernel = Mat::ones(Ksize, Ksize, CV_32F) / (float)(Ksize * Ksize);
    
    // Create a copy of the image to hold the blurred image
    Mat blurred_image = image.clone();
    
    // start timer
    auto start_time = high_resolution_clock::now();

    // Apply the kernel to each pixel in the image
    for (int i = k; i < image.rows - k; i++) {
        for (int j = k; j < image.cols - k; j++) {
            float sum_r = 0;
            float sum_g = 0;
            float sum_b = 0;
            for (int x = -k; x <= k; x++) {
                for (int y = -k; y <= k; y++) {
                    Vec3b pixel = image.at<Vec3b>(i + x, j + y);
                    sum_b += pixel[0] * kernel.at<float>(k + x, k + y);
                    sum_g += pixel[1] * kernel.at<float>(k + x, k + y);
                    sum_r += pixel[2] * kernel.at<float>(k + x, k + y);
                }
            }
            Vec3b new_pixel(sum_b, sum_g, sum_r);
            blurred_image.at<Vec3b>(i, j) = new_pixel;
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