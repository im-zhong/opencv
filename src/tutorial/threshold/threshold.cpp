// 2023/10/10
// zhangzhong
// cv lab3

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cassert>
#include <cmath>
#include <doctest/doctest.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

int threshold_value = 0;
int threshold_type = 0; // 0: Binary, 1: Binary Inverted, 2: Threshold
                        // Truncated, 3: Threshold to Zero, 4: Threshold
                        // to Zero Inverted
int adaptive_method = 0;
int const max_value = 255;
int const max_type = 4;
int const max_adaptive_threshold_type = 1; // 0: Binary, 1: Binary Inverted
int const max_binary_value = 255;
int const max_adaptive_method = 1; // 0: MEAN_C, 1: GAUSSIAN_C

Mat src, dst;
const char* winname = "Threshold Demo";
const char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n "
                            "2: Truncated \n 3: To Zero \n 4: To Zero "
                            "Inverted";
const char* trackbar_value = "Threshold Value";

void threshold_demo(int, void*) {
    // 0. Binary
    // 1. Binary Inverted
    // 2. Threshold Truncated
    // 3. Threshold to Zero
    // 4. Threshold to Zero Inverted
    cv::threshold(src, dst, threshold_value, max_binary_value, threshold_type);
    imshow(winname, dst);
}

// 阈值分割 thresholding
// https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
// https://docs.opencv.org/4.x/db/d8e/tutorial_threshold.html
void learn_simple_thresholding() {
    // 1. simple thresholding
    // For every pixel, the same threshold value is applied
    // If the pixel value is smaller than the threshold, it is set to 0,
    // otherwise it is set to a maximum value.

    // https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
    // cv::threshold(src, dst, thresh, maxval, type)
    // 1. The first argument is the source image, which should be a grayscale
    // 2. the second arguments is the output array
    // 3. The third argument is the threshold value which is used to classify
    // the pixel values
    // 4. The forth argument is the maximum value which is assigned to pixel
    // values exceeding the threshold
    // 5. The fifth argument is the thresholding type, such as cv.THRESH_BINARY

    // get image
    const char* filename = "/home/zhangzhong/src/opencv/imgs/orange.jpg";
    src = imread(samples::findFile(filename), IMREAD_GRAYSCALE);
    if (src.empty()) {
        cout << "Cannot read the image: " << filename << std::endl;
        return;
    }

    // create window
    namedWindow(winname, WINDOW_AUTOSIZE);

    // create trackbar to choose type of thresholding
    createTrackbar(trackbar_type, winname, &threshold_type, max_type,
                   threshold_demo);
    // create trackbar to choose threshold vlaue
    createTrackbar(trackbar_value, winname, &threshold_value, max_value,
                   threshold_demo);

    // call the function to initialize
    threshold_demo(0, 0);

    waitKey();
}

// block size 必须是奇数
// 我有一个办法，我们根据索引来生成一个奇数不就行了吗
int block_size = 1;
int real_block_size = 3;
int max_block_size = 21;
double C = 5;

void adaptive_threshold_demo(int, void*) {
    real_block_size = 2 * block_size + 1;
    if (real_block_size < 3) {
        real_block_size = 3;
    }
    // https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gaa42a3e6ef26247da787bf34030ed772c
    cv::adaptiveThreshold(src, dst, max_binary_value, adaptive_method,
                          threshold_type, real_block_size, C);
    imshow(winname, dst);
}

// 效果仍然很差
void learn_adaptive_thresholding() {
    // . Here, the algorithm determines the threshold for a pixel based on a
    // small region around it
    // https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3
    // cv::adaptiveThreshold(InputArray src, OutputArray dst, double maxValue,
    //                       int adaptiveMethod, int thresholdType, int
    //                       blockSize, double C);
    // 1. average thresholding
    // 2. gaussian thresholding: cv::ADAPTIVE_THRESH_GAUSSIAN_C
    // gaussian-weighted sum of the neighbourhood values minus the constant C
    // The blockSize determines the size of the neighbourhood area
    // C is a constant that is subtracted from the mean or weighted sum of the
    // neighbourhood pixels.

    const char* filename = "/home/zhangzhong/src/opencv/imgs/orange.jpg";
    src = imread(samples::findFile(filename), IMREAD_GRAYSCALE);
    if (src.empty()) {
        cout << "Cannot read the image: " << filename << std::endl;
        return;
    }

    // gaussian blur
    GaussianBlur(src, src, Size(5, 5), 0, 0, BORDER_DEFAULT);

    // create window
    namedWindow(winname, WINDOW_AUTOSIZE);

    // create trackbar to choose type of adaptive method
    createTrackbar("adaptive method", winname, &adaptive_method,
                   max_adaptive_method, adaptive_threshold_demo);
    // create trackbar to choose type of thresholding
    createTrackbar("threshold type", winname, &threshold_type,
                   max_adaptive_threshold_type, adaptive_threshold_demo);
    // 这种情况下 好像我们不需要手动选择 threshold_value了
    // create trackbar to choose block size
    createTrackbar("Block Size", winname, &block_size, max_block_size,
                   adaptive_threshold_demo);

    // call the function to initialize
    adaptive_threshold_demo(0, 0);
    waitKey();
}

void learn_ostu() {
    // Consider an image with only two distinct image values (bimodal image)
    // 双峰图像 where the histogram would only consist of two peaks.
    // A good threshold would be in the middle of those two values
    // Otsu's method determines an optimal global threshold value from the image
    // histogram.

    // 高斯滤波好像非常管用

    const char* filename = "/home/zhangzhong/src/opencv/imgs/orange.jpg";
    src = imread(samples::findFile(filename), IMREAD_GRAYSCALE);
    if (src.empty()) {
        cout << "Cannot read the image: " << filename << std::endl;
        return;
    }

    // gaussian blur
    GaussianBlur(src, src, Size(9, 9), 0, 0, BORDER_DEFAULT);

    // ostu
    double ostu_thresh_val = cv::threshold(src, dst, 0, max_binary_value,
                                           THRESH_BINARY | THRESH_OTSU);
    // create window
    namedWindow(winname, WINDOW_AUTOSIZE);
    imshow(winname, dst);
    waitKey();
}

TEST_CASE("test simple thresholding") { learn_simple_thresholding(); }

TEST_CASE("test adaptive thresholding") { learn_adaptive_thresholding(); }

TEST_CASE("test ostu thresholding") { learn_ostu(); }
