// 2023/9/21
// zhangzhong
// https://docs.opencv.org/4.x/da/d5c/tutorial_canny_detector.html

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>

using namespace cv;
using namespace std;

Mat src, src_gray;
Mat dst, detected_edges;

int lowThreshold = 0;
const int max_lowThreshold = 500;
const int RATIO = 3;
const int kernel_size = 3;
const char* window_name = "Edge Map";

static void CannyThreshold(int, void*) {
    // First, we blur the image with a filter of kernel size 3
    blur(src_gray, detected_edges, Size(3, 3));
    // Second, we apply the OpenCV function cv::Canny
    Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * RATIO,
          kernel_size, true);

    // We fill a dst image with zeros (meaning the image is completely black)
    dst = Scalar::all(0);
    // copy scr to dst, but only copy the pixels indicated by the mask
    // https://docs.opencv.org/4.x/d3/d63/classcv_1_1Mat.html#a33fd5d125b4c302b0c9aa86980791a77
    // detected_edge act as mask: Its non-zero elements indicate which matrix
    // elements need to be copied. src_gray.copyTo( dst, detected_edges);

    // 把src_gray转成彩色图像
    Mat src_color;
    cvtColor(detected_edges, src_color, COLOR_GRAY2BGR);

    // 然后我们读取groundtruth，将groundtruth中的边缘标记为红色
    // 读取groundtruth
    const char* path = "../imgs/cutleryNoisy.png";
    Mat groundtruth = imread(path, IMREAD_GRAYSCALE);
    cout << groundtruth << endl;
    // 将groundtruth中的边缘标记为红色
    for (int i = 0; i < groundtruth.rows; i++) {
        for (int j = 0; j < groundtruth.cols; j++) {
            if (groundtruth.at<uchar>(i, j) == 254) {
                src_color.at<Vec3b>(i, j)[0] = 0;
                src_color.at<Vec3b>(i, j)[1] = 0;
                src_color.at<Vec3b>(i, j)[2] = 255;
            }
        }
    }

    imshow(window_name, src_color);
}

int main(int argc, char* argv[]) {
    // load an image
    if (argc != 2) {
        std::printf("Usage: %s <path>", argv[0]);
        return 1;
    }
    std::string image_path = samples::findFile(argv[1]);
    src_gray = imread(image_path, IMREAD_GRAYSCALE);

    // The variable to be controlled by the Trackbar is lowThreshold with a
    // limit of max_lowThreshold (which we set to 100 previously) Each time the
    // Trackbar registers an action, the callback function CannyThreshold will
    // be invoked.
    namedWindow(window_name, WINDOW_NORMAL);
    createTrackbar("Min Threshold:", window_name, &lowThreshold,
                   max_lowThreshold, CannyThreshold);

    CannyThreshold(0, 0);
    waitKey();
}
