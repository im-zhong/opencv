// 2023/9/21
// zhangzhong
// Operations with images
// https://docs.opencv.org/4.8.0/d5/d98/tutorial_mat_operations.html

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>

using namespace std;
using namespace cv;

// int learning(int argc, char* argv[]) {
//     // get the first arg as path
//     if (argc != 2) {
//         std::printf("Usage: %s <path>\n", argv[0]);
//         return 1;
//     }
    
//     // load an image from a file
//     // If you read a jpg file, a 3 channel image is created by default.
//     Mat img = imread(argv[1]);
//     // if you need a grayscale image
//     Mat img_gray = imread(argv[1], IMREAD_GRAYSCALE);

//     // Format of the file is determined by its content (first few bytes)
//     // to save an image to a file
//     imwrite("lena_gray.jpg", img_gray);
//     // Format of the file is determined by its extension.
//     // Use cv::imdecode and cv::imencode to read and write an image from/to memory rather than a file.

//     // Access pixel
//     // channel-1
//     int row = 0;
//     int col = 0;
//     Scalar intensity = img_gray.at<uchar>(row, col);

//     // channel-3, BGR
//     Vec3b intensity = img.at<Vec3b>(row, col);
//     uchar blue = intensity.val[0];
//     uchar green = intensity.val[1];
//     uchar red = intensity.val[2];

//     // use the same method for floating-point images
//     // Vec3f intensity = img.at<Vec3f>(y, x);
//     // float blue = intensity.val[0];
//     // float green = intensity.val[1];
//     // float red = intensity.val[2];

//     // The same method can be used to change pixel intensities:
//     img_gray.at<uchar>(row, col) = 128;

//     // primitive operations

//     // make a black image from an existing grayscale image
//     // 这样相当于给所有的像素赋值为零
//     img = Scalar(0);

//     // select a region of interest (ROI) in an image
//     Rect r(10, 10, 100, 100);
//     Mat smallImg = img_gray(r);

//     // conversion from color to grayscale
//     // cvtColor(img, img_gray, COLOR_BGR2GRAY);

//     // change image type from 8UC1 to 32FC1
//     // 不知道参数用自己会不会出问题
//     img_gray.convertTo(img_gray, CV_32FC1);

//     // uchar, 8U type of image can be shown with imshow

//     // A 32F image need to be converted to 8U type
//     // [min-max] -> [0-255]
//     // https://docs.opencv.org/4.8.0/d3/d63/classcv_1_1Mat.html#adf88c60c5b4980e05bb556080916978b
//     // m(x,y) = saturate_cast<rType>(α(∗this)(x,y) + β)
//     // tmp = src(x, y) * alpha + beta
//     // dst(x, y) = saturate_cast<uchar>(tmp)
//     img.convertTo(img, CV_8UC1, 255.0/(max-min), -min*255.0/(max-min));
// }

int main(int argc, char* argv[]) {
    // create a image and save it
    // TIP: 如果想要初始化参数，一定要使用Scalar, 直接填0会报错
    Mat img(300, 400, CV_8UC1, Scalar(0));
    // cout << img << endl;

    // 填充一块矩形区域
    for (int row = 10; row < 90; ++row) {
        for (int col = 10; col < 90; ++col) {
            img.at<uchar>(row, col) = 255;
        }
    }

    // 展示图像
    imshow("Display window", img);
    waitKey();

    // 保存图像
    imwrite("rect.jpg", img);

    // 没问题，那就是我的算法写错了

    return 0;
}