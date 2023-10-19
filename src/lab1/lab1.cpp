// 2023/9/19
// zhangzhong
// https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;

void ConvertToGray(const Mat& img, Mat& gray_img) {
    // step2: 将图像转成灰度图
    gray_img = Mat(img.rows, img.cols, CV_8UC1);
    for (int row = 0; row < gray_img.rows; row++) {
        for (int col = 0; col < gray_img.cols; col++) {
            // 通过公式将彩色图像转成灰度图
            // method 1. Average
            // gray_img.at<uchar>(i, j) = (img.at<Vec3b>(i, j)[0] +
            // img.at<Vec3b>(i, j)[1] + img.at<Vec3b>(i, j)[2]) / 3; methdo 2.
            // Weighted Average gray = 0.299*r + 0.587*g + 0.114*b
            // 但是实际上opencv的颜色顺序是BGR，所以应该是
            // reference: https://boofcv.org/index.php?title=Example_RGB_to_Gray
            gray_img.at<uchar>(row, col) = 0.114 * img.at<Vec3b>(row, col)[0] +
                                           0.587 * img.at<Vec3b>(row, col)[1] +
                                           0.299 * img.at<Vec3b>(row, col)[2];
        }
    }
}

void WriteToFile(const Mat& img, const std::string& filename) {
    std::ofstream fout(filename);
    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            fout << (int)img.at<uchar>(row, col) << " ";
        }
        fout << std::endl;
    }
    fout.close();
}

TEST_CASE("test lab1") {
    const char* filename = "/home/zhangzhong/src/opencv/imgs/orange.jpg";
    std::string image_path = samples::findFile(filename);
    Mat img = imread(image_path, IMREAD_COLOR);

    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return;
    }

    imshow("Display window", img);
    // wait for a keystroke in the window
    waitKey(0);

    // 将图像转成灰度图
    Mat gray_img;
    // cvtColor(img, gray_img, COLOR_BGR2GRAY);
    ConvertToGray(img, gray_img);

    imshow("Display window", gray_img);
    waitKey(0);

    // 保存灰度图
    imwrite("imgs/gray_orange.jpg", gray_img);

    // step3: 将每一个像素的灰度值写入一个文本文件
    WriteToFile(gray_img, "imgs/gray_orange.txt");

    // step 4:
    // 对于每一个像素的灰度值乘以5，再加10，如果得到的值超过255，则将此数值直接用255代替，然后输出该图像
    Mat new_img = gray_img.clone();
    for (int row = 0; row < new_img.rows; row++) {
        for (int col = 0; col < new_img.cols; col++) {
            // 防止计算溢出
            int val = ((int)new_img.at<uchar>(row, col) * 5 + 10) % 256;
            // if (val > 255) {
            //     val = 255;
            // }
            new_img.at<uchar>(row, col) = val;
        }
    }
    imshow("Display window", new_img);
    waitKey(0);
    imwrite("imgs/new_gray_orange.jpg", new_img);
}
