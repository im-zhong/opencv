// 2023/9/19
// zhangzhong
// https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>
#include <string>

using namespace cv;

void convertToGray(const Mat& img, Mat& gray_img) {
    // step2: 将图像转成灰度图
    gray_img = Mat(img.rows, img.cols, CV_8UC1);
    for (int i = 0; i < gray_img.rows; i++) {
        for (int j = 0; j < gray_img.cols; j++) {
            // 通过公式将彩色图像转成灰度图
            // method 1. Average
            // gray_img.at<uchar>(i, j) = (img.at<Vec3b>(i, j)[0] +
            // img.at<Vec3b>(i, j)[1] + img.at<Vec3b>(i, j)[2]) / 3; methdo 2.
            // Weighted Average gray = 0.299*r + 0.587*g + 0.114*b
            // 但是实际上opencv的颜色顺序是BGR，所以应该是
            // reference: https://boofcv.org/index.php?title=Example_RGB_to_Gray
            gray_img.at<uchar>(i, j) = 0.114 * img.at<Vec3b>(i, j)[0] +
                                       0.587 * img.at<Vec3b>(i, j)[1] +
                                       0.299 * img.at<Vec3b>(i, j)[2];
        }
    }
}

int main(int argc, char* argv[]) {

    // get the first arg as path
    if (argc != 2) {
        std::printf("Usage: %s <path>\n", argv[0]);
        return 1;
    }

    std::string image_path = samples::findFile(argv[1]);
    Mat img = imread(image_path, IMREAD_COLOR);

    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    imshow("Display window", img);
    // wait for a keystroke in the window
    waitKey(0);

    // 将图像转成灰度图
    Mat gray_img;
    // cvtColor(img, gray_img, COLOR_BGR2GRAY);
    convertToGray(img, gray_img);

    imshow("Display window", gray_img);
    waitKey(0);

    // 保存灰度图
    imwrite("imgs/gray_orange.jpg", gray_img);

    // step3: 将每一个像素的灰度值写入一个文本文件
    std::ofstream fout("imgs/gray_orange.txt");
    for (int i = 0; i < gray_img.rows; i++) {
        for (int j = 0; j < gray_img.cols; j++) {
            fout << (int)gray_img.at<uchar>(i, j) << " ";
        }
        fout << std::endl;
    }
    fout.close();

    // step 4:
    // 对于每一个像素的灰度值乘以5，再加10，如果得到的值超过255，则将此数值直接用255代替，然后输出该图像
    Mat new_img = gray_img.clone();
    for (int i = 0; i < new_img.rows; i++) {
        for (int j = 0; j < new_img.cols; j++) {
            // 防止计算溢出
            int val = ((int)new_img.at<uchar>(i, j) * 5 + 10) % 256;
            // if (val > 255) {
            //     val = 255;
            // }
            new_img.at<uchar>(i, j) = val;
        }
    }
    imshow("Display window", new_img);
    waitKey(0);
    imwrite("imgs/new_gray_orange.jpg", new_img);

    return 0;
}
