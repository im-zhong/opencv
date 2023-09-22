// 2023/9/21
// zhangzhong
// test for filter

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    Mat sobel_x = (Mat_<double>(3, 3) << -1, 0, 1,
                                      -2, 0, 2,
                                      -1, 0, 1);
    // opencv types: https://docs.opencv.org/3.4/d2/d10/core_2include_2opencv2_2core_2hal_2interface_8h.html
    cout << "sobel_x.depth() = " << sobel_x.depth() << endl;                               

    // make a simple image
    Mat img = (Mat_<double>(5, 5) << 10, 10, 10, 0, 0,
                                    10, 10, 10, 0, 0,
                                    10, 10, 10, 0, 0,
                                    10, 10, 10, 0, 0,
                                    10, 10, 10, 0, 0);
    Mat sobel_img;
    // https://docs.opencv.org/4.8.0/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
    filter2D(img, sobel_img, img.depth(), sobel_x);
    cout << "sobel_img = " << sobel_img << endl;
    // 这就对了呀！！！
    // 原来需要指定depth的类型！！！
    // 大家都是double类型 才会有负的梯度
}