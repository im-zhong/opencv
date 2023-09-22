// 2023/9/20
// zhangzhong
// basic usage of opencv

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {

    Mat A(3, 3, CV_32F);
    Mat B;

    B.create(A.size(), A.type());
    for (int i = 0; i < B.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            B.at<float>(i, j) = 1;
        }
    }

    cout << A << endl;
    cout << B << endl;

    return 0;
}