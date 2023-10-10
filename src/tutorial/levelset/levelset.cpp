#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

#include <iostream>
using namespace std;

int main(void) {
    Mat cat_img =
        imread("/home/zhangzhong/src/opencv/imgs/orange.jpg", IMREAD_GRAYSCALE);

    threshold(cat_img, cat_img, 127, 255, THRESH_BINARY);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(cat_img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE,
                 Point(0, 0));

    Mat cat_contours = Mat::zeros(cat_img.size(), CV_8UC3);

    for (size_t i = 0; i < contours.size(); i++)
        drawContours(cat_contours, contours, (int)i, Scalar(255, 0, 0), 2, 8,
                     hierarchy, 0, Point());

    imshow("cat img", cat_contours);

    waitKey(0);

    destroyAllWindows();

    return 0;
}