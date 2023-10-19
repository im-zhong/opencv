// 2023/10/18
// lab4
// 图像特征提取

#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

using namespace cv;

// TODO: 重构。在所有的遍历中，不再使用i,j 而是使用 row, col代替
std::vector<Point> MyFindZone(const Mat& img) {
    std::vector<Point> zone;
    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            // opencv的图片在索引时 使用(row, col)
            if (img.at<uchar>(row, col) == 255) {
                // BUG:
                // 而在表示坐标时，应该是(x, y) = (col, row)
                zone.push_back(Point(col, row));
            }
        }
    }
    return zone;
}

double MyArea(const Mat& img) {
    auto zone = MyFindZone(img);
    double area = zone.size();
    return area;
}

std::vector<Point> MyFindContour(const Mat& img) {
    std::vector<Point> contour;
    for (int row = 1; row < img.rows - 1; row++) {
        for (int col = 1; col < img.cols - 1; col++) {
            if (img.at<uchar>(row, col) == 255) {
                if (img.at<uchar>(row - 1, col) == 0 ||
                    img.at<uchar>(row + 1, col) == 0 ||
                    img.at<uchar>(row, col - 1) == 0 ||
                    img.at<uchar>(row, col + 1) == 0) {
                    contour.push_back(Point(col, row));
                }
            }
        }
    }
    return contour;
}

// The function iterates over each pixel in the image and checks if it is white
// (255). If it is, the function checks if the pixel is on the edge of the image
// or if any of its neighbors are black (0). If either of these conditions are
// true, the pixel is considered to be part of the perimeter and the perimeter
// counter is incremented.
double MyPerimeter(const Mat& img) {
    auto contour = MyFindContour(img);
    double perimeter = contour.size();
    return perimeter;
}

double MyDiameter(const Mat& img) {
    double diameter = 0;
    auto contour = MyFindContour(img);
    for (const auto& point1 : contour) {
        for (const auto& point2 : contour) {
            double distance =
                sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2));
            if (distance > diameter) {
                diameter = distance;
            }
        }
    }
    return diameter;
}

double MyThinness(const Mat& img) {
    double thinness = 0;
    double area = MyArea(img);
    double diameter = MyDiameter(img);
    thinness = (diameter * diameter) / area;
    return thinness;
}

Point MyCenter(const Mat& img) {
    Point center;
    auto zone = MyFindZone(img);
    double x = 0;
    double y = 0;
    for (const auto& point : zone) {
        x += point.x;
        y += point.y;
    }
    center.x = x / zone.size();
    center.y = y / zone.size();
    return center;
}

struct AspectRatioResult {
    double ratio;
    double radian;
    int max_x;
    int min_x;
    int max_y;
    int min_y;
};

AspectRatioResult MyAspectRatio(const Mat& img) {
    auto contour = MyFindContour(img);
    int min_x = contour[0].x;
    int max_x = contour[0].x;
    int min_y = contour[0].y;
    int max_y = contour[0].y;
    for (const auto& point : contour) {
        if (point.x < min_x) {
            min_x = point.x;
        }
        if (point.x > max_x) {
            max_x = point.x;
        }
        if (point.y < min_y) {
            min_y = point.y;
        }
        if (point.y > max_y) {
            max_y = point.y;
        }
    }
    double aspectRatio = static_cast<double>(max_x - min_x) / (max_y - min_y);
    return AspectRatioResult{
        aspectRatio, 0, max_x, min_x, max_y, min_y,
    };
}

Mat MyRotateMatrix(Point center, double radian) {
    double cos = std::cos(radian);
    double sin = std::sin(radian);
    double x = center.x;
    double y = center.y;
    return (Mat_<double>(3, 3) << cos, sin, (1 - cos) * x - sin * y, -sin, cos,
            sin * x + (1 - cos) * y, 0, 0, 1);
}

Mat MyRotate(const Mat& src, const Mat& rotate_matrix) {
    Mat dest = Mat::zeros(src.size(), src.type());
    for (int row = 0; row < dest.rows; row++) {
        for (int col = 0; col < dest.cols; col++) {
            Mat point = (Mat_<double>(3, 1) << col, row, 1);
            Mat result = rotate_matrix * point;
            int x = result.at<double>(0, 0);
            int y = result.at<double>(1, 0);
            if (x >= 0 && x < dest.cols && y >= 0 && y < dest.rows) {
                dest.at<uchar>(y, x) = src.at<uchar>(row, col);
            }
        }
    }
    return dest;
}

AspectRatioResult MyMinAspectRatio(const Mat& img) {
    auto min_area = std::numeric_limits<double>::max();
    auto center = MyCenter(img);
    AspectRatioResult final_result = {};
    for (int angle = 0; angle < 90; angle++) {
        double radian = static_cast<double>(angle) * CV_PI / 180.0;
        Mat rotate_matrix = MyRotateMatrix(center, radian);
        Mat rotate_img = MyRotate(img, rotate_matrix);
        auto result = MyAspectRatio(rotate_img);

        double area =
            (result.max_x - result.min_x) * (result.max_y - result.min_y);
        if (area < min_area) {
            min_area = area;
            final_result = result;
        }

        // std::cout << "angle: " << angle << " ratio: " << result.ratio
        //           << " area: " << area << std::endl;
    }
    return final_result;
}

void MyBinarize(const Mat& src, Mat& dest) {
    dest = src.clone();
    for (int i = 0; i < dest.rows; i++) {
        for (int j = 0; j < dest.cols; j++) {
            if (dest.at<uchar>(i, j) > 0) {
                dest.at<uchar>(i, j) = 255;
            }
        }
    }
}

int main_back(int argc, char* argv[]) {
    // 读取图片
    const char* filename = "/home/zhangzhong/src/opencv/imgs/orange_split.bmp";
    Mat img = imread(filename, IMREAD_GRAYSCALE);
    assert(img.data != nullptr);

    // 因为图片并不是二值化的 我们将其进行简单的二值化
    // 只要灰度值大于0，直接标记为255
    MyBinarize(img, img);

    // 输出图片的灰度值
    std::ofstream fout;
    fout.open("gray.txt");
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            fout << int(img.at<uchar>(i, j)) << " ";
        }
        fout << std::endl;
    }

    // 输出图片的面积
    double area = MyArea(img);
    std::cout << "area: " << area << " pixels" << std::endl;

    // 输出图片的周长
    double perimeter = MyPerimeter(img);
    std::cout << "perimeter: " << perimeter << " pixels" << std::endl;

    // 输出图片的直径
    double diameter = MyDiameter(img);
    std::cout << "diameter: " << diameter << " pixels" << std::endl;

    // 输出图片薄度
    double thinness = MyThinness(img);
    std::cout << "thinness: " << thinness << std::endl;

    // 输出图片的中心
    Point center = MyCenter(img);
    std::cout << "center: " << center << std::endl;

    // 输出图片的长宽比
    auto result = MyAspectRatio(img);
    std::cout << "aspect ratio: " << result.ratio << std::endl;

    // 输出图片的最小长宽比
    result = MyMinAspectRatio(img);
    std::cout << "minimum aspect ratio: " << result.ratio << std::endl;

    return 0;
}

const char* winname = "rotate";
int angle = 0;

void MyRotateDemo(int, void*) {
    // 读取图片
    const char* filename = "/home/zhangzhong/src/opencv/imgs/orange_split.bmp";
    Mat img = imread(filename, IMREAD_GRAYSCALE);
    assert(img.data != nullptr);

    // 因为图片并不是二值化的 我们将其进行简单的二值化
    // 只要灰度值大于0，直接标记为255
    MyBinarize(img, img);

    // auto contour = MyFindContour(img);
    // for (const auto& point : contour) {
    //     cv::circle(img, point, 3, Scalar(128));
    // }

    // 旋转图片

    Point center = MyCenter(img);
    cv::circle(img, center, 5, Scalar(128));

    double radian = static_cast<double>(angle) * CV_PI / 180.0;
    Mat rotate_matrix = MyRotateMatrix(center, radian);
    // 由于计算误差 会导致旋转之后的图片产生一些空洞
    // 但是应该不影响纵横比的计算
    Mat rotate_img = MyRotate(img, rotate_matrix);
    auto aspect_ratio = MyAspectRatio(rotate_img);
    // // 最好可以把纵横比的矩形画出来
    cv::rectangle(rotate_img, Point(aspect_ratio.min_x, aspect_ratio.min_y),
                  Point(aspect_ratio.max_x, aspect_ratio.max_y), Scalar(128));

    imshow("rotate", rotate_img);
}

// 看一下旋转图片的效果
int main(int argc, char* argv[]) {

    cv::namedWindow(winname);
    cv::createTrackbar("angle", winname, &angle, 90, MyRotateDemo);
    MyRotateDemo(0, 0);
    waitKey();
    return 0;
}
