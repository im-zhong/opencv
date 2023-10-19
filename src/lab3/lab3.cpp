// 2023/10/10
// zhangzhong
// cv lab3

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cassert>
#include <cmath>
#include <doctest/doctest.h>
#include <fstream>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

struct MyCircle {
    Point center;
    int radius;
};

void WriteImageToFile(Mat gray_img, const std::string& filename) {
    std::ofstream fout(filename);
    for (int i = 0; i < gray_img.rows; i++) {
        for (int j = 0; j < gray_img.cols; j++) {
            // 这个类型可能不对呀
            fout << (int)gray_img.at<float>(i, j) << " ";
        }
        fout << std::endl;
    }
    fout.close();
}

double Jaccard(Mat img, Mat ground_truth) {
    assert(img.size() == ground_truth.size());

    // S1: segmentation, S2: ground truth
    // J(S1, S2) = |S1 ∩ S2| / |S1 ∪ S2|
    int intersection = 0;
    int union_ = 0;
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            if (img.at<uchar>(y, x) == 255 &&
                ground_truth.at<uchar>(y, x) == 255) {
                intersection++;
            }
            if (img.at<uchar>(y, x) == 255 ||
                ground_truth.at<uchar>(y, x) == 255) {
                union_++;
            }
        }
    }
    return (double)intersection / union_;
}

template <typename T>
void ROI(Mat img, float ratio_x = 0.5, float ratio_y = 0.6) {
    // only consider the up-right corner of the image
    // other parts are set to zero
    int x0 = int(img.cols * (1 - ratio_x));
    int y0 = int(img.rows * ratio_y);
    for (int y = 0; y < y0; y++) {
        for (int x = 0; x < x0; x++) {
            img.at<T>(y, x) = 0;
        }
    }
    for (int y = y0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            img.at<T>(y, x) = 0;
        }
    }
}

void GradientSlope(
    cv::Mat image, size_t edge_point_index, int min_radius, int max_radius,
    double canny_threshold = 130,
    double accumulator_threshold = 5) { // imshow("image", image);

    // 1. first, the image is passed through an edge detection phase
    Mat edge;
    // canny recommended a ratio of high:low threshold between 2:1 and 3:1
    cv::Canny(image, edge, canny_threshold / 2, canny_threshold);

    // 2. next, for every nonzero point in the edge image, the local gradient is
    // considered we compute the gradient by first computing the first-order
    // sobel-x and sobel-y derivatives
    // https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html
    // ddepth: output image depth
    Mat sobel_x, sobel_y;
    cv::Sobel(edge, sobel_x, CV_32F, 1, 0);
    cv::Sobel(edge, sobel_y, CV_32F, 0, 1);

    // TODO: use ROI to only considier the up-left corner of the image
    ROI<uchar>(edge);
    ROI<float>(sobel_x);
    ROI<float>(sobel_y);

    // // 3. next, we compute the gradient direction
    // Mat angle;
    // // TODO: check the doc, make sure is calculate the correct answer
    // cv::phase(sobel_x, sobel_y, angle, true);

    for (int y = 0; y < sobel_x.rows; y++) {
        for (int x = 0; x < sobel_x.cols; x++) {
            float dx = sobel_x.at<float>(y, x);
            float dy = sobel_y.at<float>(y, x);
            float magnitude = sqrt(dx * dx + dy * dy);
            if (magnitude == 0) {
                continue;
            }
            sobel_x.at<float>(y, x) = dx / magnitude;
            sobel_y.at<float>(y, x) = dy / magnitude;
        }
    }

    // use the opencv houghcicle the draw it on the same image
    std::vector<Vec3f> circles;
    medianBlur(image, image, 5);
    cv::HoughCircles(image, circles, HOUGH_GRADIENT, 1, image.rows / 16.0, 100,
                     30, 60, 93);
    for (size_t i = 0; i < circles.size(); i++) {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        // circle center
        circle(image, center, 1, Scalar(200), 3);
        // circle outline
        int radius = c[2];
        circle(image, center, radius, Scalar(200), 3);
    }

    // 4. using this gradient, we increment every point along the line indicated
    // by this slope from a specified minimum radius to a specified maximum
    // radius

    std::vector<Point> points;
    for (int y = 0; y < edge.rows; y++) {
        for (int x = 0; x < edge.cols; x++) {
            if (edge.at<uchar>(y, x) == 0) {
                continue;
            }
            points.push_back(Point(x, y));
        }
    }

    if (edge_point_index < points.size()) {
        Point point = points[edge_point_index];
        // draw this point on the image
        circle(image, point, 1, Scalar(255), 2);
        for (int r = min_radius; r < max_radius; r++) {
            // 两个方向都做一下
            int x0 = point.x + r * sobel_x.at<float>(point.y, point.x);
            int y0 = point.y + r * sobel_y.at<float>(point.y, point.x);
            if (x0 >= 0 && x0 < edge.cols && y0 >= 0 && y0 < edge.rows) {
                circle(image, Point(x0, y0), 1, Scalar(255), 2);
            }

            x0 = point.x - r * sobel_x.at<float>(point.y, point.x);
            y0 = point.y - r * sobel_y.at<float>(point.y, point.x);
            if (x0 >= 0 && x0 < edge.cols && y0 >= 0 && y0 < edge.rows) {
                circle(image, Point(x0, y0), 1, Scalar(255), 2);
            }
        }
    }
    imshow("circle on image", image);
}

// https://docs.opencv.org/4.x/da/d5c/tutorial_canny_detector.html
// https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html
// https://docs.opencv.org/4.x/d4/d70/tutorial_hough_circle.html
// Learning OpenCV 3. Ch12. Hough Transform
MyCircle MyHoughCircles(cv::Mat image, std::vector<Vec3f>& circles,
                        int min_radius, int max_radius,
                        double canny_threshold = 130,
                        double accumulator_threshold = 5) {

    // imshow("image", image);

    // 1. first, the image is passed through an edge detection phase
    Mat edge;
    // canny recommended a ratio of high:low threshold between 2:1 and 3:1
    // TODO: choose a better threshold for canny
    cv::Canny(image, edge, canny_threshold / 2, canny_threshold);
    // imshow("edge", edge);
    // waitKey();

    // 2. next, for every nonzero point in the edge image, the local
    // gradient is considered we compute the gradient by first computing the
    // first-order sobel-x and sobel-y derivatives
    // https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html
    // ddepth: output image depth
    Mat sobel_x, sobel_y;
    // medianBlur(image, image, 5);
    GaussianBlur(image, image, Size(5, 5), 0);
    cv::Sobel(image, sobel_x, CV_32F, 1, 0);
    cv::Sobel(image, sobel_y, CV_32F, 0, 1);
    // imshow("sobel_x", sobel_x);
    // imshow("sobel_y", sobel_y);
    // waitKey();

    // TODO: use ROI to only considier the up-left corner of the image
    ROI<uchar>(edge);
    ROI<float>(sobel_x);
    ROI<float>(sobel_y);
    // imshow("ROI edge", edge);
    // imshow("ROI sobel_x", sobel_x);
    // imshow("ROI sobel_y", sobel_y);
    // waitKey();

    // // 3. next, we compute the gradient direction
    // Mat angle;
    // // TODO: check the doc, make sure is calculate the correct answer
    // cv::phase(sobel_x, sobel_y, angle, true);

    // TODO:
    // 我们可以不用计算angle，因为在接下来的计算中，我们只需要使用到梯度向量
    // 只需要将(sobel_x, sobel_y)组成的向量进行单位化即可作为 (cos, sin)使用
    for (int y = 0; y < sobel_x.rows; y++) {
        for (int x = 0; x < sobel_x.cols; x++) {
            float dx = sobel_x.at<float>(y, x);
            float dy = sobel_y.at<float>(y, x);
            float magnitude = sqrt(dx * dx + dy * dy);
            if (magnitude == 0) {
                continue;
            }
            sobel_x.at<float>(y, x) = dx / magnitude;
            sobel_y.at<float>(y, x) = dy / magnitude;
        }
    }
    // 4. using this gradient, we increment every point along the line
    // indicated by this slope from a specified minimum radius to a
    // specified maximum radius
    Mat accumulator = Mat::zeros(image.size(), CV_32F);
    for (int y = 0; y < edge.rows; y++) {
        for (int x = 0; x < edge.cols; x++) {
            if (edge.at<uchar>(y, x) == 0) {
                continue;
            }
            for (int r = min_radius; r < max_radius; r++) {
                // x0 = x + r * cos
                int x0 = x + r * sobel_x.at<float>(y, x);
                // y0 = y + r * sin
                int y0 = y + r * sobel_y.at<float>(y, x);
                if (x0 >= 0 && x0 < edge.cols && y0 >= 0 && y0 < edge.rows) {
                    accumulator.at<float>(y0, x0) += 1;
                }

                x0 = x - r * sobel_x.at<float>(y, x);
                y0 = y - r * sobel_y.at<float>(y, x);
                if (x0 >= 0 && x0 < edge.cols && y0 >= 0 && y0 < edge.rows) {
                    // circle(image, Point(x0, y0), 1, Scalar(255), 2);
                    accumulator.at<float>(y0, x0) += 1;
                }
            }
        }
    }
    // 交点太少了呀 这是为什么呢 是不是因为梯度其实不是很精确
    WriteImageToFile(accumulator, "accumulator.txt");
    // imshow("accumulator", accumulator);
    // waitKey();
    // TODO: 把accumulator的值输出出来看一下吧

    // 5. the candidate centers are then selected from the accumulator that
    // are both above the accumulator_threshold and larger than their
    // surrounding neighbors
    Mat candidate_centers = Mat::zeros(image.size(), CV_8U);
    for (int y = 1; y < accumulator.rows - 1; y++) {
        for (int x = 1; x < accumulator.cols - 1; x++) {
            if (accumulator.at<float>(y, x) > accumulator_threshold) {
                // check if this point is the maximum in its neighborhood
                bool is_max = true;
                for (int y0 = y - 1; y0 <= y + 1; y0++) {
                    for (int x0 = x - 1; x0 <= x + 1; x0++) {
                        if (accumulator.at<float>(y0, x0) >
                            accumulator.at<float>(y, x)) {
                            is_max = false;
                            break;
                        }
                    }
                    if (!is_max) {
                        break;
                    }
                }
                if (is_max) {
                    // the circle center should in the image
                    if (x - max_radius < 0 || x + max_radius >= edge.cols ||
                        y - max_radius < 0 || y + max_radius >= edge.rows) {
                        continue;
                    }
                    candidate_centers.at<uchar>(y, x) = 255;
                }
            }
        }
    }
    // imshow("candidate_centers", candidate_centers);
    // waitKey();

    // 6. the candidate centers are sorted in descending order of their
    // accumulator values, so that the centers with the most supporint
    // pixels appear first
    vector<Point> centers;
    for (int y = 0; y < candidate_centers.rows; y++) {
        for (int x = 0; x < candidate_centers.cols; x++) {
            if (candidate_centers.at<uchar>(y, x) == 255) {
                centers.push_back(Point(x, y));
            }
        }
    }
    sort(centers.begin(), centers.end(),
         [&](const Point& a, const Point& b) -> bool {
             return accumulator.at<float>(a.y, a.x) >
                    accumulator.at<float>(b.y, b.x);
         });

    // draw the first candidate center
    // 可能没有candidate center
    if (centers.empty()) {
        return {};
    }
    Mat candidate_centers_ = Mat::zeros(image.size(), CV_8U);
    candidate_centers_.at<uchar>(centers[0].y, centers[0].x) = 255;
    // imshow("first candidate center", candidate_centers_);
    // waitKey();

    // 7. for each center, all of the nonzero pixels are considered,
    // these pixels are sorted according to their distance from the center,
    // working out from the minimum radius to the maximum radius, we select
    // a single radius that is best supported by the nonzero pixels

    // only consider the first center
    centers.resize(1);
    vector<Point> circles_;
    int radius = 0;
    for (const Point& center : centers) {
        vector<Point> points;
        for (int y = 0; y < edge.rows; y++) {
            for (int x = 0; x < edge.cols; x++) {
                if (edge.at<uchar>(y, x) == 0) {
                    continue;
                }
                float distance = (x - center.x) * (x - center.x) +
                                 (y - center.y) * (y - center.y);
                if (distance < min_radius * min_radius ||
                    distance > max_radius * max_radius) {
                    continue;
                }
                points.push_back(Point(x, y));
            }
        }
        sort(points.begin(), points.end(),
             [&](const Point& a, const Point& b) -> bool {
                 return norm(a - center) < norm(b - center);
             });

        int max_support = 0;
        for (int r = min_radius; r < max_radius; r++) {
            int support = 0;
            // points保存着所有的边缘点
            for (const Point& point : points) {
                // 如果点和半径的距离差不太多，就认为支持这个半径
                if (abs(norm(point - center) - r) < 10) {
                    support++;
                }
            }
            if (support > max_support) {
                // 更新最佳半径
                radius = r;
                max_support = support;
            }
        }
        circles_.push_back(Point(center.x, center.y));
    }

    // finally we find this best fit circle!!!
    MyCircle my_circle{circles_[0], radius};

    // draw the circle on the original image
    Mat image_ = image.clone();
    for (const Point& center : circles_) {
        // circle center
        circle(image_, center, 1, Scalar(255), 2);
        // circle outline
        cv::circle(image_, center, radius, Scalar(255), 2);
    }

    // use the opencv houghcicle the draw it on the same image
    // circles.clear();
    // medianBlur(image, image, 5);
    // cv::HoughCircles(image, circles, HOUGH_GRADIENT, 1, image.rows / 16.0,
    // 100,
    //                  30, 60, 93);
    // for (size_t i = 0; i < circles.size(); i++) {
    //     Vec3i c = circles[i];
    //     Point center = Point(c[0], c[1]);
    //     // circle center
    //     circle(image_, center, 1, Scalar(128), 3);
    //     // circle outline
    //     int radius = c[2];
    //     circle(image_, center, radius, Scalar(128), 3);
    // }
    imshow("circle on image", image_);
    // waitKey();

    return my_circle;
}

int min_radius = 60;
int max_radius = 85;
int canny_threshold = 40;
int accumulator_threshold = 5;
void MyHoughCirclesDemo(int, void*) {

    // read imgge from file, and convert it into grayscale
    const char* filename = "/home/zhangzhong/src/opencv/imgs/orange.jpg";
    Mat image = imread(samples::findFile(filename), IMREAD_GRAYSCALE);

    std::vector<cv::Vec3f> circles;

    // 用tracebar调一下参数吧
    MyCircle my_circle = MyHoughCircles(image, circles, min_radius, max_radius,
                                        canny_threshold, accumulator_threshold);

    Mat segmentation = Mat::zeros(image.size(), CV_8U);
    // https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670
    circle(segmentation, my_circle.center, my_circle.radius, Scalar(255), -1,
           FILLED);
    imwrite("segmentation.jpg", segmentation);

    // read the ground truch image
    const char* ground_truth_path =
        "/home/zhangzhong/src/opencv/imgs/orange_split.bmp";
    Mat ground_truth =
        imread(samples::findFile(ground_truth_path), IMREAD_GRAYSCALE);
    double jaccard = Jaccard(segmentation, ground_truth);
    std::cout << "Jacard: " << jaccard << std::endl;
}

int edge_point_index = 0;
void GradientSlopeDemo(int, void*) {
    // read imgge from file, and convert it into grayscale
    const char* filename = "/home/zhangzhong/src/opencv/imgs/orange.jpg";
    Mat image = imread(samples::findFile(filename), IMREAD_GRAYSCALE);

    GradientSlope(image, edge_point_index, min_radius, max_radius,
                  canny_threshold, accumulator_threshold);
}

TEST_CASE("lab3 gradient slope demo") {
    namedWindow("circle on image");
    createTrackbar("canny threshold", "circle on image", &canny_threshold, 255,
                   GradientSlopeDemo);
    createTrackbar("accumulator threshold", "circle on image",
                   &accumulator_threshold, 10, GradientSlopeDemo);
    createTrackbar("min radius", "circle on image", &min_radius, 90,
                   GradientSlopeDemo);
    createTrackbar("max radius", "circle on image", &max_radius, 200,
                   GradientSlopeDemo);
    createTrackbar("edge point index", "circle on image", &edge_point_index,
                   3000, GradientSlopeDemo);
    GradientSlopeDemo(0, 0);
    waitKey();
}

// Jacard: 0.907839
TEST_CASE("opencv houghcircle") {
    // 看看opencv的houghcircle的jaccard是多少
    const char* filename = "/home/zhangzhong/src/opencv/imgs/orange.jpg";
    Mat image = imread(samples::findFile(filename), IMREAD_GRAYSCALE);
    std::vector<Vec3f> circles;
    medianBlur(image, image, 5);
    cv::HoughCircles(image, circles, HOUGH_GRADIENT, 1, image.rows / 16.0, 100,
                     30, 60, 93);
    Mat segmentation = Mat::zeros(image.size(), CV_8U);
    MyCircle my_circle = {Point(circles[0][0], circles[0][1]),
                          (int)circles[0][2]};
    circle(segmentation, my_circle.center, my_circle.radius, Scalar(255), -1,
           FILLED);
    // read the ground truch image
    const char* ground_truth_path =
        "/home/zhangzhong/src/opencv/imgs/orange_split.bmp";
    Mat ground_truth =
        imread(samples::findFile(ground_truth_path), IMREAD_GRAYSCALE);
    double jaccard = Jaccard(segmentation, ground_truth);
    std::cout << "Jacard: " << jaccard << std::endl;
}

TEST_CASE("lab3") {
    namedWindow("circle on image");
    createTrackbar("canny threshold", "circle on image", &canny_threshold, 255,
                   MyHoughCirclesDemo);
    createTrackbar("accumulator threshold", "circle on image",
                   &accumulator_threshold, 10, MyHoughCirclesDemo);
    createTrackbar("min radius", "circle on image", &min_radius, 90,
                   MyHoughCirclesDemo);
    createTrackbar("max radius", "circle on image", &max_radius, 100,
                   MyHoughCirclesDemo);
    MyHoughCirclesDemo(0, 0);
    waitKey();
}