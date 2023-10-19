// 2023/9/19
// zhangzhong
// canny edge detection

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cassert>
#include <cmath>
#include <doctest/doctest.h>
#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

double pi = 3.14159265358979323846;

// 计算地图的模和方向角
void CalculateMagnitudeAndAngle(const Mat& x, const Mat& y, Mat& magnitude,
                                Mat& angle) {
    // 给magnitude分配内存
    // BUG
    // magnitude.create(x.size(), x.type());
    // angle.create(x.size(), x.type());
    magnitude = Mat(x.size(), CV_64FC1);
    // TODO: 这里的类型应该是double，就不会有后面的精度问题了
    angle = Mat(x.size(), CV_64FC1);

    // TODO: 应该在执行矩阵的遍历之前assert矩阵的类型是否符合预期
    assert(x.size() == y.size());

    // 可能是类型的问题
    for (int i = 0; i < x.rows; ++i) {
        for (int j = 0; j < x.cols; ++j) {
            // 计算模
            // 不对呀 这里的类型也不对
            // opencv的at函数实在是太傻逼了!!! 完全不检查类型的吗！！！
            magnitude.at<double>(i, j) =
                std::sqrt(x.at<double>(i, j) * x.at<double>(i, j) +
                          y.at<double>(i, j) * y.at<double>(i, j));
            // 计算方向角
            // https://stackoverflow.com/questions/283406/what-is-the-difference-between-atan-and-atan2-in-c
            // TODO: x != 0
            // 按照实验的要求，只需要计算0-180度的角度就够了
            // 果然，x=0的时候，atan2的结果是0
            // assert(x.at<float>(i, j) != 0);
            if (x.at<double>(i, j) != 0) {
                double a =
                    std::atan((double)y.at<double>(i, j) / x.at<double>(i, j));
                if (a > pi / 2) {
                    a = pi / 2;
                }
                if (a < -pi / 2) {
                    a = -pi / 2;
                }
                angle.at<double>(i, j) = a;
            } else if (y.at<double>(i, j) != 0) {
                angle.at<double>(i, j) = pi / 2;
            } else {
                angle.at<double>(i, j) = 0;
            }
        }
    }
}

// 你要做的就是，我给你一个角度，你给我两个坐标
// 然后NMS就可以在自己和两个坐标之间进行非极大值抑制
float NMSHelper(double angle, const Mat& magnitude, int row, int col) {

    // if (!(angle >= -pi/2 && angle <= pi/2)) {
    //     std::cout << "angle: " << angle << std::endl;
    // }
    // 浮点数精度问题 不能直接用==比较
    assert(angle >= (-pi / 2 - 0.001) && angle <= (pi / 2 + 0.001));

    int row1, col1;
    int row2, col2;
    if (angle >= -3 * pi / 8 && angle < -pi / 8) {
        // (row+1, col-1) (row-1, col+1)
        // 右上 左下
        row1 = row + 1;
        col1 = col - 1;
        row2 = row - 1;
        col2 = col + 1;
    } else if (angle >= -pi / 8 && angle < pi / 8) {
        // 左 右
        // (row, col-1) (row, col+1)
        row1 = row;
        col1 = col - 1;
        row2 = row;
        col2 = col + 1;
    } else if (angle >= pi / 8 && angle < 3 * pi / 8) {
        // 左上 右下
        // (row-1, col-1) (row+1, col+1)
        row1 = row - 1;
        col1 = col - 1;
        row2 = row + 1;
        col2 = col + 1;

    } else {
        // 上 下
        // (row-1, col) (row+1, col)
        row1 = row - 1;
        col1 = col;
        row2 = row + 1;
        col2 = col;
    }

    double m1 = magnitude.at<double>(row1, col1);
    double m2 = magnitude.at<double>(row2, col2);
    return m1 > m2 ? m1 : m2;
}

void NonMaximumSuppression(const Mat& magnitude, const Mat& angle,
                           Mat& result) {

    // 在前面的步骤中，我们计算出了幅值图像Magnitude和方向图像angle
    // 现在需要减小Magnitude的尺寸，只留下最大点?? 为什么

    // 1. 将NMS初始化为Magnitude的副本
    result = magnitude.clone();
    // type: uchar
    // angle.type: float

    // 2. 根据梯度方向，对相邻的像素进行比较 进行极大值抑制
    for (int row = 1; row < magnitude.rows - 1; ++row) {
        for (int col = 1; col < magnitude.cols - 1; ++col) {
            double curr_angle = angle.at<double>(row, col);
            double max_magnitude = NMSHelper(curr_angle, magnitude, row, col);
            double curr_magnitude = magnitude.at<double>(row, col);
            // 注意一定是和原始的Magnitude比较 而不是和NMS比较
            if (curr_magnitude < max_magnitude) {
                // TODO: 这个<float>可以去掉吗？？
                // 抑制
                result.at<double>(row, col) = 0;
            }
        }
    }

    // result图像的边缘置零
    // TIP: 0一定写成Scalar(0)
    result.row(0).setTo(Scalar(0));
    result.row(result.rows - 1).setTo(Scalar(0));
    result.col(0).setTo(Scalar(0));
    result.col(result.cols - 1).setTo(Scalar(0));
}

void DoubleThresholdDetection(Mat& nms, double low, double high,
                              Mat& strong_edge, Mat& weak_edge) {
    // 边缘图像全部初始化为0
    strong_edge = Mat::zeros(nms.size(), CV_8UC1);
    weak_edge = Mat::zeros(nms.size(), CV_8UC1);

    for (int row = 0; row < nms.rows; ++row) {
        for (int col = 0; col < nms.cols; ++col) {
            if (nms.at<double>(row, col) > high) {
                strong_edge.at<uchar>(row, col) = 255;
            } else if (nms.at<double>(row, col) > low) {
                weak_edge.at<uchar>(row, col) = 255;
            }
        }
    }
}

void DrawFloatingPointImage(Mat& img, const char* window_name) {
    assert(img.type() == CV_64FC1);

    Mat img_uchar;
    double minVal, maxVal;
    // find minimum and maximum intensities
    minMaxLoc(img, &minVal, &maxVal);
    img.convertTo(img_uchar, CV_8UC1, 255.0 / (maxVal - minVal),
                  -minVal * 255.0 / (maxVal - minVal));
    imshow(window_name, img_uchar);
    waitKey();
}

void EdgeConnection(Mat& strong_edge, Mat& weak_edge, Mat& edge) {
    // 图像中被划分为强边缘的像素点已经被确定为边缘
    // 弱边缘的像素的八邻域如果与强边缘点连接 则也被确定为边缘
    // 所以只需要遍历一次弱边缘图像即可

    // 首先，所有的强边缘点都作为边缘
    edge = strong_edge.clone();

    // 然后，遍历弱边缘图像
    for (int row = 1; row < weak_edge.rows - 1; ++row) {
        for (int col = 1; col < weak_edge.cols - 1; ++col) {

            // 首先，检查 (row, col) 是否是弱边缘点
            // 再检查 (row, col) 的八邻域是否有强边缘点
            // 如果有，那么 (row, col) 也是边缘点
            if (weak_edge.at<uchar>(row, col) == 255) {
                // 检查八邻域
                if (strong_edge.at<uchar>(row - 1, col - 1) == 255 ||
                    strong_edge.at<uchar>(row - 1, col) == 255 ||
                    strong_edge.at<uchar>(row - 1, col + 1) == 255 ||
                    strong_edge.at<uchar>(row, col - 1) == 255 ||
                    strong_edge.at<uchar>(row, col + 1) == 255 ||
                    strong_edge.at<uchar>(row + 1, col - 1) == 255 ||
                    strong_edge.at<uchar>(row + 1, col) == 255 ||
                    strong_edge.at<uchar>(row + 1, col + 1) == 255) {
                    edge.at<uchar>(row, col) = 255;
                } else {
                    edge.at<uchar>(row, col) = 0;
                }
            }
        }
    }
}

// 膨胀
Mat MyDilation(const Mat& img, const Mat& kernel) {
    auto size = kernel.size();
    // 必须是一个方形区域 必须是奇数
    assert(size.width == size.height);
    assert(size.width % 2 == 1);

    size_t pad = size.width / 2;

    // 不用padding了
    // 遍历图像
    Mat result = img.clone();
    for (int row = pad; row < img.rows - pad; ++row) {
        for (int col = pad; col < img.cols - pad; ++col) {
            // 如果当前像素是255
            // 那么将形状置为kernel的形状
            if (img.at<uchar>(row, col) == 255) {
                for (int i = 0; i < size.width; ++i) {
                    for (int j = 0; j < size.height; ++j) {
                        // 如果kernel中的像素是255
                        // 那么将result中的像素也置为255
                        if (kernel.at<uchar>(i, j) == 1) {
                            result.at<uchar>(row - pad + i, col - pad + j) =
                                255;
                        }
                    }
                }
            }
        }
    }

    return result;
}

// 腐蚀
Mat MyErosion(const Mat& img, const Mat& kernel) {
    auto size = kernel.size();
    // 必须是一个方形区域 必须是奇数
    assert(size.width == size.height);
    assert(size.width % 2 == 1);

    size_t pad = size.width / 2;

    // 不用padding了
    // 遍历图像
    Mat result = img.clone();
    for (int row = pad; row < img.rows - pad; ++row) {
        for (int col = pad; col < img.cols - pad; ++col) {
            // 如果当前像素是255
            // 那么将形状置为kernel的形状
            if (img.at<uchar>(row, col) == 255) {
                // 图像中的点必须和kernel中的点 全部对应 才能保留下来 否则就是0
                bool reserve = true;
                for (int i = 0; i < size.width; ++i) {
                    for (int j = 0; j < size.height; ++j) {
                        // 如果kernel中的像素是255
                        // 那么将result中的像素也置为255
                        if (kernel.at<uchar>(i, j) == 1) {
                            if (img.at<uchar>(row - pad + i, col - pad + j) !=
                                255) {
                                reserve = false;
                            }
                        }
                    }
                }
                if (!reserve) {
                    result.at<uchar>(row, col) = 0;
                }
            }
        }
    }

    return result;
}

void MyBinarize(const Mat& src, Mat& dest) {
    dest = src.clone();
    for (int i = 0; i < dest.rows; i++) {
        for (int j = 0; j < dest.cols; j++) {
            if (dest.at<uchar>(i, j) > 128) {
                dest.at<uchar>(i, j) = 255;
            } else {
                dest.at<uchar>(i, j) = 0;
            }
        }
    }
}

int CalculateDiff(const Mat& img1, const Mat& img2) {
    assert(img1.size() == img2.size());
    assert(img1.type() == img2.type());
    assert(img1.channels() == img2.channels());
    assert(img1.channels() == 1);

    int diff = 0;
    for (int row = 0; row < img1.rows; ++row) {
        for (int col = 0; col < img1.cols; ++col) {
            // 我们要保证两张图都是二值图
            assert(img1.at<uchar>(row, col) == 0 ||
                   img1.at<uchar>(row, col) == 255);
            assert(img2.at<uchar>(row, col) == 0 ||
                   img2.at<uchar>(row, col) == 255);
            if (img1.at<uchar>(row, col) != img2.at<uchar>(row, col)) {
                diff += 1;
            }
        }
    }

    return diff;
}

void MyCanny(const Mat& img, Mat& edge, double low_threshold,
             double high_threshold, bool verbose = false) {
    // convert to double
    Mat img_f;
    img.convertTo(img_f, CV_64F);

    // gaussian blur
    Mat gaussian_kernel =
        (Mat_<double>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1) / 16.0;
    Mat img_gaussian;
    filter2D(img_f, img_gaussian, img_f.depth(), gaussian_kernel);
    if (verbose) {
        cout << "gaussin = " << img_gaussian << endl << endl;
        cout << "img_f.depth() = " << img_f.depth() << endl;
        cout << "img to floating point = " << img_f << endl << endl;
        DrawFloatingPointImage(img_gaussian, "gaussian blur");
    }

    // calculate gradient
    // 对图像做高斯平滑 出来的结果应该是浮点数
    Mat sobel_x = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    Mat sobel_y = (Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    Mat img_sobel_x;
    Mat img_sobel_y;
    filter2D(img_gaussian, img_sobel_x, img_gaussian.depth(), sobel_x);
    filter2D(img_gaussian, img_sobel_y, img_gaussian.depth(), sobel_y);
    if (verbose) {
        cout << "img_sobel_x = " << endl << " " << img_sobel_x << endl << endl;
        DrawFloatingPointImage(img_sobel_x, "sobel x");
        cout << "img_sobel_y = " << endl << " " << img_sobel_y << endl << endl;
        DrawFloatingPointImage(img_sobel_y, "sobel y");
    }

    // calculate magnitude and angle
    Mat img_sobel_magnitude;
    Mat img_sobel_angle;
    CalculateMagnitudeAndAngle(img_sobel_x, img_sobel_y, img_sobel_magnitude,
                               img_sobel_angle);
    if (verbose) {
        cout << "img_sobel_magnitude = " << endl
             << " " << img_sobel_magnitude << endl
             << endl;
        DrawFloatingPointImage(img_sobel_magnitude, "magnitude");
        cout << "img_sobel_angle = " << endl
             << " " << img_sobel_angle << endl
             << endl;
        DrawFloatingPointImage(img_sobel_angle, "angle");
    }

    // non maximum suppression
    Mat img_nms;
    NonMaximumSuppression(img_sobel_magnitude, img_sobel_angle, img_nms);
    if (verbose) {
        cout << "img_nms = " << endl << " " << img_nms << endl << endl;
        DrawFloatingPointImage(img_nms, "nms");
    }

    // double threshold detection
    Mat strong_edge, weak_edge;
    DoubleThresholdDetection(img_nms, low_threshold, high_threshold,
                             strong_edge, weak_edge);
    if (verbose) {
        imshow("strong edge", strong_edge);
        waitKey();
        imshow("weak edge", weak_edge);
        waitKey();
    }

    // edge connection get final edge graph
    EdgeConnection(strong_edge, weak_edge, edge);
    if (verbose) {
        imshow("edge", edge);
        waitKey();
    }

    // 效果不好
    Mat dialation_kernel = (Mat_<uchar>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
    Mat erosion_kernel = Mat::ones(5, 5, CV_8UC1);
    // 膨胀
    Mat dilated_edge = MyDilation(edge, dialation_kernel);
    // 腐蚀
    Mat eroded_edge = MyErosion(dilated_edge, erosion_kernel);
    if (verbose) {
        imshow("DilatedEdge", dilated_edge);
        imshow("ErodedEdge", eroded_edge);
        waitKey();
    }
}

int low_threshold = 60;
const int max_low_threshold = 500;
const int RATIO = 2;
const char* winname = "My Canny Detector";
static void MyCannyDemo(int, void*) {
    const char* filename = "/home/zhangzhong/src/opencv/imgs/CutleryDT.png";
    std::string image_path = samples::findFile(filename);
    Mat image = imread(image_path, IMREAD_GRAYSCALE);
    Mat edge;
    MyCanny(image, edge, low_threshold, low_threshold * RATIO, false);
    imshow(winname, edge);

    Mat edge_color;
    cvtColor(edge, edge_color, COLOR_GRAY2BGR);

    // 把ground truth也画在上面 做一个对比
    Mat ground_truth =
        imread("/home/zhangzhong/src/opencv/imgs/lab2gt.png", IMREAD_GRAYSCALE);
    MyBinarize(ground_truth, ground_truth);
    for (int i = 0; i < ground_truth.rows; i++) {
        for (int j = 0; j < ground_truth.cols; j++) {
            // 将groundtruth中的边缘标记为红色
            if (ground_truth.at<uchar>(i, j) == 255) {
                edge_color.at<Vec3b>(i, j)[0] = 0;
                edge_color.at<Vec3b>(i, j)[1] = 0;
                edge_color.at<Vec3b>(i, j)[2] = 255;
            }
        }
    }
    imshow(winname, edge_color);

    // calculate diff
    int diff = CalculateDiff(edge, ground_truth);
    std::cout << diff << std::endl;
}

TEST_CASE("lab2") {
    namedWindow(winname, WINDOW_NORMAL);
    createTrackbar("min threshold:", winname, &low_threshold, max_low_threshold,
                   MyCannyDemo);
    MyCannyDemo(0, 0);
    waitKey();
}

TEST_CASE("lab2 verbose") {
    double low_threshold = 100;
    double high_threshold = 200;
    const char* filename = "/home/zhangzhong/src/opencv/imgs/CutleryDT.png";
    std::string image_path = samples::findFile(filename);
    Mat image = imread(image_path, IMREAD_GRAYSCALE);
    Mat edge;
    MyCanny(image, edge, low_threshold, high_threshold, true);
    Mat ground_truth =
        imread("/home/zhangzhong/src/opencv/imgs/lab2gt.png", IMREAD_GRAYSCALE);
    MyBinarize(ground_truth, ground_truth);
    int diff = CalculateDiff(edge, ground_truth);
    std::cout << diff << std::endl;
}
