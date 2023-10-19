// 2023/9/19
// zhangzhong
// canny edge detection

#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
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
    // 不行 create会 随机初始化
    strong_edge.create(nms.size(), CV_8UC1);
    weak_edge.create(nms.size(), CV_8UC1);
    // 将两个图像填充为零
    strong_edge = Scalar(0);
    weak_edge = Scalar(0);

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

int main_back(int argc, char* argv[]) {
    if (argc != 2) {
        std::printf("Usage: %s <path>", argv[0]);
        return 1;
    }

    // TIP: 中间过程的结果尽可能是灰度图 这样非常容易检查每一步的结果 发现错误

    // step 1. 目前先使用opencv自带的filter2D进行过滤吧 就不自己实现了
    // https://stackoverflow.com/questions/20041391/why-gauss-filter-chooses-such-values
    // 定义高斯模糊核
    Mat guassin_kernel_5 =
        (Mat_<double>(5, 5) << 1, 4, 7, 4, 1, 4, 16, 26, 16, 4, 7, 26, 41, 26,
         7, 4, 16, 26, 16, 4, 1, 4, 7, 4, 1);
    guassin_kernel_5 /= 273;

    Mat gaussian_kernel_3 = (Mat_<double>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1);
    gaussian_kernel_3 /= 16;

    // step 2. 读取图像
    std::string image_path = samples::findFile(argv[1]);
    Mat img = imread(image_path, IMREAD_GRAYSCALE);

    // 咱给他转成doubel类型的矩阵
    // 然后封装一个函数 可以从double类型转成灰度图
    Mat img_f;
    img.convertTo(img_f, CV_64FC1);
    cout << "img to floating point = " << img_f << endl << endl;
    cout << "img_f.depth() = " << img_f.depth() << endl;

    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    // step 3. 高斯模糊
    // 确实变模糊了
    Mat img_gaussian;
    filter2D(img, img_gaussian, img_f.depth(), guassin_kernel_5);
    cout << "gaussin = " << img_gaussian << endl << endl;

    // imshow("Display window", img_gaussian);
    DrawFloatingPointImage(img_gaussian, "gaussian");
    waitKey(0);

    // step 4. 计算梯度
    Mat sobel_x = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    Mat sobel_y = (Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

    // 最简单的方法应该是分别得到x和y的梯度，然后再求模和方向角
    // TODO: sobel_x和sobel_y的类型应该是double
    // 因为如果是保存在uchar里面的话，我们是没有变化有负数的
    // 这样就会导致梯度的方向计算错误
    Mat img_sobel_x;
    filter2D(img_gaussian, img_sobel_x, img.depth(), sobel_x);
    // 输出一下梯度图像
    imshow("Display window", img_sobel_x);
    waitKey();
    // 也输出一下图像的值吧
    // 其结果是一个灰度图像
    cout << "img_sobel_x = " << endl << " " << img_sobel_x << endl << endl;

    Mat img_sobel_y;
    filter2D(img_gaussian, img_sobel_y, img.depth(), sobel_y);
    imshow("Display window", img_sobel_y);
    waitKey();
    cout << "img_sobel_y = " << endl << " " << img_sobel_y << endl << endl;

    // 计算梯度的模
    Mat img_sobel_magnitude;
    Mat img_sobel_angle;
    CalculateMagnitudeAndAngle(img_sobel_x, img_sobel_y, img_sobel_magnitude,
                               img_sobel_angle);
    // 那么讲道理 我们的总的梯度图像 也应该是一个灰度图像
    // 也就是说我们要做 saturate_cast
    // 这一步就错了
    // 就是因为没有做saturate_cast
    imshow("Display window", img_sobel_magnitude);
    waitKey();

    // step 5. 非极大值抑制
    // TODO: 这一步的处理结果不对
    Mat img_nms;
    NonMaximumSuppression(img_sobel_magnitude, img_sobel_angle, img_nms);
    imshow("Display window", img_nms);
    waitKey();

    // step 6. 双阈值检测
    // 需要指定两个阈值
    double low_threshold = 22;
    double high_threshold = 44;
    Mat strong_edge, weak_edge;
    DoubleThresholdDetection(img_nms, low_threshold, high_threshold,
                             strong_edge, weak_edge);
    imshow("StrongEdge", strong_edge);
    waitKey();
    imshow("WeakEdge", weak_edge);
    waitKey();

    // step 7. 边缘连接
    // TODO
    // EdgeConnection(weak_edge, strong_edge);

    // TODO: 目前来看图片的边缘检测是正确的，但是可能输出的时候多了三个通道
    // 而不是灰度图 最后输出边缘图像 imshow("Display window", strong_edge);
    // imwrite("imgs/strong_edge.jpg", strong_edge);

    return 0;
}

// 计算我们边缘检测的图像和真正的边缘图像之间的差距loss
double CalculateLoss(const Mat& edge, const Mat& ground_truth) {
    // 首先 两幅图像的大小应该是一样的
    assert(edge.size() == ground_truth.size());
    // 然后 两幅图像的类型应该是一样的
    assert(edge.type() == ground_truth.type());
    // 最后 两幅图像的通道数应该是一样的
    assert(edge.channels() == ground_truth.channels());

    // 计算loss
    double loss = 0;
    for (int row = 0; row < edge.rows; ++row) {
        for (int col = 0; col < edge.cols; ++col) {
            if (edge.at<uchar>(row, col) != ground_truth.at<uchar>(row, col)) {
                loss += 1;
            }
        }
    }

    return loss;
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

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::printf("Usage: %s <path>", argv[0]);
        return 1;
    }

    Mat gaussian_kernel = (Mat_<double>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1);
    gaussian_kernel /= 16.0;
    cout << "gaussin kernel = " << endl
         << " " << gaussian_kernel << endl
         << endl;

    std::string image_path = samples::findFile(argv[1]);
    Mat img = imread(image_path, IMREAD_GRAYSCALE);

    Mat img_f;
    img.convertTo(img_f, CV_64FC1);
    cout << "img_f.depth() = " << img_f.depth() << endl;
    // cout << "img to floating point = " << img_f << endl << endl;
    DrawFloatingPointImage(img_f, "test windows");

    // 对图像做高斯平滑 出来的结果应该是浮点数
    // 感觉高斯平滑没做好啊
    Mat img_gaussian;
    filter2D(img_f, img_gaussian, img_f.depth(), gaussian_kernel);
    // 这就对了 原来是输出错了矩阵
    cout << "gaussin = " << img_gaussian << endl << endl;

    Mat sobel_x = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    Mat sobel_y = (Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    Mat img_sobel_x;
    filter2D(img_gaussian, img_sobel_x, img_gaussian.depth(), sobel_x);
    cout << "img_sobel_x = " << endl << " " << img_sobel_x << endl << endl;
    DrawFloatingPointImage(img_sobel_x, "img_sobel_x");

    Mat img_sobel_y;
    filter2D(img_gaussian, img_sobel_y, img_gaussian.depth(), sobel_y);
    cout << "img_sobel_y = " << endl << " " << img_sobel_y << endl << endl;
    DrawFloatingPointImage(img_sobel_y, "img_sobel_y");

    // 计算梯度的模
    Mat img_sobel_magnitude;
    Mat img_sobel_angle;
    CalculateMagnitudeAndAngle(img_sobel_x, img_sobel_y, img_sobel_magnitude,
                               img_sobel_angle);
    cout << "img_sobel_magnitude = " << endl
         << " " << img_sobel_magnitude << endl
         << endl;
    DrawFloatingPointImage(img_sobel_magnitude, "img_sobel_magnitude");

    cout << "img_sobel_angle = " << endl
         << " " << img_sobel_angle << endl
         << endl;
    DrawFloatingPointImage(img_sobel_angle, "img_sobel_angle");

    // 非极大值抑制
    Mat img_nms;
    NonMaximumSuppression(img_sobel_magnitude, img_sobel_angle, img_nms);
    cout << "img_nms = " << endl << " " << img_nms << endl << endl;
    DrawFloatingPointImage(img_nms, "img_nms");
    // 这次终于对了 果然是类型的问题!!!

    // 双阈值检测
    double low_threshold = 22;
    double high_threshold = 44;
    Mat strong_edge, weak_edge;
    DoubleThresholdDetection(img_nms, low_threshold, high_threshold,
                             strong_edge, weak_edge);
    imshow("StrongEdge", strong_edge);
    waitKey();
    imshow("WeakEdge", weak_edge);
    waitKey();

    // 边缘连接
    Mat edge;
    EdgeConnection(strong_edge, weak_edge, edge);
    imshow("Edge", edge);
    waitKey();

    // 效果不好
    // // 腐蚀
    // Mat dialation_kernel = (Mat_<uchar>(3, 3) << 0, 0, 0, 1, 1, 1, 0, 0, 0);
    // Mat erosion_kernel = Mat::ones(7, 7, CV_8UC1);

    // // 膨胀
    // Mat dilated_edge = MyDilation(edge, dialation_kernel);
    // imshow("DilatedEdge", dilated_edge);

    // Mat eroded_edge = MyErosion(dilated_edge, erosion_kernel);
    // imshow("ErodedEdge", eroded_edge);

    // 计算loss
    Mat ground_truth =
        imread("/home/zhangzhong/src/opencv/imgs/lab2gt.png", IMREAD_GRAYSCALE);

    MyBinarize(ground_truth, ground_truth);
    std::ofstream fout("ground_truth.txt");
    fout << ground_truth;
    imshow("GroundTruth", ground_truth);
    // 输出ground_truth到txt文件

    waitKey();
    int diff = CalculateDiff(edge, ground_truth);
    std::cout << diff << std::endl;

    // use opencv canny detection
    // https://docs.opencv.org/4.x/da/d5c/tutorial_canny_detector.html
    // https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    // https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de
    Mat img_canny;
    Canny(img, img_canny, low_threshold, high_threshold, 3, true);
    imshow("Canny", img_canny);
    waitKey();

    // 用trackbar来调整阈值即可
    return 0;
}
