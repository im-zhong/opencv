// 2023/9/19
// zhangzhong
// Mask operations on matrices
// https://docs.opencv.org/4.8.0/d7/d37/tutorial_mat_mask_operations.html

// we recalculate each pixel's value in an image according to a mask matrix (also known as kernel).
// This mask holds values that will adjust how much influence neighboring pixels (and the current pixel) have on the new pixel value
// we make a weighted average, with our specified values.

// You use the mask by putting the center of the mask matrix 
// (in the upper case noted by the zero-zero index) on the pixel you want to calculate 
// and sum up the pixel values multiplied with the overlapped matrix values.

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>

using namespace std;
using namespace cv;

static void help(char* progName)
{
 cout << endl
 << "This program shows how to filter images with mask: the write it yourself and the"
 << "filter2d way. " << endl
 << "Usage:" << endl
 << progName << " [image_path -- default lena.jpg] [G -- grayscale] " << endl << endl;
}

void Sharpen(const Mat& myImage, Mat& Result) {
    // // accept only uchar images
    // TODO: 那我有一个问题，我是RGB三个通道，但是每个通道都是uchar的算吗?
    CV_Assert(myImage.depth() == CV_8U); 

    const int nChannels = myImage.channels();
    // 相当于自动分配合适的内存空间
    // TODO: 查一下这个函数有什么行为 他会复制之前的值吗？
    Result.create(myImage.size(),myImage.type());

    // 这里的处理非常简单
    // 没有任何填充，直接跳过了边界
    for (int j = 1; j < myImage.rows-1; ++j) {
        // 获得当前行，当前行的上一行，当前行的下一行
        const uchar* previous = myImage.ptr<uchar>(j-1);
        const uchar* current  = myImage.ptr<uchar>(j);
        const uchar* next     = myImage.ptr<uchar>(j+1);

        // 拿到result矩阵对应的输出行
        uchar* output = Result.ptr<uchar>(j);
        // 遍历当前行
        for (int i = nChannels; i < nChannels*(myImage.cols-1); ++i) {
            // kernel weighted sum
            // saturate_cast 保证数据最终处于 0-255
            *output++ = saturate_cast<uchar>(5*current[i]
                         -current[i-nChannels] - current[i+nChannels] - previous[i] - next[i]);
        }
    }

    // 第一行，最后一行
    // 第一列，最后一列(如果channels=3, 那么就是头三列 和 最后三列)
    // 这些列在前面的计算中略过了
    // 所以我们需要在这里清零
    Result.row(0).setTo(Scalar(0));
    Result.row(Result.rows-1).setTo(Scalar(0));
    // TODO: 不对呀，这里为什么没有考虑channels=3的问题呢?
    Result.col(0).setTo(Scalar(0));
    Result.col(Result.cols-1).setTo(Scalar(0));
}

int main(int argc, char* argv[]) {

    help(argv[0]);

    if (argc != 2) {
        std::printf("Usage: %s <path>", argv[0]);
        return 1;
    }
    
    Mat src, dst0, dst1;

    // 命令行参数G指定是否将图像转成灰度图
    if (argc >= 3 && string("G") == string(argv[2]))
        src = imread(argv[1], IMREAD_GRAYSCALE);
    else
        src = imread(argv[1], IMREAD_COLOR);

    // 检查图像是否正常读入
    if (src.empty()) {
        std::cout << "Error opening image" << std::endl;
        return EXIT_FAILURE;
    }

    // 运行程序会显示两个窗口
    namedWindow("Input", WINDOW_AUTOSIZE);
    namedWindow("Output", WINDOW_AUTOSIZE);

    imshow( "Input", src );

    // 调用手写实现的sharpen来做constract enhancement
    // 手写的相比filter的区别就是边界全部置零 会导致图片的看起来小了一圈
    // 而filter会对边界进行padding 处理的更好
    Sharpen( src, dst0 );

    imshow( "Output", dst0 );
    // 默认是零
    waitKey();

    // 使用opencv的filter2D来做constract enhancement
    // 首先定义kernel
    Mat kernel = (Mat_<char>(3,3) <<  0, -1,  0,
                                     -1,  5, -1,
                                      0, -1,  0);
    // 在这里可以发现，我们并没有给dst1分配内存空间
    // 实际上opencv会自动给dst1分配内存空间
    // .depth()是保存图像像素的数据类型, 比如CV_8U
    filter2D(src, dst1, src.depth(), kernel);    
    imshow( "Output", dst1 );
    waitKey();                    

    return 0;
}
