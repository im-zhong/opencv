// // 2023/9/16
// // zhangzhong

// #include <opencv2/opencv.hpp>
// #include <iostream>

// using namespace cv;

// int main(int argc, char* argv[]) {
//     if (argc != 2) {
//         printf("usage: DisplayImage.out <Image_Path>\n");
//         return -1;
//     }
//     Mat image;
//     image = imread(argv[1], 1);
//     if (!image.data) {
//         printf("No image data \n");
//         return -1;
//     }
//     namedWindow("Display Image", WINDOW_AUTOSIZE);
//     imshow("Display Image", image);
//     waitKey(0);
//     return 0;
// }

// 1
// // 2023/9/19
// // zhangzhong
// // https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html

// // defined the basic building blocks of the library
// #include <opencv2/opencv.hpp>
// // provides functions for reading and writing
// #include <opencv2/imgcodecs.hpp>
// // contains the functions to show an image in a window
// #include <opencv2/highgui.hpp>

// #include <iostream>
// #include <string>

// using namespace cv;

// int main(int argc, char* argv[]) {
//     // get the first arg as path
//     if (argc != 2) {
//         std::printf("Usage: %s <path>", argv[0]);
//         return 1;
//     }

//     // loads the image using the file path specified by the first argument
//     std::string image_path = samples::findFile(argv[1]);
//     // IMREAD_COLOR: RGB
//     // IMREAD_UNCHANGED: loads the image as is (including the alpha channel
//     if present)
//     // IMREAD_GRAYSCALE: loads the image as an intensity one
//     // the image data will be stored in a cv::Mat object.
//     Mat img = imread(image_path, IMREAD_COLOR);

//     // a check is executed, if the image was loaded correctly.
//     if (img.empty()) {
//         std::cout << "Could not read the image: " << image_path << std::endl;
//         return 1;
//     }

//     // the image is shown using a call to the cv::imshow function
//     imshow("Display window", img);

//     // wait for a keystroke in the window
//     // Zero means to wait forever
//     // The return value is the key that was pressed.
//     int k = waitKey(0);

//     if (k == 's') {
//         // image is written to a file
//         imwrite("starry_night.png", img);
//     }
//     return 0;
// }

// // 2
// // 2023/9/19
// // zhangzhong
// //
// https://docs.opencv.org/4.8.0/d6/d6d/tutorial_mat_the_basic_image_container.html

// // OpenCV is an image processing library
// // all images inside a computer world may be reduced to numerical matrices
// // and other information describing the matrix itself.

// // Mat: RAII
// // 1. the matrix header (containing information such as the size of the
// matrix, the method used for storing, at which address is the matrix stored,
// and so on)
// // 2. a pointer to the matrix containing the pixel values (taking any
// dimensionality depending on the method chosen for storing)

// // OpenCV uses a reference counting system
// // Mat.copy: shallow copy, only copy header
// // Mat.clone: deep copy

// #include <opencv2/core.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui.hpp>

// #include <iostream>
// #include <string>

// int main(int argc, char* argv[]) {
//     cv::Mat A, C;
//     A = cv::imread(argv[1], cv::IMREAD_COLOR);

//     // use the copy constructor
//     // shallow copy
//     cv::Mat B(A);

//     // assignment operator
//     // shallow copy
//     C = A;

//     // All the above objects,
//     // point to the same single data matrix

//     // Nevertheless, their header parts are different
//     // provide different access methods to the same underlying data

//     // ROI: region of interest
//     // you can create headers which refer to only a subsection of the full
//     data cv::Mat D(A, cv::Rect(10, 10, 100, 100)); cv::Mat E =
//     A(cv::Range::all(), cv::Range(1,3)); // using row and column boundaries

//     // deepcopy
//     cv::Mat F = A.clone();
//     cv::Mat G;
//     A.copyTo(G);
//     // now modify F and G will not affect A

//     // rememeber:
//     // Output image allocation for OpenCV functions is automatic (unless
//     specified otherwise).
//     // You do not need to think about memory management with OpenCV's C++
//     interface.
//     // The assignment operator and the copy constructor only copy the header.
//     // The underlying matrix of an image may be copied using the
//     cv::Mat::clone() and cv::Mat::copyTo() functions.

//     // TODO: 为什么用了cv前缀反而不行了, 除非这是个宏
//     // Create a Mat object explicitly
//     // cv::Mat M(2, 2, cv::CV_8UC3, cv::Scalar(0, 0, 255));
//     using namespace cv;

//     Mat M(2,2, // For two dimensional and multichannel images we first define
//     their size: row and column count wise. CV_8UC3, // specify the data type
//     to use for storing the elements and the number of channels per matrix
//     point
//                 // CV_[The number of bits per item][Signed or Unsigned][Type
//                 Prefix]C[The channel number]
//                 // CV_8UC3: unsigned char types that are 8 bit long nd each
//                 pixel has three of these to form the three channels.
//     Scalar(0,0,255)); // four element short vector, initialize all matrix
//     points with a custom value.
//     // << operator only works for two dimensional matrices
//     std::cout << "M = " << std::endl
//                 << M << std::endl;

//     int sz[3] = {2,2,2};
//     // // more that two dimensions
//     Mat L(3,    // specify its dimension
//     sz,         // pass a pointer containing the size for each dimension
//     CV_8UC(1), Scalar::all(0)); // remain the same as above
//     // so can not use << operator
//     // std::cout << "L = " << std::endl
//     //             << L << std::endl;

//     // cv::Mat::create() function
//     // what dose this function mean?
//     M.create(4,4, CV_8UC(2));

//     using namespace std;
//     // MATLAB style
//     Mat EE = Mat::eye(4, 4, CV_64F);
//     cout << "E = " << endl << " " << EE << endl << endl;
//     Mat O = Mat::ones(2, 2, CV_32F);
//     cout << "O = " << endl << " " << O << endl << endl;
//     Mat Z = Mat::zeros(3,3, CV_8UC1);
//     cout << "Z = " << endl << " " << Z << endl << endl;

// For small matrices you may use comma separated initializers
// or initializer lists (C++11 support is required in the last case):
// Mat CC = (Mat_<double>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);

//     // create a new header for an existing Mat object
//     // cv::Mat::clone(), cv::Mat::copyTo()
//     Mat RowClone = C.row(1).clone();
//     cout << "RowClone = " << endl << " " << RowClone << endl << endl;

//     // fill out a matrix with random value with cv::randu()
//     // set lower and upper limit
//     Mat R = Mat(3, 2, CV_8UC3);
//     randu(R, Scalar::all(0), Scalar::all(255));
//     cout << "R = " << endl << " " << R << endl << endl;

//     // output formatting
//     cout << "R (default) = " << endl << R << endl << endl;
//     cout << "R (python) = " << endl << format(R, Formatter::FMT_PYTHON) <<
//     endl << endl; cout << "R (csv) = " << endl << format(R,
//     Formatter::FMT_CSV ) << endl << endl; cout << "R (numpy) = " << endl <<
//     format(R, Formatter::FMT_NUMPY ) << endl << endl; cout << "R (c) = " <<
//     endl << format(R, Formatter::FMT_C ) << endl << endl;
// }

// 3
// 2023/9/19
// zhangzhong
// https://docs.opencv.org/4.8.0/db/da5/tutorial_how_to_scan_images.html

// for larger images it would be wise to calculate all possible values
// beforehand and during the assignment just make the assignment, by using a
// lookup table.
//  Lookup tables are simple arrays (having one or more dimensions) that for a
//  given input value variation holds the final output value

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <string>

using namespace cv;
using namespace std;

// 1. The efficient way
Mat& ScanImageAndReduce(Mat& I, const uchar* const table) {
    // accept only char type matrices
    CV_Assert(I.depth() == CV_8U);

    int channels = I.channels();

    int nRows = I.rows;
    int nCols = I.cols * channels;

    // The .data member of a Mat object returns the pointer to the first row,
    // first column If this pointer is null you have no valid input in that
    // object.

    // in many cases the memory is large enough to store the rows in a
    // successive fashion the rows may follow one after another, creating a
    // single long row.
    if (I.isContinuous()) {
        nCols *= nRows;
        nRows = 1;
    }

    // Here we basically just acquire a pointer to the start of each row and go
    // through it until it ends
    int i, j;
    uchar* p;
    for (i = 0; i < nRows; ++i) {
        // get the pointer to the ith row
        p = I.ptr<uchar>(i);
        for (j = 0; j < nCols; ++j) {
            // p[j] is the pixel value!
            p[j] = table[p[j]];
        }
    }

    return I;
}

// 2. The iterator (safe) method
Mat& ScanImageAndReduceIterator(Mat& I, const uchar* const table) {
    // accept only char type matrices
    CV_Assert(I.depth() == CV_8U);

    int channels = I.channels();

    switch (channels) {
    case 1: {
        MatIterator_<uchar> it, end;
        for (it = I.begin<uchar>(), end = I.end<uchar>; it != end; ++it) {
            // *it is the pixel value!
            *it = table[*it];
        }
        break;
    }

    case 3: {
        MatIterator_<Vec3b> it, end;
        for (it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it) {
            // BGR
            (*it)[0] = table[(*it)[0]];
            (*it)[1] = table[(*it)[1]];
            (*it)[2] = table[(*it)[2]];
        }
    }
    }

    return I;
}

// 3. On-the-fly address calculation with reference returning
// It was made to acquire or modify somehow random elements in the image
// Its basic usage is to specify the row and column number of the item you want
// to access cv::Mat::at()
Mat& ScanImageAndReduceRandomAccess(Mat& I, const uchar* const table) {
    // accept only char type matrices
    CV_Assert(I.depth() == CV_8U);

    const int channels = I.channels();
    switch (channels) {
    case 1: {
        for (int i = 0; i < I.rows; ++i) {
            for (int j = 0; j < I.cols; ++j) {
                // The function takes your input type and coordinates and
                // calculates the address of the queried item
                I.at<uchar>(i, j) = table[I.at<uchar>(i, j)];
            }
        }
        break;
    }

    case 3: {
        Mat_<Vec3b> _I = I;
        for (int i = 0; i < I.rows; ++i) {
            for (int j = 0; j < I.cols; ++j) {
                _I(i, j)[0] = table[_I(i, j)[0]];
                _I(i, j)[1] = table[_I(i, j)[1]];
                _I(i, j)[2] = table[_I(i, j)[2]];
            }
        }
        // convert back
        I = _I;
        break;
    }
    }

    return I;
}

// 4. the best way
// cv::LUT() multi-thread
Mat ScanImageAndReduceLUT(Mat& I, const uchar* const table) {
    // make a new Mat and copy table into this Mat
    Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for (int i = 0; i < 256; ++i) {
        p[i] = table[i];
    }

    // I: input
    // J: output
    LUT(I, lookUpTable, J);
    return J;
}

int main(int argc, char* argv[]) {
    // Usage: how_to_scan_images imageName.jpg intValueToReduce [G]
    // The final argument is optional. If given the image will be loaded in
    // grayscale format, otherwise the BGR color space is used read in an image
    // passed as a command line argument (it may be either color or grayscale)
    // apply the reduction with the given command line argument integer value

    // 1. calculate the lookup table
    int divideWith = 0; // convert our input string to number - C++ style
    stringstream s;
    s << argv[2];
    s >> divideWith;

    if (!s || !divideWith) {
        cout << "Invalid number entered for dividing. " << endl;
        return -1;
    }

    // use pixel value to index the lookup table
    uchar table[256];
    for (int i = 0; i < 256; ++i) {
        table[i] = (uchar)(divideWith * (i / divideWith));
    }

    // TODO:
    // 实现读入图像并使用上述四种方法对图像进行处理，然后输出图像和统计执行时间
}

// color space convertion
// https://docs.opencv.org/4.8.0/d8/d01/group__imgproc__color__conversions.html