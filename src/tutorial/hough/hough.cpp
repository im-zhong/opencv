// 2023/10/19
// zhangzhong

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <doctest/doctest.h>
using namespace cv;
using namespace std;

TEST_CASE("test hough") {
    const char* filename = "/home/zhangzhong/src/opencv/imgs/orange.jpg";
    // Loads an image
    Mat src = imread(samples::findFile(filename), IMREAD_COLOR);
    // Check if image is loaded fine
    if (src.empty()) {
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default %s] \n", filename);
        return;
    }
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    medianBlur(gray, gray, 5);
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
                 gray.rows / 16.0, // change this value to detect circles with
                                   // different distances to each other
                 100, 30, 60, 93   // change the last two parameters
                 // (min_radius & max_radius) to detect larger circles
    );
    for (size_t i = 0; i < circles.size(); i++) {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        // circle center
        circle(src, center, 1, Scalar(0, 100, 100), 3, LINE_AA);
        // circle outline
        int radius = c[2];
        circle(src, center, radius, Scalar(255, 0, 255), 3, LINE_AA);
    }
    imshow("detected circles", src);
    waitKey();
}
