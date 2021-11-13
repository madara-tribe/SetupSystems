#include <iostream>
#include <opencv2/opencv.hpp>

#define IMAGE_WIDTH 416 
#define IMAGE_HEIGHT 416
#define IMAGE "sample.png"

using namespace cv;
int main()
{
    Mat img;
    const auto shapes = Size(IMAGE_WIDTH, IMAGE_HEIGHT);
    img = cv::imread(IMAGE);
    resize(img, img, shapes, INTER_NEAREST);
    std::cout << img.size << std::endl;
    return 0;
}
