#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <opencv2/opencv.hpp>

#pragma once

namespace im_proc{
    int computeMedian(std::vector<int> elements);
    cv::Mat compute_median(std::vector<cv::Mat> vec); 
};

#endif