
#include <chrono>
#include <unistd.h>
#include <iostream>
#include <random>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define FPS 10
#define SLEEP_TIME 1000000/FPS
high_resolution_clock::time_point start;
high_resolution_clock::time_point start_move;
#define NOW high_resolution_clock::now()
#define TIME duration_cast<duration<double>>(NOW - start).count()
#define TIME_MOVE duration_cast<duration<double>>(NOW - start_move).count()

cv::VideoCapture cap;
cv::Mat frame = cv::Mat::zeros(1080, 1920, CV_8UC3);
std::vector<cv::Mat> frames;
cv::Mat grayMedianFrame;

void contour(cv::Mat tmp_mat){
    cv::Mat  grey_mat;
    cv::GaussianBlur(tmp_mat,grey_mat,cv::Size(7,7),0,0);

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i > hierarchy;
    cv::Mat edged;
    cv::Canny(grey_mat, edged, 20,140);
    cv::findContours(edged, contours, hierarchy,cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Draw the merged contours on the image
    cv::Mat merged_contours_mat = cv::Mat::zeros(grey_mat.size(), CV_8UC1);
    for (auto contour : contours)
    {
        if (!contour.empty())
        {
            drawContours(merged_contours_mat, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), 2);
        }
    }



    cv::imshow("video", merged_contours_mat);
    cv::waitKey(1);
}

void background(std::string path){

  std::string backgroundpath = path + "/background.jpg";
  if (access(backgroundpath.c_str(), F_OK) == 0) {
    frame = cv::imread(backgroundpath);
    cv::cvtColor(frame, grayMedianFrame, cv::COLOR_BGR2GRAY);
    return;
  }

  default_random_engine generator;
  uniform_int_distribution<int>distribution(0, 
  cap.get(cv::CAP_PROP_FRAME_COUNT));
 
  vector<cv::Mat> frames;
 
  while(frames.size() < 20)
  {
    int fid = distribution(generator);
    bool b = cap.set(cv::CAP_PROP_POS_FRAMES, fid);
    if(!b) continue;
    int id = cap.get(CV_CAP_PROP_POS_FRAMES);
    if(id != fid)continue;
    cv::Mat f;
    cap.read(f);
    if(f.empty())
      continue;
    // cv::imshow("frame", frame);
    // cv::waitKey(0);
    frames.push_back(f);
  }

  cv::Mat medianFrame = im_proc::compute_median(frames);
  cv::imshow("frame", medianFrame);
  cv::waitKey(1000);

  //  Reset frame number to 0
  cap.set(cv::CAP_PROP_POS_FRAMES, 0);
 
  // Convert background to grayscale

  cvtColor(medianFrame, grayMedianFrame, cv::COLOR_BGR2GRAY);
  cv::imwrite(backgroundpath, grayMedianFrame);
}

void framediff(){
  // Calculate absolute difference of current frame and the median frame
  cv::Mat dframe;
  // Convert current frame to grayscale
  cvtColor(frame, dframe, cv::COLOR_BGR2GRAY);
 

  absdiff(dframe, grayMedianFrame, dframe);
  
  // Threshold to binarize
  threshold(dframe, dframe, 30, 255, cv::THRESH_BINARY);
 
  // Display Image
  contour(dframe);
}

void run(){

    cap >> frame;
    framediff();

}

int main(int argc, char** argv)
{
  start = NOW;
  std::cout << "-------" << std::endl;
  std::string videopath(argv[1]);
  cap = cv::VideoCapture(videopath+"/video.mp4");

  if(!cap.isOpened()){
    std::cout << "Cannot read video at " << videopath << std::endl;
    exit(0);
  }
 
  int fps = cap.get(cv::CAP_PROP_FPS);
  printf("IS OPENED %d %d\n",cap.isOpened(),fps);
 
  background(videopath);
  while(1){
    start = NOW;
    
    run();

    int us = std::max(0.,SLEEP_TIME - TIME*1e6);
    usleep(us);

  }

  std::cout << "-------" << std::endl;
  std::cout << TIME << std::endl;
}