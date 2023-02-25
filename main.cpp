
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
high_resolution_clock::time_point begining;
#define NOW high_resolution_clock::now()
#define TIME duration_cast<duration<double>>(NOW - start).count()
#define TIME_GLOBAL duration_cast<duration<double>>(NOW - begining).count()

cv::VideoCapture cap;
cv::VideoWriter writer;
cv::Mat frame = cv::Mat::zeros(1080, 1920, CV_8UC3);
cv::Mat blackframe = cv::Mat::zeros(1080, 1920, CV_8UC3);
cv::Size worksize = cv::Size(400,400);
std::vector<cv::Mat> frames;
cv::Mat grayMedianFrame,grayMedianFrameDownscale;

bool doConvexHullsOverlap(const std::vector<cv::Point>& hull1, const std::vector<cv::Point>& hull2, double threshold) {
    std::vector<cv::Point> intersection;
    cv::intersectConvexConvex(hull1, hull2, intersection);
    return intersection.size()>1 && cv::contourArea(intersection)<=std::min(cv::contourArea(hull1),cv::contourArea(hull2));
}

std::vector<std::vector<cv::Point>> contour(cv::Mat tmp_mat){
    cv::Mat  grey_mat;
    cv::GaussianBlur(tmp_mat,grey_mat,cv::Size(15,15),0,0);

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i > hierarchy;
    cv::Mat edged;
    cv::Canny(grey_mat, edged, 20,140);
    cv::findContours(edged, contours, hierarchy,cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);


  std::vector<std::vector<cv::Point> > cvcontours;
  for(auto contour : contours){

    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull); 
    for(auto& p : hull){
      p.x *= (1920./tmp_mat.size().width);
      p.y *= (1080./tmp_mat.size().height); 
    }
    cvcontours.push_back(hull);
  }


  return cvcontours;
  // // Create mask image
  // cv::Mat maskImg = cv::Mat::zeros(grey_mat.size(), CV_8UC1);

  // // Draw contours on mask image
  // cv::drawContours(maskImg, cvcontours, -1, cv::Scalar(255), cv::FILLED);
  // cv::resize(maskImg,maskImg,frame.size());
  // Apply mask to input image
 // return maskImg;

}

void background(std::string path){

  std::string backgroundpath = path + "/background.jpg";
  if (access(backgroundpath.c_str(), F_OK) == 0) {
    grayMedianFrame = cv::imread(backgroundpath);
    cv::cvtColor(grayMedianFrame, grayMedianFrame, cv::COLOR_BGR2GRAY);
    cv::resize(grayMedianFrame,grayMedianFrameDownscale,worksize);
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
    cv::resize(f, f, worksize);
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

std::vector<std::vector<cv::Point>> framediff(cv::Mat background){
  // Calculate absolute difference of current frame and the median frame
  cv::Mat dframe;
  // Convert current frame to grayscale
  cv::resize(frame, dframe, background.size());
  cvtColor(dframe, dframe, cv::COLOR_BGR2GRAY);
 
  absdiff(dframe, background, dframe);
  
  // Threshold to binarize
  threshold(dframe, dframe, 30, 255, cv::THRESH_BINARY);

  return contour(dframe);

}

cv::Mat contour2mask(std::vector<std::vector<cv::Point>>&contours){
  
  // Create mask image
  cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
  // Draw contours on mask image
  cv::drawContours(mask, contours, -1, cv::Scalar(255), cv::FILLED);
  return mask;
}

void run(){

  cap >> frame;
  cv::Mat mask,maskImgDownscale;
  auto contoursDownscale = framediff(grayMedianFrameDownscale);
  auto contours = framediff(grayMedianFrame);

  mask = contour2mask(contours); 
  maskImgDownscale = contour2mask(contoursDownscale); 
  cv::imshow("mask", mask);
  cv::imshow("maskds", maskImgDownscale);

  auto it = contours.begin();
  while(it != contours.end()){
    bool b = true;
    for(auto c : contoursDownscale){
      if(doConvexHullsOverlap(c,*it,0)){
        contours.erase(it);
        b = false;
        break;
      }
    }
    if(b)it++;
  }

  mask = contour2mask(contours);  

  cv::Mat outputImg;
  //cv::bitwise_not(maskImgDownscale,maskImgDownscale);
  // cv::bitwise_not(mask,mask);
  // cv::Mat colorImage;
  // cv::cvtColor(merged_contours_mat, colorImage, cv::COLOR_GRAY2RGB);
  cv::bitwise_and(frame, frame, outputImg, mask);
  //cv::bitwise_and(outputImg, blackframe, outputImg, maskImgDownscale);
  //   cv::imshow("video", outputImg);
  // cv::waitKey(1);

  writer.write(outputImg);
  cv::imshow("bis", outputImg);
  cv::waitKey(1);
}

int main(int argc, char** argv)
{

  start = NOW;
  std::cout << "-------" << std::endl;
  std::string videopath(argv[1]);
  float endtime = std::atoi(argv[2]);
  cap = cv::VideoCapture(videopath+"/video.mp4");
  writer = cv::VideoWriter(videopath+"/video_br.mkv", cv::VideoWriter::fourcc('H','2','6','4'), 10, cv::Size(1920, 1080));

  if(!cap.isOpened()){
    std::cout << "Cannot read video at " << videopath << std::endl;
    exit(0);
  }
 
  int fps = cap.get(cv::CAP_PROP_FPS);
  printf("IS OPENED %d %d\n",cap.isOpened(),fps);
 
  background(videopath);
  begining = NOW;
  while(1){
    start = NOW;
    
    run();

    int us = std::max(0.,SLEEP_TIME - TIME*1e6);
    usleep(us);

    if(TIME_GLOBAL > endtime){
      break;
    }

  }

  cap.release();
  writer.release();
  cv::destroyAllWindows();

  std::cout << "-------" << std::endl;
  std::cout << TIME_GLOBAL << std::endl;
}