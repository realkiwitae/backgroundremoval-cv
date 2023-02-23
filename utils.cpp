#include "utils.h"

int im_proc::computeMedian(std::vector<int> v) 
{
  size_t n = v.size() / 2;
  std::nth_element(v.begin(), v.begin()+n, v.end());
  int vn = v[n];
  if(v.size()%2 == 1)
  {
    return vn;
  }else
  {
    std::nth_element(v.begin(), v.begin()+n-1, v.end());
    return 0.5*(vn+v[n-1]);
  }
}
 
cv::Mat im_proc::compute_median(std::vector<cv::Mat> vec) 
{
  // Note: Expects the image to be CV_8UC3
  cv::Mat medianImg(vec[0].rows, vec[0].cols, CV_8UC3, cv::Scalar(0, 0, 0));
 
  for(int row=0; row<vec[0].rows; row++) 
  {
    for(int col=0; col<vec[0].cols; col++) 
    {
      std::vector<int> elements_B;
      std::vector<int> elements_G;
      std::vector<int> elements_R;
 
      for(size_t imgNumber=0; imgNumber<vec.size(); imgNumber++) 
      {
        int B = vec[imgNumber].at<cv::Vec3b>(row, col)[0];
        int G = vec[imgNumber].at<cv::Vec3b>(row, col)[1];
        int R = vec[imgNumber].at<cv::Vec3b>(row, col)[2];
 
        elements_B.push_back(B);
        elements_G.push_back(G);
        elements_R.push_back(R);
      }
      
      medianImg.at<cv::Vec3b>(row, col)[0]= computeMedian(elements_B);
      medianImg.at<cv::Vec3b>(row, col)[1]= computeMedian(elements_G);
      medianImg.at<cv::Vec3b>(row, col)[2]= computeMedian(elements_R);
    }
  }
  return medianImg;
}