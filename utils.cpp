#include <iostream>
#include <vector>
#include <iostream>
#include <vector>
#include <array>
#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <chrono>
#include <cmath>
#include <bits/stdc++.h>
using namespace cv;
using std::vector;
using std::array;
using std::string;
using std::fstream;
using std::stringstream;
using Eigen::MatrixXd;

void draw_matches(Mat &img1,Mat &img2, vector<KeyPoint> &kp1, vector<KeyPoint> &kp2,vector<int> &kp1_ind_inl, vector<int> &kp2_ind_inl, vector<DMatch> &matches)
{
  vector<KeyPoint> kp1_inl;
  vector<KeyPoint> kp2_inl;
  vector<DMatch> match_inl;
  DMatch temp_match;
  int count = 0;
  for (int i = 0; i < kp1_ind_inl.size(); i++)
  {
   kp1_inl.push_back(kp1[kp1_ind_inl[i]]);
   kp2_inl.push_back(kp2[kp2_ind_inl[i]]);
   for(int j = 0; j < matches.size(); j++)
    {
     if (matches[j].queryIdx == kp1_ind_inl[i] and matches[j].trainIdx == kp2_ind_inl[i])
     {
      temp_match = matches[j];
      temp_match.queryIdx = i;
      temp_match.trainIdx = i;
      match_inl.push_back(temp_match);
      break;
     }
    }
  }

    
  Mat out_img;
  drawMatches(img1,kp1_inl,img2,kp2_inl,match_inl,out_img);
  imshow("matches", out_img);
  waitKey(0);
}
double findMedian(vector<float> arr, int n)
{
  if (n % 2 == 0) 
  {
    std::nth_element(arr.begin(),arr.begin() + n / 2, arr.end());
    std::nth_element(arr.begin(), arr.begin() + (n - 1) / 2, arr.end());
    return (double)(arr[(n - 1) / 2] + arr[n / 2]) / 2.0;
  }
  else 
  {
    std::nth_element(arr.begin(),arr.begin() + n / 2, arr.end());
    return (double)arr[n / 2];
  }
}

void getProjMat(Mat &R, Mat &T, Mat &K, Mat &Proj)
{
  Mat RT;
  Mat TT;
  transpose(R, RT);
  TT = -RT * T;
  for (int i = 0; i < 3; i++)
  { 
    for(int j = 0; j < 3; j++)
    {
      Proj.at<double>(i,j) = RT.at<double>(i,j);
    }
    Proj.at<double>(i,3) = TT.at<double>(i);
  }
  Proj = K * Proj;
}
