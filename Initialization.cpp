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
//#include "utils.cpp"
#include <tuple>
using namespace cv;
using std::vector;
using std::array;
using std::string;
using std::fstream;
using std::stringstream;
using Eigen::MatrixXd;


double calc_scoreH(Mat M, vector<Point2f> &matches_kp1, vector<Point2f> &matches_kp2, int kp1_ind[], int kp2_ind[], vector<int> &kp1_ind_inl, vector<int> &kp2_ind_inl)
{
  float score = 0;
  float score_f = 0;
  float score_b = 0;
  float error_f = 0;
  float error_b = 0;
  int count = 0;
  Mat Mf;
  Mat temp;
  invert(M,Mf);
  for (int i = 0; i < matches_kp1.size(); i++)
  {
    vector<double> mkp1 = {matches_kp1[i].x, matches_kp1[i].y, 1.0 };
    Mat m1 (mkp1);
    vector<double> mkp2 = {matches_kp2[i].x, matches_kp2[i].y, 1.0 };
    Mat m2 (mkp2);
    
    temp = M * m2;
    temp.at<double>(0) = temp.at<double>(0) / temp.at<double>(2);
    temp.at<double>(1) = temp.at<double>(1) / temp.at<double>(2);
    temp.at<double>(2) = 1.0;
    error_b = std::pow(norm(temp - m1, NORM_L2), 2);
    
    temp = Mf * m1;
    temp.at<double>(0) = temp.at<double>(0) / temp.at<double>(2);
    temp.at<double>(1) = temp.at<double>(1) / temp.at<double>(2);
    temp.at<double>(2) = 1.0;
    error_f = std::pow(norm(temp - m2, NORM_L2), 2);
    
    
    if (error_f < 5.99)
    {
      score_f = 5.99 - error_f;
    }
    else
    {
      score_f = 0;
    }
    if (error_b < 5.99)
    {
      score_b = 5.99 - error_b; 
    }
    else
    {
      score_b = 0;
    }
    if (score_f > 0 and score_b > 0)
    {
      kp1_ind_inl.push_back(kp1_ind[i]);
      kp2_ind_inl.push_back(kp2_ind[i]);
      count = count + 1;
    }
    score = score + score_f + score_b;
  }
  return score;
}


double calc_scoreF(Mat M, vector<Point2f> &matches_kp1, vector<Point2f> &matches_kp2, int kp1_ind[], int kp2_ind[], vector<int> &kp1_ind_inl, vector<int> &kp2_ind_inl)
{
  float score = 0;
  float score_1 = 0;
  float score_2 = 0;
  float d1 = 0;
  float d2 = 0;
  int count = 0;
  Mat M_t; 
  transpose(M, M_t);
  for (int i = 0; i < matches_kp1.size(); i++ )
  {
    vector<double> mkp1 = {matches_kp1[i].x, matches_kp1[i].y, 1.0 };
    Mat m1 (mkp1);
    vector<double> mkp2 = {matches_kp2[i].x, matches_kp2[i].y, 1.0 };
    Mat m2 (mkp2);
    Mat m1_t;
    Mat m2_t;
    transpose(m1, m1_t);
    transpose(m2, m2_t);
    
    Mat ep2 = m1_t * M;  
    d2 = std::pow(ep2.at<double>(0) * matches_kp2[i].x + ep2.at<double>(1) * matches_kp2[i].y + ep2.at<double>(2) , 2) / (std::pow(ep2.at<double>(0), 2) + std::pow(ep2.at<double>(1), 2));
    
    Mat ep1 = m2_t * M_t; 
    d1 = std::pow(ep1.at<double>(0) * matches_kp1[i].x + ep1.at<double>(1) * matches_kp1[i].y + ep1.at<double>(2) , 2) /  (std::pow(ep1.at<double>(0), 2) + std::pow(ep1.at<double>(1), 2));
    
    if (d1 <  3.84)
    {
      score_1 = 5.99 - d1;
    }
    else
    {
    score_1 = 0;
    }
    if (d2 <  3.84)
    {
      score_2 = 5.99 - d2;
    }
    else
    {
    score_2 = 0;
    }
    if (score_1 > 0 and score_2 > 0)
    {
      kp1_ind_inl.push_back(kp1_ind[i]);
      kp2_ind_inl.push_back(kp2_ind[i]);
      count = count + 1;  
    }
    score = score + score_1 + score_2;
  }
  return score;  
}

std::tuple<float, float> filter_transformation(vector<Mat> &R, vector<Mat> &T, vector<int> &kp1_ind_inlH, vector<int> &kp2_ind_inlH, vector<KeyPoint> &kp1, vector<KeyPoint> &kp2, Mat& K1, Mat& Rn, Mat& Tn)
{
  Mat R_12; 
  Mat T_12;
  Mat R_temp;
  Mat T_temp;
  Mat H(3,4, CV_64F);
  Mat Proj1 (3,4,CV_64F);
  Mat Proj2 (3,4,CV_64F);
  Mat temp;
 
  R_temp = Mat::eye(3,3,CV_64F);
  T_temp = Mat::zeros(3,1, CV_64F);
  for (int j = 0; j < 3; j++)
  { 
    for(int k = 0; k < 3; k++)
    {
      Proj1.at<double>(j,k) = R_temp.at<double>(j,k);
    }
    Proj1.at<double>(j,3) = T_temp.at<double>(j);
  }
  Proj1 = K1 * Proj1;
  Mat kp1_inl(2,kp1_ind_inlH.size(), CV_64F);
  Mat kp2_inl(2,kp2_ind_inlH.size(), CV_64F);
  for (int i = 0; i < kp1_ind_inlH.size(); i++)
  {
    kp1_inl.at<double>(0,i) = kp1[kp1_ind_inlH[i]].pt.x;
    kp1_inl.at<double>(1,i) = kp1[kp1_ind_inlH[i]].pt.y;
    kp2_inl.at<double>(0,i) = kp2[kp2_ind_inlH[i]].pt.x;
    kp2_inl.at<double>(1,i) = kp2[kp2_ind_inlH[i]].pt.y;
  }
  
  float c1 = 0;
  float c2 = 0;
  int count_arr[4];
  int max_index;
  int count_c;
  int count_m = 0;
  int count2_m = 0;
  for (int i = 0; i < R.size(); i++)
  {
   count_c = 0;
   transpose(R[i], R_12);
   T_12 = -R_12 * T[i];
   for (int j = 0; j < 3; j++)
   { 
    for(int k = 0; k < 3; k++)
    {
      H.at<double>(j,k) = R_12.at<double>(j,k);
    }
    H.at<double>(j,3) = T_12.at<double>(j);
   }
   Proj2 = K1 * H;
   Mat world_pts(4,kp1_ind_inlH.size(), CV_64F);
   triangulatePoints(Proj1, Proj2, kp1_inl,kp2_inl,world_pts);
   
   for (int j = 0; j < kp1_ind_inlH.size(); j++)
   {
    Mat pt (3,1, CV_64F);
    pt.at<double>(0) = world_pts.at<double>(0,j) / world_pts.at<double>(3,j);
    pt.at<double>(1) = world_pts.at<double>(1,j) / world_pts.at<double>(3,j);
    pt.at<double>(2) = world_pts.at<double>(2,j) / world_pts.at<double>(3,j);
    c1 = pt.at<double>(2);
    temp = R_12 * (pt - T[i]);
    c2 = temp.at<double>(2);
    if (c1 > 0 and c2 > 0)
    {
      count_c = count_c + 1;
    }
   }
   //std::cout << "Count is: " << count_c <<std::endl;
   count_arr[i] = count_c;
   if (count_c > count_m)
   {
    count_m = count_c;
    Rn = R[i];
    Tn = T[i];
    max_index = i;
    }
  }
  count_c = 0;
  for (int l = 0; l < 4; l++)
  {
    count_c = count_arr[l];
    if (l!= max_index and count_c > count2_m)
      count2_m = count_c;
  }
  float percm = (float(count_m) / float(kp1_ind_inlH.size())) * 100.00;
  float perc2m = (float(count2_m) / float(kp1_ind_inlH.size())) * 100.00;
  return std::make_tuple(percm, perc2m);
}


bool initialize(vector<KeyPoint> &kp1, vector<KeyPoint> &kp2, Mat &desc1, Mat &desc2, Mat &img1, Mat &img2, Mat &K, Mat &Rn, Mat &Tn, vector<int> &ind1, vector<int> &ind2)
{
  bool status = false;
  Ptr <BFMatcher> matcher = BFMatcher::create(NORM_HAMMING, true);
  vector<DMatch> matches;
  matcher -> match(desc1,desc2,matches);
  const int num_matches = matches.size(); 
  if (num_matches < 100)
  {
    status = false;
    return status;
  }
  int kp1_ind[num_matches];
  int kp2_ind[num_matches];
  vector<Point2f> matches_kp1;
  vector<Point2f> matches_kp2;
  for (int i = 0; i < matches.size(); i++)
  {
    kp1_ind[i] = matches[i].queryIdx;
    kp2_ind[i] = matches[i].trainIdx;
    matches_kp1.push_back(kp1[kp1_ind[i]].pt);
    matches_kp2.push_back(kp2[kp2_ind[i]].pt);
  }
  
  /*auto start = std::chrono::high_resolution_clock::now();
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  */
  
  Mat outH;
  int count = 0;
  
  Mat H = findHomography(matches_kp2, matches_kp1, outH, RANSAC, 2.44745);
  for (int j = 0; j < num_matches; j++)
  {
    if (outH.at<uchar>(j) == 1)
    {
      count = count + 1;
    }
  }
  vector<int> kp1_ind_inlH;
  vector<int> kp2_ind_inlH;
  float score_H = calc_scoreH(H, matches_kp1, matches_kp2, kp1_ind, kp2_ind, kp1_ind_inlH, kp2_ind_inlH);
  //draw_matches(img1, img2, kp1, kp2, kp1_ind_inlH, kp2_ind_inlH, matches);
  //std::cout<<"model score of H  is : " << score_H << std::endl;
  
  Mat outF;   
  Mat F = findFundamentalMat(matches_kp2, matches_kp1, outF, FM_RANSAC, 1.95959, 0.99);
  
  count = 0;
  for (int j = 0; j < num_matches; j++)
  {
    if (outF.at<uchar>(j) == 1)
    {
      count = count + 1;
    }
  }
  //std::cout << "No of inliers from fundamental ransac is: " << count << std::endl;
  vector<int> kp1_ind_inlF;
  vector<int> kp2_ind_inlF;
  float score_F = calc_scoreF(F, matches_kp1, matches_kp2, kp1_ind, kp2_ind, kp1_ind_inlF, kp2_ind_inlF);
  //std::cout<<"model score of F  is : " << score_F << std::endl;
  //draw_matches(img1, img2, kp1, kp2, kp1_ind_inlF, kp2_ind_inlF, matches);
  
  float heuristic = score_H / (score_H + score_F);
  //std::cout << "The heuristic score is: " << heuristic << std::endl;
  
  vector<Mat> R;
  vector<Mat> T;
  float percm , perc2m;
  
  if (heuristic > 0.45)
  {
    decomposeHomographyMat(H, K , R, T, noArray());
    
    std::tie(percm,perc2m) = filter_transformation(R, T, kp1_ind_inlH, kp2_ind_inlH, kp1, kp2, K, Rn, Tn);
    
    if (percm > 90 and perc2m < 60)
    {
      ind1 = kp1_ind_inlH;
      ind2 = kp2_ind_inlH;
      std::cout << "Initialised through Homography matrix " << std::endl;
      std::cout << "Percentage of points in front of camera is : " << percm << std::endl;
      std::cout << "No of valid points are: " << ind1.size() << std::endl;
      status = true;
    }
    else
      status = false;
    R.clear();
    T.clear();
  }
  
  else
   {
    Mat KT(3,3,CV_64F);
    transpose(K, KT);
    Mat E = KT * F * K;
    Mat RE1;
    Mat RE2;
    Mat TE;
    decomposeEssentialMat(E, RE1, RE2, TE);
    R.push_back(RE1);
    R.push_back(RE1);
    R.push_back(RE2);
    R.push_back(RE2);
    T.push_back(TE);
    T.push_back(-TE);
    T.push_back(TE);
    T.push_back(-TE);
   
    MatrixXd E_eigen = MatrixXd::Zero(3,3);
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        E_eigen(i,j) = E.at<double>(i,j);
      }
    }
    Eigen::JacobiSVD<MatrixXd> svd(E_eigen, Eigen::ComputeThinV | Eigen::ComputeThinU);
    MatrixXd s(3,3);
    s << 1, 0, 0, 0, 1, 0, 0, 0, 0;
    MatrixXd V_t(3,3);
    V_t = svd.matrixV().transpose();
    E_eigen = svd.matrixU() * s * V_t ;
    
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
       E.at<double>(i,j) =  E_eigen(i,j);
      }
    }

    std::tie(percm,perc2m) = filter_transformation(R, T, kp1_ind_inlF, kp2_ind_inlF, kp1, kp2, K, Rn, Tn);
    
    if (percm > 90 and perc2m < 60)
    {
      ind1 = kp1_ind_inlF;
      ind2 = kp2_ind_inlF;
      std::cout << "Initialised through Fundamental matrix " << std::endl;
      std::cout << "Percentage of points in front of camera is :" << percm << std::endl;
      std::cout << "No of valid points are: " << ind1.size() << std::endl;
      status = true;
    }
    else
      status = false; 
    R.clear();
    T.clear();
   }
  
  return status;    
}
