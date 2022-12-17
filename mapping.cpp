#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <limits>
#include <algorithm> 

using namespace cv;
using std::vector;
using std::array;
using std::string;
using std::fstream;
using std::stringstream;
using Eigen::MatrixXd;
/*
void cullMapPts(WorldPt &worldPt, KeyFrame &keyFrame)
{
  for (int i = 0; i < worldPt.pos.size(); i++)
  {
    int count = 0; 
    vector<int> frameInd;
    for (int j = 0; j < keyFrame.frames.size(); j++)
    {
      vector<int> currIndex3d;
      vector<int>::iterator it; 
      for (int k = 0; k < keyFrame.frames[j].indexPairs.size(); k++)
      {
        currIndex3d.push_back(keyFrame.frames[j].indexPairs[k][1]);
      }
      for (int k = 0; k < curIndex3d.size(); k++)
      {
        it = find(currIndex3d.begin(), currIndex3d.end(), i);
        if (it !=  currIndex3d.end())
        { 
          frameInd.push_back({j,it - v.begin()});
          count = count + 1;
          if (count > 3)
            {break;}    
        }     
      }
    }
    if (count < 3)
    {
      removeOutliers(worldPt, keyFrame, int 3dind,);
      i = i -1;
    }   
  }
*/  

void addKeyFrame(Mat &R, Mat &T, vector<int> &indexKeyPts, vector<int> &indexCorres3dPts, vector<KeyPoint> &kpCurr, Mat &descCurr, KeyFrame &keyFrame)
{
  vector<vector<int>> indexPairs;
  for (int i = 0; i < indexKeyPts.size(); i++)
  {
    indexPairs.push_back({indexKeyPts[i], indexCorres3dPts[i]});  
  }  
  keyFrame.addKeyFrames(R, T, indexPairs, kpCurr, descCurr);
  
  vector<int> lastIndex3d;  
  for (int i = 0; i < keyFrame.frames.back().indexPairs.size(); i++)
  {
    lastIndex3d.push_back(keyFrame.frames.back().indexPairs[i][1]);
  }
  
  vector<int>::iterator it;
  for (int i = 0; i < keyFrame.frames.size() -1; i++)
  {
    int count = 0;
    for (int j = 0; j < keyFrame.frames[i].indexPairs.size(); j++)
    {
      int elem = keyFrame.frames[i].indexPairs[j][1];
      it = find(lastIndex3d.begin(), lastIndex3d.end(), elem);
      if (it != lastIndex3d.end())
        {count = count + 1;}
    }
    if (count > 5)
    {keyFrame.addCovisibility(keyFrame.frames.size() -1 , i , count);}
  }  
}

void addNewPoints(WorldPt &worldPt, KeyFrame & keyFrame, vector<int> indexKeyPts, Mat K)
{
  Mat unmatchedDesc;
  vector<int> unmatchedInd;
  for (int i = 0; i < keyFrame.frames.back().features.size(); i++)
  {
    vector<int>::iterator it;  
    it = find(indexKeyPts.begin(), indexKeyPts.end(), i);
    if (it == indexKeyPts.end())
    {
      unmatchedInd.push_back(i);
      unmatchedDesc.push_back(keyFrame.frames.back().descriptors.row(i));
    }  
  }
  
  
  for (int i = 0; i < keyFrame.frames.size(); i++)
  {
    Mat unmatchedDescCurr;
    vector<int> unmatchedIndCurr;
    vector<int> matchedIndCurr;
    
    for (int j = 0; j < keyFrame.frames[i].indexPairs.size(); j++)
    {
     matchedIndCurr.push_back(keyFrame.frames[i].indexPairs[j][0]);
    }
    
    for (int j = 0; j < keyFrame.frames[i].features.size(); j++)
    {
      vector<int>::iterator it;  
      it = find(matchedIndCurr.begin(), matchedIndCurr.end(), j);
      if (it == matchedIndCurr.end())
      {
       unmatchedIndCurr.push_back(j);
       unmatchedDescCurr.push_back(keyFrame.frames.back().descriptors.row(j));
      }   
    }
    
    Ptr <BFMatcher> matcher = BFMatcher::create(NORM_HAMMING, true);
    vector<DMatch> matches;
    matcher -> match(unmatchedDesc, unmatchedDescCurr, matches);
    
    int kp1_ind[matches.size()];
    int kp2_ind[matches.size()];
    vector<Point2f> matches_kp1;
    vector<Point2f> matches_kp2;
    for (int j = 0; j < matches.size(); j++)
    {
      kp1_ind[j] = unmatchedInd[matches[j].queryIdx];
      kp2_ind[j] = unmatchedIndCurr[matches[j].trainIdx];
      matches_kp1.push_back(keyFrame.frames.back().features[kp1_ind[j]].pt);
      matches_kp2.push_back(keyFrame.frames[i].features[kp2_ind[j]].pt);
    }
    
     
    vector<int> inl1;
    vector<int> inl2;
    Mat outF;
    Mat F = findFundamentalMat(matches_kp2, matches_kp1, outF, FM_RANSAC, 1.95959, 0.99);
    for (int j = 0; j < matches.size(); j++)
    {
      if (outF.at<uchar>(j) == 1)
      {
        inl1.push_back(kp1_ind[j]);  
        inl2.push_back(kp2_ind[j]);  
      }
    }
    if (inl1.size() > 10)
    {
      Mat R1 = keyFrame.frames.back().R;
      Mat T1 = keyFrame.frames.back().T;
      
      Mat R2 = keyFrame.frames[i].R;
      Mat T2 = keyFrame.frames[i].T;
      vector<vector<int>>  indexPairsComb;
      bool status = worldPt.addWorldPts(R1, R2, T1, T2, inl1, inl2, K, keyFrame.frames.back().features, keyFrame.frames[i].features, indexPairsComb, 1.0, 3.0);
      double medianDepth = worldPt.scaleMap();
      keyFrame.scalePoses(medianDepth);
      keyFrame.updateIndexPairs(keyFrame.frames.size() - 1, i, indexPairsComb);
      keyFrame.addCovisibility(keyFrame.frames.size() -1 , i , indexPairsComb.size());
      worldPt.updateDirectionDistance(keyFrame.frames);
      worldPt.updateRepresentativeView(keyFrame.frames);
    }
  }
}
