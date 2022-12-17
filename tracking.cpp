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

void matchFeaturesInRadius1(vector<int> &index3d, vector<int> &index2d, Frame &lastKeyFrame, WorldPt &worldPt, vector<KeyPoint> &kpCurr,  Mat &descCurr, Mat &R, Mat &T, Mat &K, int width, int height, double searchRadius, vector<int> &indexKeyPts, vector<int> &indexCorres3dPts)
{
 Mat lastKeyDesc; 
 Mat P(3, 4, CV_64F);
 Mat estimPt(1, 2, CV_64F);
 Mat temp(3, 1, CV_64F);
 Mat currworldPt(4, 1, CV_64F);
 getProjMat(R, T, K, P);
 vector<int> nearestKeyPts;
 double dist; 
 Ptr <BFMatcher> matcher = BFMatcher::create(NORM_HAMMING, true);
 int count = 0;
 for (int i = 0; i < index3d.size(); i++)
 {
  Mat desc1;
  vector<vector<DMatch>> matches;
  nearestKeyPts.clear();
  lastKeyDesc = lastKeyFrame.descriptors.row(index2d[i]);
  currworldPt.at<double>(0) = worldPt.pos[index3d[i]][0];
  currworldPt.at<double>(1) = worldPt.pos[index3d[i]][1];
  currworldPt.at<double>(2) = worldPt.pos[index3d[i]][2];
  currworldPt.at<double>(3) = 1.0;

  temp = P * currworldPt;
  estimPt.at<double>(0) = temp.at<double>(0) / temp.at<double>(2);
  estimPt.at<double>(1) = temp.at<double>(1) / temp.at<double>(2);
  //std::cout << "estimated point is : " << estimPt << std::endl;
  for (int j = 0; j < kpCurr.size(); j++)
  {
    double x = double(estimPt.at<double>(0) - kpCurr[j].pt.x);
    double y = double(estimPt.at<double>(1) - kpCurr[j].pt.y);
    dist = sqrt(std::pow(x, 2.0) + std::pow(y, 2.0));
    if (dist < searchRadius)
    {
      nearestKeyPts.push_back(j);
      desc1.push_back(descCurr.row(j));
    }
  }


  if (nearestKeyPts.size() > 0)
  {
    vector<DMatch> tempMatch;
    matcher -> radiusMatch(lastKeyDesc, desc1, matches, 40);
    tempMatch = matches[0];
    if (tempMatch.size() > 0)
    {
       indexKeyPts.push_back(nearestKeyPts[tempMatch[0].trainIdx]);
       indexCorres3dPts.push_back(index3d[i]);
    }
  }
  else
    continue;
 }
}


void trackLastKeyFrame( vector<KeyPoint> &kpCurr, Mat &descCurr, KeyFrame &keyFrame, WorldPt &worldPt, Mat &K,  vector<int> &indexKeyPts, vector<int> &indexCorres3dPts, Mat &R, Mat &T, int width, int height)
{
  Frame lastKeyFrame = keyFrame.frames.back();
  vector<int> index3d;
  vector<int> index2d;
  
  Mat descInView;
  
  for (int i = 0; i < lastKeyFrame.indexPairs.size(); i++)
  {
   index2d.push_back(lastKeyFrame.indexPairs[i][0]); 
   index3d.push_back(lastKeyFrame.indexPairs[i][1]);
   descInView.push_back(lastKeyFrame.descriptors.row(lastKeyFrame.indexPairs[i][0]));
  }
  Ptr <BFMatcher> matcher = BFMatcher::create(NORM_HAMMING, true);
  vector<DMatch> matches;
  matcher -> match(descInView, descCurr, matches);

  
  Mat worldPtInView(matches.size(), 3, CV_64F);
  Mat CurrImgPt (matches.size(), 2, CV_64F);
  
  for (int i = 0; i < matches.size(); i++)
  {
    worldPtInView.at<double>(i,0) = worldPt.pos[index3d[matches[i].queryIdx]][0]; 
    worldPtInView.at<double>(i,1) = worldPt.pos[index3d[matches[i].queryIdx]][1]; 
    worldPtInView.at<double>(i,2) = worldPt.pos[index3d[matches[i].queryIdx]][2];
    CurrImgPt.at<double>(i,0) = kpCurr[matches[i].trainIdx].pt.x; 
    CurrImgPt.at<double>(i,1) = kpCurr[matches[i].trainIdx].pt.y; 
  }
  
  Mat rvec;
  Mat tvec;
  Mat Rt;
  Mat inliers;
  solvePnPRansac(worldPtInView, CurrImgPt, K, noArray(), rvec, tvec, inliers);
  solvePnPRefineLM(worldPtInView, CurrImgPt, K, noArray(),rvec, tvec);
  Rodrigues(rvec, Rt);
  transpose(Rt, R);
  T = -R * tvec;
  
  float r = 4;
  double searchRadius = r * 1.2; 
 
  
  matchFeaturesInRadius1(index3d, index2d, lastKeyFrame, worldPt, kpCurr,  descCurr, R, T, K, width, height, searchRadius, indexKeyPts, indexCorres3dPts);
  
  Mat Corres3dPts(indexCorres3dPts.size(), 3, CV_64F);
  Mat KeyPts(indexKeyPts.size(), 2, CV_64F);
  
  for (int i = 0; i < indexKeyPts.size(); i++)
  {
    Corres3dPts.at<double>(i,0) =  worldPt.pos[indexCorres3dPts[i]][0]; 
    Corres3dPts.at<double>(i,1) =  worldPt.pos[indexCorres3dPts[i]][1]; 
    Corres3dPts.at<double>(i,2) =  worldPt.pos[indexCorres3dPts[i]][2];
    KeyPts.at<double>(i,0) = kpCurr[indexKeyPts[i]].pt.x;
    KeyPts.at<double>(i,1) = kpCurr[indexKeyPts[i]].pt.y;
  }
  
  solvePnPRefineLM(Corres3dPts, KeyPts, K, noArray(), rvec, tvec);
  Rodrigues(rvec, Rt);
  transpose(Rt, R);
  T = -R * tvec;
}

void updateLocalMap(WorldPt &worldPt, KeyFrame &keyFrame, vector<int> &localMapPtInd, vector<int> &localMapKeyIds, int &RefKeyFrame, vector<int> &indexCorres3dPts)
{
  localMapPtInd.clear();
  localMapKeyIds.clear();
  RefKeyFrame = -1;
  vector<int> icpTemp = indexCorres3dPts;
  std::sort(icpTemp.begin(), icpTemp.end());
  int countc = 0;
  int countm = 0;
  for (int i = 0; i < keyFrame.frames.size(); i++)
  {
    countc = 0;
    vector<int> temp;
    vector<int> out(icpTemp.size());
    fill_n(out.begin(), out.size(), -1);
    for (int j = 0; j < keyFrame.frames[i].indexPairs.size(); j++)
    {
      temp.push_back(keyFrame.frames[i].indexPairs[j][1]); 
    }
    std::sort(temp.begin(), temp.end());
    set_intersection(icpTemp.begin(), icpTemp.end(), temp.begin(), temp.end(), out.begin());
    for (int j = 0; j < out.size(); j++)
    {
      if (out[j] != -1)
        countc = countc + 1;
      else
        break;
    }
    if (countc > countm)
    { 
      RefKeyFrame = i;
      countm = countc;
    }  
  }
  
  for (int i = 0; i < keyFrame.frames[RefKeyFrame].covisibility.size(); i++)
  {
    int frameIndex = keyFrame.frames[RefKeyFrame].covisibility[i][0];
    localMapKeyIds.push_back(frameIndex);
    vector<int> temp;
    int countc1 = 0;
    int countm1 = 0;
    int winnerFrame;
    for (int j = 0; j < keyFrame.frames[frameIndex].covisibility.size(); j++)
    {
      countc1 = keyFrame.frames[frameIndex].covisibility[j][1];
      if (countc1 > countm1)
      {
        countm1 = countc1;
        winnerFrame =  keyFrame.frames[frameIndex].covisibility[j][0];
      }
    }
    localMapKeyIds.push_back(winnerFrame);  
  }
  vector<int>::iterator it;
  for (int i = 0; i < localMapKeyIds.size(); i++)
  {
    for (int j = 0; j < keyFrame.frames[localMapKeyIds[i]].indexPairs.size(); j++)
    {
      int elem = keyFrame.frames[localMapKeyIds[i]].indexPairs[j][1];
      it = find(localMapPtInd.begin(), localMapPtInd.end(), elem);
      if (it == localMapPtInd.end())
      {
        localMapPtInd.push_back(elem);
      }    
    }       
  }
}

void removeOutliers(WorldPt &worldPt,vector<int> &new3dPtsInd, Mat &R, Mat &T, Mat &K, int width, int height, vector<int> &new3dPtsInl, vector<double> &predScales )
{
  Mat P(3, 4, CV_64F);
  getProjMat(R, T, K, P);
  Mat estimPt(1, 2, CV_64F);
  Mat temp(3, 1, CV_64F);
  Mat currworldPt(4, 1, CV_64F);
  Mat ray1(3,1, CV_64F);
  Mat ray2(3,1, CV_64F);
  float cosAngle = 0;
  float maxParallax = 60;
  float minDist;
  float maxDist;
  float currDist;
  int level;
  
  for (int i = 0; i < new3dPtsInd.size(); i++)
  {
    currworldPt.at<double>(0) = worldPt.pos[new3dPtsInd[i]][0];
    currworldPt.at<double>(1) = worldPt.pos[new3dPtsInd[i]][1];
    currworldPt.at<double>(2) = worldPt.pos[new3dPtsInd[i]][2];
    currworldPt.at<double>(3) = 1.0;
    temp = P * currworldPt;
    estimPt.at<double>(0) = temp.at<double>(0) / temp.at<double>(2);
    estimPt.at<double>(1) = temp.at<double>(1) / temp.at<double>(2);
    bool cond1 = (estimPt.at<double>(0) >= 0.0) and (estimPt.at<double>(0) <= double(width)) and (estimPt.at<double>(1) >= 0.0) and (estimPt.at<double>(1) <= double(height));
   
   ray1.at<double>(0) = worldPt.viewDirection[new3dPtsInd[i]][0];
   ray1.at<double>(1) = worldPt.viewDirection[new3dPtsInd[i]][1];
   ray1.at<double>(2) = worldPt.viewDirection[new3dPtsInd[i]][2];
   ray2.at<double>(0) = currworldPt.at<double>(0) - T.at<double>(0);
   ray2.at<double>(1) = currworldPt.at<double>(1) - T.at<double>(1);
   ray2.at<double>(2) = currworldPt.at<double>(2) - T.at<double>(2);
   cosAngle = ray1.dot(ray2) / (norm(ray1) * norm(ray2));  
   bool cond2 = (cosAngle >= cos((maxParallax / 180.0) * M_PI));
   
   minDist = worldPt.dmin[new3dPtsInd[i]] / 1.2;
   maxDist = worldPt.dmax[new3dPtsInd[i]] * 1.2;
   
   currDist = sqrt(pow(ray2.at<double>(0),2) + pow(ray2.at<double>(1),2) + pow(ray2.at<double>(2),2));
   bool cond3 = (currDist < maxDist) and (currDist > minDist);
   if (cond1 == true and cond2 == true and cond3 == true)
   {
    new3dPtsInl.push_back(new3dPtsInd[i]);
    level = ceil(log(maxDist/currDist) / log(1.2));
    if (level < 0)
      { level = 0;}
    if (level >= 7)
      { level = 7;}
    predScales.push_back(pow(1.2, level));
   }
  }
}


void matchFeaturesInRadius2(vector<int> &index3d, vector<int> &index2d_unmatched, Mat &localDesc, WorldPt &worldPt, vector<KeyPoint> &kpCurr,  Mat &descCurr, Mat &R, Mat &T, Mat &K, int width, int height, vector<double> &searchRadius, vector<int> &newmatchedKeyPtInd, vector<int> &newmatchedCorres3dInd)
{
 Mat localDescCurr;
 Mat P(3, 4, CV_64F);
 Mat estimPt(1, 2, CV_64F);
 Mat temp(3, 1, CV_64F);
 Mat currworldPt(4, 1, CV_64F);
 getProjMat(R, T, K, P);
 vector<int> nearestKeyPts;
 double dist; 
 Ptr <BFMatcher> matcher = BFMatcher::create(NORM_HAMMING, true);
 int count = 0;
 for (int i = 0; i < index3d.size(); i++)
 {
  Mat desc1;
  vector<vector<DMatch>> matches;
  nearestKeyPts.clear();
  localDescCurr = localDesc.row(i);
  currworldPt.at<double>(0) = worldPt.pos[index3d[i]][0];
  currworldPt.at<double>(1) = worldPt.pos[index3d[i]][1];
  currworldPt.at<double>(2) = worldPt.pos[index3d[i]][2];
  currworldPt.at<double>(3) = 1.0;

  temp = P * currworldPt;
  estimPt.at<double>(0) = temp.at<double>(0) / temp.at<double>(2);
  estimPt.at<double>(1) = temp.at<double>(1) / temp.at<double>(2);
  for (int j = 0; j < index2d_unmatched.size(); j++)
  {
    double x = double(estimPt.at<double>(0) - kpCurr[index2d_unmatched[j]].pt.x);
    double y = double(estimPt.at<double>(1) - kpCurr[index2d_unmatched[j]].pt.y);
    dist = sqrt(std::pow(x, 2.0) + std::pow(y, 2.0));
    if (dist < searchRadius[i])
    {
      nearestKeyPts.push_back(index2d_unmatched[j]);
      desc1.push_back(descCurr.row(index2d_unmatched[j]));
    }
  }


  if (nearestKeyPts.size() > 0)
  {
    vector<DMatch> tempMatch;
    matcher -> radiusMatch(localDescCurr, desc1, matches, 40);
    tempMatch = matches[0];
    if (tempMatch.size() > 0)
    {
       newmatchedKeyPtInd.push_back(nearestKeyPts[tempMatch[0].trainIdx]);
       newmatchedCorres3dInd.push_back(index3d[i]);
    }
  }
  else
    continue;
 }
}


void trackLocalMap(WorldPt &worldPt, KeyFrame &keyFrame, vector<KeyPoint> &kpCurr, Mat &descCurr, Mat &R, Mat &T, Mat &K, vector<int> &localMapPtInd, vector<int> &localMapKeyIds, int &RefKeyFrame, vector<int> &indexKeyPts, vector<int> &indexCorres3dPts, bool &isLastKeyFrame, int &framesPassed, int &minPtsCurrFrame, int width, int height, int index)

{
  if (isLastKeyFrame == true)
  {
    updateLocalMap(worldPt, keyFrame, localMapPtInd, localMapKeyIds, RefKeyFrame, indexCorres3dPts);
  }
  vector<int> new3dPtsInd;
  vector<int>::iterator it;
  for (int i = 0; i < localMapPtInd.size(); i++)
  { 
    it = find(indexCorres3dPts.begin(), indexCorres3dPts.end(), localMapPtInd[i]);
    if (it == indexCorres3dPts.end())
    {
      new3dPtsInd.push_back(localMapPtInd[i]);
    }
  }
  
  vector<int> new3dPtsInl;
  vector<double> predScales;
  removeOutliers(worldPt, new3dPtsInd, R, T, K, width, height, new3dPtsInl, predScales);
  
  Mat localDesc;
  vector<double> searchRadius;
  double viewAngle;
  Mat cameraVector = R.row(2);
  for (int i = 0; i < new3dPtsInl.size(); i++)
  {
    int worldInd = new3dPtsInl[i];
    int repframe = worldPt.representative[worldInd][0];
    int keyPointInd = worldPt.representative[worldInd][1];
    localDesc.push_back(keyFrame.frames[repframe].descriptors.row(keyPointInd));
    Mat currRay(3,1,CV_64F);

    currRay.at<double>(0) = worldPt.pos[worldInd][0] - T.at<double>(0);
    currRay.at<double>(1) = worldPt.pos[worldInd][1] - T.at<double>(1);
    currRay.at<double>(2) = worldPt.pos[worldInd][2] - T.at<double>(2);
    double angle;
    viewAngle = acos(cameraVector.at<double>(0) * currRay.at<double>(0) +  cameraVector.at<double>(1) * currRay.at<double>(1) +  cameraVector.at<double>(2) * currRay.at<double>(2)) / norm(currRay);
    viewAngle = (angle / M_PI) * 180;
    if (viewAngle < 3)
      {searchRadius.push_back(2.5);}
    searchRadius.push_back(predScales[i]);
  }
  
  vector<int> index2d_unmatched;
  vector<int>::iterator it1;

  for (int i = 0; i < kpCurr.size(); i++)
  {
    it1 = find(indexKeyPts.begin(), indexKeyPts.end(), i);
    if (it1 == indexKeyPts.end())
    {
      index2d_unmatched.push_back(i);
    }
  }
  
  vector<int> newmatchedKeyPtInd;
  vector<int> newmatchedCorres3dInd;
  
  
  matchFeaturesInRadius2(new3dPtsInl, index2d_unmatched, localDesc, worldPt, kpCurr, descCurr, R, T, K, width, height, searchRadius, newmatchedKeyPtInd, newmatchedCorres3dInd);
  
  for (int i = 0; i < newmatchedKeyPtInd.size(); i++)
  {
    indexKeyPts.push_back(newmatchedKeyPtInd[i]);
    indexCorres3dPts.push_back(newmatchedCorres3dInd[i]);
  }
    
  Mat Corres3dPts(indexCorres3dPts.size(), 3, CV_64F);
  Mat KeyPts(indexKeyPts.size(), 2, CV_64F);
  
  for (int i = 0; i < indexKeyPts.size(); i++)
  {
    Corres3dPts.at<double>(i,0) =  worldPt.pos[indexCorres3dPts[i]][0]; 
    Corres3dPts.at<double>(i,1) =  worldPt.pos[indexCorres3dPts[i]][1]; 
    Corres3dPts.at<double>(i,2) =  worldPt.pos[indexCorres3dPts[i]][2];
    KeyPts.at<double>(i,0) = kpCurr[indexKeyPts[i]].pt.x;
    KeyPts.at<double>(i,1) = kpCurr[indexKeyPts[i]].pt.y;
  }
  
  Mat rvec;
  Mat tvec;
  Mat Rt;
  solvePnPRefineLM(Corres3dPts, KeyPts, K, noArray(), rvec, tvec);
  Rodrigues(rvec, Rt);
  transpose(Rt, R);
  T = -R * tvec;
  
  solvePnPRefineLM(Corres3dPts, KeyPts, K, noArray(), rvec, tvec);
  Rodrigues(rvec, Rt);
  transpose(Rt, R);
  T = -R * tvec;
  bool cond1 = index > (keyFrame.frames.size() - 1 + framesPassed);
  bool cond2 = indexCorres3dPts.size() < minPtsCurrFrame;
  bool cond3 = indexCorres3dPts.size() <  (0.9 * keyFrame.frames[RefKeyFrame].indexPairs.size());
  isLastKeyFrame = (cond1 or cond2) and (cond3);
}
