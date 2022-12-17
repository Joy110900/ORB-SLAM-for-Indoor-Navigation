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

using namespace cv;
using std::vector;
using std::array;
using std::string;
using std::fstream;
using std::stringstream;
using Eigen::MatrixXd;

struct Frame
{
  Mat R = Mat::zeros(3, 3, CV_64F);
  Mat T = Mat::zeros(3, 1, CV_64F);
  vector<vector<int>> indexPairs; //first index corresponds to keypoint, 2nd corresponds to world point
  vector<KeyPoint> features; 
  Mat descriptors;
  vector<vector<int>> covisibility; // first index corresponds to frame index, 2nd corresponds to num of common points 
};

class KeyFrame
{
  public:
    vector<Frame> frames;
    
    void addKeyFrames(Mat &R, Mat &T, vector<vector<int>>& indexPairs, vector<KeyPoint> &kp, Mat &desc)
    {
      Frame tempFrame;
      tempFrame.R = R;
      tempFrame.T = T;
      tempFrame.features = kp;
      tempFrame.descriptors = desc;
      tempFrame.indexPairs = indexPairs;
      frames.push_back(tempFrame);
    } 
   void addCovisibility(int srcFrame, int dstFrame, int commonPts)
   {
    frames[srcFrame].covisibility.push_back({dstFrame, commonPts});
    frames[dstFrame].covisibility.push_back({srcFrame, commonPts});
   }
   void scalePoses(double medianDepth)
    {
     for (int i = 0; i < frames.size(); i++ )
     {
       frames[i].T.at<double>(0) =  frames[i].T.at<double>(0) / medianDepth;
       frames[i].T.at<double>(1) =  frames[i].T.at<double>(1) / medianDepth;
       frames[i].T.at<double>(2) =  frames[i].T.at<double>(2) / medianDepth;
     }
    }
    void updateIndexPairs(int frame1, int frame2, vector<vector<int>> indexPairsComb)
    {
      for (int i = 0; i < indexPairsComb.size(); i++)
      {
        frames[frame1].indexPairs.push_back({indexPairsComb[i][0], indexPairsComb[i][2]});
        frames[frame2].indexPairs.push_back({indexPairsComb[i][1], indexPairsComb[i][2]});
      }
    }
};

class WorldPt
{
  public:
    vector<vector<double>> pos;
    vector<vector<double>> viewDirection;
    vector<double> dmax;
    vector<double> dmin;
    vector<vector<int>> representative; // 1st index denots frame, 2nd index denotes keypoint in the frame
    
    bool addWorldPts(Mat &R1, Mat &R2, Mat &T1, Mat&T2, const vector<int> &ind1, const vector<int> &ind2, Mat &K, vector<KeyPoint> &kp1, vector<KeyPoint> &kp2, vector<vector<int>>& indexPairsComb, float maxReprojError, float minParallax )
    {
     Mat proj1 (3,4,CV_64F);
     Mat proj2 (3,4,CV_64F);
     getProjMat(R1, T1, K, proj1);
     getProjMat(R2, T2, K, proj2);
     
     Mat kp1Inl(2,ind1.size(), CV_64F);
     Mat kp2Inl(2,ind2.size(), CV_64F);
     for (int i = 0; i < ind1.size(); i++)
     {
      kp1Inl.at<double>(0,i) = kp1[ind1[i]].pt.x;
      kp1Inl.at<double>(1,i) = kp1[ind1[i]].pt.y;
      kp2Inl.at<double>(0,i) = kp2[ind2[i]].pt.x;
      kp2Inl.at<double>(1,i) = kp2[ind2[i]].pt.y;
     }
     
     Mat worldPts(4,ind1.size(), CV_64F);
     triangulatePoints(proj1, proj2, kp1Inl,kp2Inl,worldPts);
     
     Mat R1T;
     Mat R2T;
     transpose(R1, R1T);
     transpose(R2, R2T);
     Mat temp;
     Mat currPt(3,1, CV_64F);
     Mat currImg1Pt(2,1, CV_64F);
     Mat currImg2Pt(2,1, CV_64F);
     Mat ray1(3,1, CV_64F);
     Mat ray2(3,1, CV_64F);
     vector<bool> isParallax;
     float d1 = 0;
     float d2 = 0;
     float proj1Error = 0;
     float proj2Error = 0;
     Mat proj1ErrorVec(3, 1, CV_64F);
     Mat proj2ErrorVec(3, 1, CV_64F);
     float avgReprojError = 0;
     float cosAngle = 0;
     bool status;
     vector<float> pt;
     for (int i = 0; i < ind1.size(); i++)
     {
      currImg1Pt = kp1Inl.col(i);
      currImg2Pt = kp2Inl.col(i);
      
      proj1ErrorVec = proj1 * worldPts.col(i);
      proj1ErrorVec.at<double>(0) = proj1ErrorVec.at<double>(0) / proj1ErrorVec.at<double>(2); 
      proj1ErrorVec.at<double>(1) = proj1ErrorVec.at<double>(1) / proj1ErrorVec.at<double>(2); 
      proj1Error = sqrt(pow(proj1ErrorVec.at<double>(0) - currImg1Pt.at<double>(0), 2) + pow(proj1ErrorVec.at<double>(1) - currImg1Pt.at<double>(1), 2));
      
      proj2ErrorVec = proj2 * worldPts.col(i);
      proj2ErrorVec.at<double>(0) = proj2ErrorVec.at<double>(0) / proj2ErrorVec.at<double>(2); 
      proj2ErrorVec.at<double>(1) = proj2ErrorVec.at<double>(1) / proj2ErrorVec.at<double>(2); 
      proj2Error = sqrt(pow(proj2ErrorVec.at<double>(0) - currImg2Pt.at<double>(0), 2) + pow(proj2ErrorVec.at<double>(1) - currImg2Pt.at<double>(1), 2));
 
      avgReprojError = (proj1Error + proj2Error) / 2.0; 
      currPt.at<double>(0) = worldPts.at<double>(0,i) / worldPts.at<double>(3,i);
      currPt.at<double>(1) = worldPts.at<double>(1,i) / worldPts.at<double>(3,i);
      currPt.at<double>(2) = worldPts.at<double>(2,i) / worldPts.at<double>(3,i);

      
      temp = R1T * (currPt - T1); 
      d1 = temp.at<double>(2);
      temp = R2T * (currPt - T2);
      d2 = temp.at<double>(2);
      
      ray1 = currPt - T1;
      ray2 = currPt- T2;
      cosAngle = ray1.dot(ray2) / (norm(ray1) * norm(ray2));  
      if ( (d1 > 0 and d2 > 0) and (avgReprojError < maxReprojError) and (cosAngle <= cos((minParallax / 180.0) * M_PI)))
      {
        pos.push_back({currPt.at<double>(0), currPt.at<double>(1), currPt.at<double>(2) });
        indexPairsComb.push_back({ind1[i], ind2[i], int(pos.size()-1)});
      }
     }
     if (pos.size() > 0)
      status = true;
     else
      status = false;
     std::cout << "Num of points added to map are: " << pos.size() << std::endl;
     return status;
    }
    double scaleMap()
    {
      vector<float> posNorms;
      Mat temp(3, 1 , CV_64F);
      double currNorm;
      double medianDepth;
      for (int i = 0; i < pos.size(); i++)
      {
        temp.at<double>(0) = pos[i][0];
        temp.at<double>(1) = pos[i][1];
        temp.at<double>(2) = pos[i][2];
        currNorm = norm(temp);
        posNorms.push_back(currNorm);
      }
      medianDepth = findMedian(posNorms, int(pos.size()));
      for (int i = 0; i < pos.size(); i++)
      {
        pos[i][0] = pos[i][0] / medianDepth;
        pos[i][1] = pos[i][1] / medianDepth;
        pos[i][2] = pos[i][2] / medianDepth;
      }
      return medianDepth;  
    }
    void updateDirectionDistance(vector<Frame> &frames)
    {
     for (int i  = 0; i < pos.size(); i++)
     {
      double diff[3];
      double distArr[frames.size()];
      double currDirection[3];
      double meanDirection[3] = {};
      double meanDirectionDist = 0;
      for (int j = 0; j < frames.size(); j++)
      {
        diff[0] = pos[i][0] - frames[j].T.at<double>(0);
        diff[1] = pos[i][1] - frames[j].T.at<double>(1);
        diff[2] = pos[i][2] - frames[j].T.at<double>(2);
        distArr[j] = sqrt(diff[0]* diff[0] + diff[1]* diff[1] + diff[2]* diff[2]);
        currDirection[0] = diff[0] / distArr[j];
        currDirection[1] = diff[1] / distArr[j];
        currDirection[2] = diff[2] / distArr[j];
        meanDirection[0] = meanDirection[0] + currDirection[0];
        meanDirection[1] = meanDirection[1] + currDirection[1];
        meanDirection[2] = meanDirection[2] + currDirection[2];
      }
      dmax.push_back(*std::max_element(distArr, distArr + frames.size()));
      dmin.push_back(*std::min_element(distArr, distArr + frames.size()));
      
      meanDirection[0] = meanDirection[0] / frames.size();
      meanDirection[1] = meanDirection[1] / frames.size();
      meanDirection[2] = meanDirection[2] / frames.size();
      
      meanDirectionDist = sqrt(meanDirection[0] * meanDirection[0]  + meanDirection[1] * meanDirection[1]  + meanDirection[2] * meanDirection[2]);
      
      meanDirection[0] = meanDirection[0] / meanDirectionDist;
      meanDirection[1] = meanDirection[1] / meanDirectionDist;
      meanDirection[2] = meanDirection[2] / meanDirectionDist;
      viewDirection.push_back({meanDirection[0], meanDirection[1], meanDirection[2]});
     }
    }
    
    void updateRepresentativeView(vector<Frame> &frames)
    {
      for (int i = 0; i < pos.size(); i++)
      {
        vector<Mat> desc;
        vector<vector<int>> imgKeypt;
        int rep[2] = {-1,-1};
        
        for (int j = 0; j < frames.size(); j++)
        {
          for (int k = 0; k < frames[j].indexPairs.size(); k++)
          {
           if (int(frames[j].indexPairs[k][1]) == int(i))
           {
            imgKeypt.push_back({j, frames[j].indexPairs[k][0]});
            desc.push_back(frames[j].descriptors.row(frames[j].indexPairs[k][0]));
            break;
           } 
          }
        }
        double mindist = 1000000;
        for (int j = 0; j < desc.size(); j++)
        {
          double currdist = 0;
          for (int k = 0; k < desc.size(); k++)
          {
            currdist = currdist + norm(desc[j], desc[k], NORM_HAMMING);
          }
          if (currdist < mindist)
          {
            mindist = currdist;
            rep[0] = imgKeypt[j][0];
            rep[1] = imgKeypt[j][1];
          }
        }
        if (rep[0] != -1)
        {
          representative.push_back({rep[0], rep[1]});
        }
      }
    }
    
};


