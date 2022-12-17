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
#include "utils.cpp"
#include "structure.cpp"
#include "Initialization.cpp"
#include "tracking.cpp"
#include "mapping.cpp"


using namespace cv;
using std::vector;
using std::array;
using std::string;
using std::fstream;
using std::stringstream;
using Eigen::MatrixXd;



void read_data(vector<array<string,12>> &data)
{
  fstream file_1("/home/abizer/RBE_COURSES/RBE595_PFAV/course project/dataset/rgbd_dataset_freiburg1_xyz/rgb_depth.txt", fstream::in );
  fstream file_2("/home/abizer/RBE_COURSES/RBE595_PFAV/course project/dataset/rgbd_dataset_freiburg1_xyz/depth_groundtruth.txt", fstream::in);
  string myText;
  string myText1;
  while (getline (file_1, myText) and getline(file_2, myText1)) 
  {
    stringstream ss(myText);
    stringstream ss1(myText1);
    string temp;
    array<string,12> temp_arr = {"0","0","0","0","0","0","0","0","0","0","0","0"}; 
    int i =0;
    while(getline(ss, temp, ' '))
    {
     if (i ==0 or i==1)
     {
      temp_arr[i] = temp; 
     } 
     else
      break;
     i = i + 1;  
    }
    while(getline(ss1, temp, ' '))
    { 
      temp_arr[i] = temp;
      i = i + 1;
    }
    data.push_back(temp_arr);
  }
  file_1.close();  
  file_2.close();
}
int main()
{
  //Initialize pose, intrinsics and class objects
  Mat K = (Mat_<double>(3,3) << 517.3, 0.000000, 318.6, 0.000000, 516.5, 255.3, 0.000000, 0.000000, 1.000000);
  Mat dist_coeff = (Mat_<double>(5,1) << 0.2624, -0.9531, -0.0054, 0.0026, 1.1613);
  int width = 640;
  int height = 480;
  int framesPassed = 20;
  int minPtsCurrFrame = 80;
  
 // Mat initPose = (Mat_<double>(1,7) << 1.3452, 0.6273, 1.6627, 0.6582, 0.6109, -0.2950, -0.3265);  
  Eigen::Quaternion quat(1.0, 0.0, 0.0, 0.0);
  Eigen::Matrix <double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> temp = quat.normalized().toRotationMatrix();
  
  Mat initR(temp.rows(), temp.cols(), CV_64F, temp.data());
  //Mat initT = (Mat_<double>(3,1) << 1.3452, 0.6273, 1.6627);
  Mat initT = (Mat_<double>(3,1) << 0, 0, 0);
 
  WorldPt worldPt;  
  KeyFrame keyFrame;
  
  // read data file and store rgb image names, and respective depth image names and  groundtruth.
  vector<array<string,12>> data;
  read_data(data);
  
  //read first frame
  string folder_path =  string("/home/abizer/RBE_COURSES/RBE595_PFAV/course project/dataset/rgbd_dataset_freiburg1_xyz/");
  string img_name = data[0][1];
  string img_path = folder_path + img_name;
  Mat img1;
  Mat img1t = imread(img_path, IMREAD_UNCHANGED);
  cvtColor(img1t,img1t,COLOR_BGR2GRAY);
  undistort(img1t,img1, K , dist_coeff);
  
  //imshow("image", img );
  //waitKey(0);

  //extract features and descriptors of the first frame
  Ptr <ORB> detector = ORB::create(1000, 1.2, 8);
  vector<KeyPoint> kp1;
  Mat desc1;
  detector -> detectAndCompute(img1,noArray(),kp1,desc1); 
  
  
  /* read subsequent frames, extract features and calculate the pose. If the pose
calculation is not successful, keep on reading frames until we get a successful pose.
 */
  
  bool status = false;
  int index = 1;
  Mat Rc(3, 3, CV_64F);
  Mat Tc(3, 1, CV_64F);
  vector<KeyPoint> kp2;
  Mat desc2;
  vector<vector<int>> indexPairsComb;
  
  while(!status)
  {
    img_name = data[index][1];
    std::cout << "image name is: " << img_name <<std::endl;
    std::cout << "index is : " << index << std::endl;
    img_path = folder_path + img_name;
    Mat img2;
    Mat img2t = imread(img_path, IMREAD_UNCHANGED);
    cvtColor(img2t,img2t,COLOR_BGR2GRAY);
    undistort(img2t,img2, K , dist_coeff);

    detector -> detectAndCompute(img2,noArray(),kp2,desc2); 
   
    Mat RcRel(3,3,CV_64F);
    Mat TcRel(3,1,CV_64F);
    vector<int> ind1;
    vector<int> ind2;
    status = initialize(kp1, kp2, desc1, desc2, img1, img2, K, RcRel, TcRel, ind1, ind2);
    
    if (status == true)
    {  
      Rc = initR * RcRel;
      Tc = initT + initR * TcRel; 
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> temp(Rc.ptr<double>(), Rc.rows, Rc.cols);
      
      Eigen::Matrix3d Rn_abs_eig = temp; 
      Eigen::Quaterniond quatn_abs (Rn_abs_eig);
      std::cout << "The absolute pose is (quaternion only): " << quatn_abs << std::endl;
      status = worldPt.addWorldPts(initR, Rc, initT, Tc, ind1, ind2, K, kp1, kp2, indexPairsComb, 1.0, 1.0);
    }
    index = index + 1;
  }
  
  vector<vector<int>> indexPairs1;
  vector<vector<int>> indexPairs2;
  for (int i = 0; i < indexPairsComb.size(); i++)
  {
    indexPairs1.push_back({indexPairsComb[i][0], indexPairsComb[i][2]});    
    indexPairs2.push_back({indexPairsComb[i][1], indexPairsComb[i][2]});  
  }

  keyFrame.addKeyFrames(initR, initT, indexPairs1, kp1, desc1);
  keyFrame.addKeyFrames(Rc, Tc, indexPairs2, kp2, desc2);
  keyFrame.addCovisibility(0, 1, worldPt.pos.size());
  double medianDepth = worldPt.scaleMap();
  keyFrame.scalePoses(medianDepth);
  worldPt.updateDirectionDistance(keyFrame.frames);
  worldPt.updateRepresentativeView(keyFrame.frames);
  
  
  // Main loop
  vector<int> localMapPtInd;
  vector<int> localMapKeyIds;
  int RefKeyFrame;
  bool isLastKeyFrame = true;
   
  while(index <= data.size() -1)
  {
    vector<KeyPoint> kpCurr;
    vector<int> indexKeyPts;
    vector<int> indexCorres3dPts;
    Mat descCurr;
    Mat R;
    Mat T;
    img_name = data[index][1];
    std::cout << "image name is: " << img_name <<std::endl;
    std::cout << "index is : " << index << std::endl;
    img_path = folder_path + img_name;
    Mat imgCurr;
    Mat imgCurrt = imread(img_path, IMREAD_UNCHANGED);
    cvtColor(imgCurrt,imgCurrt,COLOR_BGR2GRAY);
    undistort(imgCurrt, imgCurr, K , dist_coeff);
    detector -> detectAndCompute(imgCurr,noArray(),kpCurr,descCurr); 
    trackLastKeyFrame(kpCurr, descCurr, keyFrame, worldPt, K, indexKeyPts, indexCorres3dPts, R, T, width, height);
    trackLocalMap(worldPt, keyFrame, kpCurr, descCurr, R, T, K, localMapPtInd, localMapKeyIds, RefKeyFrame, indexKeyPts, indexCorres3dPts, isLastKeyFrame, framesPassed, minPtsCurrFrame, width, height, index);
    if (isLastKeyFrame == false)
    {
      index = index + 1;  
      continue;
    }
    addKeyFrame(R, T, indexKeyPts, indexCorres3dPts, kpCurr, descCurr, keyFrame);
    addNewPoints(worldPt, keyFrame, indexKeyPts, K);
    
  }
  
  
  
  
  
  
  return 0;
}
