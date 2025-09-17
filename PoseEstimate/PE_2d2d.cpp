/*
 * 应用ORB提取的特征点，进行2d2d姿态估计
 * 流程: 通过2d2d点对关系，计算出基本矩阵和本质矩阵，再SVD分解成RT
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void compute_feature_match(
    const cv::Mat& img1,
    const cv::Mat& img2,
    std::vector<cv::KeyPoint> &keypoints1,
    std::vector<cv::KeyPoint> &keypoints2,
    std::vector<cv::DMatch> &matches){

  int nFeatures = 50; // 最大关键点数量
  cv::Ptr<cv::ORB> orb = cv::ORB::create(nFeatures);
  orb->detect(img1, keypoints1);
  orb->detect(img2, keypoints2);

  cv::Mat descriptors1, descriptors2;
  orb->compute(img1, keypoints1, descriptors1);
  orb->compute(img2, keypoints2, descriptors2);

  std::vector<cv::DMatch> match;
  cv::BFMatcher matcher(cv::NORM_HAMMING, /*crossCheck=*/true);
  matcher.match(descriptors1, descriptors2, match);
  std::cout << "match size: " << match.size() << std::endl;

  // 匹配点对筛选
  double min_dist = 10000, max_dist = 0;

  // 找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
  for (int i = 0; i < match.size(); i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  // 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  for (int i = 0; i < match.size(); i++) {
    if (match[i].distance <= std::max(2 * min_dist, 30.0)) {
      matches.push_back(match[i]);
    }
  }

}

void pose_estimate_2d2d(
    std::vector<cv::KeyPoint> &keypoints1,
    std::vector<cv::KeyPoint> &keypoints2,
    std::vector<cv::DMatch> &matches,
    cv::Mat& R, cv::Mat& t){

  cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  std::vector<cv::Point2f> points1, points2;
  for (int i = 0; i < matches.size(); i++){
    points1.emplace_back(keypoints1[matches[i].queryIdx].pt);
    points2.emplace_back(keypoints2[matches[i].trainIdx].pt);
  }

  // 计算基本矩阵F
  cv::Mat fundamental_matrix;
  fundamental_matrix = findFundamentalMat(points1, points2, cv::FM_8POINT);
  std::cout << "fundamental_matrix is " << std::endl << fundamental_matrix << std::endl;

  // 计算本质矩阵E
  cv::Mat essential_matrix;
  essential_matrix = findEssentialMat(points1, points2, K);
  std::cout << "essential_matrix is " << std::endl << essential_matrix << std::endl;

//  //-- 计算单应矩阵(在三维空间中，点在同一平面内)
//  //-- 但是本例中场景不是平面，单应矩阵意义不大
//  cv::Mat homography_matrix;
//  homography_matrix = findHomography(points1, points2, RANSAC, 3);
//  std::cout << "homography_matrix is " << std::endl << homography_matrix << std::endl;

  //-- 从本质矩阵中恢复旋转和平移信息.
  // 此函数仅在Opencv3中提供
  cv::Point2d principal_point(325.1, 249.7);  //相机光心, TUM dataset标定值
  double focal_length = 521;      //相机焦距, TUM dataset标定值
  recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
  std::cout << "R is " << std::endl << R << std::endl;
  std::cout << "t is " << std::endl << t << std::endl;
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d
      (
          (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
          (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
      );
}

int main(int argc, char **argv){

  const std::string file1 = R"(C:\Users\76423\Desktop\test\ImageProcess\PoseEstimate\image\1.png)";
  const std::string file2 = R"(C:\Users\76423\Desktop\test\ImageProcess\PoseEstimate\image\2.png)";
  // 读取两张图像
  cv::Mat img1 = cv::imread(file1, cv::IMREAD_COLOR);
  cv::Mat img2 = cv::imread(file2, cv::IMREAD_COLOR);
  if (img1.empty() || img2.empty()) {
    std::cerr << "Cannot read images!" << std::endl;
    return -1;
  }

  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  // 得到筛选后的匹配对
  std::vector<cv::DMatch> matches;
  compute_feature_match(img1, img2,keypoints1,keypoints2,matches);
  std::cout << "一共找到了" << matches.size() << "组匹配点" << std::endl;

  //-- 估计两张图像间运动
  cv::Mat R, t;
  pose_estimate_2d2d(keypoints1, keypoints2, matches, R, t);

  //-- 验证E=t^R*scale
  Mat t_x =
      (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
          t.at<double>(2, 0), 0, -t.at<double>(0, 0),
          -t.at<double>(1, 0), t.at<double>(0, 0), 0);

  cout << "t^R=" << endl << t_x * R << endl;

  //-- 验证对极约束
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  for (DMatch m: matches) {
    Point2d pt1 = pixel2cam(keypoints1[m.queryIdx].pt, K);
    Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
    Point2d pt2 = pixel2cam(keypoints2[m.trainIdx].pt, K);
    Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
    Mat d = y2.t() * t_x * R * y1;
    cout << "epipolar constraint = " << d << endl;
  }

  return 0;
}

