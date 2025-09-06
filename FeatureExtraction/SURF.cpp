///*
// * SURF:加速稳健特征
// *   这里我们调用OpenCV实现该算法效果
// */
//
//#include <opencv2/opencv.hpp>
//#include <opencv2/xfeatures2d.hpp>
//#include <iostream>
//
//int main() {
//
//  const std::string file1 = R"(C:\Users\76423\Desktop\test\ImageProcess\FeatureExtraction\image\1.png)";
//  const std::string file2 = R"(C:\Users\76423\Desktop\test\ImageProcess\FeatureExtraction\image\2.png)";
//
//  // 读取图像（灰度）
//  cv::Mat img1 = cv::imread(file1, cv::IMREAD_COLOR);
//  cv::Mat img2 = cv::imread(file2, cv::IMREAD_COLOR);
//
//  if (img1.empty() || img2.empty()) {
//    std::cerr << "无法读取图像!" << std::endl;
//    return -1;
//  }
//
//  // ===== 1. 创建 SURF 检测器 =====
//  double hessianThreshold = 200; // 阈值越大，检测到的关键点越少
//  cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(hessianThreshold);
//
//  // ===== 2. 检测关键点 =====
//  std::vector<cv::KeyPoint> keypoints1, keypoints2;
//  surf->detect(img1, keypoints1);
//  surf->detect(img2, keypoints2);
//
//  // ===== 3. 计算描述子 =====
//  cv::Mat descriptors1, descriptors2;
//  surf->compute(img1, keypoints1, descriptors1);
//  surf->compute(img2, keypoints2, descriptors2);
//
//  // ===== 4. 匹配特征点 =====
//  cv::BFMatcher matcher(cv::NORM_L2, true); // L2 距离，crossCheck=true
//  std::vector<cv::DMatch> matches;
//  matcher.match(descriptors1, descriptors2, matches);
//
//  // ===== 5. 可视化匹配 =====
//  cv::Mat imgMatches;
//  cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);
//  cv::imshow("SURF Matches", imgMatches);
//  cv::waitKey(0);
//
//  return 0;
//}


#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <iostream>

int main() {
  std::cout << cv::getBuildInformation() << std::endl;
  auto surf = cv::xfeatures2d::SURF::create();
  std::cout << "SURF created successfully!" << std::endl;
}