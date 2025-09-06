#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

int main()
{

  const std::string file1 = R"(C:\Users\76423\Desktop\test\ImageProcess\FeatureExtraction\image\1.png)";

  // 读取图像
  cv::Mat img = cv::imread(file1, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "Cannot read image!" << std::endl;
    return -1;
  }

  // 1. 创建 FAST 检测器
  int threshold = 20;      // 阈值
  bool nonmaxSuppression = true; // 是否非极大值抑制
  cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(
      threshold, nonmaxSuppression, cv::FastFeatureDetector::TYPE_9_16
  );

  // 2. 检测关键点
  std::vector<cv::KeyPoint> keypoints;
  fast->detect(img, keypoints);

  // 3. 绘制关键点
  cv::Mat imgKeypoints;
  cv::drawKeypoints(img, keypoints, imgKeypoints, cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  // 4. 显示结果
  std::cout << "Detected FAST keypoints: " << keypoints.size() << std::endl;
  cv::imshow("FAST Keypoints", imgKeypoints);
  cv::waitKey(0);

  return 0;
}
