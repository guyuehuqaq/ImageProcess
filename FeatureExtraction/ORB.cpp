/*
 * Description: ORB（Oriented FAST and Rotated BRIEF）
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

int main()
{

  const std::string file1 = R"(C:\Users\76423\Desktop\test\ImageProcess\FeatureExtraction\image\1.png)";
  const std::string file2 = R"(C:\Users\76423\Desktop\test\ImageProcess\FeatureExtraction\image\2.png)";

  // 读取两张图像
  cv::Mat img1 = cv::imread(file1, cv::IMREAD_COLOR);
  cv::Mat img2 = cv::imread(file2, cv::IMREAD_COLOR);

  if (img1.empty() || img2.empty()) {
    std::cerr << "Cannot read images!" << std::endl;
    return -1;
  }

  // 1. 创建 ORB 检测器
  int nFeatures = 500; // 最大关键点数量
  cv::Ptr<cv::ORB> orb = cv::ORB::create(nFeatures);

  // 2. 检测关键点
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  orb->detect(img1, keypoints1);
  orb->detect(img2, keypoints2);

  // 3. 计算描述子
  cv::Mat descriptors1, descriptors2;
  orb->compute(img1, keypoints1, descriptors1);
  orb->compute(img2, keypoints2, descriptors2);

  // 4. 特征匹配（汉明距离）
  cv::BFMatcher matcher(cv::NORM_HAMMING, /*crossCheck=*/true);
  std::vector<cv::DMatch> matches;
  matcher.match(descriptors1, descriptors2, matches);

  // 5. 按距离排序
  std::sort(matches.begin(), matches.end(),
            [](const cv::DMatch &a, const cv::DMatch &b) {
              return a.distance < b.distance;
            });

  // 6. 绘制前 50 个匹配
  cv::Mat imgMatches;
  cv::drawMatches(img1, keypoints1,
                  img2, keypoints2,
                  std::vector<cv::DMatch>(matches.begin(), matches.begin() + std::min(50, (int)matches.size())),
                  imgMatches,
                  cv::Scalar::all(-1), cv::Scalar::all(-1),
                  std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  cv::imshow("ORB Matches", imgMatches);
  cv::waitKey(0);

  return 0;
}
