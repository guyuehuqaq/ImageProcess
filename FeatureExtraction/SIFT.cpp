/*
 * SIFT:尺度不变特征转换，特征提取的一种方法
 *   这里我们调用OpenCV实现该算法效果
 */

#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>


int main() {

  // 读取图像
  const std::string file1 = R"(C:\Users\76423\Desktop\test\ImageProcess\FeatureExtraction\image\1.png)";
  const std::string file2 = R"(C:\Users\76423\Desktop\test\ImageProcess\FeatureExtraction\image\2.png)";
  cv::Mat img1 = cv::imread(file1, cv::IMREAD_COLOR);
  cv::Mat img2 = cv::imread(file2, cv::IMREAD_COLOR);

  if (img1.empty() || img2.empty()) {
    std::cout << "Could not open images." << std::endl;
    return -1;
  }

  // 2. 创建 SIFT 检测器
  cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

  // 3. 检测关键点并计算描述子
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;

  sift->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
  sift->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

  std::cout << "Image1 keypoints: " << keypoints1.size() << std::endl;
  std::cout << "Image2 keypoints: " << keypoints2.size() << std::endl;

  // 4. 使用 BFMatcher 进行匹配（L2 距离）
  cv::BFMatcher matcher(cv::NORM_L2);
  std::vector<cv::DMatch> matches;
  matcher.match(descriptors1, descriptors2, matches);

  // 5. 按距离排序，筛选前 N 个匹配
  std::sort(matches.begin(), matches.end(),
            [](const cv::DMatch &a, const cv::DMatch &b) {
              return a.distance < b.distance;
            });

  const int numGoodMatches = 50; // 可以根据需要调整
  std::vector<cv::DMatch> goodMatches(matches.begin(),
                                      matches.begin() + std::min(numGoodMatches, (int)matches.size()));

  // 6. 绘制匹配结果
  cv::Mat imgMatches;
  cv::drawMatches(img1, keypoints1, img2, keypoints2,
                  goodMatches, imgMatches, cv::Scalar::all(-1),
                  cv::Scalar::all(-1), std::vector<char>(),
                  cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  cv::imshow("SIFT Matches", imgMatches);
  cv::waitKey(0);

  return 0;
}
