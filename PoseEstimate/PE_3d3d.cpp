#include <iostream>
#include <Eigen/SVD>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "PE_3d3d.h"

using namespace cv;
using namespace std;

void PrintRT(const cv::Mat& R, const cv::Mat& t) {
  std::cout << "Rotation matrix R:" << std::endl;
  for (int i = 0; i < R.rows; ++i) {
    for (int j = 0; j < R.cols; ++j) {
      std::cout << R.at<double>(i, j) << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Translation vector t:" << std::endl;
  for (int i = 0; i < t.rows; ++i) {
    std::cout << t.at<double>(i, 0) << " ";
  }
  std::cout << std::endl;
}

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

Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d(
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
  );
}

void pose_estimate_3d3d(const std::vector<cv::Point3f>& pts1,
                        const std::vector<cv::Point3f>& pts2,
                        cv::Mat& R, cv::Mat& t) {
  // 1. 计算质心
  cv::Point3f p1_mean, p2_mean;
  int N = pts1.size();
  for (int i = 0; i < N; i++) {
    p1_mean += pts1[i];
    p2_mean += pts2[i];
  }
  p1_mean /= N;
  p2_mean /= N;

  // 2. 去中心化
  std::vector<cv::Point3f> q1(N), q2(N);
  for (int i = 0; i < N; i++) {
    q1[i] = pts1[i] - p1_mean;
    q2[i] = pts2[i] - p2_mean;
  }

  // 3. 计算协方差矩阵
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  for (int i = 0; i < N; i++) {
    Eigen::Vector3d v1(q1[i].x, q1[i].y, q1[i].z);
    Eigen::Vector3d v2(q2[i].x, q2[i].y, q2[i].z);
    W += v2 * v1.transpose();  // 注意这里交换顺序
  }

  // 4. SVD求R
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  Eigen::Matrix3d R_ = U * V.transpose();
  if (R_.determinant() < 0) {
    R_ = -R_;
  }

  // 5. 计算平移
  Eigen::Vector3d t_ = Eigen::Vector3d(p2_mean.x, p2_mean.y, p2_mean.z)
                       - R_ * Eigen::Vector3d(p1_mean.x, p1_mean.y, p1_mean.z);

  // 6. 转成cv::Mat
  R = (cv::Mat_<double>(3, 3) <<
                              R_(0, 0), R_(0, 1), R_(0, 2),
      R_(1, 0), R_(1, 1), R_(1, 2),
      R_(2, 0), R_(2, 1), R_(2, 2));

  t = (cv::Mat_<double>(3, 1) << t_(0), t_(1), t_(2));
}


void BundleAdjustmentUsingCeres(const vector<Point3f>& pts1,
                                const vector<Point3f>& pts2,
                                Mat& R, Mat& t){
  BundleAdjustmentData data;

  // 1. 转换点
  data.pts1.reserve(pts1.size());
  data.pts2.reserve(pts2.size());
  for (size_t i = 0; i < pts1.size(); ++i) {
    data.pts1.emplace_back(pts1[i].x, pts1[i].y, pts1[i].z);
    data.pts2.emplace_back(pts2[i].x, pts2[i].y, pts2[i].z);
  }

  Eigen::Matrix3d rot_vec;
  cv::cv2eigen(R, rot_vec);
  Eigen::AngleAxisd aa(rot_vec);
  Eigen::Vector3d rot_vec1 = aa.axis() * aa.angle();
  Eigen::Vector3d t_eigen;
  cv::cv2eigen(t, t_eigen);
  Eigen::Matrix<double,6,1> rt;
  rt.segment<3>(0) = rot_vec1;
  rt.segment<3>(3) = t_eigen;

  ceres::Solver::Summary summary;
  ceres::Problem problem;
  ceres::LossFunction* loss_function = new ceres::HuberLoss(1);
  problem.AddParameterBlock(reinterpret_cast<double *>(rt.data()), 6);
  for (int i = 0; i < pts1.size(); i++){
    ceres::CostFunction* cost_function = nullptr;
    cost_function = PosCostFunction::Create(data.pts1[i], data.pts2[i]);
    problem.AddResidualBlock(cost_function,
                             loss_function,
                             reinterpret_cast<double *>(rt.data()));
  }

  ceres::Solver::Options solver_options;
  solver_options.minimizer_progress_to_stdout = true;  // 打印优化过程
  solver_options.linear_solver_type = ceres::DENSE_QR; // 小型问题用 DENSE_QR
  solver_options.max_num_iterations = 100;
  ceres::Solve(solver_options, &problem, &summary);

  // --- 优化结果转换回 R, t ---
  Eigen::Vector3d rot_vec_opt = rt.segment<3>(0);
  Eigen::Vector3d t_opt       = rt.segment<3>(3);

  Eigen::AngleAxisd aa_opt(rot_vec_opt.norm(), rot_vec_opt.normalized());
  Eigen::Matrix3d R_opt = aa_opt.toRotationMatrix();

  // --- Eigen -> OpenCV ---
  cv::eigen2cv(R_opt, R);
  cv::eigen2cv(t_opt, t);

  // --- 打印优化结果 ---
  std::cout << "Optimized Rotation R:" << std::endl;
  std::cout << R << std::endl;
  std::cout << "Optimized Translation t:" << std::endl;
  std::cout << t << std::endl;
}

void BundleAdjustmentUsingG2O(const vector<Point3f>& pts1,
                              const vector<Point3f>& pts2,
                              Mat& R, Mat& t){
  // --- Eigen 转换 ---
  Eigen::Matrix3d R_eigen;
  cv::cv2eigen(R, R_eigen);
  Eigen::Vector3d t_eigen;
  cv::cv2eigen(t, t_eigen);
  Sophus::SE3d pose(R_eigen, t_eigen);

  // 构建图优化，先设定g2o
  typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;
  auto linearSolver = std::make_unique< g2o::LinearSolverDense<Block::PoseMatrixType> >();
  auto solver_ptr = std::make_unique<Block>(std::move(linearSolver));
  auto* solver =
      new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
  g2o::SparseOptimizer optimizer;     // 图模型
  optimizer.setAlgorithm(solver);   // 设置求解器
  optimizer.setVerbose(true);       // 打开调试输出

  // Vertex
  auto *v = new VertexPose();
  v->setId(0);
  v->setEstimate(pose);
  optimizer.addVertex(v);

  // Edges
  for(int i = 0; i < pts1.size(); i++){
    Eigen::Vector3d p1(pts1[i].x, pts1[i].y, pts1[i].z);
    Eigen::Vector3d p2(pts2[i].x, pts2[i].y, pts2[i].z);
    auto* edge = new Edge3DPoint(p1);
    edge->setVertex(0, v);
    edge->setMeasurement(p2);
    edge->setInformation(Eigen::Matrix3d::Identity());
    optimizer.addEdge(edge);
  }

  // --- 执行优化 ---
  optimizer.initializeOptimization();
  optimizer.optimize(50);

  // --- 打印结果 ---
  Sophus::SE3d pose_opt = v->estimate();
  Eigen::Matrix3d R_opt = pose_opt.rotationMatrix();
  Eigen::Vector3d t_opt = pose_opt.translation();
  cv::eigen2cv(R_opt, R);
  cv::eigen2cv(t_opt, t);
  std::cout << "Optimized R:" << std::endl << R << std::endl;
  std::cout << "Optimized t:" << std::endl << t << std::endl;

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
  std::cout << "find" << matches.size() << "matches" << std::endl;

  // 根据深度图信息得到3d点
  // 读取深度图
  const std::string file_depth1 = R"(C:\Users\76423\Desktop\test\ImageProcess\PoseEstimate\image\1_depth.png)";
  const std::string file_depth2 = R"(C:\Users\76423\Desktop\test\ImageProcess\PoseEstimate\image\2_depth.png)";
  Mat depth1 = imread(file_depth1, IMREAD_UNCHANGED);       // 深度图为16位无符号数，单通道图像
  Mat depth2 = imread(file_depth2, IMREAD_UNCHANGED);       // 深度图为16位无符号数，单通道图像
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<Point3f> pts1, pts2;
  for (DMatch m:matches) {
    ushort d1 = depth1.ptr<unsigned short>(int(keypoints1[m.queryIdx].pt.y))[int(keypoints1[m.queryIdx].pt.x)];
    ushort d2 = depth2.ptr<unsigned short>(int(keypoints2[m.trainIdx].pt.y))[int(keypoints2[m.trainIdx].pt.x)];
    if (d1 == 0 || d2 == 0)   // bad depth
      continue;
    Point2d p1 = pixel2cam(keypoints1[m.queryIdx].pt, K);
    Point2d p2 = pixel2cam(keypoints2[m.trainIdx].pt, K);
    /*
     TUM RGB-D 数据集：深度图存的数值单位是 1/5000 米
       也就是说，depth = 5000 ⇒ 实际深度 = 1 m
       所以要除以 5000.0 才能得到米
     Kinect v1：深度存储是毫米（1mm = 0.001m）
       这里就应该除以 1000.0
     RealSense：有的模式下深度单位是 1/1000 米，也可能配置成别的
     */
    float dd1 = float(d1) / 5000.0;
    float dd2 = float(d2) / 5000.0;
    pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
    pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
  }

  Mat R,t;
  pose_estimate_3d3d(pts1, pts2, R, t);
  PrintRT(R, t);

//  BundleAdjustmentUsingCeres(pts1, pts2, R, t);

  BundleAdjustmentUsingG2O(pts1, pts2, R, t);

  return 0;
}


