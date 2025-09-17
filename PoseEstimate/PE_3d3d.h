#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <utility>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <chrono>
#include <sophus/se3.hpp>

struct BundleAdjustmentData {
  std::vector<Eigen::Vector3d> pts1;
  std::vector<Eigen::Vector3d> pts2;
  double rt[6];  // 参数: [rx, ry, rz, tx, ty, tz]
};

class PosCostFunction{
public:
  PosCostFunction(Eigen::Vector3d pts1,
                  Eigen::Vector3d pts2)
                  : pts1_(std::move(pts1)), pts2_(std::move(pts2)){}

  template<typename T>
  bool operator()(const T* rt, T* residuals) const{
    Eigen::Matrix<T,3,1> t(rt[3], rt[4], rt[5]);

    Eigen::Matrix<T,3,1> p1 = pts1_.cast<T>();
    Eigen::Matrix<T,3,1> p2 = pts2_.cast<T>();
    Eigen::Matrix<T,3,1> p1_rot;
    ceres::AngleAxisRotatePoint(rt, p1.data(), p1_rot.data());
    Eigen::Matrix<T,3,1> p1_trans = p1_rot + t;

    residuals[0] = p1_trans[0] - p2[0];
    residuals[1] = p1_trans[1] - p2[1];
    residuals[2] = p1_trans[2] - p2[2];

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& pts1,
                                     const Eigen::Vector3d& pts2) {
    return new ceres::AutoDiffCostFunction<
        PosCostFunction,
        3,  // residual size = 每个点 3D误差
        6   // 参数块 = 旋转(3) + 平移(3)
    >(new PosCostFunction(pts1, pts2));
  }

private:
  const Eigen::Vector3d pts1_;
  const Eigen::Vector3d pts2_;

};


// 定义优化参数顶点的类型, 6维，李代数形式
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  void setToOriginImpl() override{
    _estimate = Sophus::SE3d(Eigen::Matrix3d::Identity(),
                             Eigen::Vector3d::Zero());
  }

  void oplusImpl(const double* update) override {
    Eigen::Matrix<double, 6, 1> u;
    for (int i = 0; i < 6; ++i) u[i] = update[i];
    _estimate = Sophus::SE3d::exp(u) * _estimate;
  }

  bool read(std::istream& /*in*/) override { return true; }
  bool write(std::ostream& /*out*/) const override { return true; }
};

class Edge3DPoint : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Edge3DPoint(Eigen::Vector3d point) : _point(std::move(point)){}

  void computeError() override{
    const auto *pose = dynamic_cast<const VertexPose *>(_vertices[0]);
    _error = _measurement - pose->estimate() * _point;
  }

  void linearizeOplus() override {
    auto *pose = dynamic_cast<VertexPose *>(_vertices[0]);
    Sophus::SE3d T = pose->estimate();
    Eigen::Vector3d xyz_trans = T.so3() * _point;
    _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    _jacobianOplusXi.block<3, 3>(0, 3) = -Sophus::SO3d::hat(xyz_trans);
  }

  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }

private:
  const Eigen::Vector3d _point;
};