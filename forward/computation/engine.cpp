#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>  // 必须包含此头文件
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>
#include <algorithm>

namespace py = pybind11;

// 1. 严格实现 GaussianW2
double GaussianW2(const Eigen::VectorXd& m0, const Eigen::VectorXd& m1, 
                  const Eigen::MatrixXd& Sigma0, const Eigen::MatrixXd& Sigma1) {
    Eigen::MatrixXd Sigma00 = Sigma0.sqrt();
    Eigen::MatrixXd Sigma010 = (Sigma00 * Sigma1 * Sigma00).sqrt();
    double d = (m0 - m1).squaredNorm() + Sigma0.trace() + Sigma1.trace() - 2.0 * Sigma010.trace();
    return std::max(0.0, d);
}

// 2. 严格实现 GaussianBarycenterW2 (Banzato 迭代)
std::pair<Eigen::VectorXd, Eigen::MatrixXd> GaussianBarycenterW2(
    const std::vector<Eigen::VectorXd>& mu, 
    const std::vector<Eigen::MatrixXd>& Sigma, 
    const Eigen::VectorXd& alpha, 
    int N) {
    
    int d = mu[0].size();
    int K = mu.size();
    
    Eigen::VectorXd mun = Eigen::VectorXd::Zero(d);
    for (int j = 0; j < K; ++j) mun += alpha(j) * mu[j];
    
    Eigen::MatrixXd Sigman = Eigen::MatrixXd::Identity(d, d);
    for (int n = 0; n < N; ++n) {
        Eigen::MatrixXd Sigmandemi = Sigman.sqrt();
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(d, d);
        for (int j = 0; j < K; ++j) {
            T += alpha(j) * (Sigmandemi * Sigma[j] * Sigmandemi).sqrt();
        }
        Sigman = T;
    }
    return {mun, Sigman};
}

// 3. 严格实现 Entropic Partial OT
Eigen::MatrixXd entropic_partial_ot(
    const Eigen::VectorXd& a, 
    const Eigen::VectorXd& b, 
    Eigen::MatrixXd M, 
    double Lambda, double reg, int iter_max, double stop_thr) {
    
    M.array() -= 2.0 * Lambda;
    int n = a.size();
    int m = b.size();

    Eigen::VectorXd a_ext(n + 1); a_ext << a, 1.0;
    Eigen::VectorXd b_ext(m + 1); b_ext << b, 1.0;
    
    Eigen::MatrixXd M_ext = Eigen::MatrixXd::Zero(n + 1, m + 1);
    M_ext.block(0, 0, n, m) = M;
    Eigen::MatrixXd K = (-M_ext / reg).array().exp();

    Eigen::VectorXd u = Eigen::VectorXd::Ones(n + 1);
    Eigen::VectorXd v = Eigen::VectorXd::Ones(m + 1);
    
    for (int i = 0; i < iter_max; ++i) {
        Eigen::VectorXd u_old = u;
        u = a_ext.cwiseQuotient(K * v);
        v = b_ext.cwiseQuotient(K.transpose() * u);
        if ((u - u_old).template lpNorm<Eigen::Infinity>() < stop_thr) break;
    }
    return (u.asDiagonal() * K * v.asDiagonal()).block(0, 0, n, m);
}

PYBIND11_MODULE(cpp_engine, m) {
    m.def("GaussianW2", &GaussianW2);
    m.def("GaussianBarycenterW2", &GaussianBarycenterW2);
    m.def("entropic_partial_ot", &entropic_partial_ot);
}
