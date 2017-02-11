#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>
//#include "eigenmvn-master/eigenmvn.h"

using namespace std;
using namespace Eigen;

static float h_(Vector2f &x, Vector2f &p)
{
  return (x - p).norm();
}

static Vector3f h(Vector2f &x, Vector2f &p0, Vector2f &p1, Vector2f &p2)
{
  Vector3f y(h_(x, p0), h_(x, p1), h_(x, p2));
  return y;
}

static void Jh0(Vector2f &x, Vector2f &p0, Vector2f &p1, Vector2f &p2,
               MatrixXf &j)
{
  j.transpose() << (x - p0) / h_(x, p0),
                   (x - p1) / h_(x, p2),
                   (x - p2) / h_(x, p2);
//cout << "j:\n" << j << endl;
}

int main(void)
{
  
  /* 初期化 */
  // 観測数
  const int T = 1000;
  // 観測値の座標
  Vector2f p0(0.0f, 0.0f);
  Vector2f p1(10.0f, 0.0f);
  Vector2f p2(0.0f ,10.0f);
  // 初期位置
  Vector2f x(0.0f, 0.0f);
  // 状態
  Vector2f X[T];
  // 観測  
  Vector3f Y[T];

  // state x = A * x_ + B * u + w, w~N(0,Q)
  Matrix2f A = MatrixXf::Identity(2,2);
  Matrix2f B = MatrixXf::Identity(2,2);
  Matrix2f Q = MatrixXf::Identity(2,2);
  Vector2f u(2.0f, 2.0f);

//cout << "A:\n" << A << endl;
//cout << "B:\n" << B << endl;
//cout << "Q:\n" << Q << endl;

  // observation Y = h(x) + v, v~N(0,R)
  Matrix3f R;
  R << 2.0f, 0.0f, 0.0f,
       0.0f, 2.0f, 0.0f,
       0.0f, 0.0f, 0.2f;

  // EKF
  Vector2f mu = VectorXf::Zero(2);
  Vector2f mu_;
  Matrix2f Sigma = MatrixXf::Zero(2,2);
  Matrix2f Sigma_;
  Vector2f M[T];
  Vector3f yi;
  MatrixXf C(3,2);
  MatrixXf K(2,3);
  Matrix3f S;
  MatrixXf xxx;

  Vector2f av2(-0.05f, -0.05f);
  Vector3f av3(-0.05f, -0.05f, -0.05f);

  for (int i; i < T; i++) {
    // 推定
    M[i] = mu;
//cout << "mu:\n" << mu << endl;
    // 観測データの生成
    x = A * x + B * u; // + np.random.multivariate_normal([0,0],Q,1).T
    x += VectorXf::Random(2) / 10.0f - av2; // noise
    X[i] = x;
    // p0, p1, p2 = np.random.multivariate_normal([0,0,0],R,1).T
    Y[i] = h(x, p0, p1, p2);
    Y[i] += VectorXf::Random(3) / 10.0f - av3; // noise
    // prediction
    mu_ = A * mu + B * u;
//cout << "mu_:\n" << mu_ << endl;
    Sigma_ = Q + A * Sigma * A.transpose();
//cout << "Sigma_:\n" << Sigma_ << endl;
    // update
    Jh0(mu_, p0, p1, p2, C); /* 解析的 */
//cout << "C:\n" << C << endl;
    //Jh1(mu_, p0, p1, p2, C); /* 数値的 */
    yi = Y[i] - C * mu_;
    S = C * Sigma_ * C.transpose() + R;
    K = Sigma_ * C.transpose() * S.inverse();
//cout << "K:\n" << K << endl;
    mu = mu_ + K * yi;
    Sigma = Sigma_ - K * C * Sigma_;
//cout << "Sigma:\n" << Sigma << endl;
    
    cout << "\tM:" << M[i](0) << ',' << M[i](1);
    cout << "\tX:" << X[i](0) << ',' << X[i](1);
    cout << "\tY:" << Y[i](0) << ',' << Y[i](1) << endl;
  }

  return 0;
}
