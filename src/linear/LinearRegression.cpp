#include <ml/linear/LinearRegression.hpp>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace ml {


static std::vector<double> col_mean(const Matrix& X) {
  const size_t n = X.size(), d = X[0].size();
  std::vector<double> m(d, 0.0);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < d; ++j) m[j] += X[i][j];
  for (size_t j = 0; j < d; ++j) m[j] /= (n ? double(n) : 1.0);
  return m;
}

static double mean(const Vector& v) {
  double s = 0.0;
  for (double x : v) s += x;
  return v.empty() ? 0.0 : s / double(v.size());
}

static Matrix xtx_centered(const Matrix& Xc, double lambda_l2) {
  const size_t n = Xc.size(), d = Xc[0].size();
  Matrix A(d, Vector(d, 0.0));
  for (size_t i = 0; i < n; ++i) {
    for (size_t a = 0; a < d; ++a) {
      const double xa = Xc[i][a];
      for (size_t b = 0; b < d; ++b) {
        A[a][b] += xa * Xc[i][b];
      }
    }
  }
  // Ridge: + lambda * I
  for (size_t j = 0; j < d; ++j) A[j][j] += lambda_l2;
  return A;
}

static Vector xty_centered(const Matrix& Xc, const Vector& yc) {
  const size_t n = Xc.size(), d = Xc[0].size();
  Vector b(d, 0.0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < d; ++j) b[j] += Xc[i][j] * yc[i];
  }
  return b;
}


static Vector solve_linear_system(Matrix A, Vector b) {
  const size_t d = A.size();
  if (d == 0 || A[0].size() != d || b.size() != d)
    throw std::invalid_argument("solve_linear_system: bad dimensions");

  
  for (size_t col = 0; col < d; ++col) {
    
    size_t piv = col;
    double best = std::fabs(A[col][col]);
    for (size_t r = col + 1; r < d; ++r) {
      if (std::fabs(A[r][col]) > best) {
        best = std::fabs(A[r][col]);
        piv = r;
      }
    }
    if (best == 0.0) throw std::runtime_error("Singular matrix in normal equations");
    if (piv != col) { std::swap(A[piv], A[col]); std::swap(b[piv], b[col]); }

    
    const double diag = A[col][col];
    for (size_t c = col; c < d; ++c) A[col][c] /= diag;
    b[col] /= diag;

    for (size_t r = 0; r < d; ++r) {
      if (r == col) continue;
      const double factor = A[r][col];
      if (factor == 0.0) continue;
      for (size_t c = col; c < d; ++c) A[r][c] -= factor * A[col][c];
      b[r] -= factor * b[col];
    }
  }

  return b;
}

// --------- LinearRegression ---------

void LinearRegression::fit(const Matrix& X, const Vector& y, const FitParams&) {
  if (X.empty() || X[0].empty()) throw std::invalid_argument("X is empty");
  if (X.size() != y.size()) throw std::invalid_argument("X and y size mismatch");

  const size_t n = X.size(), d = X[0].size();


  Vector x_mean(d, 0.0);
  double y_mean = 0.0;

  Matrix Xc = X;
  Vector yc = y;

  if (params_.fit_intercept) {
    x_mean = col_mean(X);
    y_mean = mean(y);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < d; ++j) Xc[i][j] -= x_mean[j];
      yc[i] -= y_mean;
    }
  }

  
  Matrix A = xtx_centered(Xc, params_.lambda_l2);
  Vector bb = xty_centered(Xc, yc);

  
  w_ = solve_linear_system(A, bb);

  
  if (params_.fit_intercept) {
    double wb = 0.0;
    for (size_t j = 0; j < d; ++j) wb += w_[j] * x_mean[j];
    b_ = y_mean - wb;
  } else {
    b_ = 0.0;
  }
}

Vector LinearRegression::predict(const Matrix& X) const {
  if (X.empty()) return {};
  const size_t n = X.size(), d = X[0].size();
  if (!w_.empty() && w_.size() != d)
    throw std::invalid_argument("predict: feature dimension mismatch");

  Vector yhat(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    double s = b_;
    for (size_t j = 0; j < w_.size(); ++j) s += w_[j] * X[i][j];
    yhat[i] = s;
  }
  return yhat;
}

double LinearRegression::score(const Matrix& X, const Vector& y) const {
  if (X.size() != y.size()) throw std::invalid_argument("score: X and y size mismatch");
  Vector yhat = predict(X);
  double mu = mean(y);
  double ss_res = 0.0, ss_tot = 0.0;
  for (size_t i = 0; i < y.size(); ++i) {
    const double e = y[i] - yhat[i];
    ss_res += e * e;
    const double t = y[i] - mu;
    ss_tot += t * t;
  }
  return (ss_tot == 0.0) ? 0.0 : 1.0 - ss_res / ss_tot;
}

} // namespace ml
