#include <iostream>
#include <ml/linear/LinearRegression.hpp>

int main() {
  ml::Matrix X{{1.0, 2.0}, {2.0, 0.0}, {3.0, 1.0}};
  ml::Vector y{5.0, 3.0, 6.0};

  ml::LinearRegression::Params p;
  p.fit_intercept = true;
  ml::LinearRegression lr(p);

  lr.fit(X, y);                          
  auto preds = lr.predict(X);            
  auto r2    = lr.score(X, y);         

  std::cout << "preds: ";
  for (double v : preds) std::cout << v << " ";
  std::cout << "\nR2: " << r2 << "\n";

  std::cout << "weights: ";
  for (double w : lr.coefficients()) std::cout << w << " ";
  std::cout << "\nbias: " << lr.intercept() << "\n";
  return 0;
}



