#include <iostream>
#include <ml/core/Types.hpp>
#include <ml/core/Model.hpp>

int main() {
    std::cout << "✅ QuarkML headers included successfully.\n";
    ml::Matrix X{{1.0, 2.0}, {3.0, 4.0}};
    ml::Vector y{5.0, 6.0};
    std::cout << "Rows=" << X.size() << ", Cols=" << X[0].size()
              << ", y.size=" << y.size() << "\n";
    return 0;
}
