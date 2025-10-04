#pragma once
#include <iosfwd>
#include <optional>
#include "Types.hpp"

namespace ml {

// Optional training knobs
struct FitParams {
    int    max_iter   = 1000;
    double tol        = 1e-6;
    double alpha      = 0.01;   // e.g., GD learning rate
    double lambda_l2  = 0.0;    // e.g., Ridge
};

// Base model interface
class Model {
public:
    virtual ~Model() = default;

    // Supervised default: X, y. Unsupervised models may ignore y.
    virtual void   fit(const Matrix& X, const Vector& y, const FitParams& params = {}) = 0;
    virtual Vector predict(const Matrix& X) const = 0;

    // Default scoring: regression uses R^2; classification uses accuracy; unsupervised may return 0 or a relevant metric.
    virtual double score(const Matrix& X, const Vector& y) const = 0;

    // (Optional) persistence
    virtual void save(std::ostream& os) const {}
    virtual void load(std::istream& is) {}
};

} // namespace ml
