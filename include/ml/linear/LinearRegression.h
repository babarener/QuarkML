#pragma once
#include <string>
#include <vector>

#include "ml/core/SerializableModel.h"

namespace ml::linear {

/// A simple dense Linear Regression model (y = X w + b),
/// trained by batch Gradient Descent. Provides save/load for reuse.
class LinearRegression : public ml::core::SerializableModel {
public:
    LinearRegression(double lr = 0.01, int epochs = 1000, bool fit_intercept = true);

    /// Train on features X (N x D) and targets y (N).
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y);

    /// Predict for features X (N x D) -> y_hat (N).
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;

    /// R^2 score on a dataset. Returns 1.0 if y variance is zero.
    double score(const std::vector<std::vector<double>>& X,
                 const std::vector<double>& y) const;

    /// Serialize parameters/metadata to a human-readable text file:
    /// Example:
    ///   # QuarkML LinearRegression v1
    ///   n_features=3
    ///   fit_intercept=true
    ///   bias=0.12345
    ///   weights=0.0100000000,-0.2200000000,1.3070000000
    void save(const std::string& path) const override;

    /// Load a model from the file saved by `save(path)`.
    static LinearRegression load(const std::string& path);

    // Accessors
    const std::vector<double>& weights() const { return weights_; }
    double bias() const { return bias_; }
    int n_features() const { return n_features_; }
    double learning_rate() const { return lr_; }
    int epochs() const { return epochs_; }
    bool fit_intercept() const { return fit_intercept_; }

private:
    // parameters
    std::vector<double> weights_; // length D
    double bias_{0.0};

    // training hyperparameters (not required for inference, but kept for reference)
    double lr_{0.01};
    int epochs_{1000};
    bool fit_intercept_{true};

    // cached metadata
    int n_features_{0};

    // helpers
    static void check_dimensions(const std::vector<std::vector<double>>& X,
                                 const std::vector<double>& y);
    static double dot(const std::vector<double>& a, const std::vector<double>& b);
};

} // namespace ml::linear
