#include "ml/linear/LinearRegression.h"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <iomanip>
#include "ml/io/ModelIO.h"

namespace ml::linear {

LinearRegression::LinearRegression(double lr, int epochs, bool fit_intercept)
    : lr_(lr), epochs_(epochs), fit_intercept_(fit_intercept) {}

void LinearRegression::check_dimensions(const std::vector<std::vector<double>>& X,
                                        const std::vector<double>& y) {
    if (X.empty()) throw std::invalid_argument("fit: X is empty");
    const size_t n = X.size();
    const size_t d = X[0].size();
    if (d == 0) throw std::invalid_argument("fit: X has zero features");
    if (y.size() != n) throw std::invalid_argument("fit: y length != X rows");
    for (const auto& row : X) {
        if (row.size() != d) throw std::invalid_argument("fit: inconsistent row width in X");
        // allow NaNs/Inf? For simplicity we don't check here.
    }
}

double LinearRegression::dot(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("dot: size mismatch");
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

void LinearRegression::fit(const std::vector<std::vector<double>>& X,
                           const std::vector<double>& y) {
    check_dimensions(X, y);
    const size_t n = X.size();
    const size_t d = X[0].size();

    n_features_ = static_cast<int>(d);
    weights_.assign(d, 0.0);
    if (fit_intercept_) bias_ = 0.0; else bias_ = 0.0; // explicit

    for (int ep = 0; ep < epochs_; ++ep) {
        std::vector<double> grad_w(d, 0.0);
        double grad_b = 0.0;

        // batch gradient
        for (size_t i = 0; i < n; ++i) {
            const double yhat = dot(weights_, X[i]) + (fit_intercept_ ? bias_ : 0.0);
            const double err  = yhat - y[i];
            for (size_t j = 0; j < d; ++j) grad_w[j] += err * X[i][j];
            if (fit_intercept_) grad_b += err;
        }

        // average gradients
        const double inv_n = 1.0 / static_cast<double>(n);
        for (size_t j = 0; j < d; ++j) weights_[j] -= lr_ * (grad_w[j] * inv_n);
        if (fit_intercept_) bias_ -= lr_ * (grad_b * inv_n);
    }
}

std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& X) const {
    if (X.empty()) return {};
    const size_t d = X[0].size();
    if (static_cast<int>(d) != n_features_) {
        throw std::invalid_argument("predict: feature dimension mismatch (expected " +
                                    std::to_string(n_features_) + ", got " + std::to_string(d) + ")");
    }
    std::vector<double> out;
    out.reserve(X.size());
    for (const auto& row : X) {
        if (row.size() != d) throw std::invalid_argument("predict: inconsistent row width in X");
        double yhat = dot(weights_, row) + (fit_intercept_ ? bias_ : 0.0);
        out.push_back(yhat);
    }
    return out;
}

double LinearRegression::score(const std::vector<std::vector<double>>& X,
                               const std::vector<double>& y) const {
    if (X.size() != y.size()) throw std::invalid_argument("score: X and y size mismatch");
    if (X.empty()) throw std::invalid_argument("score: empty dataset");

    auto yhat = predict(X);

    // compute R^2
    double mean_y = 0.0;
    for (double v : y) mean_y += v;
    mean_y /= static_cast<double>(y.size());

    double ss_tot = 0.0, ss_res = 0.0;
    for (size_t i = 0; i < y.size(); ++i) {
        const double diff_tot = y[i] - mean_y;
        const double diff_res = y[i] - yhat[i];
        ss_tot += diff_tot * diff_tot;
        ss_res += diff_res * diff_res;
    }
    if (ss_tot == 0.0) return 1.0; // all targets equal -> define R^2 = 1
    return 1.0 - (ss_res / ss_tot);
}

void LinearRegression::save(const std::string& path) const {
    // File format (text):
    // # QuarkML LinearRegression v1
    // n_features=3
    // fit_intercept=true
    // bias=0.12345
    // weights=0.0100000000,-0.2200000000,1.3070000000
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs) throw std::runtime_error("LinearRegression::save: cannot open file: " + path);

    io::write_header(ofs, "LinearRegression", ml::core::SerializableModel::kModelFormatVersion);
    io::write_kv(ofs, "n_features", std::to_string(n_features_));
    io::write_kv(ofs, "fit_intercept", fit_intercept_ ? "true" : "false");
    ofs << std::setprecision(10) << std::fixed;
    io::write_kv(ofs, "bias", std::to_string(bias_));
    io::write_vec(ofs, "weights", weights_, /*precision=*/10);
    ofs.flush();
    if (!ofs) throw std::runtime_error("LinearRegression::save: write failed for: " + path);
}

LinearRegression LinearRegression::load(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("LinearRegression::load: cannot open file: " + path);

    // We read the full file, ignoring the first header line (handled by parse_kv_file).
    auto kv = io::parse_kv_file(ifs);

    auto it_nf = kv.find("n_features");
    auto it_fit = kv.find("fit_intercept");
    auto it_bias = kv.find("bias");
    auto it_w = kv.find("weights");

    if (it_nf == kv.end() || it_fit == kv.end() || it_bias == kv.end() || it_w == kv.end()) {
        throw std::runtime_error("LinearRegression::load: missing required keys "
                                 "(n_features, fit_intercept, bias, weights)");
    }

    // parse
    int n_features = 0;
    try {
        n_features = std::stoi(it_nf->second);
    } catch (...) {
        throw std::runtime_error("LinearRegression::load: invalid n_features: " + it_nf->second);
    }

    bool fit_intercept = false;
    const std::string fit_str = it_fit->second;
    if (fit_str == "true" || fit_str == "1") fit_intercept = true;
    else if (fit_str == "false" || fit_str == "0") fit_intercept = false;
    else throw std::runtime_error("LinearRegression::load: invalid fit_intercept: " + fit_str);

    double bias = 0.0;
    try {
        size_t idx = 0;
        bias = std::stod(it_bias->second, &idx);
        if (idx != it_bias->second.size())
            throw std::runtime_error("trailing chars");
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("LinearRegression::load: invalid bias: ")
                                 + it_bias->second + " (" + e.what() + ")");
    }

    std::vector<double> weights = io::parse_vec(it_w->second);
    if (static_cast<int>(weights.size()) != n_features) {
        throw std::runtime_error("LinearRegression::load: weights length (" +
                                 std::to_string(weights.size()) +
                                 ") != n_features (" + std::to_string(n_features) + ")");
    }

    LinearRegression model; // default hyperparameters; for inference they don't matter
    model.weights_ = std::move(weights);
    model.bias_ = bias;
    model.fit_intercept_ = fit_intercept;
    model.n_features_ = n_features;
    return model;
}

} // namespace ml::linear

