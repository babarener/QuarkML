#include <iostream>
#include <vector>
#include "ml/linear/LinearRegression.h"

int main() {
    using namespace ml::linear;

    // --- Step 1: Prepare training data (y = 2x + 1 + noise)
    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
    std::vector<double> y = {3.1, 5.0, 7.2, 9.1, 10.9};

    // --- Step 2: Train a Ridge Regression model
    LinearRegression model(/*lr=*/0.01, /*epochs=*/3000,
                           /*fit_intercept=*/true,
                           /*lambda=*/0.1);
    model.fit(X, y);

    // --- Step 3: Save the trained model
    std::string path = "models/linreg_ridge.qmlf";
    model.save(path);
    std::cout << "Model saved to: " << path << std::endl;

    // --- Step 4: Load it back from file
    auto loaded = LinearRegression::load(path);
    std::cout << "Model loaded. n_features=" << loaded.n_features()
              << ", lambda=" << loaded.l2_lambda() << std::endl;

    // --- Step 5: Predict
    auto preds = loaded.predict(X);

    std::cout << "Predictions after reload:\n";
    for (size_t i = 0; i < preds.size(); ++i) {
        std::cout << "x=" << X[i][0] << " -> y_hat=" << preds[i]
                  << " (true=" << y[i] << ")\n";
    }

    std::cout << "R^2 score = " << loaded.score(X, y) << std::endl;
    return 0;
}
