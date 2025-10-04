# QuarkML

A tiny yet rigorous C++20 machine learning library built from scratch.

**Status:** (Implementation in progress)

[![Build](https://img.shields.io/badge/build-WIP-lightgrey)]()
[![C++](https://img.shields.io/badge/C%2B%2B-20-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

QuarkML is a minimal, educational, and engineering-grade machine learning library written in modern C++20.  
It aims to reimplement fundamental ML algorithms from scratch with clear structure, unified APIs, and production-style organization.

### Planned Features

- Linear Regression — OLS via Normal/QR/SVD, optional Gradient Descent, Ridge (L2 regularization)
- K-Means Clustering — random & k-means++ initialization, multiple restarts, inertia
- Decision Tree — CART for classification & regression, pre-pruning options

Implementations are under development.  
The current repository mainly contains the project architecture and interface scaffolding.

---

## Project Goals

| Goal | Description |
|------|--------------|
| Unified API | Every model follows the same `fit / predict / score` interface |
| Readable Implementation | Focus on algorithm clarity, not just performance |
| Modern C++ | Leverage templates, RAII, and strong typing |
| Engineering Practice | CMake, unit tests, CI/CD, docs, benchmarks |

Non-Goals (v0):
- Competing with full frameworks like scikit-learn or XGBoost  
- GPU acceleration or large-scale parallelism

---

## Architecture Overview

```
QuarkML/
├─ CMakeLists.txt
├─ .gitignore
├─ LICENSE
├─ .clang-format / .editorconfig
│
├─ include/ml/
│  ├─ core/
│  │  ├─ Model.hpp
│  │  ├─ Dataset.hpp
│  │  ├─ Metrics.hpp
│  │  └─ Utils.hpp
│  ├─ linear/LinearRegression.hpp
│  ├─ cluster/KMeans.hpp
│  ├─ tree/DecisionTree.hpp
│  ├─ io/Csv.hpp
│  └─ preprocessing/Scaler.hpp
│
├─ src/
│  ├─ core/
│  ├─ linear/LinearRegression.cpp
│  ├─ cluster/KMeans.cpp
│  ├─ tree/DecisionTree.cpp
│  ├─ io/Csv.cpp
│  └─ preprocessing/Scaler.cpp
│
├─ examples/
│  ├─ linear_regression_basic.cpp
│  ├─ kmeans_iris.cpp
│  └─ decision_tree_toy.cpp
│
├─ tests/
│  ├─ test_linear.cpp
│  ├─ test_kmeans.cpp
│  └─ test_tree.cpp
│
├─ benchmarks/
│  ├─ bench_kmeans.cpp
│  └─ bench_tree.cpp
│
├─ docs/
│  └─ Doxyfile
│
├─ data/
│  └─ .gitkeep
│
└─ .github/workflows/ci.yml
```

---

## API Concept

```
class Model {
public:
    virtual void fit(const Matrix& X, const Vector& y) = 0;
    virtual Vector predict(const Matrix& X) const = 0;
    virtual double score(const Matrix& X, const Vector& y) const = 0;
    virtual ~Model() = default;
};
```

Example (planned):

```
#include <ml/linear/LinearRegression.hpp>
#include <ml/io/Csv.hpp>
#include <ml/core/Metrics.hpp>

int main() {
    auto [X, y] = ml::Csv::read_xy("data/housing.csv");
    ml::LinearRegression lr;
    lr.fit(X, y);
    auto preds = lr.predict(X);
    std::cout << "R² score: " << ml::r2_score(y, preds) << "\n";
}
```

---

## Roadmap

### v0 (MVP)
- [ ] Core utilities (Dataset, Metrics, Utils)
- [ ] Linear Regression (OLS, QR/SVD, Ridge)
- [ ] K-Means (k-means++, inertia)
- [ ] Decision Tree (CART with pre-pruning)
- [ ] CSV reader + scaler
- [ ] First examples + minimal tests
- [ ] CI & Doxygen docs

### v1
- [ ] Cross-validation utilities
- [ ] Model persistence (save/load)
- [ ] Benchmarks & performance tests
- [ ] Improved documentation site

### v1.x
- [ ] Logistic Regression / Naive Bayes
- [ ] PCA / SVD utilities
- [ ] Random Forest / Gradient Boosting
- [ ] Python bindings via pybind11

---

## Build (planned)

Requirements:
- Compiler: Clang ≥ 14 / GCC ≥ 11 / MSVC ≥ 19.3x
- CMake: ≥ 3.20
- Optional: Eigen for linear algebra backend

```
cmake -S . -B build
cmake --build build -j
```

---

## Contributing

- Follow `.clang-format`
- Keep modules independent
- Add unit tests for new code
- Document public APIs with Doxygen comments

---

## License

Licensed under the MIT License.  
See `LICENSE` for full text.

---

## Acknowledgements

Inspired by:
- scikit-learn (Python) — unified ML API  
- Eigen (C++) — efficient numerical backend  
- educational ML reimplementations emphasizing clarity  

QuarkML aims to be a readable, instructive, and well-engineered reference for fundamental ML algorithms in C++.
