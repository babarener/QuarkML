#pragma once
#include <string>

namespace ml::core {

/// Base interface for models that can be serialized to disk.
/// Concrete models should implement `save(path)` and provide a static `load(path)`.
struct SerializableModel {
    virtual ~SerializableModel() = default;

    /// Serialize model parameters and metadata to a file.
    virtual void save(const std::string& path) const = 0;

    /// Model file format version (bump when making backward-incompatible changes).
    static constexpr int kModelFormatVersion = 1;
};

} // namespace ml::core

