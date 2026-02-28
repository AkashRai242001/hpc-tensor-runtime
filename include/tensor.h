#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <memory>

/*
 A minimal Tensor abstraction.

 Responsibilities:
 - Own or reference memory
 - Store shape & strides
 - Provide safe access to raw data
*/

class Tensor {
public:
    // Constructor: allocate a contiguous tensor
    Tensor(const std::vector<size_t>& shape);

    // Destructor
    ~Tensor();

    // Disable copy (expensive & unsafe for now)
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Enable move
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

    // Metadata access
    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& strides() const;
    size_t numel() const;

    // Raw data access
    float* data();
    const float* data() const;

    float& operator()(const std::vector<size_t> &indices);
    const float& operator()(const std::vector<size_t> &indices) const;

private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t numel_;

    float* data_;  // raw contiguous buffer
};

