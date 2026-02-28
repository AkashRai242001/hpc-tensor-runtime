#include "tensor.h"
#include <numeric>
#include <cstring>
#include <stdexcept>

// Helper: compute total number of elements
static size_t compute_numel(const std::vector<size_t>& shape) {
    return std::accumulate(
        shape.begin(),
        shape.end(),
        static_cast<size_t>(1),
        std::multiplies<size_t>()
    );
}

// Helper: compute contiguous strides
static std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size());
    size_t stride = 1;

    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

Tensor::Tensor(const std::vector<size_t>& shape)
    : shape_(shape),
      strides_(compute_strides(shape)),
      numel_(compute_numel(shape)),
      data_(nullptr)
{
    if (numel_ == 0) {
        throw std::runtime_error("Tensor with zero elements is not allowed");
    }

    data_ = new float[numel_];
    std::memset(data_, 0, numel_ * sizeof(float));
}

Tensor::~Tensor() {
    delete[] data_;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      numel_(other.numel_),
      data_(other.data_)
{
    other.data_ = nullptr;
    other.numel_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        delete[] data_;

        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        numel_ = other.numel_;
        data_ = other.data_;

        other.data_ = nullptr;
        other.numel_ = 0;
    }
    return *this;
}
// element accessor
float& Tensor::operator()(const std::vector<size_t> &indices) {
    if (indices.size() != shape_.size()) {
        throw std::runtime_error("invalid number of indices");
    }
    size_t offset = 0;
    for (size_t i = 0; i< indices.size(); i++){
        if (indices[i] > shape_[i]) {
            throw std::out_of_range("Tensor index out of range");
        }
        offset += indices[i] * strides_[i];
    }
    return data_[offset];
}

const float& Tensor::operator()(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::runtime_error("Invalid number of indices");
    }

    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Tensor index out of bounds");
        }
        offset += indices[i] * strides_[i];
    }

    return data_[offset];
}

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

const std::vector<size_t>& Tensor::strides() const {
    return strides_;
}

size_t Tensor::numel() const {
    return numel_;
}

float* Tensor::data() {
    return data_;
}

const float* Tensor::data() const {
    return data_;
}

