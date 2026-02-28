#include <gtest/gtest.h>
#include "tensor.h"

using namespace std;

TEST(TensorBasics, ShapeAndStrides) {
    Tensor t({2, 3});

    EXPECT_EQ(t.shape().size(), 2);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);

    EXPECT_EQ(t.strides()[0], 3);
    EXPECT_EQ(t.strides()[1], 1);
}

TEST(TensorPrint, PrintingData) {
    Tensor t({2, 3});
    int k = 0;
    for (size_t i = 0; i < t.shape()[0]; i++) {
        for (size_t j = 0; j < t.shape()[1]; j++) {
            t({i, j}) = i*j + (k);
            k++;
        }
    }
    cout << "------printing Data----\n";
    for (size_t i = 0; i < t.shape()[0]; i++) {
        for (size_t j = 0; j < t.shape()[1]; j++) {
            cout << t({i, j}) << ", ";
        }
    }    
}

TEST(NDTensor, NDTesnorPrint) {
    Tensor t({2, 3, 4});
    int k = 0;
    for (size_t i = 0; i < t.shape()[0]; i++) {
        for (size_t j = 0; j < t.shape()[1]; j++) {
            for (size_t k = 0; k < t.shape()[2]; k++) {
                t({i, j, k}) = i*j + (k);
                k++;
            }
        }
    }
    cout << "------printing 3D Data----\n";
    for (size_t i = 0; i < t.shape()[0]; i++) {
        for (size_t j = 0; j < t.shape()[1]; j++) {
            for (size_t k = 0; k < t.shape()[2]; k++) {
                cout << t({i, j, k}) << ", ";
            }        
        }
    }    
}