#pragma once

#include "Matrix.h"
#include <cstddef>

class OpenCLCompute
{
public:
    static void performMultiplication(
        const float* leftMatrix,
        const float* rightMatrix,
        float* outputMatrix,
        std::size_t rowsLeft,
        std::size_t commonDim,
        std::size_t colsRight,
        Matrix::MultiplicationMode mode
    );
};