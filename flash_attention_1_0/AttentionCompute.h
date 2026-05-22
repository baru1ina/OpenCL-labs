#pragma once
#include "Matrix.h"

class AttentionCompute
{
public:
    enum class Mode
    {
        NAIVE,
        OPTIMIZED
    };

    static Matrix computeForward(const Matrix& queries,
        const Matrix& keys,
        const Matrix& values,
        Mode mode);
};