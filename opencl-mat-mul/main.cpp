#include <cmath>
#include <cstdio>
#include <exception>
#include <iostream>

#include "Matrix.h"

static bool validateMatrix(const Matrix& mat, float expectedValue, float tolerance = 1e-6f)
{
    for (std::size_t row = 0; row < mat.getHeight(); ++row)
    {
        for (std::size_t col = 0; col < mat.getWidth(); ++col)
        {
            if (std::abs(mat.getElement(row, col) - expectedValue) > tolerance)
            {
                return false;
            }
        }
    }
    return true;
}

int main()
{
    try
    {
        const std::size_t dimension = 1024;

        Matrix firstMatrix = Matrix::createConstant(1.0f, dimension * 2, dimension);
        Matrix secondMatrix = Matrix::createConstant(1.0f, dimension, dimension);

        Matrix resultMatrix_intr = firstMatrix.multiply(secondMatrix, Matrix::MultiplicationMode::WARP_INTRINSICS);

        if (validateMatrix(resultMatrix_intr, static_cast<float>(dimension)))
        {
            std::printf("CORRECT");
        }
        else
        {
            std::printf("WRONG");
        }
        std::printf(" ANSWER FOR WARP_INTRINSICS\n");

        Matrix resultMatrix_tile = firstMatrix.multiply(secondMatrix, Matrix::MultiplicationMode::TILED);

        if (validateMatrix(resultMatrix_tile, static_cast<float>(dimension)))
        {
            std::printf("CORRECT");
        }
        else
        {
            std::printf("WRONG");
        }
        std::printf(" ANSWER FOR TILED\n");
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "Error: %s\n", error.what());
        return -1;
    }

    return 0;
}