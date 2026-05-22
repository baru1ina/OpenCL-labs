#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>

#include "AttentionCompute.h"
#include "Matrix.h"

static float computeMaxError(const Matrix& first, const Matrix& second)
{
    float maxDifference = 0.f;
    for (size_t row = 0; row < first.getHeight(); ++row)
    {
        for (size_t col = 0; col < first.getWidth(); ++col)
        {
            float diff = std::abs(first.getElement(row, col) - second.getElement(row, col));
            maxDifference = std::max(maxDifference, diff);
        }
    }
    return maxDifference;
}

static bool areMatricesClose(const Matrix& first, const Matrix& second, float tolerance = 1e-3f)
{
    return first.getHeight() == second.getHeight() &&
        first.getWidth() == second.getWidth() &&
        computeMaxError(first, second) <= tolerance;
}

static void displayMatrix(const Matrix& matrix)
{
    for (size_t row = 0; row < matrix.getHeight(); ++row)
    {
        for (size_t col = 0; col < matrix.getWidth(); ++col)
        {
            std::cout << std::fixed << std::setprecision(4) << matrix.getElement(row, col) << '\t';
        }
        std::cout << '\n';
    }
}

int main()
{
    try
    {
        constexpr size_t SEQUENCE_LEN = 64;
        constexpr size_t HIDDEN_DIM = 64;

        Matrix queries = Matrix::createRandom(SEQUENCE_LEN, HIDDEN_DIM, 1, -1.f, 1.f);
        Matrix keys = Matrix::createRandom(SEQUENCE_LEN, HIDDEN_DIM, 2, -1.f, 1.f);
        Matrix values = Matrix::createRandom(SEQUENCE_LEN, HIDDEN_DIM, 3, -1.f, 1.f);

        Matrix naiveResult = AttentionCompute::computeForward(queries, keys, values, AttentionCompute::Mode::NAIVE);
        Matrix optimizedResult = AttentionCompute::computeForward(queries, keys, values, AttentionCompute::Mode::OPTIMIZED);

        float maxError = computeMaxError(naiveResult, optimizedResult);
        std::cout << "Maximum absolute error: " << maxError << '\n';
        std::cout << (areMatricesClose(naiveResult, optimizedResult) ? "CORRECT ANSWER" : "WRONG ANSWER") << '\n';
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << std::endl;
        return -1;
    }

    return 0;
}