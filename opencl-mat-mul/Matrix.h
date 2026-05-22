#pragma once

#include <cstddef>

class Matrix {
public:
    enum class MultiplicationMode
    {
        NAIVE,
        TILED,
        WARP_INTRINSICS
    };

    Matrix(std::size_t rows, std::size_t cols);
    Matrix(const Matrix& source);
    Matrix& operator=(const Matrix& source);

    static Matrix createConstant(float value, std::size_t rows, std::size_t cols);

    Matrix multiply(const Matrix& other, MultiplicationMode mode) const;

    float getElement(std::size_t row, std::size_t col) const;
    float& getElement(std::size_t row, std::size_t col);

    std::size_t getHeight() const;
    std::size_t getWidth() const;

    ~Matrix();

private:
    float* m_dataBuffer;
    std::size_t m_rowCount;
    std::size_t m_colCount;
};