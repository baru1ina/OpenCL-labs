#pragma once

#include <cstddef>

class Matrix {
public:
    enum class MultiplicationMode
    {
        NAIVE,
        TILED
    };

    Matrix(std::size_t rows, std::size_t cols);
    Matrix(const Matrix& source);
    Matrix& operator=(const Matrix& source);

    static Matrix createConstant(float value, std::size_t rows, std::size_t cols);
    static Matrix createRandom(std::size_t rows, std::size_t cols, unsigned int seed = 42, float minValue = -1.f, float maxValue = 1.f);

    //Matrix multiply(const Matrix& other, MultiplicationMode mode) const;

    Matrix full(float val, size_t h, size_t w);

    float getElement(std::size_t row, std::size_t col) const;
    float& getElement(std::size_t row, std::size_t col);

    std::size_t getHeight() const;
    std::size_t getWidth() const;

    float const* getData() const;
    float* getData();

    ~Matrix();

private:
    float* m_dataBuffer;
    std::size_t m_rowCount;
    std::size_t m_colCount;
};