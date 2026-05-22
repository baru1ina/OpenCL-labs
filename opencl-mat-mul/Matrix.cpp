#include "Matrix.h"
#include "OpenCLCompute.h"

#include <cstring>
#include <stdexcept>

namespace
{
    inline void copyFloatArray(float* dest, const float* src, std::size_t count)
    {
        std::memcpy(dest, src, count * sizeof(float));
    }
}

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : m_dataBuffer(new float[rows * cols]), m_rowCount(rows), m_colCount(cols)
{
}

Matrix::Matrix(const Matrix& source)
    : Matrix(source.m_rowCount, source.m_colCount)
{
    copyFloatArray(m_dataBuffer, source.m_dataBuffer, m_rowCount * m_colCount);
}

Matrix& Matrix::operator=(const Matrix& source)
{
    if (this == &source)
    {
        return *this;
    }

    float* newBuffer = new float[source.m_rowCount * source.m_colCount];
    copyFloatArray(newBuffer, source.m_dataBuffer, source.m_rowCount * source.m_colCount);

    delete[] m_dataBuffer;
    m_dataBuffer = newBuffer;
    m_rowCount = source.m_rowCount;
    m_colCount = source.m_colCount;
    return *this;
}

Matrix Matrix::createConstant(float value, std::size_t rows, std::size_t cols)
{
    Matrix result(rows, cols);
    for (std::size_t idx = 0; idx < rows * cols; ++idx)
    {
        result.m_dataBuffer[idx] = value;
    }
    return result;
}

Matrix Matrix::multiply(const Matrix& other, MultiplicationMode mode) const
{
    if (m_colCount != other.m_rowCount)
    {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
    }

    Matrix result(m_rowCount, other.m_colCount);

    OpenCLCompute::performMultiplication(
        m_dataBuffer, other.m_dataBuffer, result.m_dataBuffer,
        m_rowCount, m_colCount, other.m_colCount,
        mode
    );

    return result;
}

std::size_t Matrix::getHeight() const
{
    return m_rowCount;
}

std::size_t Matrix::getWidth() const
{
    return m_colCount;
}

float Matrix::getElement(std::size_t row, std::size_t col) const
{
    return m_dataBuffer[row * m_colCount + col];
}

float& Matrix::getElement(std::size_t row, std::size_t col)
{
    return m_dataBuffer[row * m_colCount + col];
}

Matrix::~Matrix()
{
    delete[] m_dataBuffer;
    m_dataBuffer = nullptr;
}