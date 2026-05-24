#pragma once

#include <CL/cl.h>
#include <stdexcept>
#include <string>

class OpenCLException : public std::runtime_error
{
public:
    OpenCLException(cl_int error, const std::string& message, unsigned int line);

    cl_int error() const noexcept { return m_error; }
    unsigned int line() const noexcept { return m_line; }

    static void throwIfFailed(cl_int error, const char* expression, unsigned int line);
    static const char* errorToString(cl_int error) noexcept;

private:
    cl_int m_error;
    unsigned int m_line;
};

#define OPENCL_FAIL(expr) OpenCLException::throwIfFailed((expr), #expr, __LINE__)
