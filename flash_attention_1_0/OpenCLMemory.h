#pragma once
#include "OpenCLException.h"

class OpenCLMemory
{
public:
    explicit OpenCLMemory(cl_mem memory = nullptr) : m_memory(memory) {}

    OpenCLMemory(const OpenCLMemory&) = delete;
    OpenCLMemory& operator=(const OpenCLMemory&) = delete;

    OpenCLMemory(OpenCLMemory&& other) noexcept : m_memory(other.m_memory)
    {
        other.m_memory = nullptr;
    }

    OpenCLMemory& operator=(OpenCLMemory&& other) noexcept
    {
        if (this != &other)
        {
            release();
            m_memory = other.m_memory;
            other.m_memory = nullptr;
        }
        return *this;
    }

    cl_mem get() const { return m_memory; }

    cl_mem* getPointer()
    {
        release();
        return &m_memory;
    }

    void release()
    {
        if (m_memory)
        {
            clReleaseMemObject(m_memory);
            m_memory = nullptr;
        }
    }

    ~OpenCLMemory()
    {
        release();
    }

private:
    cl_mem m_memory;
};