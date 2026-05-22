#pragma once

#include <cstddef>

class KernelSource
{
public:
    static const char* getSource();
    static std::size_t getSourceLength();
};