#include "OpenCLCompute.h"
#include "GPUContext.h"
#include "KernelSource.h"
#include "OpenCLException.h"

#include <CL/cl.h>
#include <stdexcept>
#include <cstring>

namespace Utilities
{
    inline std::size_t roundUp(std::size_t value, std::size_t multiple)
    {
        return ((value + multiple - 1) / multiple) * multiple;
    }

    inline void copyData(float* dest, const float* src, std::size_t count)
    {
        std::memcpy(dest, src, count * sizeof(float));
    }
}

namespace
{
    constexpr std::size_t BLOCK_SIZE = 16;

    void runKernel(
        GPUContext& gpu,
        cl_kernel kernel,
        const float* A, const float* B, float* C,
        std::size_t L, std::size_t M, std::size_t N,
        Matrix::MultiplicationMode mode)
    {
        if (L > UINT32_MAX || M > UINT32_MAX || N > UINT32_MAX)
        {
            throw std::overflow_error("Matrix dimensions exceed 32-bit limit");
        }

        cl_int status = CL_SUCCESS;

        std::size_t sizeA = L * M * sizeof(float);
        std::size_t sizeB = M * N * sizeof(float);
        std::size_t sizeC = L * N * sizeof(float);

        cl_mem bufA = clCreateBuffer(gpu.getContext(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeA, const_cast<float*>(A), &status);
        OPENCL_FAIL(status);

        cl_mem bufB = clCreateBuffer(gpu.getContext(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeB, const_cast<float*>(B), &status);
        OPENCL_FAIL(status);

        cl_mem bufC = clCreateBuffer(gpu.getContext(),
            CL_MEM_WRITE_ONLY, sizeC, nullptr, &status);
        OPENCL_FAIL(status);

        cl_uint L32 = static_cast<cl_uint>(L);
        cl_uint M32 = static_cast<cl_uint>(M);
        cl_uint N32 = static_cast<cl_uint>(N);

        OPENCL_FAIL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA));
        OPENCL_FAIL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB));
        OPENCL_FAIL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC));
        OPENCL_FAIL(clSetKernelArg(kernel, 3, sizeof(cl_uint), &L32));
        OPENCL_FAIL(clSetKernelArg(kernel, 4, sizeof(cl_uint), &M32));
        OPENCL_FAIL(clSetKernelArg(kernel, 5, sizeof(cl_uint), &N32));

        std::size_t localSize[2] = { BLOCK_SIZE, BLOCK_SIZE };
        std::size_t globalSize[2] = {
            Utilities::roundUp(N, BLOCK_SIZE),
            Utilities::roundUp(L, BLOCK_SIZE)
        };

        if (mode == Matrix::MultiplicationMode::WARP_INTRINSICS)
        {
            localSize[0] = 32;
            localSize[1] = 1;
            globalSize[0] = N * localSize[0];
            globalSize[1] = L;
        }

        OPENCL_FAIL(clEnqueueNDRangeKernel(gpu.getQueue(), kernel, 2, nullptr, globalSize, localSize, 0, nullptr, nullptr));
        OPENCL_FAIL(clFinish(gpu.getQueue()));
        OPENCL_FAIL(clEnqueueReadBuffer(gpu.getQueue(), bufC, CL_TRUE, 0, sizeC, C, 0, nullptr, nullptr));

        clReleaseMemObject(bufA);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufC);
    }
}

void OpenCLCompute::performMultiplication(
    const float* leftMatrix,
    const float* rightMatrix,
    float* outputMatrix,
    std::size_t rowsLeft,
    std::size_t commonDim,
    std::size_t colsRight,
    Matrix::MultiplicationMode mode)
{
    GPUContext& gpu = GPUContext::getInstance();

    cl_kernel selectedKernel = nullptr;
    switch (mode)
    {
    case Matrix::MultiplicationMode::NAIVE:
        selectedKernel = gpu.getNaiveKernel();
        break;
    case Matrix::MultiplicationMode::TILED:
        selectedKernel = gpu.getTiledKernel();
        break;
    case Matrix::MultiplicationMode::WARP_INTRINSICS:
        selectedKernel = gpu.getWarpIntrinsicKernel();
        break;
    default:
        throw std::invalid_argument("Unsupported multiplication mode");
    }

    runKernel(gpu, selectedKernel, leftMatrix, rightMatrix, outputMatrix,
        rowsLeft, commonDim, colsRight, mode);
}