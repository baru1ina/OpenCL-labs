#include "AttentionCompute.h"
#include "GPUContext.h"
#include "KernelSource.h"
#include "OpenCLMemory.h"
#include "OpenCLException.h"

#include <cmath>
#include <stdexcept>

namespace
{
    constexpr int WORK_GROUP_SIZE = 128;
    constexpr int MAX_HIDDEN_DIM = 128;

    class KernelWrapper
    {
    public:
        KernelWrapper(const char* name, cl_kernel kernel) : m_name(name), m_kernel(kernel) {}

        cl_kernel get() const { return m_kernel; }

        template<typename T>
        void setArg(cl_uint index, const T& value)
        {
            OPENCL_FAIL(clSetKernelArg(m_kernel, index, sizeof(T), &value));
        }

        void setBufferArg(cl_uint index, cl_mem buffer)
        {
            OPENCL_FAIL(clSetKernelArg(m_kernel, index, sizeof(cl_mem), &buffer));
        }

    private:
        const char* m_name;
        cl_kernel m_kernel;
    };

    void executeAttention(const float* queryData,
        const float* keyData,
        const float* valueData,
        float* outputData,
        int sequenceLen,
        int hiddenDim,
        AttentionCompute::Mode mode)
    {
        GPUContext& gpu = GPUContext::getInstance();
        cl_int status = CL_SUCCESS;

        std::size_t bufferBytes = static_cast<std::size_t>(sequenceLen) *
            static_cast<std::size_t>(hiddenDim) * sizeof(float);

        OpenCLMemory queryBuffer(clCreateBuffer(gpu.getContext(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            bufferBytes, const_cast<float*>(queryData), &status));
        OPENCL_FAIL(status);

        OpenCLMemory keyBuffer(clCreateBuffer(gpu.getContext(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            bufferBytes, const_cast<float*>(keyData), &status));
        OPENCL_FAIL(status);

        OpenCLMemory valueBuffer(clCreateBuffer(gpu.getContext(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            bufferBytes, const_cast<float*>(valueData), &status));
        OPENCL_FAIL(status);

        OpenCLMemory outputBuffer(clCreateBuffer(gpu.getContext(),
            CL_MEM_WRITE_ONLY,
            bufferBytes, nullptr, &status));
        OPENCL_FAIL(status);

        const char* kernelName = (mode == AttentionCompute::Mode::NAIVE)
            ? "naiveAttentionKernel"
            : "optimizedAttentionKernel";

        cl_kernel rawKernel = gpu.getKernel(kernelName);
        KernelWrapper kernel(kernelName, rawKernel);

        float scaleFactor = 1.f / std::sqrt(static_cast<float>(hiddenDim));

        kernel.setBufferArg(0, queryBuffer.get());
        kernel.setBufferArg(1, keyBuffer.get());
        kernel.setBufferArg(2, valueBuffer.get());
        kernel.setBufferArg(3, outputBuffer.get());
        kernel.setArg<int>(4, sequenceLen);
        kernel.setArg<int>(5, hiddenDim);
        kernel.setArg<float>(6, scaleFactor);

        std::size_t localSize = WORK_GROUP_SIZE;
        std::size_t globalSize = static_cast<std::size_t>(sequenceLen) * localSize;

        OPENCL_FAIL(clEnqueueNDRangeKernel(gpu.getQueue(), rawKernel, 1, nullptr,
            &globalSize, &localSize, 0, nullptr, nullptr));
        OPENCL_FAIL(clFinish(gpu.getQueue()));

        OPENCL_FAIL(clEnqueueReadBuffer(gpu.getQueue(), outputBuffer.get(), CL_TRUE, 0, bufferBytes,
            outputData, 0, nullptr, nullptr));
    }
}

Matrix AttentionCompute::computeForward(const Matrix& queries,
    const Matrix& keys,
    const Matrix& values,
    Mode mode)
{
    if (queries.getHeight() != keys.getHeight() || queries.getHeight() != values.getHeight())
    {
        throw std::invalid_argument("Q, K, V must have identical number of rows (sequence length)");
    }

    if (queries.getWidth() != keys.getWidth() || queries.getWidth() != values.getWidth())
    {
        throw std::invalid_argument("Q, K, V must have identical hidden dimension");
    }

    if (queries.getWidth() > MAX_HIDDEN_DIM)
    {
        throw std::invalid_argument("Hidden dimension D must be <= 128 for this implementation");
    }

    Matrix result(queries.getHeight(), queries.getWidth());

    executeAttention(queries.getData(), keys.getData(), values.getData(), result.getData(),
        static_cast<int>(queries.getHeight()),
        static_cast<int>(queries.getWidth()),
        mode);

    return result;
}