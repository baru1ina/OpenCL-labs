#include "GPUContext.h"
#include "OpenCLException.h"
#include "KernelSource.h"

#include <vector>
#include <string>
#include <iostream>
#include <sstream>

namespace
{
    std::string getPlatformInfo(cl_platform_id platform, cl_platform_info param)
    {
        std::size_t size = 0;
        OPENCL_FAIL(clGetPlatformInfo(platform, param, 0, nullptr, &size));
        std::string result(size, '\0');
        OPENCL_FAIL(clGetPlatformInfo(platform, param, size, result.data(), nullptr));
        if (!result.empty() && result.back() == '\0') result.pop_back();
        return result;
    }

    std::string getDeviceInfo(cl_device_id device, cl_device_info param)
    {
        std::size_t size = 0;
        OPENCL_FAIL(clGetDeviceInfo(device, param, 0, nullptr, &size));
        std::string result(size, '\0');
        OPENCL_FAIL(clGetDeviceInfo(device, param, size, result.data(), nullptr));
        if (!result.empty() && result.back() == '\0') result.pop_back();
        return result;
    }
}

GPUContext& GPUContext::getInstance()
{
    static GPUContext instance;
    return instance;
}

GPUContext::GPUContext()
    : m_platform(nullptr), m_device(nullptr), m_context(nullptr)
    , m_queue(nullptr), m_program(nullptr)
    , m_naiveKernel(nullptr), m_tiledKernel(nullptr), m_warpIntrinsicKernel(nullptr)
{
    selectPlatformAndDevice();
    createContextAndQueue();
    buildProgram();
}

GPUContext::~GPUContext()
{
    if (m_naiveKernel) clReleaseKernel(m_naiveKernel);
    if (m_tiledKernel) clReleaseKernel(m_tiledKernel);
    if (m_warpIntrinsicKernel) clReleaseKernel(m_warpIntrinsicKernel);
    if (m_program) clReleaseProgram(m_program);
    if (m_queue) clReleaseCommandQueue(m_queue);
    if (m_context) clReleaseContext(m_context);
}

void GPUContext::selectPlatformAndDevice()
{
    cl_uint platformCount = 0;
    OPENCL_FAIL(clGetPlatformIDs(0, nullptr, &platformCount));
    if (platformCount == 0)
    {
        throw std::runtime_error("No OpenCL platforms available");
    }

    std::vector<cl_platform_id> platforms(platformCount);
    OPENCL_FAIL(clGetPlatformIDs(platformCount, platforms.data(), nullptr));

    bool foundIntel = false;
    for (cl_platform_id platform : platforms)
    {
        cl_uint deviceCount = 0;
        cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
        if (status == CL_DEVICE_NOT_FOUND || deviceCount == 0) continue;

        OPENCL_FAIL(status);
        std::vector<cl_device_id> devices(deviceCount);
        OPENCL_FAIL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr));

        for (cl_device_id device : devices)
        {
            std::string vendor = getDeviceInfo(device, CL_DEVICE_VENDOR);
            if (vendor.find("Intel") != std::string::npos)
            {
                m_platform = platform;
                m_device = device;
                foundIntel = true;
                break;
            }
        }
        if (foundIntel) break;
    }

    if (!foundIntel)
    {
        for (cl_platform_id platform : platforms)
        {
            cl_uint deviceCount = 0;
            cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
            if (status == CL_DEVICE_NOT_FOUND || deviceCount == 0) continue;

            OPENCL_FAIL(status);
            std::vector<cl_device_id> devices(deviceCount);
            OPENCL_FAIL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr));

            m_platform = platform;
            m_device = devices.front();
            foundIntel = true;
            break;
        }
    }

    if (!foundIntel)
    {
        throw std::runtime_error("No GPU device found with OpenCL support");
    }
}

void GPUContext::createContextAndQueue()
{
    cl_int error = CL_SUCCESS;
    m_context = clCreateContext(nullptr, 1, &m_device, nullptr, nullptr, &error);
    OPENCL_FAIL(error);

    m_queue = clCreateCommandQueue(m_context, m_device, 0, &error);
    OPENCL_FAIL(error);
}

void GPUContext::buildProgram()
{
    cl_int error = CL_SUCCESS;
    const char* source = KernelSource::getSource();
    std::size_t sourceLen = KernelSource::getSourceLength();

    m_program = clCreateProgramWithSource(m_context, 1, &source, &sourceLen, &error);
    OPENCL_FAIL(error);

    const char* options = "-DTILE_SIZE=16 -DWARP_SIZE=32";
    error = clBuildProgram(m_program, 1, &m_device, options, nullptr, nullptr);

    if (error != CL_SUCCESS)
    {
        std::size_t logSize = 0;
        clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string buildLog(logSize, '\0');
        if (logSize > 0)
        {
            clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
        }
        std::ostringstream oss;
        oss << "Failed to build program";
        if (!buildLog.empty()) oss << "\nLog:\n" << buildLog;
        throw std::runtime_error(oss.str());
    }

    m_naiveKernel = clCreateKernel(m_program, "simpleMatMul", &error);
    OPENCL_FAIL(error);

    m_tiledKernel = clCreateKernel(m_program, "tiledMatMul", &error);
    OPENCL_FAIL(error);

    m_warpIntrinsicKernel = clCreateKernel(m_program, "warpIntrinsicMatMul", &error);
    OPENCL_FAIL(error);

    std::cout << "Platform: " << getPlatformInfo(m_platform, CL_PLATFORM_NAME) << '\n';
    std::cout << "Device: " << getDeviceInfo(m_device, CL_DEVICE_NAME) << '\n';
}