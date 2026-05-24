#include "OpenCLRuntime.h"

#include "OpenCLException.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace
{
    std::string readPlatformString(cl_platform_id platform, cl_platform_info param)
    {
        std::size_t size = 0;
        OPENCL_FAIL(clGetPlatformInfo(platform, param, 0, nullptr, &size));

        std::string value(size, '\0');
        OPENCL_FAIL(clGetPlatformInfo(platform, param, size, value.data(), nullptr));

        if (!value.empty() && value.back() == '\0')
        {
            value.pop_back();
        }
        return value;
    }

    std::string readDeviceString(cl_device_id device, cl_device_info param)
    {
        std::size_t size = 0;
        OPENCL_FAIL(clGetDeviceInfo(device, param, 0, nullptr, &size));

        std::string value(size, '\0');
        OPENCL_FAIL(clGetDeviceInfo(device, param, size, value.data(), nullptr));

        if (!value.empty() && value.back() == '\0')
        {
            value.pop_back();
        }
        return value;
    }

    bool tryPickDevice(
        const std::vector<cl_platform_id>& platforms,
        cl_device_type type,
        OpenCLRuntime::DeviceSelection& out)
    {
        for (cl_platform_id platform : platforms)
        {
            cl_uint deviceCount = 0;
            cl_int status = clGetDeviceIDs(platform, type, 0, nullptr, &deviceCount);

            if (status == CL_DEVICE_NOT_FOUND || deviceCount == 0)
            {
                continue;
            }

            OPENCL_FAIL(status);

            std::vector<cl_device_id> devices(deviceCount);
            OPENCL_FAIL(clGetDeviceIDs(platform, type, deviceCount, devices.data(), nullptr));

            out.platform = platform;
            out.device = devices.front();
            out.type = type;
            return true;
        }

        return false;
    }
}

namespace OpenCLRuntime
{
    std::string getPlatformInfo(cl_platform_id platform, cl_platform_info param)
    {
        return readPlatformString(platform, param);
    }

    std::string getDeviceInfo(cl_device_id device, cl_device_info param)
    {
        return readDeviceString(device, param);
    }

    DeviceSelection selectDevice(const std::vector<cl_device_type>& priority)
    {
        cl_uint platformCount = 0;
        OPENCL_FAIL(clGetPlatformIDs(0, nullptr, &platformCount));
        if (platformCount == 0)
        {
            throw std::runtime_error("OpenCL platforms not found");
        }

        std::vector<cl_platform_id> platforms(platformCount);
        OPENCL_FAIL(clGetPlatformIDs(platformCount, platforms.data(), nullptr));

        DeviceSelection selected;
        for (cl_device_type type : priority)
        {
            if (tryPickDevice(platforms, type, selected))
            {
                return selected;
            }
        }

        throw std::runtime_error("OpenCL device not found");
    }

    void createContextAndQueue(
        cl_platform_id platform,
        cl_device_id device,
        cl_context& context,
        cl_command_queue& queue)
    {
        cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform),
            0
        };

        cl_int error = CL_SUCCESS;
        context = clCreateContext(props, 1, &device, nullptr, nullptr, &error);
        OPENCL_FAIL(error);

        queue = clCreateCommandQueue(context, device, 0, &error);
        OPENCL_FAIL(error);
    }

    void printSelectedDevice(cl_platform_id platform, cl_device_id device)
    {
        std::cout << "OpenCL platform: " << getPlatformInfo(platform, CL_PLATFORM_NAME) << '\n';
        std::cout << "OpenCL device: " << getDeviceInfo(device, CL_DEVICE_NAME) << '\n';
        std::cout << "OpenCL C version: " << getDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION) << '\n';
    }
}
