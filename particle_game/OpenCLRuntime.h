#pragma once

#include <CL/cl.h>
#include <string>
#include <vector>

namespace OpenCLRuntime
{
    struct DeviceSelection
    {
        cl_platform_id platform = nullptr;
        cl_device_id device = nullptr;
        cl_device_type type = 0;
    };

    std::string getPlatformInfo(cl_platform_id platform, cl_platform_info param);
    std::string getDeviceInfo(cl_device_id device, cl_device_info param);

    DeviceSelection selectDevice(const std::vector<cl_device_type>& priority);
    void createContextAndQueue(
        cl_platform_id platform,
        cl_device_id device,
        cl_context& context,
        cl_command_queue& queue);

    void printSelectedDevice(cl_platform_id platform, cl_device_id device);
}
