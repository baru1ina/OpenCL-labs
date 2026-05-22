#pragma once

#include <CL/cl.h>

class GPUContext
{
public:
    static GPUContext& getInstance();

    cl_context getContext() const { return m_context; }
    cl_command_queue getQueue() const { return m_queue; }
    cl_kernel getNaiveKernel() const { return m_naiveKernel; }
    cl_kernel getTiledKernel() const { return m_tiledKernel; }
    cl_kernel getWarpIntrinsicKernel() const { return m_warpIntrinsicKernel; }

    GPUContext(const GPUContext&) = delete;
    GPUContext& operator=(const GPUContext&) = delete;

    ~GPUContext();

private:
    GPUContext();

    void selectPlatformAndDevice();
    void createContextAndQueue();
    void buildProgram();

    cl_platform_id m_platform;
    cl_device_id m_device;
    cl_context m_context;
    cl_command_queue m_queue;
    cl_program m_program;
    cl_kernel m_naiveKernel;
    cl_kernel m_tiledKernel;
    cl_kernel m_warpIntrinsicKernel;
};