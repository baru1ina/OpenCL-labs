#pragma once
#include <CL/cl.h>

class GPUContext
{
public:
    static GPUContext& getInstance();

    cl_context getContext() const { return m_context; }
    cl_command_queue getQueue() const { return m_queue; }
    cl_kernel getKernel(const char* name) const;

    GPUContext(const GPUContext&) = delete;
    GPUContext& operator=(const GPUContext&) = delete;

    ~GPUContext();

private:
    GPUContext();

    void selectDevice();
    void createContextAndQueue();
    void buildProgram();
    cl_kernel createKernel(const char* name) const;

    cl_platform_id m_platform;
    cl_device_id m_device;
    cl_context m_context;
    cl_command_queue m_queue;
    cl_program m_program;
};