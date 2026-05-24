#include "particles.h"

#include "OpenCLException.h"
#include "OpenCLRuntime.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace
{
    std::string readTextFile(const std::string& path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Failed to open file: " + path);

        std::ostringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    bool fileExists(const std::string& path)
    {
        std::ifstream file(path, std::ios::binary);
        return file.good();
    }

    std::string directoryOf(const std::string& path)
    {
        const std::size_t pos = path.find_last_of("\\/");
        if (pos == std::string::npos)
            return ".";
        return path.substr(0, pos);
    }

    std::string executableDirectory()
    {
        char buffer[MAX_PATH]{};
        DWORD len = GetModuleFileNameA(nullptr, buffer, MAX_PATH);
        if (len == 0 || len >= MAX_PATH)
            return ".";
        return directoryOf(std::string(buffer, len));
    }

    std::string loadKernelSource()
    {
        const std::string exeDir = executableDirectory();
        const std::string paths[] = {
            "particles.cl",
            ".\\particles.cl",
            exeDir + "\\particles.cl",
            exeDir + "\\..\\..\\particles.cl",
            exeDir + "\\..\\..\\..\\particles.cl",
            "..\\particles.cl",
            "..\\..\\particles.cl"
        };

        for (const auto& path : paths)
        {
            if (fileExists(path))
                return readTextFile(path);
        }

        std::ostringstream oss;
        oss << "Failed to open particles.cl. Working directory or executable directory does not contain the kernel file. Tried:";
        for (const auto& path : paths)
            oss << "\n  " << path;
        throw std::runtime_error(oss.str());
    }

    std::string getPlatformInfoString(cl_platform_id platform, cl_platform_info param)
    {
        std::size_t size = 0;
        OPENCL_FAIL(clGetPlatformInfo(platform, param, 0, nullptr, &size));
        std::string value(size, '\0');
        OPENCL_FAIL(clGetPlatformInfo(platform, param, size, value.data(), nullptr));
        if (!value.empty() && value.back() == '\0')
            value.pop_back();
        return value;
    }

    std::string getDeviceInfoString(cl_device_id device, cl_device_info param)
    {
        std::size_t size = 0;
        OPENCL_FAIL(clGetDeviceInfo(device, param, 0, nullptr, &size));
        std::string value(size, '\0');
        OPENCL_FAIL(clGetDeviceInfo(device, param, size, value.data(), nullptr));
        if (!value.empty() && value.back() == '\0')
            value.pop_back();
        return value;
    }

    std::size_t roundUp(std::size_t value, std::size_t multiple)
    {
        return ((value + multiple - 1) / multiple) * multiple;
    }

    ParticleSystem::Float4 f4(float3 v, float w = 0.0f)
    {
        return ParticleSystem::Float4{ v.x, v.y, v.z, w };
    }

    struct Int4 { int x, y, z, w; };

    Int4 i4(int3 v, int w = 0)
    {
        return Int4{ v.x, v.y, v.z, w };
    }

    cl_mem createBufferChecked(cl_context context, cl_mem_flags flags, std::size_t bytes, void* hostPtr, const char* label)
    {
        cl_int error = CL_SUCCESS;
        cl_mem mem = clCreateBuffer(context, flags, bytes, hostPtr, &error);
        if (error != CL_SUCCESS)
        {
            std::ostringstream oss;
            oss << "Failed to create OpenCL buffer: " << label;
            throw OpenCLException(error, oss.str(), __LINE__);
        }
        return mem;
    }

    class ClBuffer
    {
    public:
        explicit ClBuffer(cl_mem mem = nullptr) : mem_(mem) {}

        ~ClBuffer()
        {
            if (mem_)
                clReleaseMemObject(mem_);
        }

        ClBuffer(const ClBuffer&) = delete;
        ClBuffer& operator=(const ClBuffer&) = delete;

        cl_mem get() const { return mem_; }

    private:
        cl_mem mem_ = nullptr;
    };
}

ParticleSystem::ParticleSystem(int particleCount, GridInfo grid)
    : particleCount_(particleCount), grid_(grid)
{
    cellTotal_ = grid.dims.x * grid.dims.y * grid.dims.z;
}

ParticleSystem::~ParticleSystem()
{
    releaseAll();
}

bool ParticleSystem::setError(const std::string& message)
{
    lastError_ = message;
    std::cerr << message << '\n';
    return false;
}

bool ParticleSystem::initializeOpenCL()
{
    try
    {
        const OpenCLRuntime::DeviceSelection selected = OpenCLRuntime::selectDevice({
            CL_DEVICE_TYPE_GPU,
            CL_DEVICE_TYPE_CPU
        });

        platform_ = selected.platform;
        device_ = selected.device;

        OpenCLRuntime::createContextAndQueue(platform_, device_, context_, queue_);
        OpenCLRuntime::printSelectedDevice(platform_, device_);
        deviceName_ = OpenCLRuntime::getDeviceInfo(device_, CL_DEVICE_NAME);

        return true;
    }
    catch (const std::exception& e)
    {
        return setError(e.what());
    }
}

bool ParticleSystem::buildOpenCLProgram()
{
    try
    {
        const std::string kernelSource = loadKernelSource();
        const char* source = kernelSource.c_str();
        const std::size_t sourceLength = kernelSource.size();

        cl_int error = CL_SUCCESS;
        program_ = clCreateProgramWithSource(context_, 1, &source, &sourceLength, &error);
        OPENCL_FAIL(error);

        const std::string buildOptions =
            "-cl-std=CL1.2"
            " -DMAX_PARTICLES_PER_CELL=" + std::to_string(MAX_PARTICLES_PER_CELL) +
            " -DMAX_PLANES=" + std::to_string(MAX_PLANES);

        error = clBuildProgram(program_, 1, &device_, buildOptions.c_str(), nullptr, nullptr);
        if (error != CL_SUCCESS)
        {
            std::size_t logSize = 0;
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
            std::string buildLog(logSize, '\0');

            if (logSize > 0)
                clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);

            std::ostringstream oss;
            oss << "Failed to build OpenCL program";
            if (!buildLog.empty())
                oss << "\nBuild log:\n" << buildLog;

            throw OpenCLException(error, oss.str(), __LINE__);
        }

        auto createKernel = [&](const char* kernelName, cl_kernel& out)
        {
            cl_int kernelError = CL_SUCCESS;
            out = clCreateKernel(program_, kernelName, &kernelError);
            OPENCL_FAIL(kernelError);
        };

        createKernel("initParticles", initKernel_);
        createKernel("clearGrid", clearGridKernel_);
        createKernel("buildGrid", buildGridKernel_);
        createKernel("collideDifferentTypes", collideKernel_);
        createKernel("updateParticles", updateKernel_);

        return true;
    }
    catch (const std::exception& e)
    {
        return setError(e.what());
    }
}

bool ParticleSystem::createBuffers()
{
    try
    {
        const std::size_t n = static_cast<std::size_t>(particleCount_);
        const std::size_t cells = static_cast<std::size_t>(cellTotal_);

        d_pos_ = createBufferChecked(context_, CL_MEM_READ_WRITE, n * sizeof(Float4), nullptr, "positions");
        d_vel_ = createBufferChecked(context_, CL_MEM_READ_WRITE, n * sizeof(Float4), nullptr, "velocities");
        d_life_ = createBufferChecked(context_, CL_MEM_READ_WRITE, n * sizeof(float), nullptr, "life");
        d_type_ = createBufferChecked(context_, CL_MEM_READ_WRITE, n * sizeof(int), nullptr, "type");
        d_rng_ = createBufferChecked(context_, CL_MEM_READ_WRITE, n * sizeof(unsigned int), nullptr, "rng");
        d_cellCount_ = createBufferChecked(context_, CL_MEM_READ_WRITE, cells * sizeof(int), nullptr, "cellCount");
        d_cellParticles_ = createBufferChecked(context_, CL_MEM_READ_WRITE, cells * MAX_PARTICLES_PER_CELL * sizeof(int), nullptr, "cellParticles");
        d_score_ = createBufferChecked(context_, CL_MEM_READ_WRITE, sizeof(int), nullptr, "score");

        return true;
    }
    catch (const std::exception& e)
    {
        return setError(e.what());
    }
}

bool ParticleSystem::initialize(const Emitter& emitter, unsigned int seed, std::string* errorText)
{
    if (!initializeOpenCL() || !buildOpenCLProgram() || !createBuffers())
    {
        if (errorText)
            *errorText = lastError_;
        return false;
    }

    try
    {
        resetScore();

        int arg = 0;
        Float4 emitterCenter = f4(emitter.center);
        Float4 emitterSize = f4(emitter.size);
        Float4 emitterVelocity = f4(emitter.baseVelocity);

        OPENCL_FAIL(clSetKernelArg(initKernel_, arg++, sizeof(cl_mem), &d_pos_));
        OPENCL_FAIL(clSetKernelArg(initKernel_, arg++, sizeof(cl_mem), &d_vel_));
        OPENCL_FAIL(clSetKernelArg(initKernel_, arg++, sizeof(cl_mem), &d_life_));
        OPENCL_FAIL(clSetKernelArg(initKernel_, arg++, sizeof(cl_mem), &d_type_));
        OPENCL_FAIL(clSetKernelArg(initKernel_, arg++, sizeof(cl_mem), &d_rng_));
        OPENCL_FAIL(clSetKernelArg(initKernel_, arg++, sizeof(int), &particleCount_));
        OPENCL_FAIL(clSetKernelArg(initKernel_, arg++, sizeof(unsigned int), &seed));
        OPENCL_FAIL(clSetKernelArg(initKernel_, arg++, sizeof(Float4), &emitterCenter));
        OPENCL_FAIL(clSetKernelArg(initKernel_, arg++, sizeof(Float4), &emitterSize));
        OPENCL_FAIL(clSetKernelArg(initKernel_, arg++, sizeof(Float4), &emitterVelocity));

        const std::size_t local = static_cast<std::size_t>(threads_);
        const std::size_t global = roundUp(static_cast<std::size_t>(particleCount_), local);
        OPENCL_FAIL(clEnqueueNDRangeKernel(queue_, initKernel_, 1, nullptr, &global, &local, 0, nullptr, nullptr));
        OPENCL_FAIL(clFinish(queue_));
        return true;
    }
    catch (const std::exception& e)
    {
        if (errorText)
            *errorText = e.what();
        return setError(e.what());
    }
}

void ParticleSystem::resetScore()
{
    if (!queue_ || !d_score_)
        return;

    int zero = 0;
    clEnqueueWriteBuffer(queue_, d_score_, CL_TRUE, 0, sizeof(int), &zero, 0, nullptr, nullptr);
}

bool ParticleSystem::step(float dt, const SceneObjects& scene, const SimulationSettings& settings, const Emitter& emitter)
{
    if (!queue_)
        return false;

    try
    {
        const std::size_t local = static_cast<std::size_t>(threads_);
        const std::size_t particleGlobal = roundUp(static_cast<std::size_t>(particleCount_), local);
        const std::size_t gridGlobal = roundUp(static_cast<std::size_t>(cellTotal_), local);

        int arg = 0;
        OPENCL_FAIL(clSetKernelArg(clearGridKernel_, arg++, sizeof(cl_mem), &d_cellCount_));
        OPENCL_FAIL(clSetKernelArg(clearGridKernel_, arg++, sizeof(int), &cellTotal_));
        OPENCL_FAIL(clEnqueueNDRangeKernel(queue_, clearGridKernel_, 1, nullptr, &gridGlobal, &local, 0, nullptr, nullptr));

        Float4 gridOrigin = f4(grid_.origin);
        Int4 gridDims = i4(grid_.dims);

        arg = 0;
        OPENCL_FAIL(clSetKernelArg(buildGridKernel_, arg++, sizeof(cl_mem), &d_pos_));
        OPENCL_FAIL(clSetKernelArg(buildGridKernel_, arg++, sizeof(int), &particleCount_));
        OPENCL_FAIL(clSetKernelArg(buildGridKernel_, arg++, sizeof(cl_mem), &d_cellCount_));
        OPENCL_FAIL(clSetKernelArg(buildGridKernel_, arg++, sizeof(cl_mem), &d_cellParticles_));
        OPENCL_FAIL(clSetKernelArg(buildGridKernel_, arg++, sizeof(Float4), &gridOrigin));
        OPENCL_FAIL(clSetKernelArg(buildGridKernel_, arg++, sizeof(float), &grid_.cellSize));
        OPENCL_FAIL(clSetKernelArg(buildGridKernel_, arg++, sizeof(Int4), &gridDims));
        OPENCL_FAIL(clEnqueueNDRangeKernel(queue_, buildGridKernel_, 1, nullptr, &particleGlobal, &local, 0, nullptr, nullptr));

        arg = 0;
        OPENCL_FAIL(clSetKernelArg(collideKernel_, arg++, sizeof(cl_mem), &d_pos_));
        OPENCL_FAIL(clSetKernelArg(collideKernel_, arg++, sizeof(cl_mem), &d_vel_));
        OPENCL_FAIL(clSetKernelArg(collideKernel_, arg++, sizeof(cl_mem), &d_type_));
        OPENCL_FAIL(clSetKernelArg(collideKernel_, arg++, sizeof(int), &particleCount_));
        OPENCL_FAIL(clSetKernelArg(collideKernel_, arg++, sizeof(cl_mem), &d_cellCount_));
        OPENCL_FAIL(clSetKernelArg(collideKernel_, arg++, sizeof(cl_mem), &d_cellParticles_));
        OPENCL_FAIL(clSetKernelArg(collideKernel_, arg++, sizeof(Float4), &gridOrigin));
        OPENCL_FAIL(clSetKernelArg(collideKernel_, arg++, sizeof(float), &grid_.cellSize));
        OPENCL_FAIL(clSetKernelArg(collideKernel_, arg++, sizeof(Int4), &gridDims));
        OPENCL_FAIL(clSetKernelArg(collideKernel_, arg++, sizeof(float), &settings.particleRadius));
        OPENCL_FAIL(clEnqueueNDRangeKernel(queue_, collideKernel_, 1, nullptr, &particleGlobal, &local, 0, nullptr, nullptr));

        std::array<Float4, MAX_PLANES> planePoints{};
        std::array<Float4, MAX_PLANES> planeNormals{};
        for (int i = 0; i < MAX_PLANES; ++i)
        {
            planePoints[i] = f4(scene.planes[i].point);
            planeNormals[i] = f4(scene.planes[i].normal);
        }

        std::array<float, MAX_PLANES> planeRadii{};
        for (int i = 0; i < MAX_PLANES; ++i)
        {
            planeRadii[i] = scene.planes[i].radius;
        }

        ClBuffer dPlaneRadii(createBufferChecked(
            context_,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * MAX_PLANES,
            planeRadii.data(),
            "planeRadii"));

        ClBuffer dPlanePoints(createBufferChecked(
            context_,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(Float4) * MAX_PLANES,
            planePoints.data(),
            "planePoints"));

        ClBuffer dPlaneNormals(createBufferChecked(
            context_,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(Float4) * MAX_PLANES,
            planeNormals.data(),
            "planeNormals"));

        Float4 basketMin = f4(scene.basket.minCorner);
        Float4 basketMax = f4(scene.basket.maxCorner);
        Float4 emitterCenter = f4(emitter.center);
        Float4 emitterSize = f4(emitter.size);
        Float4 emitterVelocity = f4(emitter.baseVelocity);

        cl_mem planePointsMem = dPlanePoints.get();
        cl_mem planeNormalsMem = dPlaneNormals.get();
        cl_mem planeRadiiMem = dPlaneRadii.get();

        arg = 0;
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(cl_mem), &d_pos_));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(cl_mem), &d_vel_));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(cl_mem), &d_life_));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(cl_mem), &d_type_));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(cl_mem), &d_rng_));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(cl_mem), &d_score_));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(int), &particleCount_));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(float), &dt));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(Float4), &emitterCenter));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(Float4), &emitterSize));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(Float4), &emitterVelocity));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(cl_mem), &planePointsMem));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(cl_mem), &planeNormalsMem));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(cl_mem), &planeRadiiMem));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(int), &scene.planeCount));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(Float4), &basketMin));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(Float4), &basketMax));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(float), &settings.particleRadius));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(float), &settings.worldMinY));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(float), &settings.damping));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(float), &settings.rollingFriction));
        OPENCL_FAIL(clSetKernelArg(updateKernel_, arg++, sizeof(float), &settings.gravity));

        OPENCL_FAIL(clEnqueueNDRangeKernel(queue_, updateKernel_, 1, nullptr, &particleGlobal, &local, 0, nullptr, nullptr));

        OPENCL_FAIL(clFinish(queue_));
        return true;
    }
    catch (const std::exception& e)
    {
        return setError(e.what());
    }
}

int ParticleSystem::getScore() const
{
    int value = 0;
    if (queue_ && d_score_)
        clEnqueueReadBuffer(queue_, d_score_, CL_TRUE, 0, sizeof(int), &value, 0, nullptr, nullptr);
    return value;
}

std::vector<ParticleSnapshot> ParticleSystem::downloadParticles(int maxCount) const
{
    const int count = std::min(maxCount, particleCount_);
    std::vector<ParticleSnapshot> out(count);
    if (!queue_ || !d_pos_ || !d_type_ || count <= 0)
        return out;

    std::vector<Float4> positions(count);
    std::vector<int> types(count);

    clEnqueueReadBuffer(queue_, d_pos_, CL_TRUE, 0, sizeof(Float4) * count, positions.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue_, d_type_, CL_TRUE, 0, sizeof(int) * count, types.data(), 0, nullptr, nullptr);

    for (int i = 0; i < count; ++i)
    {
        out[i].position = make_float3(positions[i].x, positions[i].y, positions[i].z);
        out[i].type = types[i];
    }

    return out;
}

void ParticleSystem::releaseAll()
{
    auto releaseMem = [](cl_mem& mem)
    {
        if (mem)
        {
            clReleaseMemObject(mem);
            mem = nullptr;
        }
    };

    releaseMem(d_pos_);
    releaseMem(d_vel_);
    releaseMem(d_life_);
    releaseMem(d_type_);
    releaseMem(d_rng_);
    releaseMem(d_cellCount_);
    releaseMem(d_cellParticles_);
    releaseMem(d_score_);

    if (initKernel_) { clReleaseKernel(initKernel_); initKernel_ = nullptr; }
    if (clearGridKernel_) { clReleaseKernel(clearGridKernel_); clearGridKernel_ = nullptr; }
    if (buildGridKernel_) { clReleaseKernel(buildGridKernel_); buildGridKernel_ = nullptr; }
    if (collideKernel_) { clReleaseKernel(collideKernel_); collideKernel_ = nullptr; }
    if (updateKernel_) { clReleaseKernel(updateKernel_); updateKernel_ = nullptr; }
    if (program_) { clReleaseProgram(program_); program_ = nullptr; }
    if (queue_) { clReleaseCommandQueue(queue_); queue_ = nullptr; }
    if (context_) { clReleaseContext(context_); context_ = nullptr; }

    device_ = nullptr;
    platform_ = nullptr;
}
