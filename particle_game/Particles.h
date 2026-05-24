#pragma once

#include <string>
#include <vector>

#include <CL/cl.h>

#include "Geometry.h"

struct ParticleSnapshot {
    float3 position;
    int type;
};

class ParticleSystem {
public:
    ParticleSystem(int particleCount, GridInfo grid);
    ~ParticleSystem();

    ParticleSystem(const ParticleSystem&) = delete;
    ParticleSystem& operator=(const ParticleSystem&) = delete;

    bool initialize(const Emitter& emitter, unsigned int seed, std::string* errorText = nullptr);
    void resetScore();
    bool step(float dt, const SceneObjects& scene, const SimulationSettings& settings, const Emitter& emitter);

    int getScore() const;
    std::vector<ParticleSnapshot> downloadParticles(int maxCount) const;
    std::string deviceName() const { return deviceName_; }
    std::string lastError() const { return lastError_; }

public:
    struct Float4 { float x, y, z, w; };

private:
    int particleCount_ = 0;
    GridInfo grid_{};
    int cellTotal_ = 0;
    int threads_ = 128;

    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;

    cl_kernel initKernel_ = nullptr;
    cl_kernel clearGridKernel_ = nullptr;
    cl_kernel buildGridKernel_ = nullptr;
    cl_kernel collideKernel_ = nullptr;
    cl_kernel updateKernel_ = nullptr;

    cl_mem d_pos_ = nullptr;
    cl_mem d_vel_ = nullptr;
    cl_mem d_life_ = nullptr;
    cl_mem d_type_ = nullptr;
    cl_mem d_rng_ = nullptr;
    cl_mem d_cellCount_ = nullptr;
    cl_mem d_cellParticles_ = nullptr;
    cl_mem d_score_ = nullptr;

    std::string deviceName_;
    std::string lastError_;

    bool initializeOpenCL();
    bool buildOpenCLProgram();
    bool createBuffers();
    void releaseAll();
    bool setError(const std::string& message);
};
