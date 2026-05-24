#pragma once

#include <chrono>
#include <string>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include "Geometry.h"
#include "Particles.h"
#include "Renderer.h"

class Game {
public:
    Game();
    ~Game();

    bool initialize(HWND hwnd, int width, int height);
    void frame();
    void resize(HWND hwnd);
    void shutdown();
    bool isRunning() const { return running_; }
    void onMouseWheel(int delta);

private:
    enum class SelectedObject {
        Plane0,
        Plane1
    };

    struct PlaneConstraints {
        float minX = -8.0f, maxX = 8.0f;
        float minY = -12.0f, maxY = 4.0f;
        float minZ = -8.0f, maxZ = 8.0f;
        float minNormalY = 0.1f;   
        float maxNormalY = 1.5f;    
        float minNormalXZ = -1.0f;
        float maxNormalXZ = 1.0f;
    };

    PlaneConstraints planeConstraints_;

    void clampPlanePosition(Plane& plane);
    void clampPlaneNormal(Plane& plane);

    SimulationSettings settings_{};
    GridInfo grid_{};
    Emitter emitter_{};
    SceneObjects scene_{};
    ParticleSystem particles_;
    Renderer renderer_;

    SelectedObject selected_ = SelectedObject::Plane0;
    bool running_ = true;
    bool paused_ = false;
    bool previousPDown_ = false;
    bool initialized_ = false;

    std::chrono::high_resolution_clock::time_point startTime_{};
    std::chrono::high_resolution_clock::time_point lastFrameTime_{};

    void handleInput(float dt);
    void applyMove(float dx, float dy, float dz);
    void rotateSelectedPlane(float nxDelta, float nzDelta);
    Plane& selectedPlane();
    const Plane& selectedPlane() const;
    void updateWindowTitle(float elapsedSeconds, int score);
    std::string selectedName() const;

    void handleCameraInput(float dt);
};
