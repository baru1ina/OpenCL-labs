#include "game.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <windows.h> 

constexpr int KEY_UP = VK_UP;
constexpr int KEY_DOWN = VK_DOWN;
constexpr int KEY_LEFT = VK_LEFT;
constexpr int KEY_RIGHT = VK_RIGHT;
constexpr int KEY_NUMPAD_PLUS = VK_ADD;
constexpr int KEY_NUMPAD_MINUS = VK_SUBTRACT;
constexpr int KEY_C = 'C'; 

namespace {

float3 make3(float x, float y, float z) {
    return make_float3(x, y, z);
}

float lengthCpu(const float3& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float3 normalizeCpu(float3 v) {
    float len = lengthCpu(v);
    if (len < 1e-6f) {
        return make3(0.0f, 1.0f, 0.0f);
    }
    return make3(v.x / len, v.y / len, v.z / len);
}

SimulationSettings makeSettings() {
    SimulationSettings s{};
    s.particleCount = 20000;
    s.targetScore = 1200;
    s.gameDuration = 15.0f;
    s.particleRadius = 0.105f;
    s.worldMinY = -18.0f;
    s.damping = 0.98f;
    s.rollingFriction = 0.5f;
    s.gravity = 9.8f;
    return s;
}

GridInfo makeGrid() {
    GridInfo grid{};
    grid.origin = make3(-16.0f, -20.0f, -16.0f);
    grid.cellSize = 1.0f;
    grid.dims = make_int3(32, 32, 32);
    return grid;
}

Emitter makeEmitter() {
    Emitter e{};
    e.center = make3(0.0f, 9.5f, 0.0f);
    e.size = make3(4.8f, 0.35f, 4.8f);
    e.baseVelocity = make3(0.0f, -8.0f, 0.0f);
    return e;
}

SceneObjects makeScene() {
    SceneObjects scene{};
    scene.planeCount = 2;

    scene.planes[0].point = make3(0.0f, -1.0f, 0.0f);
    scene.planes[0].normal = normalizeCpu(make3(0.45f, 1.0f, 0.15f));
    scene.planes[0].radius = 6.0f;

    scene.planes[1].point = make3(-4.0f, -5.0f, 0.0f);
    scene.planes[1].normal = normalizeCpu(make3(0.25f, 1.0f, 0.45f));
    scene.planes[1].radius = 5.0f;  
    scene.basket.minCorner = make3(-2.0f, -16.0f, -2.0f);
    scene.basket.maxCorner = make3(2.0f, -14.0f, 2.0f);

    return scene;
}

}

Game::Game()
    : settings_(makeSettings()),
    grid_(makeGrid()),
    emitter_(makeEmitter()),
    scene_(makeScene()),
    particles_(settings_.particleCount, grid_) {

    planeConstraints_.minX = -10.0f;
    planeConstraints_.maxX = 10.0f;
    planeConstraints_.minY = -14.0f; 
    planeConstraints_.maxY = 6.0f;
    planeConstraints_.minZ = -10.0f;
    planeConstraints_.maxZ = 10.0f;

    planeConstraints_.minNormalY = 0.15f;  
    planeConstraints_.maxNormalY = 1.2f; 
}

Game::~Game() {
    shutdown();
}

bool Game::initialize(HWND hwnd, int width, int height) {

    if (!GetConsoleWindow()) {
        AllocConsole();
        FILE* f;
        freopen_s(&f, "CONOUT$", "w", stdout);
        freopen_s(&f, "CONOUT$", "w", stderr);
        std::cout << "Debug console initialized. Press F1 for camera info." << std::endl;
    }

    if (!renderer_.initialize(hwnd, width, height)) {
        return false;
    }

    //renderer_.setCameraPosition(15.0f, 0.0f, -25.0f);
    renderer_.setCameraPosition(-8.3f, 7.9f, 36.4f);

    std::string openclError;
    if (!particles_.initialize(emitter_, 1337UL, &openclError)) {
        std::string message = "OpenCL initialization failed:\n\n" + openclError;
        MessageBoxA(hwnd, message.c_str(), "particles", MB_OK | MB_ICONERROR);
        return false;
    }

    startTime_ = std::chrono::high_resolution_clock::now();
    lastFrameTime_ = startTime_;
    initialized_ = true;
    return true;
}

void Game::frame() {
    if (!initialized_ || !running_) {
        return;
    }

    using clock = std::chrono::high_resolution_clock;
    auto now = clock::now();
    std::chrono::duration<float> frameDelta = now - lastFrameTime_;
    lastFrameTime_ = now;

    constexpr float maxDt = 1.0f / 30.0f;
    float dt = std::min(frameDelta.count(), maxDt);
    float elapsed = std::chrono::duration<float>(now - startTime_).count();

    handleInput(dt);

    if (!paused_) {
        particles_.step(dt, scene_, settings_, emitter_);
    }

    int score = particles_.getScore();
    auto snapshot = particles_.downloadParticles(settings_.particleCount);

    renderer_.render(scene_, emitter_, snapshot, settings_.particleRadius);
    updateWindowTitle(elapsed, score);

    if (score >= settings_.targetScore) {
        renderer_.setTitle("Victory! Basket filled. Press Esc to exit.");
        paused_ = true;
    }

    if (elapsed >= settings_.gameDuration) {
        renderer_.setTitle("Time is over. Press Esc to exit.");
        paused_ = true;
    }
}

void Game::resize(HWND hwnd) {
    renderer_.resize(hwnd);
}

void Game::shutdown() {
    if (initialized_) {
        renderer_.terminate();
        initialized_ = false;
    }
}

void Game::applyMove(float dx, float dy, float dz) {
    Plane& plane = selectedPlane();
    plane.point.x += dx;
    plane.point.y += dy;
    plane.point.z += dz;
    clampPlanePosition(plane); 
}

void Game::rotateSelectedPlane(float nxDelta, float nzDelta) {
    Plane& plane = selectedPlane();
    plane.normal.x += nxDelta;
    plane.normal.z += nzDelta;
    clampPlaneNormal(plane); 
}

void Game::updateWindowTitle(float elapsedSeconds, int score) {
    float timeLeft = std::max(0.0f, settings_.gameDuration - elapsedSeconds);
    std::ostringstream oss;
    oss << "(Babakhina) particle-game OpenCL | Time: " << static_cast<int>(timeLeft + 0.5f)
        << "s | Score: " << score << "/" << settings_.targetScore
        << " | Selected: " << selectedName()
        << " | " << (paused_ ? "paused" : "running");
    renderer_.setTitle(oss.str());
}

Plane& Game::selectedPlane() {
    return scene_.planes[selected_ == SelectedObject::Plane0 ? 0 : 1];
}

const Plane& Game::selectedPlane() const {
    return scene_.planes[selected_ == SelectedObject::Plane0 ? 0 : 1];
}

std::string Game::selectedName() const {
    switch (selected_) {
        case SelectedObject::Plane0: return "plane 1";
        case SelectedObject::Plane1: return "plane 2";
    }
    return "unknown";
}

void Game::handleInput(float dt) {
    if (renderer_.isKeyDown(KEY_ESCAPE)) {
        running_ = false;
        PostQuitMessage(0);
        return;
    }

    if (renderer_.isKeyDown(KEY_1)) selected_ = SelectedObject::Plane0;
    if (renderer_.isKeyDown(KEY_2)) selected_ = SelectedObject::Plane1;

    bool pDown = renderer_.isKeyDown(KEY_P);
    if (pDown && !previousPDown_) {
        paused_ = !paused_;
    }
    previousPDown_ = pDown;

    float moveSpeed = 6.0f;
    float tiltSpeed = 1.6f;

    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;

    if (renderer_.isKeyDown(KEY_A)) dx -= moveSpeed * dt;
    if (renderer_.isKeyDown(KEY_D)) dx += moveSpeed * dt;
    if (renderer_.isKeyDown(KEY_Q)) dy += moveSpeed * dt;
    if (renderer_.isKeyDown(KEY_E)) dy -= moveSpeed * dt;
    if (renderer_.isKeyDown(KEY_W)) dz -= moveSpeed * dt;
    if (renderer_.isKeyDown(KEY_S)) dz += moveSpeed * dt;

    if (dx != 0.0f || dy != 0.0f || dz != 0.0f) {
        applyMove(dx, dy, dz);
    }

    if (renderer_.isKeyDown(KEY_R)) rotateSelectedPlane(tiltSpeed * dt, 0.0f);
    if (renderer_.isKeyDown(KEY_F)) rotateSelectedPlane(-tiltSpeed * dt, 0.0f);
    if (renderer_.isKeyDown(KEY_T)) rotateSelectedPlane(0.0f, tiltSpeed * dt);
    if (renderer_.isKeyDown(KEY_G)) rotateSelectedPlane(0.0f, -tiltSpeed * dt);

    handleCameraInput(dt);
}

void Game::handleCameraInput(float dt) {
    (void)dt; 

    static bool rightButtonWasPressed = false;
    static int lastMouseX = 0, lastMouseY = 0;

    bool rightButtonPressed = renderer_.isKeyDown(VK_RBUTTON);

    if (rightButtonPressed && !rightButtonWasPressed) {
        SetCapture(renderer_.getHWND());
        ShowCursor(FALSE);
        POINT mousePos;
        GetCursorPos(&mousePos);
        ScreenToClient(renderer_.getHWND(), &mousePos);
        lastMouseX = mousePos.x;
        lastMouseY = mousePos.y;
    }
    else if (!rightButtonPressed && rightButtonWasPressed) {
        ReleaseCapture();
        ShowCursor(TRUE);
    }

    rightButtonWasPressed = rightButtonPressed;

    if (rightButtonPressed) {
        POINT mousePos;
        GetCursorPos(&mousePos);
        ScreenToClient(renderer_.getHWND(), &mousePos);

        int deltaX = mousePos.x - lastMouseX;
        int deltaY = mousePos.y - lastMouseY;

        if (deltaX != 0 || deltaY != 0) {
            const float mouseSensitivity = 0.003f;
            renderer_.rotateCamera(deltaX * mouseSensitivity, -deltaY * mouseSensitivity);
        }

        lastMouseX = mousePos.x;
        lastMouseY = mousePos.y;
    }

    static bool wasF1Pressed = false;
    if (renderer_.isKeyDown(VK_F1)) {
        if (!wasF1Pressed) {
            std::cout << "\n=== Camera Debug Info ===" << std::endl;
            std::cout << "Position: (" << renderer_.getCameraX() << ", "
                << renderer_.getCameraY() << ", " << renderer_.getCameraZ() << ")" << std::endl;
            wasF1Pressed = true;
        }
    }
    else {
        wasF1Pressed = false;
    }

    static bool wasCPressed = false;
    if (renderer_.isKeyDown(KEY_C)) {
        if (!wasCPressed) {
            renderer_.resetCamera();
            wasCPressed = true;
        }
    }
    else {
        wasCPressed = false;
    }
}

void Game::onMouseWheel(int delta) {
    if (!initialized_) return;
    float wheelSpeed = 0.6f; 
    float step = (delta > 0) ? wheelSpeed : -wheelSpeed;

    float lr = renderer_.getLRAngle();
    float ud = renderer_.getUDAngle();

    float forwardX = cosf(ud) * sinf(lr);
    float forwardY = sinf(ud);
    float forwardZ = cosf(ud) * cosf(lr);

    renderer_.moveCamera(forwardX * step, forwardY * step, forwardZ * step);
}

void Game::clampPlanePosition(Plane& plane) {
    plane.point.x = std::clamp(plane.point.x,
        planeConstraints_.minX,
        planeConstraints_.maxX);
    plane.point.y = std::clamp(plane.point.y,
        planeConstraints_.minY,
        planeConstraints_.maxY);
    plane.point.z = std::clamp(plane.point.z,
        planeConstraints_.minZ,
        planeConstraints_.maxZ);
}

void Game::clampPlaneNormal(Plane& plane) {
    plane.normal.x = std::clamp(plane.normal.x,
        planeConstraints_.minNormalXZ,
        planeConstraints_.maxNormalXZ);
    plane.normal.z = std::clamp(plane.normal.z,
        planeConstraints_.minNormalXZ,
        planeConstraints_.maxNormalXZ);

    plane.normal.y = std::clamp(plane.normal.y,
        planeConstraints_.minNormalY,
        planeConstraints_.maxNormalY);

    plane.normal = normalizeCpu(plane.normal);
}