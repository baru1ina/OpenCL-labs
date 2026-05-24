#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>  

#include <string>
#include <vector>

#include "Geometry.h"
#include "Particles.h"

enum KeyCode {
    KEY_ESCAPE = VK_ESCAPE,
    KEY_1 = '1',
    KEY_2 = '2',
    KEY_W = 'W',
    KEY_A = 'A',
    KEY_S = 'S',
    KEY_D = 'D',
    KEY_Q = 'Q',
    KEY_E = 'E',
    KEY_R = 'R',
    KEY_F = 'F',
    KEY_T = 'T',
    KEY_G = 'G',
    KEY_P = 'P'
};

class Renderer {
public:
    Renderer() = default;
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    bool initialize(HWND hwnd, int width, int height);
    void terminate();
    void resize(HWND hwnd);

    bool isKeyDown(int virtualKey) const;
    void setTitle(const std::string& title);

    void render(
        const SceneObjects& scene,
        const Emitter& emitter,
        const std::vector<ParticleSnapshot>& particles,
        float particleRadius
    );

    void moveCamera(float dx, float dy, float dz);
    void rotateCamera(float yaw, float pitch); 
    void setCameraPosition(float x, float y, float z);
    void resetCamera();

    HWND getHWND() const { return hwnd_; }

    float getLRAngle() const { return lrAngle_; }
    float getUDAngle() const { return udAngle_; }
    float getCameraX() const { return cameraPosition_.x; }
    float getCameraY() const { return cameraPosition_.y; }
    float getCameraZ() const { return cameraPosition_.z; }

private:
    HWND hwnd_ = nullptr;
    int width_ = 1280;
    int height_ = 720;

    ID3D11Device* device_ = nullptr;
    ID3D11DeviceContext* context_ = nullptr;
    IDXGISwapChain* swapChain_ = nullptr;
    ID3D11RenderTargetView* renderTargetView_ = nullptr;
    ID3D11Texture2D* depthStencilTexture_ = nullptr;
    ID3D11DepthStencilView* depthStencilView_ = nullptr;

    ID3D11InputLayout* inputLayout_ = nullptr;
    ID3D11VertexShader* solidVertexShader_ = nullptr;
    ID3D11VertexShader* particleVertexShader_ = nullptr;
    ID3D11GeometryShader* particleGeometryShader_ = nullptr;
    ID3D11PixelShader* pixelShader_ = nullptr;

    ID3D11Buffer* constantBuffer_ = nullptr;
    ID3D11Buffer* particleVertexBuffer_ = nullptr;
    ID3D11Buffer* lineVertexBuffer_ = nullptr;
    size_t particleVertexCapacity_ = 0;
    size_t lineVertexCapacity_ = 0;

    ID3D11RasterizerState* rasterizerState_ = nullptr;
    ID3D11DepthStencilState* depthState_ = nullptr;
    ID3D11BlendState* blendState_ = nullptr;

    DirectX::XMFLOAT3 cameraPosition_;
    float cameraSpeed_ = 3.0f;
    float lrAngle_ = 3.0f;
    float udAngle_ = -0.28f;  

    bool createDeviceAndSwapChain();
    bool createRenderTargets();
    bool initBufferShader();
    bool createStates();
    bool createConstantBuffer();
    void releaseRenderTargets();
    void cleanup();

    bool compileShaderFromFile(const std::wstring& path, const char* target, ID3DBlob** outBlob) const;

    void drawParticles(const std::vector<ParticleSnapshot>& particles, float particleRadius);
    void drawSceneObjects(const SceneObjects& scene, const Emitter& emitter);

    bool ensureDynamicVertexBuffer(ID3D11Buffer** buffer, size_t& capacity, size_t requiredBytes);
    bool uploadDynamicVertexBuffer(ID3D11Buffer* buffer, const void* data, size_t bytes);

};
