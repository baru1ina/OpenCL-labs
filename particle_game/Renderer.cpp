#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "renderer.h"
#include <dxgi.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

using namespace DirectX;

void Renderer::moveCamera(float dx, float dy, float dz) {
    cameraPosition_.x += dx * cameraSpeed_;
    cameraPosition_.y += dy * cameraSpeed_;
    cameraPosition_.z += dz * cameraSpeed_;

    //cameraPosition_.x = std::clamp(cameraPosition_.x, -30.0f, 30.0f);
    //cameraPosition_.y = std::clamp(cameraPosition_.y, -20.0f, 20.0f);
    //cameraPosition_.z = std::clamp(cameraPosition_.z, -40.0f, 40.0f);
}

void Renderer::rotateCamera(float yaw, float pitch) {
    lrAngle_ += yaw;
    udAngle_ += pitch;

    if (lrAngle_ > XM_2PI) lrAngle_ -= XM_2PI;
    if (lrAngle_ < -XM_2PI) lrAngle_ += XM_2PI;

    if (udAngle_ > XM_PIDIV2) udAngle_ = XM_PIDIV2;
    if (udAngle_ < -XM_PIDIV2) udAngle_ = -XM_PIDIV2;
}

void Renderer::setCameraPosition(float x, float y, float z) {
    cameraPosition_ = XMFLOAT3(x, y, z);
}

void Renderer::resetCamera() {
    cameraPosition_.x = 0.0f;
    cameraPosition_.y = 0.0f;
    cameraPosition_.z = 0.0f;
    lrAngle_ = 0.0f;
    udAngle_ = 0.0f;
    cameraSpeed_ = 3.0f;
}

namespace {

constexpr float kPi = 3.14159265358979323846f;

struct DxVertex {
    float x, y, z;
    float r, g, b, a;
};

struct SceneConstantBuffer {
    XMMATRIX mvp;
    XMFLOAT2 invViewport;
    float pointSize;
    float padding;
};

template <typename T>
void safeRelease(T*& ptr) {
    if (ptr) {
        ptr->Release();
        ptr = nullptr;
    }
}

float3 make3(float x, float y, float z) {
    return make_float3(x, y, z);
}

float dotCpu(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float3 crossCpu(float3 a, float3 b) {
    return make3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

float lengthCpu(float3 v) {
    return std::sqrt(dotCpu(v, v));
}

float3 normalizeCpu(float3 v) {
    float len = lengthCpu(v);
    if (len < 1e-6f) {
        return make3(0.0f, 1.0f, 0.0f);
    }
    return make3(v.x / len, v.y / len, v.z / len);
}

DxVertex makeVertex(float3 p, float r, float g, float b, float a = 1.0f) {
    return DxVertex{p.x, p.y, p.z, r, g, b, a};
}

void addLine(std::vector<DxVertex>& lines, float3 a, float3 b, float r, float g, float bl) {
    lines.push_back(makeVertex(a, r, g, bl));
    lines.push_back(makeVertex(b, r, g, bl));
}

std::string hresultToString(HRESULT hr) {
    std::ostringstream oss;
    oss << "HRESULT 0x" << std::hex << static_cast<unsigned long>(hr);
    return oss.str();
}


std::wstring shaderPath(const wchar_t* fileName) {
    return std::wstring(fileName);
}

} 

Renderer::~Renderer() {
    cleanup();
}

bool Renderer::initialize(HWND hwnd, int width, int height) {
    hwnd_ = hwnd;
    width_ = width;
    height_ = height;

    if (!createDeviceAndSwapChain()) {
        return false;
    }
    if (!createRenderTargets()) {
        return false;
    }
    if (!initBufferShader()) {
        return false;
    }
    if (!createStates()) {
        return false;
    }
    if (!createConstantBuffer()) {
        return false;
    }
    return true;
}

bool Renderer::createDeviceAndSwapChain() {
    HRESULT result = S_OK;

    IDXGIFactory* factory = nullptr;
    result = CreateDXGIFactory(__uuidof(IDXGIFactory), reinterpret_cast<void**>(&factory));
    if (FAILED(result)) {
        std::cerr << "CreateDXGIFactory failed: " << hresultToString(result) << "\n";
        return false;
    }

    IDXGIAdapter* selectedAdapter = nullptr;
    IDXGIAdapter* adapter = nullptr;
    UINT adapterIndex = 0;
    while (factory->EnumAdapters(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND) {
        DXGI_ADAPTER_DESC desc{};
        adapter->GetDesc(&desc);
        if (wcscmp(desc.Description, L"Microsoft Basic Render Driver") != 0) {
            selectedAdapter = adapter;
            break;
        }
        adapter->Release();
        adapter = nullptr;
        ++adapterIndex;
    }

    UINT flags = 0;
#ifdef _DEBUG
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_FEATURE_LEVEL level{};
    D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0 };

    result = D3D11CreateDevice(
        selectedAdapter,
        selectedAdapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        flags,
        levels,
        1,
        D3D11_SDK_VERSION,
        &device_,
        &level,
        &context_
    );

#ifdef _DEBUG
    if (FAILED(result) && (flags & D3D11_CREATE_DEVICE_DEBUG)) {
        flags &= ~D3D11_CREATE_DEVICE_DEBUG;
        result = D3D11CreateDevice(
            selectedAdapter,
            selectedAdapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            flags,
            levels,
            1,
            D3D11_SDK_VERSION,
            &device_,
            &level,
            &context_
        );
    }
#endif

    if (SUCCEEDED(result)) {
        DXGI_SWAP_CHAIN_DESC swapDesc{};
        swapDesc.BufferCount = 2;
        swapDesc.BufferDesc.Width = static_cast<UINT>(std::max(1, width_));
        swapDesc.BufferDesc.Height = static_cast<UINT>(std::max(1, height_));
        swapDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        swapDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swapDesc.OutputWindow = hwnd_;
        swapDesc.SampleDesc.Count = 1;
        swapDesc.SampleDesc.Quality = 0;
        swapDesc.Windowed = TRUE;
        swapDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
        result = factory->CreateSwapChain(device_, &swapDesc, &swapChain_);
    }

    if (selectedAdapter) selectedAdapter->Release();
    if (factory) factory->Release();

    if (FAILED(result)) {
        std::cerr << "Direct3D device/swapchain initialization failed: " << hresultToString(result) << "\n";
        cleanup();
        return false;
    }

    return true;
}

bool Renderer::createRenderTargets() {
    ID3D11Texture2D* backBuffer = nullptr;
    HRESULT hr = swapChain_->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&backBuffer));
    if (FAILED(hr)) {
        std::cerr << "SwapChain GetBuffer failed: " << hresultToString(hr) << "\n";
        return false;
    }

    hr = device_->CreateRenderTargetView(backBuffer, nullptr, &renderTargetView_);
    safeRelease(backBuffer);
    if (FAILED(hr)) {
        std::cerr << "CreateRenderTargetView failed: " << hresultToString(hr) << "\n";
        return false;
    }

    D3D11_TEXTURE2D_DESC depthDesc{};
    depthDesc.Width = static_cast<UINT>(std::max(1, width_));
    depthDesc.Height = static_cast<UINT>(std::max(1, height_));
    depthDesc.MipLevels = 1;
    depthDesc.ArraySize = 1;
    depthDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    depthDesc.SampleDesc.Count = 1;
    depthDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;

    hr = device_->CreateTexture2D(&depthDesc, nullptr, &depthStencilTexture_);
    if (FAILED(hr)) {
        std::cerr << "Create depth texture failed: " << hresultToString(hr) << "\n";
        return false;
    }

    hr = device_->CreateDepthStencilView(depthStencilTexture_, nullptr, &depthStencilView_);
    if (FAILED(hr)) {
        std::cerr << "CreateDepthStencilView failed: " << hresultToString(hr) << "\n";
        return false;
    }

    D3D11_VIEWPORT vp{};
    vp.Width = static_cast<FLOAT>(std::max(1, width_));
    vp.Height = static_cast<FLOAT>(std::max(1, height_));
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0.0f;
    vp.TopLeftY = 0.0f;
    context_->RSSetViewports(1, &vp);

    return true;
}

bool Renderer::compileShaderFromFile(const std::wstring& path, const char* target, ID3DBlob** outBlob) const {
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_PACK_MATRIX_ROW_MAJOR;
#ifdef _DEBUG
    flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

    ID3DBlob* errors = nullptr;
    HRESULT hr = D3DCompileFromFile(path.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
                                   "main", target, flags, 0, outBlob, &errors);
    if (FAILED(hr)) {
        if (errors) {
            OutputDebugStringA(static_cast<const char*>(errors->GetBufferPointer()));
            std::cerr << static_cast<const char*>(errors->GetBufferPointer()) << "\n";
            errors->Release();
        }
        std::wcerr << L"D3DCompileFromFile failed for " << path << L"\n";
        return false;
    }
    if (errors) errors->Release();
    return true;
}

bool Renderer::initBufferShader() {
    static const D3D11_INPUT_ELEMENT_DESC inputDesc[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0}
    };

    ID3DBlob* solidVSCode = nullptr;
    if (!compileShaderFromFile(shaderPath(L"SolidVertex.vs"), "vs_5_0", &solidVSCode)) return false;

    HRESULT hr = device_->CreateVertexShader(solidVSCode->GetBufferPointer(), solidVSCode->GetBufferSize(), nullptr, &solidVertexShader_);
    if (FAILED(hr)) {
        safeRelease(solidVSCode);
        std::cerr << "CreateVertexShader SolidVertex failed: " << hresultToString(hr) << "\n";
        return false;
    }

    hr = device_->CreateInputLayout(inputDesc, 2, solidVSCode->GetBufferPointer(), solidVSCode->GetBufferSize(), &inputLayout_);
    safeRelease(solidVSCode);
    if (FAILED(hr)) {
        std::cerr << "CreateInputLayout failed: " << hresultToString(hr) << "\n";
        return false;
    }

    ID3DBlob* particleVSCode = nullptr;
    if (!compileShaderFromFile(shaderPath(L"ParticleVertex.vs"), "vs_5_0", &particleVSCode)) return false;
    hr = device_->CreateVertexShader(particleVSCode->GetBufferPointer(), particleVSCode->GetBufferSize(), nullptr, &particleVertexShader_);
    safeRelease(particleVSCode);
    if (FAILED(hr)) {
        std::cerr << "CreateVertexShader ParticleVertex failed: " << hresultToString(hr) << "\n";
        return false;
    }

    ID3DBlob* particleGSCode = nullptr;
    if (!compileShaderFromFile(shaderPath(L"ParticleGeometry.gs"), "gs_5_0", &particleGSCode)) return false;
    hr = device_->CreateGeometryShader(particleGSCode->GetBufferPointer(), particleGSCode->GetBufferSize(), nullptr, &particleGeometryShader_);
    safeRelease(particleGSCode);
    if (FAILED(hr)) {
        std::cerr << "CreateGeometryShader ParticleGeometry failed: " << hresultToString(hr) << "\n";
        return false;
    }

    ID3DBlob* pixelCode = nullptr;
    if (!compileShaderFromFile(shaderPath(L"ColorPixel.ps"), "ps_5_0", &pixelCode)) return false;
    hr = device_->CreatePixelShader(pixelCode->GetBufferPointer(), pixelCode->GetBufferSize(), nullptr, &pixelShader_);
    safeRelease(pixelCode);
    if (FAILED(hr)) {
        std::cerr << "CreatePixelShader ColorPixel failed: " << hresultToString(hr) << "\n";
        return false;
    }

    return true;
}

void Renderer::resize(HWND hwnd) {
    if (!swapChain_ || !context_ || !hwnd) {
        return;
    }

    RECT rc{};
    GetClientRect(hwnd, &rc);
    width_ = std::max<LONG>(1, rc.right - rc.left);
    height_ = std::max<LONG>(1, rc.bottom - rc.top);

    context_->OMSetRenderTargets(0, nullptr, nullptr);
    releaseRenderTargets();

    HRESULT hr = swapChain_->ResizeBuffers(2, static_cast<UINT>(width_), static_cast<UINT>(height_), DXGI_FORMAT_R8G8B8A8_UNORM, 0);
    if (FAILED(hr)) {
        MessageBoxW(hwnd, L"ResizeBuffers failed", L"DirectX error", MB_OK | MB_ICONERROR);
        return;
    }

    if (!createRenderTargets()) {
        MessageBoxW(hwnd, L"Back buffer reconfiguration failed", L"DirectX error", MB_OK | MB_ICONERROR);
    }
}

bool Renderer::createStates() {
    D3D11_RASTERIZER_DESC rasterDesc{};
    rasterDesc.FillMode = D3D11_FILL_SOLID;
    rasterDesc.CullMode = D3D11_CULL_NONE;
    rasterDesc.DepthClipEnable = TRUE;

    HRESULT hr = device_->CreateRasterizerState(&rasterDesc, &rasterizerState_);
    if (FAILED(hr)) {
        std::cerr << "CreateRasterizerState failed: " << hresultToString(hr) << "\n";
        return false;
    }

    D3D11_DEPTH_STENCIL_DESC depthDesc{};
    depthDesc.DepthEnable = TRUE;
    depthDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    depthDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;

    hr = device_->CreateDepthStencilState(&depthDesc, &depthState_);
    if (FAILED(hr)) {
        std::cerr << "CreateDepthStencilState failed: " << hresultToString(hr) << "\n";
        return false;
    }

    D3D11_BLEND_DESC blendDesc{};
    blendDesc.RenderTarget[0].BlendEnable = TRUE;
    blendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    blendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    blendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    blendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    blendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

    hr = device_->CreateBlendState(&blendDesc, &blendState_);
    if (FAILED(hr)) {
        std::cerr << "CreateBlendState failed: " << hresultToString(hr) << "\n";
        return false;
    }

    return true;
}

bool Renderer::createConstantBuffer() {
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = sizeof(SceneConstantBuffer);
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

    HRESULT hr = device_->CreateBuffer(&desc, nullptr, &constantBuffer_);
    if (FAILED(hr)) {
        std::cerr << "Create constant buffer failed: " << hresultToString(hr) << "\n";
        return false;
    }
    return true;
}

bool Renderer::isKeyDown(int virtualKey) const {
    if (!hwnd_) return false;
    return (GetAsyncKeyState(virtualKey) & 0x8000) != 0;
}

void Renderer::setTitle(const std::string& title) {
    if (hwnd_) {
        SetWindowTextA(hwnd_, title.c_str());
    }
}

void Renderer::render(
    const SceneObjects& scene,
    const Emitter& emitter,
    const std::vector<ParticleSnapshot>& particles,
    float particleRadius
) {
    if (!device_ || !context_) {
        return;
    }

    if (width_ <= 0 || height_ <= 0) {
        return;
    }

    D3D11_VIEWPORT viewport{};
    viewport.TopLeftX = 0.0f;
    viewport.TopLeftY = 0.0f;
    viewport.Width = static_cast<float>(width_);
    viewport.Height = static_cast<float>(height_);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;

    float clearColor[4] = {0.035f, 0.04f, 0.052f, 1.0f};
    context_->OMSetRenderTargets(1, &renderTargetView_, depthStencilView_);
    context_->RSSetViewports(1, &viewport);
    context_->ClearRenderTargetView(renderTargetView_, clearColor);
    context_->ClearDepthStencilView(depthStencilView_, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

    /*XMVECTOR eye = XMVectorSet(-18.0f, 9.0f, -22.0f, 1.0f);
    XMVECTOR at = XMVectorSet(0.5f, -3.0f, 1.0f, 1.0f);
    XMVECTOR up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
    XMMATRIX view = XMMatrixLookAtLH(eye, at, up);
    XMMATRIX proj = XMMatrixPerspectiveFovLH(XMConvertToRadians(50.0f),
                                             static_cast<float>(width_) / static_cast<float>(height_),
                                             0.1f, 100.0f);*/

    XMVECTOR direction = XMVectorSet(
        cosf(udAngle_) * sinf(lrAngle_),
        sinf(udAngle_),
        cosf(udAngle_) * cosf(lrAngle_),
        0.0f
    );

    XMVECTOR eyePos = XMVectorSet(cameraPosition_.x, cameraPosition_.y, cameraPosition_.z, 0.0f);
    XMVECTOR focusPoint = XMVectorAdd(eyePos, direction);
    XMVECTOR upDir = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
    XMMATRIX view = XMMatrixLookAtLH(eyePos, focusPoint, upDir);

    XMMATRIX proj = XMMatrixPerspectiveFovLH(XMConvertToRadians(50.0f),
        static_cast<float>(width_) / static_cast<float>(height_),
        0.1f, 100.0f);

    SceneConstantBuffer cb{};
    cb.mvp = view * proj;
    cb.invViewport = XMFLOAT2(2.0f / static_cast<float>(width_), 2.0f / static_cast<float>(height_));
    cb.pointSize = std::max(5.0f, particleRadius * 70.0f);

    context_->UpdateSubresource(constantBuffer_, 0, nullptr, &cb, 0, 0);
    context_->VSSetConstantBuffers(0, 1, &constantBuffer_);
    context_->GSSetConstantBuffers(0, 1, &constantBuffer_);

    context_->IASetInputLayout(inputLayout_);
    context_->RSSetState(rasterizerState_);
    context_->OMSetDepthStencilState(depthState_, 0);
    float blendFactor[4] = {0, 0, 0, 0};
    context_->OMSetBlendState(blendState_, blendFactor, 0xffffffff);

    drawSceneObjects(scene, emitter);
    drawParticles(particles, particleRadius);

    HRESULT hr = swapChain_->Present(1, 0);
    if (FAILED(hr)) {
        std::cerr << "SwapChain Present failed: " << hresultToString(hr) << "\n";
    }
}

bool Renderer::ensureDynamicVertexBuffer(ID3D11Buffer** buffer, size_t& capacity, size_t requiredBytes) {
    if (requiredBytes == 0) {
        return true;
    }

    if (*buffer && capacity >= requiredBytes) {
        return true;
    }

    safeRelease(*buffer);
    capacity = std::max(requiredBytes, capacity * 2 + sizeof(DxVertex) * 1024);

    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = static_cast<UINT>(capacity);
    desc.Usage = D3D11_USAGE_DYNAMIC;
    desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    HRESULT hr = device_->CreateBuffer(&desc, nullptr, buffer);
    if (FAILED(hr)) {
        std::cerr << "Create dynamic vertex buffer failed: " << hresultToString(hr) << "\n";
        return false;
    }
    return true;
}

bool Renderer::uploadDynamicVertexBuffer(ID3D11Buffer* buffer, const void* data, size_t bytes) {
    if (bytes == 0) {
        return true;
    }

    D3D11_MAPPED_SUBRESOURCE mapped{};
    HRESULT hr = context_->Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    if (FAILED(hr)) {
        std::cerr << "Map vertex buffer failed: " << hresultToString(hr) << "\n";
        return false;
    }

    std::memcpy(mapped.pData, data, bytes);
    context_->Unmap(buffer, 0);
    return true;
}

void Renderer::drawParticles(const std::vector<ParticleSnapshot>& particles, float particleRadius) {
    if (particles.empty()) {
        return;
    }

    std::vector<DxVertex> vertices;
    vertices.reserve(particles.size());
    for (const auto& particle : particles) {
        if (particle.type == BOUNCE) {
            //vertices.push_back(makeVertex(particle.position, 0.25f, 0.75f, 1.0f, 0.85f));
            vertices.push_back(makeVertex(particle.position, 0.8f, 0.5f, 0.9f, 0.85f));
        } else {
            vertices.push_back(makeVertex(particle.position, 0.6f, 0.9f, 0.3f, 0.85f));
            //vertices.push_back(makeVertex(particle.position, 1.0f, 0.72f, 0.25f, 0.85f));
        }
    }

    size_t bytes = vertices.size() * sizeof(DxVertex);
    if (!ensureDynamicVertexBuffer(&particleVertexBuffer_, particleVertexCapacity_, bytes)) return;
    if (!uploadDynamicVertexBuffer(particleVertexBuffer_, vertices.data(), bytes)) return;

    UINT stride = sizeof(DxVertex);
    UINT offset = 0;
    context_->IASetVertexBuffers(0, 1, &particleVertexBuffer_, &stride, &offset);
    context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
    context_->VSSetShader(particleVertexShader_, nullptr, 0);
    context_->GSSetShader(particleGeometryShader_, nullptr, 0);
    context_->PSSetShader(pixelShader_, nullptr, 0);
    context_->Draw(static_cast<UINT>(vertices.size()), 0);
    context_->GSSetShader(nullptr, nullptr, 0);
}

void Renderer::drawSceneObjects(const SceneObjects& scene, const Emitter& emitter) {
    std::vector<DxVertex> lines;
    lines.reserve(4096);

    auto addBox = [&](const Box& box, float r, float g, float b) {
        float3 c = box.center;
        float3 h = box.halfSize;
        float3 p[8] = {
            make3(c.x - h.x, c.y - h.y, c.z - h.z),
            make3(c.x + h.x, c.y - h.y, c.z - h.z),
            make3(c.x + h.x, c.y + h.y, c.z - h.z),
            make3(c.x - h.x, c.y + h.y, c.z - h.z),
            make3(c.x - h.x, c.y - h.y, c.z + h.z),
            make3(c.x + h.x, c.y - h.y, c.z + h.z),
            make3(c.x + h.x, c.y + h.y, c.z + h.z),
            make3(c.x - h.x, c.y + h.y, c.z + h.z)
        };
        int edges[12][2] = {
            {0,1}, {1,2}, {2,3}, {3,0},
            {4,5}, {5,6}, {6,7}, {7,4},
            {0,4}, {1,5}, {2,6}, {3,7}
        };
        for (auto& edge : edges) {
            addLine(lines, p[edge[0]], p[edge[1]], r, g, b);
        }
    };

    auto addPlane = [&](const Plane& plane, float size) {
        float3 n = normalizeCpu(plane.normal);
        float3 helper = std::fabs(n.y) < 0.9f ? make3(0.0f, 1.0f, 0.0f) : make3(1.0f, 0.0f, 0.0f);
        float3 u = normalizeCpu(crossCpu(helper, n));
        float3 v = normalizeCpu(crossCpu(n, u));

        float3 p0 = plane.point + u * (-size) + v * (-size);
        float3 p1 = plane.point + u * ( size) + v * (-size);
        float3 p2 = plane.point + u * ( size) + v * ( size);
        float3 p3 = plane.point + u * (-size) + v * ( size);

        addLine(lines, p0, p1, 0.78f, 0.78f, 0.82f);
        addLine(lines, p1, p2, 0.78f, 0.78f, 0.82f);
        addLine(lines, p2, p3, 0.78f, 0.78f, 0.82f);
        addLine(lines, p3, p0, 0.78f, 0.78f, 0.82f);

        constexpr int gridLines = 8;
        for (int i = -gridLines; i <= gridLines; ++i) {
            float t = size * static_cast<float>(i) / static_cast<float>(gridLines);
            addLine(lines, plane.point + u * t + v * (-size), plane.point + u * t + v * ( size), 0.32f, 0.34f, 0.38f);
            addLine(lines, plane.point + u * (-size) + v * t, plane.point + u * ( size) + v * t, 0.32f, 0.34f, 0.38f);
        }
    };

    for (int i = -16; i <= 16; ++i) {
        addLine(lines, make3(static_cast<float>(i), -16.0f, -16.0f), make3(static_cast<float>(i), -16.0f, 16.0f), 0.18f, 0.18f, 0.22f);
        addLine(lines, make3(-16.0f, -16.0f, static_cast<float>(i)), make3(16.0f, -16.0f, static_cast<float>(i)), 0.18f, 0.18f, 0.22f);
    }

    Box emitterBox{};
    emitterBox.center = emitter.center;
    emitterBox.halfSize = emitter.size * 0.5f;
    addBox(emitterBox, 0.25f, 0.55f, 1.0f);

    for (int i = 0; i < scene.planeCount && i < MAX_PLANES; ++i) {
        addPlane(scene.planes[i], i == 0 ? 7.0f : 5.0f);
    }

    Box basketBox{};
    basketBox.center = make3(
        (scene.basket.minCorner.x + scene.basket.maxCorner.x) * 0.5f,
        (scene.basket.minCorner.y + scene.basket.maxCorner.y) * 0.5f,
        (scene.basket.minCorner.z + scene.basket.maxCorner.z) * 0.5f
    );
    basketBox.halfSize = make3(
        (scene.basket.maxCorner.x - scene.basket.minCorner.x) * 0.5f,
        (scene.basket.maxCorner.y - scene.basket.minCorner.y) * 0.5f,
        (scene.basket.maxCorner.z - scene.basket.minCorner.z) * 0.5f
    );
    addBox(basketBox, 1.0f, 0.2f, 0.25f);

    if (lines.empty()) {
        return;
    }

    size_t bytes = lines.size() * sizeof(DxVertex);
    if (!ensureDynamicVertexBuffer(&lineVertexBuffer_, lineVertexCapacity_, bytes)) return;
    if (!uploadDynamicVertexBuffer(lineVertexBuffer_, lines.data(), bytes)) return;

    UINT stride = sizeof(DxVertex);
    UINT offset = 0;
    context_->IASetVertexBuffers(0, 1, &lineVertexBuffer_, &stride, &offset);
    context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
    context_->VSSetShader(solidVertexShader_, nullptr, 0);
    context_->GSSetShader(nullptr, nullptr, 0);
    context_->PSSetShader(pixelShader_, nullptr, 0);
    context_->Draw(static_cast<UINT>(lines.size()), 0);
}

void Renderer::releaseRenderTargets() {
    safeRelease(renderTargetView_);
    safeRelease(depthStencilView_);
    safeRelease(depthStencilTexture_);
}

void Renderer::terminate() {
    cleanup();
}

void Renderer::cleanup() {
    if (context_) {
        context_->OMSetRenderTargets(0, nullptr, nullptr);
        context_->ClearState();
        context_->Flush();
    }

    safeRelease(particleVertexBuffer_);
    safeRelease(lineVertexBuffer_);
    safeRelease(constantBuffer_);
    safeRelease(blendState_);
    safeRelease(depthState_);
    safeRelease(rasterizerState_);
    safeRelease(pixelShader_);
    safeRelease(particleGeometryShader_);
    safeRelease(particleVertexShader_);
    safeRelease(solidVertexShader_);
    safeRelease(inputLayout_);
    releaseRenderTargets();
    safeRelease(swapChain_);
    safeRelease(context_);
    safeRelease(device_);

    hwnd_ = nullptr;
}

