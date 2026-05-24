#pragma once

#include <cmath>
#include <algorithm>

constexpr int MAX_PLANES = 4;
constexpr int BOUNCE = 0;
constexpr int ROLL = 1;
constexpr int MAX_PARTICLES_PER_CELL = 64;

struct float3 {
    float x, y, z;
};

struct int3 {
    int x, y, z;
};

inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
inline int3 make_int3(int x, int y, int z) { return {x, y, z}; }

inline float3 operator+(float3 a, float3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline float3 operator-(float3 a, float3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline float3 operator*(float3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float3 operator/(float3 a, float s) { return {a.x / s, a.y / s, a.z / s}; }
inline float dot3(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float length3(float3 a) { return std::sqrt(dot3(a, a)); }
inline float3 normalize3(float3 a) { float l = length3(a); return l < 1e-6f ? make_float3(0,1,0) : a / l; }
inline float3 cross3(float3 a, float3 b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

struct Box {
    float3 center;
    float3 halfSize;
};

struct Plane {
    float3 point;      
    float3 normal;
    float radius = 5.0f;
};

struct Basket {
    float3 minCorner;
    float3 maxCorner;
};

struct Emitter {
    float3 center;
    float3 size;
    float3 baseVelocity;
};

struct SceneObjects {
    Plane planes[MAX_PLANES];
    int planeCount = 0;
    Basket basket;
};

struct SimulationSettings {
    int particleCount = 20000;
    int targetScore = 1200;
    float gameDuration = 15.0f;
    float particleRadius = 0.08f;
    float worldMinY = -18.0f;
    float damping = 0.98f;
    float rollingFriction = 0.985f;
    float gravity = 9.8f;
};

struct GridInfo {
    float3 origin;
    float cellSize;
    int3 dims;
};
