#define BOUNCE 0
#define ROLL 1
#ifndef MAX_PARTICLES_PER_CELL
#define MAX_PARTICLES_PER_CELL 64
#endif
#ifndef MAX_PLANES
#define MAX_PLANES 4
#endif

typedef struct { int x; int y; int z; int w; } Int4;

uint xorshift(uint* state) {
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

float rand01(uint* state) {
    return (float)(xorshift(state) & 0x00ffffff) / 16777216.0f;
}

float4 respawnPos(uint* rng, float4 center, float4 size) {
    float rx = rand01(rng) - 0.5f;
    float ry = rand01(rng) - 0.5f;
    float rz = rand01(rng) - 0.5f;
    return (float4)(center.x + rx * size.x, center.y + ry * size.y, center.z + rz * size.z, 0.0f);
}

float4 respawnVel(uint* rng, float4 baseVel) {
    return (float4)(baseVel.x + (rand01(rng) - 0.5f) * 0.65f,
                    baseVel.y,
                    baseVel.z + (rand01(rng) - 0.5f) * 0.65f,
                    0.0f);
}

float4 reflectVelocity(float4 v, float4 n, float damping) {
    return (v - 2.0f * dot(v.xyz, n.xyz) * n) * damping;
}

float4 slideVelocity(float4 v, float4 n, float friction) {
    float4 normalPart = dot(v.xyz, n.xyz) * n;
    float4 tangentPart = v - normalPart;
    return tangentPart * friction;
}

int3 getCellCoord(float4 p, float4 origin, float cellSize) {
    return (int3)((int)floor((p.x - origin.x) / cellSize),
                  (int)floor((p.y - origin.y) / cellSize),
                  (int)floor((p.z - origin.z) / cellSize));
}

int getCellIdCoord(int3 c, Int4 dims) {
    if (c.x < 0 || c.y < 0 || c.z < 0 || c.x >= dims.x || c.y >= dims.y || c.z >= dims.z) return -1;
    return (c.z * dims.y + c.y) * dims.x + c.x;
}

int getCellId(float4 p, float4 origin, float cellSize, Int4 dims) {
    return getCellIdCoord(getCellCoord(p, origin, cellSize), dims);
}

__kernel void initParticles(
    __global float4* pos,
    __global float4* vel,
    __global float* life,
    __global int* type,
    __global uint* rng,
    int n,
    uint seed,
    float4 emitterCenter,
    float4 emitterSize,
    float4 emitterVelocity)
{
    int i = get_global_id(0);
    if (i >= n) return;

    uint state = seed ^ (uint)(i * 747796405u + 2891336453u);
    state = xorshift(&state);

    pos[i] = respawnPos(&state, emitterCenter, emitterSize);
    vel[i] = respawnVel(&state, emitterVelocity);
    
    float lifetime = 5.0f + rand01(&state) * 4.0f;
    life[i] = lifetime;

    float age = rand01(&state) * lifetime;
    life[i] = lifetime - age;
    
    type[i] = (i & 1) == 0 ? BOUNCE : ROLL;
    rng[i] = state;
}

__kernel void clearGrid(__global int* cellCount, int cellTotal) {
    int i = get_global_id(0);
    if (i < cellTotal) cellCount[i] = 0;
}

__kernel void buildGrid(
    __global const float4* pos,
    int n,
    __global int* cellCount,
    __global int* cellParticles,
    float4 origin,
    float cellSize,
    Int4 dims)
{
    int i = get_global_id(0);
    if (i >= n) return;
    int cellId = getCellId(pos[i], origin, cellSize, dims);
    if (cellId < 0) return;
    int index = atomic_inc(&cellCount[cellId]);
    if (index < MAX_PARTICLES_PER_CELL) {
        cellParticles[cellId * MAX_PARTICLES_PER_CELL + index] = i;
    }
}

__kernel void collideDifferentTypes(
    __global const float4* pos,
    __global float4* vel,
    __global const int* type,
    int n,
    __global const int* cellCount,
    __global const int* cellParticles,
    float4 origin,
    float cellSize,
    Int4 dims,
    float radius)
{
    int i = get_global_id(0);
    if (i >= n) return;

    float4 p = pos[i];
    float4 v = vel[i];
    int t = type[i];
    int3 c = getCellCoord(p, origin, cellSize);
    float minDist = radius * 2.0f;

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int3 nc = (int3)(c.x + dx, c.y + dy, c.z + dz);
                int cellId = getCellIdCoord(nc, dims);
                if (cellId < 0) continue;

                int count = cellCount[cellId];
                if (count > MAX_PARTICLES_PER_CELL) count = MAX_PARTICLES_PER_CELL;
                for (int k = 0; k < count; ++k) {
                    int j = cellParticles[cellId * MAX_PARTICLES_PER_CELL + k];
                    if (j == i || type[j] == t) continue;

                    float4 delta = p - pos[j];
                    float dist = length(delta.xyz);
                    if (dist > 0.0001f && dist < minDist) {
                        float4 normal = delta / dist;
                        float relNormal = dot((v - vel[j]).xyz, normal.xyz);
                        if (relNormal < 0.0f) {
                            v -= normal * relNormal;
                        }
                    }
                }
            }
        }
    }
    v.w = 0.0f;
    vel[i] = v;
}

bool insideBasket(float4 p, float4 bmin, float4 bmax) {
    return p.x >= bmin.x && p.x <= bmax.x &&
           p.y >= bmin.y && p.y <= bmax.y &&
           p.z >= bmin.z && p.z <= bmax.z;
}

float4 collidePlane(float4 p, float4* v, int t, float4 point, float4 normal, float particleRadius, float planeRadius, float damping, float friction) {
    float4 n = normalize(normal);
    float4 toPlaneCenter = p - point;
    float distToPlane = dot(toPlaneCenter.xyz, n.xyz);
    
    if (distToPlane < particleRadius) {
        float4 projPoint = p - n * distToPlane;
        float4 centerToProj = projPoint - point;
        
        float distFromCenter = length(centerToProj.xyz - n.xyz * dot(centerToProj.xyz, n.xyz));
        
        if (distFromCenter <= planeRadius) {
            p -= n * (distToPlane - particleRadius);
            *v = t == BOUNCE ? reflectVelocity(*v, n, damping) : slideVelocity(*v, n, friction);
        }
    }
    return p;
}


__kernel void updateParticles(
    __global float4* pos,
    __global float4* vel,
    __global float* life,
    __global int* type,
    __global uint* rng,
    volatile __global int* score,
    int n,
    float dt,
    float4 emitterCenter,
    float4 emitterSize,
    float4 emitterVelocity,
    __global const float4* planePoints,
    __global const float4* planeNormals,
    __global const float* planeRadii,
    int planeCount,
    float4 basketMin,
    float4 basketMax,
    float particleRadius,
    float worldMinY,
    float damping,
    float rollingFriction,
    float gravity)
{
    int i = get_global_id(0);
    if (i >= n) return;

    float4 p = pos[i];
    float4 v = vel[i];
    int t = type[i];
    uint state = rng[i];

    v.y -= gravity * dt;
    p += v * dt;
    p.w = 0.0f;
    v.w = 0.0f;
    life[i] -= dt;

    int pc = planeCount;
    if (pc > MAX_PLANES) pc = MAX_PLANES;
    for (int k = 0; k < pc; ++k) {
        p = collidePlane(p, &v, t, planePoints[k], planeNormals[k], particleRadius, planeRadii[k], damping, rollingFriction);
    }

    bool scored = insideBasket(p, basketMin, basketMax);
    if (scored) {
        atomic_inc(score);
    }

    if (scored || life[i] <= 0.0f || p.y < worldMinY) {
        p = respawnPos(&state, emitterCenter, emitterSize);
        v = respawnVel(&state, emitterVelocity);
        life[i] = 5.0f + rand01(&state) * 4.0f;
    }

    pos[i] = p;
    vel[i] = v;
    rng[i] = state;
}
