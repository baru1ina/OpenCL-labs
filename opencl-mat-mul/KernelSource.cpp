#include "KernelSource.h"

const char* KernelSource::getSource()
{
    return R"CLC(
#ifdef cl_khr_subgroups
    #pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

    __kernel void simpleMatMul(
        __global const float* A,
        __global const float* B,
        __global float* C,
        const uint rowsA,
        const uint colsA,
        const uint colsB)
    {
        const uint colC = get_global_id(0);
        const uint rowC = get_global_id(1);

        if (rowC >= rowsA || colC >= colsB)
        {
            return;
        }

        float total = 0.0f;
        for (uint k = 0; k < colsA; ++k)
        {
            total += A[rowC * colsA + k] * B[k * colsB + colC];
        }
        C[rowC * colsB + colC] = total;
    }

    __kernel void tiledMatMul(
        __global const float* A,
        __global const float* B,
        __global float* C,
        const uint rowsA,
        const uint colsA,
        const uint colsB)
    {
        const uint colC = get_global_id(0);
        const uint rowC = get_global_id(1);
        const uint localCol = get_local_id(0);
        const uint localRow = get_local_id(1);

        __local float tileA[TILE_SIZE][TILE_SIZE];
        __local float tileB[TILE_SIZE][TILE_SIZE + 1];

        float accumulator = 0.0f;
        const uint numTiles = (colsA + TILE_SIZE - 1) / TILE_SIZE;

        for (uint tileIdx = 0; tileIdx < numTiles; ++tileIdx)
        {
            const uint colA = tileIdx * TILE_SIZE + localCol;
            const uint rowB = tileIdx * TILE_SIZE + localRow;

            tileA[localRow][localCol] = (rowC < rowsA && colA < colsA) ? A[rowC * colsA + colA] : 0.0f;
            tileB[localRow][localCol] = (rowB < colsA && colC < colsB) ? B[rowB * colsB + colC] : 0.0f;

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint i = 0; i < TILE_SIZE; ++i)
            {
                accumulator += tileA[localRow][i] * tileB[i][localCol];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (rowC < rowsA && colC < colsB)
        {
            C[rowC * colsB + colC] = accumulator;
        }
    }

    __kernel void warpIntrinsicMatMul(
        __global const float* A,
        __global const float* B,
        __global float* C,
        const uint rowsA,
        const uint colsA,
        const uint colsB)
    {
        const uint lane = get_local_id(0);
        const uint colC = get_group_id(0);
        const uint rowC = get_group_id(1);

        if (rowC >= rowsA || colC >= colsB)
        {
            return;
        }

#if defined(cl_khr_subgroups)
        __local float subgroupSums[WARP_SIZE];

        float partial = 0.0f;
        for (uint k = lane; k < colsA; k += WARP_SIZE)
        {
            partial += A[rowC * colsA + k] * B[k * colsB + colC];
        }

        const float subgroupTotal = sub_group_reduce_add(partial);
        if (sub_group_local_id() == 0)
        {
            subgroupSums[get_sub_group_id()] = subgroupTotal;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane == 0)
        {
            float total = 0.0f;
            for (uint i = 0; i < get_num_sub_groups(); ++i)
            {
                total += subgroupSums[i];
            }
            C[rowC * colsB + colC] = total;
        }
#else
        if (lane == 0)
        {
            float total = 0.0f;
            for (uint k = 0; k < colsA; ++k)
            {
                total += A[rowC * colsA + k] * B[k * colsB + colC];
            }
            C[rowC * colsB + colC] = total;
        }
#endif
    }

    )CLC";
}

std::size_t KernelSource::getSourceLength()
{
    static std::size_t len = 0;
    if (len == 0)
    {
        const char* src = getSource();
        len = 0;
        while (src[len] != '\0') ++len;
    }
    return len;
}