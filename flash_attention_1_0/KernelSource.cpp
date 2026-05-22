#include "KernelSource.h"

const char* KernelSource::getAttentionSource()
{
    return R"CLC(
#define WORK_GROUP_SIZE 128
#define KV_BLOCK_SIZE 32
#define MAX_HIDDEN_DIM 128
#define NEG_INFINITY (-3.4028234663852886e+38f)

inline float dotProduct(__global const float* vecA,
                        __global const float* vecB,
                        int rowA,
                        int rowB,
                        int dim)
{
    float total = 0.0f;
    for (int idx = 0; idx < dim; ++idx)
        total += vecA[rowA * dim + idx] * vecB[rowB * dim + idx];
    return total;
}

__kernel void naiveAttentionKernel(__global const float* queries,
                                   __global const float* keys,
                                   __global const float* values,
                                   __global float* output,
                                   int seqLen,
                                   int hiddenDim,
                                   float scale)
{
    int rowIdx = get_group_id(0);
    int localId = get_local_id(0);

    if (rowIdx >= seqLen || localId != 0)
        return;

    float maxScore = NEG_INFINITY;
    for (int j = 0; j < seqLen; ++j)
    {
        float score = dotProduct(queries, keys, rowIdx, j, hiddenDim) * scale;
        maxScore = fmax(maxScore, score);
    }

    float denominator = 0.0f;
    for (int j = 0; j < seqLen; ++j)
    {
        float score = dotProduct(queries, keys, rowIdx, j, hiddenDim) * scale;
        denominator += exp(score - maxScore);
    }

    for (int col = 0; col < hiddenDim; ++col)
    {
        float accumulator = 0.0f;
        for (int j = 0; j < seqLen; ++j)
        {
            float score = dotProduct(queries, keys, rowIdx, j, hiddenDim) * scale;
            float probability = exp(score - maxScore) / denominator;
            accumulator += probability * values[j * hiddenDim + col];
        }
        output[rowIdx * hiddenDim + col] = accumulator;
    }
}

__kernel void optimizedAttentionKernel(__global const float* queries,
                                        __global const float* keys,
                                        __global const float* values,
                                        __global float* output,
                                        int seqLen,
                                        int hiddenDim,
                                        float scale)
{
    int rowIdx = get_group_id(0);
    int localId = get_local_id(0);

    if (rowIdx >= seqLen)
        return;

    __local float queryTile[MAX_HIDDEN_DIM];
    __local float keyBlock[KV_BLOCK_SIZE][MAX_HIDDEN_DIM];
    __local float valueBlock[KV_BLOCK_SIZE][MAX_HIDDEN_DIM];
    __local float blockScores[KV_BLOCK_SIZE];
    __local float reductionBuffer[WORK_GROUP_SIZE];
    __local float sharedMax;
    __local float sharedDenom;
    __local float sharedAlpha;

    if (localId < hiddenDim)
        queryTile[localId] = queries[rowIdx * hiddenDim + localId];
    barrier(CLK_LOCAL_MEM_FENCE);

    float accumulator = 0.0f;
    float currentMax = NEG_INFINITY;
    float currentDenom = 0.0f;

    for (int blockStart = 0; blockStart < seqLen; blockStart += KV_BLOCK_SIZE)
    {
        int blockSize = min(KV_BLOCK_SIZE, seqLen - blockStart);

        for (int idx = localId; idx < blockSize * hiddenDim; idx += WORK_GROUP_SIZE)
        {
            int localRow = idx / hiddenDim;
            int localCol = idx % hiddenDim;
            int globalRow = blockStart + localRow;
            keyBlock[localRow][localCol] = keys[globalRow * hiddenDim + localCol];
            valueBlock[localRow][localCol] = values[globalRow * hiddenDim + localCol];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int localRow = 0; localRow < blockSize; ++localRow)
        {
            float partial = (localId < hiddenDim) ? queryTile[localId] * keyBlock[localRow][localId] : 0.0f;
            reductionBuffer[localId] = partial;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int stride = WORK_GROUP_SIZE / 2; stride > 0; stride >>= 1)
            {
                if (localId < stride)
                    reductionBuffer[localId] += reductionBuffer[localId + stride];
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (localId == 0)
                blockScores[localRow] = reductionBuffer[0] * scale;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (localId == 0)
        {
            float newMax = currentMax;
            for (int localRow = 0; localRow < blockSize; ++localRow)
                newMax = fmax(newMax, blockScores[localRow]);

            float alpha = (currentDenom == 0.0f) ? 0.0f : exp(currentMax - newMax);
            float newDenom = currentDenom * alpha;
            for (int localRow = 0; localRow < blockSize; ++localRow)
                newDenom += exp(blockScores[localRow] - newMax);

            sharedMax = newMax;
            sharedDenom = newDenom;
            sharedAlpha = alpha;

            currentMax = newMax;
            currentDenom = newDenom;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (localId < hiddenDim)
        {
            accumulator *= sharedAlpha;
            for (int localRow = 0; localRow < blockSize; ++localRow)
            {
                float unnormalizedProb = exp(blockScores[localRow] - sharedMax);
                accumulator += unnormalizedProb * valueBlock[localRow][localId];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId < hiddenDim)
        output[rowIdx * hiddenDim + localId] = accumulator / sharedDenom;
}
)CLC";
}