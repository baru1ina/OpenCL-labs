cbuffer SceneCB : register(b0)
{
    row_major float4x4 gMvp;
    float2 gInvViewport;
    float gPointSize;
    float gPadding;
};

struct VSOutput
{
    float3 pos : POSITION;
    float4 color : COLOR0;
};

struct PSInput
{
    float4 pos : SV_POSITION;
    float4 color : COLOR0;
};

static const float PI = 3.14159265f;

[maxvertexcount(48)]
void main(point VSOutput input[1], inout TriangleStream<PSInput> stream)
{
    float4 center = mul(float4(input[0].pos, 1.0f), gMvp);
    if (center.w <= 0.001f)
    {
        return;
    }

    const int segments = 16;
    float2 radius = gPointSize * gInvViewport * center.w;

    [unroll]
    for (int i = 0; i < segments; ++i)
    {
        float a0 = 2.0f * PI * (float)i / (float)segments;
        float a1 = 2.0f * PI * (float)(i + 1) / (float)segments;

        float2 rim0 = float2(cos(a0), sin(a0)) * radius;
        float2 rim1 = float2(cos(a1), sin(a1)) * radius;

        PSInput v;

        v.pos = center;
        v.color = saturate(input[0].color * float4(1.18f, 1.18f, 1.18f, 1.0f));
        stream.Append(v);

        v.pos = center + float4(rim0.x, rim0.y, 0.0f, 0.0f);
        v.color = input[0].color * float4(0.72f, 0.72f, 0.72f, 1.0f);
        stream.Append(v);

        v.pos = center + float4(rim1.x, rim1.y, 0.0f, 0.0f);
        v.color = input[0].color * float4(0.72f, 0.72f, 0.72f, 1.0f);
        stream.Append(v);

        stream.RestartStrip();
    }
}
