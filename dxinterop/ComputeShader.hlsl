Texture2D<float> DepthSRV : register(t0);
Texture2D<float> LastDepthSRV : register(t1);
Texture2D<uint> StencilSRV : register(t2);

RWTexture2D<float4> DepthUAV : register(u0);
RWTexture2D<uint> StencilUAV : register(u1);

cbuffer VS : register(b0)
{
    float4 array[7];
    float1 pad1;
    float1 pad2;
    float1 screen_x;
    float1 screen_y;
}

cbuffer CurrentMatrix : register(b1)
{
    float3x3 M1;
}

cbuffer LastMatrix : register(b2)
{
    float3x3 M2;
}

[numthreads(32, 32, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{   
    float3 DepthCoord = 
    float3(
        ((DTid.x / screen_x) - 0.5f) * 2,
        ((DTid.y / screen_y) - 0.5f) * 2,
        DepthSRV[DTid.xy]
    );
    
    
    
    DepthUAV[DTid.xy].a = DepthSRV[DTid.xy].r;

    StencilUAV[DTid.xy].r = StencilSRV[DTid.xy].r;    
}