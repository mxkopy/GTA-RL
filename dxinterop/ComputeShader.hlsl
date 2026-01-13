Texture2D<float> DepthSRV : register(t0);
Texture2D<float> LastDepthSRV : register(t1);
Texture2D<uint> StencilSRV : register(t2);

RWTexture2D<float> DepthUAV : register(u0);
RWTexture2D<float> LastDepthUAV : register(u1);
RWTexture2D<uint> StencilUAV : register(u2);

[numthreads(32, 32, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    DepthUAV[DTid.xy].r = DepthSRV[DTid.xy].r;

    StencilUAV[DTid.xy].r = StencilSRV[DTid.xy].r;    
}