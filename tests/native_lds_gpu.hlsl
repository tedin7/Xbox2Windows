RWByteAddressBuffer output : register(u0);
groupshared uint lds[1024];

[numthreads(32,1,1)]
void main(uint lane : SV_GroupIndex) {
    uint exec=1, v0=4*lane, v1=100+lane, v2=200+lane, v3=0, v4=0;
    uint v5=0, v6=0, v7=0, v8=0;
    lds[320+lane]=300+lane;
    lds[448+lane]=400+lane;
    lds[128+lane]=500+lane;
    lds[160+lane]=600+lane;
    if(lane==0) { lds[512]=0xffffffff; lds[513]=0; }
    GroupMemoryBarrierWithGroupSync();
    // stride-64 write, then atomic min and max on two shared words
    #pragma x2w_lds d83c0301 00020100
    v3=2048;
    #pragma x2w_lds d81c0000 00000103
    #pragma x2w_lds d8200004 00000203
    v3=4*(640+3*lane);
    v5=700+lane; v6=800+lane; v7=900+lane;
    #pragma x2w_lds db780000 00000503
    v3=4*(736+4*lane);
    v8=1000+lane;
    #pragma x2w_lds db7c0000 00000503
    exec=0;
    v8=0;
    #pragma x2w_lds db7c0000 00000503
    exec=1;
    GroupMemoryBarrierWithGroupSync();
    uint other=(lane+1)%32;
    uint base=116*lane;
    uint wide=736+4*other;
    output.Store4(base+76,uint4(lds[wide],lds[wide+1],lds[wide+2],lds[wide+3]));
    output.Store2(base,uint2(lds[64+other],lds[192+other]));
    v4=4*other;
    // The first read overwrites the incoming address register.
    #pragma x2w_lds d8e00705 04000004
    output.Store4(base+8,uint4(v4,v5,lds[512],lds[513]));
    // One inactive store cannot disturb the initialized value.
    exec=0;
    #pragma x2w_lds d8340200 00000100
    exec=1;
    v6=4*other;
    #pragma x2w_lds d8dca080 07000006
    output.Store2(base+24,uint2(v7,v8));
    v4=4*(640+3*other);
    #pragma x2w_lds d9d80000 04000004
    output.Store3(base+32,uint3(v4,v5,lds[642+3*other]));
    v1=asuint(float(lane+1)); v2=asuint(2.0f);
    #pragma x2w_vop2 06060501
    #pragma x2w_vop2 08080501
    #pragma x2w_vop2 0a0a0501
    #pragma x2w_vop2 100c0501
    output.Store4(base+44,uint4(v3,v4,v5,v6));
    v1=0xff000003; v2=0x80000000|lane;
    #pragma x2w_vop2 16060501
    #pragma x2w_vop2 340804ff 00000021
    #pragma x2w_vop2 4a0a04c1
    #pragma x2w_vop2 4a0c04ff 8000000b
    exec=0;
    #pragma x2w_vop2 4a0c0481
    output.Store4(base+60,uint4(v3,v4,v5,v6));
    exec=1;
    v0=4*(864+4*lane);
    v1=101+lane; v2=202+lane; v5=303+lane; v6=404+lane;
    // Each offset is measured in 64-bit elements. Reverse them for the read.
    #pragma x2w_lds d9380100 00050100
    v0=4*(864+4*other);
    GroupMemoryBarrierWithGroupSync();
    #pragma x2w_lds d9dc0001 00000000
    output.Store4(base+92,uint4(v0,v1,v2,v3));
    // Each lane atomically subtracts from its own word. Alias DATA0 with VDST.
    v0=4*(992+lane);
    lds[992+lane]=lane+100;
    v1=lane+1;
    #pragma x2w_lds d8840000 01000100
    output.Store2(base+108,uint2(v1,lds[992+lane]));
}
