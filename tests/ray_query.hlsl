RaytracingAccelerationStructure scene : register(t0);
RWStructuredBuffer<uint4> results : register(u0);
struct Query { float4 originMin; float4 directionMax; uint mask; uint flags; };
StructuredBuffer<Query> queries : register(t1);
// Bit 31 asks the fixture to report facing in the hit word.
static const uint ReportFacing = 0x80000000u;

RayDesc make_ray(Query query)
{
    RayDesc ray;
    ray.Origin = query.originMin.xyz;
    ray.Direction = query.directionMax.xyz;
    ray.TMin = query.originMin.w;
    ray.TMax = query.directionMax.w;
    return ray;
}

#ifdef DXR_PIPELINE
struct Payload { uint4 value; };

[shader("raygeneration")]
void RayGen()
{
    uint i = DispatchRaysIndex().x;
    Query query = queries[i];
    Payload payload;
    payload.value = 0xffffffff;
    uint cullFlags = query.flags & (RAY_FLAG_CULL_FRONT_FACING_TRIANGLES | RAY_FLAG_CULL_BACK_FACING_TRIANGLES);
    TraceRay(scene, RAY_FLAG_FORCE_OPAQUE | cullFlags, query.mask, 0, 1, 0, make_ray(query), payload);
    if ((query.flags & ReportFacing) == 0) payload.value.x &= ~2u;
    results[i] = payload.value;
}

[shader("miss")]
void Miss(inout Payload payload)
{
    payload.value = 0;
}

[shader("closesthit")]
void ClosestHit(inout Payload payload, BuiltInTriangleIntersectionAttributes attributes)
{
    uint status = HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE ? 3 : 1;
    payload.value = uint4(status, InstanceID(), PrimitiveIndex(), asuint(RayTCurrent()));
}
#else
// AS_TABLE binds the structure through a descriptor table, as a title that published it with a view does.
#ifdef AS_TABLE
#define RAY_ROOT "DescriptorTable(SRV(t0)),UAV(u0),SRV(t1)"
#else
#define RAY_ROOT "SRV(t0),UAV(u0),SRV(t1)"
#endif
[RootSignature(RAY_ROOT)]
[numthreads(1, 1, 1)]
void CSMain(uint3 thread : SV_DispatchThreadID)
{
    uint i = thread.x;
    Query query = queries[i];
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene, query.flags & 0x3ff, query.mask, make_ray(query));
    while (q.Proceed()) {}
    uint4 value = uint4(0, 0, 0, 0);
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        uint status = 1;
        if ((query.flags & ReportFacing) != 0 && q.CommittedTriangleFrontFace()) status = 3;
        value = uint4(status, q.CommittedInstanceID(), q.CommittedPrimitiveIndex(), asuint(q.CommittedRayT()));
    }
    results[i] = value;
}
#endif
