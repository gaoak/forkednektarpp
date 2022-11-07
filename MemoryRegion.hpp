#pragma once

struct BackendCPU;
struct BackendCUDA;

#if NEKTAR_USE_CUDA
using DefaultBackend = BackendCUDA;
#else
using DefaultBackend = BackendCPU;
#endif

template<typename tData, typename tBackend = DefaultBackend>
class MemoryRegion;
