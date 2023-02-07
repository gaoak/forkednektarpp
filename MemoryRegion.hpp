#pragma once

struct BackendCPU;

using DefaultBackend = BackendCPU;

template <typename tData, typename tBackend = DefaultBackend>
class MemoryRegion;
