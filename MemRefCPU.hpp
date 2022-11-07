#pragma once
#include "MemRef.hpp"

template<typename tData>
class MemRef<tData, BackendCPU>
{
public:
    MemRef() {}
    MemRef(size_t n)
    {
        m_host = new tData[n];
    }
    double *m_host = nullptr;

    double *GetPtr()
    {
        return m_host;
    }
};
