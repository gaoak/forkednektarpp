#include "MemRef.hpp"

template<typename tData, typename tBackend = DefaultMemRef>
class MemRef;

template<typename tData>
class MemRef<tData, BackendCPU>;
{
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
