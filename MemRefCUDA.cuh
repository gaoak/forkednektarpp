#include "MemRef.hpp"
#include <cuda.h>

template<typename tData>
class MemRef<tData, BackendCUDA>;
{
    MemRef(size_t n)
    {
        m_host = new tData[n];
        m_device = cudaMalloc(sizeof(tData) * n);
    }

    double *m_host = nullptr;
    double *m_device = nullptr;
    bool m_ondevice = false;

    double *GetPtr()
    {
        return m_ondevice ? m_device : m_host;
    }

    void HostToDevice()
    {
        // whatever
        m_ondevice = true;
    }

    void DeviceToHost()
    {
        // inverse whatever
        m_ondevice = false;
    }

    bool GetOnDevice() const
    {
        return m_ondevice;
    }
};

