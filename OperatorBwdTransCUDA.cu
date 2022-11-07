#include "OperatorBwdTransCUDA.hpp"

// Utility to check errors
#ifdef NDEBUG
#   define cudaErrChk(ans) { ans; }
#else
#   define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
#endif
inline void cudaAssert(cudaError_t cErr, const char *file, int line)
{
    if (cErr)
    {
        constexpr unsigned int size = 80;
        char msg [size];
        snprintf(msg, size,
          "cudaAssert: %s: %s:%d\n", cudaGetErrorString(cErr), file, line);
        throw  std::runtime_error(msg);
    }
}

__global__
void dodouble(int n, double *x, double *y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
    {
        y[i] = 2.0*x[i];
    }
}

template<typename TData>
void Operator<TData, OpBwdTrans, MethodLocMat, BackendCUDA>::apply_impl(
    Field<TData, StateCoeff, BackendCUDA> &in, Field<TData, StatePhys, BackendCUDA> &out)
{
    auto &storage_in = in.GetStorage();
    auto &storage_out = out.GetStorage();

    double *in_ptr = storage_in.m_device;
    double *out_ptr = storage_out.m_device;

    if (!storage_in.GetOnDevice())
    {
        storage_in.HostToDevice();
    }

    int N = storage_in.m_size;

    dodouble<<<(N + 255) / 256, 256>>>(N, in_ptr, out_ptr);
    cudaErrChk(cudaPeekAtLastError());

    cudaDeviceSynchronize();
}

template void Operator<double, OpBwdTrans, MethodLocMat, BackendCUDA>::apply_impl(
    Field<double, StateCoeff, BackendCUDA> &in, Field<double, StatePhys, BackendCUDA> &out);

