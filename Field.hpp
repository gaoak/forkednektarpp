#include <vector>
#include <string>

#include "MemRef.hpp"
#include "MemRefCPU.hpp"

// Device options
struct DeviceCPU;
struct DeviceCUDA;

#if NEKTAR_USE_CUDA
using DefaultDevice = DeviceCUDA;
#include "MemRefCUDA.hpp"
#else
using DefaultDevice = DeviceCPU;
#endif

// State options
struct StatePhys;
struct StateCoeff;
using DefaultState = StatePhys;

struct CollectionInfo {
    size_t offset;
    // other polynomial order stuff
};

struct VariableInfo {
    std::string name;
};

template<typename TType = double, typename TState = DefaultState, typename tBackend = DefaultBackend>
class Field
{
    public:
        Field() : m_storage(10)
        {
        }
        Field(const Field&) = default;
        ~Field() = default;

        MemRef<TType, tBackend> &GetStorage()
        {
            return m_storage;
        }

    private:
        MemRef<TType, tBackend> m_storage;

        std::vector<CollectionInfo> coll_info;
        size_t nvar;
        std::vector<VariableInfo> var_info;
};
