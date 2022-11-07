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

enum ShapeType {
    eQuadrilateral,
    eTriangle
};

struct BlockAttributes {
    ShapeType shape;
    std::array<int, 3> dofs;
    size_t num_elements;
};

template<typename TType = double, typename TState = DefaultState, typename tBackend = DefaultBackend>
class Field
{
    public:
        Field(std::vector<BlockAttributes> &blocks) : block_attributes(blocks), m_storage(blocks.size())
        {
            for (int i = 0; i < blocks.size(); ++i)
            {
                auto &block = blocks[i];
                size_t blockSize = block.num_elements * block.dofs[0] * block.dofs[1] * block.dofs[2]; // wasteful but an upper bound
                m_storage[i] = MemRef<TType, tBackend>(blockSize);
            }
        }
        Field(const Field&) = default;
        ~Field() = default;

        MemRef<TType, tBackend> &GetStorage(size_t i)
        {
            return m_storage[i];
        }

        size_t GetNumComponents()
        {
            return component_names.size();
        }

    private:
        std::vector<MemRef<TType, tBackend>> m_storage;
        std::vector<BlockAttributes> block_attributes;
        std::vector<std::string> component_names;
};
