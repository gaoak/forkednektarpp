#include <vector>
#include <string>

#include "MemoryRegion.hpp"
#include "MemoryRegionCPU.hpp"

#if NEKTAR_USE_CUDA
#include "MemoryRegionCUDA.hpp"
#else
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

template<typename TType = double, typename TState = DefaultState, typename TBackend = DefaultBackend>
class Field
{
    public:
        Field(std::vector<BlockAttributes> &blocks, int num_components = 1) : block_attributes(blocks)
        {
            size_t storage_size = 0;
            for (int i = 0; i < blocks.size(); ++i)
            {
                auto &block = blocks[i];
                size_t blockSize = block.num_elements * block.dofs[0] * block.dofs[1] * block.dofs[2]; // wasteful but an upper bound
                storage_size += blockSize;
            }

            m_storage = MemoryRegion<TType, TBackend>(storage_size * num_components);
        }
        Field(const Field&) = delete;
        ~Field() = default;

        MemoryRegion<TType, TBackend> &GetStorage()
        {
            return m_storage;
        }

        size_t GetNumComponents()
        {
            return component_names.size();
        }

    private:
        MemoryRegion<TType, TBackend> m_storage;
        //std::vector<MemoryView<TType>> m_views;
        std::vector<BlockAttributes> block_attributes;
        std::vector<std::string> component_names;
};
