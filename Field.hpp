#pragma once

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "MemoryRegionCPU.hpp"

enum class ShapeType
{
    eQuadrilateral,
    eTriangle
};

struct BlockAttributes
{
    ShapeType shape;
    std::array<int, 3> dofs;
    size_t num_elements;
};

enum class FieldState
{
    Phys,
    Coeff
};

static constexpr FieldState DefaultState = FieldState::Phys;

template <typename TType = double, FieldState TState = DefaultState> class Field
{
public:
    Field(const Field &) = delete;
    virtual ~Field()     = default;

    Field(Field &&rhs)
        : m_storage(std::move(rhs.m_storage)),
          block_attributes(std::move(rhs.block_attributes)),
          component_names(std::move(component_names))
    {
    }

    Field &operator=(Field &&rhs)
    {
        m_storage        = std::move(rhs.m_storage);
        block_attributes = std::move(rhs.block_attributes);
        component_names  = std::move(rhs.component_names);

        return *this;
    }

    template <template <typename> class TMemoryRegion = MemoryRegionCPU>
    static Field<TType, TState> create(std::vector<BlockAttributes> &blocks,
                                       int num_components = 1)
    {
        auto field = Field(blocks, num_components);

        size_t storage_size = 0;
        for (int i = 0; i < blocks.size(); ++i)
        {
            auto &block      = blocks[i];
            size_t blockSize = block.num_elements * block.dofs[0] *
                               block.dofs[1] *
                               block.dofs[2]; // wasteful but an upper bound
            storage_size += blockSize;
        }

        field.m_storage = std::make_unique<TMemoryRegion<TType>>(
            storage_size * num_components);

        return field;
    }

    template <template <typename> class TMemoryRegion = MemoryRegionCPU>
    TMemoryRegion<TType> &GetStorage()
    {
        static_assert(std::is_base_of<MemoryRegionCPU<TType>,
                                      TMemoryRegion<TType>>::value,
                      "TMemoryRegion must derive MemoryRegionCPU<TType>");

        try
        {
            return dynamic_cast<TMemoryRegion<TType> &>(*m_storage);
        }
        catch (const std::bad_cast &e)
        {
            throw std::runtime_error(
                "Failed to cast memory storage from type " +
                std::string(typeid(*m_storage).name()) + " to type " +
                std::string(typeid(TMemoryRegion<TType>).name()));
        }
    }

    size_t GetNumComponents()
    {
        return component_names.size();
    }

private:
    Field(std::vector<BlockAttributes> &blocks, int num_components = 1)
        : block_attributes(blocks)
    {
    }

    std::unique_ptr<MemoryRegionCPU<TType>> m_storage;
    // std::vector<MemoryView<TType>> m_views;
    std::vector<BlockAttributes> block_attributes;
    std::vector<std::string> component_names;
};
