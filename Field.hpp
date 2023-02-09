#pragma once

#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "LibUtilities/ErrorUtil.hpp"
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

/**
 * @brief A Field represents expansion data to be operated on
 * @tparam TState Either FieldState::Phys or FieldState::Coeff, represents
 * whether the Field is storing physical values or coefficients
 */
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

    /**
     * @brief Static templated creation method. Templated on the actual type of
     * memory region, e.g. MemoryRegionCUDA.
     */
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

        // Create new TMemoryRegion and polymorphically store as MemoryRegionCPU
        field.m_storage = std::make_unique<TMemoryRegion<TType>>(
            storage_size * num_components);

        return field;
    }

    /**
     * @brief Get the underlying storage of the field as the requested type.
     * Perform MemoryRegion conversions if necessary
     * @return MemoryRegion storage converted to the requested type
     */
    template <template <typename> class TMemoryRegion = MemoryRegionCPU>
    TMemoryRegion<TType> &GetStorage()
    {
        using T = TMemoryRegion<TType>;

        static_assert(std::is_base_of<MemoryRegionCPU<TType>, T>::value,
                      "TMemoryRegion must derive MemoryRegionCPU<TType>");
        try
        {
            // This cast fails if e.g. a MemoryRegionCUDA is requested from a
            // MemoryRegionCPU backed Field
            auto &ret = dynamic_cast<T &>(*m_storage);

            // Debug warning, a (possibly) undesired conversion occured
            WARNINGL0(typeid(*m_storage) == typeid(T),
                      std::string("Requested backing storage of type ") +
                          typeid(T).name() + " != actual storage type " +
                          typeid(*m_storage).name());
            return ret;
        }
        catch (const std::bad_cast &e)
        {
            WARNINGL0(false, std::string("Converting backing storage from ") +
                                 typeid(*m_storage).name() + " to " +
                                 typeid(T).name())

            // This is just here so that the fromCPU method does
            // not need to be declared for MemoryRegionCPU
            if constexpr (!std::is_same<T, MemoryRegionCPU<TType>>::value)
            {
                // Dynamic cast threw an exception, attempt to allocate the
                // requested TMemoryRegion from old data

                m_storage->ToCPU(); // Make sure memory is on the CPU

                m_storage = std::make_unique<T>(T::fromCPU(
                    std::move(*m_storage))); // Create new TMemoryRegion from
                                             // the CPU memory
            }

            return dynamic_cast<T &>(*m_storage);
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
