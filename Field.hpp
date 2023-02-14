#pragma once

#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "LibUtilities/ErrorUtil.hpp"
#include "MemoryRegionCPU.hpp"

/**
 * @brief Element shape enum used when defining Blocks of elements.
 */
enum class ShapeType
{
    eQuadrilateral,
    eTriangle
};

/**
 * @brief Captures the structure of a block of elements of identical shape
 * and order.
 */
struct BlockAttributes
{
    ShapeType shape;
    std::array<int, 3> dofs;
    size_t num_elements;
};

/**
 * @brief Possible states for Field data.
 * 
 * These identify the mathematical representation of the field data. The two
 * main states are *Phys*, representing the field at the quadrature points,
 * and *Coeff*, representing the field in terms of its spectral/hp element
 * basis coefficients.
 */
enum class FieldState
{
    Phys,
    Coeff
};

static constexpr FieldState DefaultState = FieldState::Phys;

/**
 * @brief A Field represents expansion data to be operated on.
 * @tparam TType  The floating-point representation used by the field.
 * @tparam TState A FieldState value representing the state of the field.
 */
template <typename TType = double, FieldState TState = DefaultState>
class Field
{
public:
    Field(const Field &) = delete;
    virtual ~Field()     = default;

    /**
     * @brief Construct a new Field object by moving storage from an existing
     * Field object.
     * 
     * @param rhs 
     */
    Field(Field &&rhs)
        : m_storage(std::move(rhs.m_storage)),
          block_attributes(std::move(rhs.block_attributes)),
          component_names(std::move(component_names))
    {
    }

    /**
     * @brief Move assignment operator.
     * 
     * @param rhs 
     * @return Field& 
     */
    Field &operator=(Field &&rhs)
    {
        m_storage        = std::move(rhs.m_storage);
        block_attributes = std::move(rhs.block_attributes);
        component_names  = std::move(rhs.component_names);

        return *this;
    }

    /**
     * @brief Static templated creation method.
     * 
     * @tparam TMemoryRegion Type of memory region to use
     * @param blocks         Field data specification.
     * @param num_components Number of components for a vector field.
     * @return Field<TType, TState> 
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
     * @return MemoryRegion storage converted to the requested type
     * 
     * This routine performs MemoryRegion conversions if necessary to enable
     * casting of, for example a CUDA memory region to a CPU memory region to
     * support the use of a CPU-only operator if necessary.
     * 
     * A runtime warning is provided if a transfer of data from device to host
     * is required to achieve the conversion.
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

    /**
     * @brief Gets the number of components for a vector field.
     * 
     * @return size_t 
     */
    size_t GetNumComponents()
    {
        return component_names.size();
    }

private:
    /**
     * @brief Construct a new Field object.
     * 
     * @param blocks    Field data layout specification
     * @param num_components Number of components for a vector field.
     */
    Field(std::vector<BlockAttributes> &blocks, int num_components = 1)
        : block_attributes(blocks)
    {
    }

    std::unique_ptr<MemoryRegionCPU<TType>> m_storage;
    // std::vector<MemoryView<TType>> m_views;
    std::vector<BlockAttributes> block_attributes;
    std::vector<std::string> component_names;
};
