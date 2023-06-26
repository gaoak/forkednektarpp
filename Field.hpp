#pragma once

#include <array>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "MemoryRegionCPU.hpp"
#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <MultiRegions/ExpList.h>

/**
 * @brief Captures the structure of a block of elements of identical shape
 * and order.
 */
struct BlockAttributes
{
    BlockAttributes(size_t num_elements, size_t num_pts)
        : num_elements(num_elements), num_pts(num_pts),
          block_size(num_elements * num_pts)
    {
    }

    size_t num_elements;
    size_t num_pts;
    size_t block_size;
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

std::vector<BlockAttributes> GetBlockAttributes(
    FieldState state, const Nektar::MultiRegions::ExpListSharedPtr explist);

/**
 * @brief A Field represents expansion data to be operated on.
 * @tparam TType  The floating-point representation used by the field.
 * @tparam TState A FieldState value representing the state of the field.
 */
template <typename TType = double, FieldState TState = DefaultState> class Field
{
    using ExpansionSharedPtr = Nektar::StdRegions::StdExpansionSharedPtr;

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
          component_names(std::move(rhs.component_names)),
          m_curVecWidth(rhs.m_curVecWidth)
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
        m_curVecWidth    = std::move(rhs.m_curVecWidth);

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
    static Field<TType, TState> create(std::vector<BlockAttributes> blocks,
                                       int num_components = 1)
    {
        auto field = Field(std::move(blocks), num_components);

        size_t storage_size = std::accumulate(
            field.block_attributes.begin(), field.block_attributes.end(), 0,
            [](size_t acc, const BlockAttributes &block) {
                return acc + block.block_size;
            });

        // Create new TMemoryRegion and polymorphically store as MemoryRegionCPU
        field.m_storage = std::make_unique<TMemoryRegion<TType>>(
            storage_size * num_components);

        return field;
    }

    template <template <typename> class TMemoryRegion = MemoryRegionCPU>
    static Field<TType, TState> fromArray(
        std::vector<BlockAttributes> blocks,
        Nektar::Array<Nektar::OneD, TType> const &array)
    {
        auto field = create<MemoryRegionCPU>(blocks);

        ASSERTL0(
            field.GetStorage().size() == array.size(),
            "Array size does not match size calculated from provided elements.")

        std::copy(array.begin(), array.end(), field.GetStorage().GetCPUPtr());

        field.template GetStorage<TMemoryRegion>(); // convert storage to
                                                    // TMemoryRegion
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
     * @brief Reshapes the storage to a prescribed vector width.
     *
     * This routine reorders the elemental data to interleave elements to a
     * prescribed vector width VW. This therefore puts the same DOF for groups
     * of VW elements contiguously in memory, enabling the efficient use of
     * vectorised instructions. At the moment this routine reshapes from VW0 to
     * VW1 by first reshaping from VW0 to a vector width of 1, and then to VW1.
     *
     * @tparam  VectorWidth     Target vector width.
     * @tparam  Align           Memory alignment to use.
     */
    template <size_t VectorWidth,
              size_t Align = __STDCPP_DEFAULT_NEW_ALIGNMENT__>
    void ReshapeStorage()
    {
        // No reshape required, early return
        if (m_curVecWidth == VectorWidth)
            return;

        ReshapeToScalar<Align>();

        // Early return if "scalar" shape is required
        if (VectorWidth == 1)
            return;

        // New memory region with requested alignment
        MemoryRegionCPU<TType> reshapedStorage(m_storage->size(), Align);

        TType *inptr  = m_storage->GetCPUPtr();
        TType *outptr = reshapedStorage.GetCPUPtr();

        for (const auto &block : block_attributes)
        {
            ASSERTL1(
                block.num_elements % VectorWidth == 0,
                "Number of elements not divisible by VectorWidth, padding not "
                "implemented yet.");

            const size_t numMetaBlocks = block.num_elements / VectorWidth;
            for (size_t metaBlock = 0; metaBlock < numMetaBlocks; ++metaBlock)
            {
                InterleaveFromScalar<VectorWidth>(inptr, block.num_pts, outptr);
                inptr += block.num_pts * VectorWidth;
                outptr += block.num_pts * VectorWidth;
            }
        }

        m_curVecWidth = VectorWidth;
        m_storage     = std::make_unique<MemoryRegionCPU<TType>>(
            std::move(reshapedStorage));
    }

    std::vector<BlockAttributes> const &GetBlocks() const
    {
        return block_attributes;
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

    Nektar::Array<Nektar::OneD, TType> toArray() const
    {
        return Nektar::Array<Nektar::OneD, TType>(m_storage->size(),
                                                  m_storage->GetCPUPtr());
    }

private:
    /**
     * @brief Reshapes the current storage interleaving to a non-interleaved
     * arrangement.
     */
    template <size_t Align = __STDCPP_DEFAULT_NEW_ALIGNMENT__>
    void ReshapeToScalar()
    {
        if (m_curVecWidth == 1)
            return;

        MemoryRegionCPU<TType> reshapedStorage(m_storage->size(), Align);

        TType *inptr  = m_storage->GetCPUPtr();
        TType *outptr = reshapedStorage.GetCPUPtr();

        for (const auto &block : block_attributes)
        {
            const size_t numMetaBlocks = block.num_elements / m_curVecWidth;

            for (size_t metaBlock = 0; metaBlock < numMetaBlocks; ++metaBlock)
            {
                Deinterleave(inptr, block.num_pts, outptr);
                inptr += m_curVecWidth * block.num_pts;
                outptr += m_curVecWidth * block.num_pts;
            }
        }

        m_curVecWidth = 1;
        m_storage     = std::make_unique<MemoryRegionCPU<TType>>(
            std::move(reshapedStorage));
    }

    /**
     * @brief Interleave the data block to the given vector width
     * @tparam  VectorWidth Target vector width.
     * @param   in          Input array with VW of 1.
     * @param   dataLen     Length of data blocks to be interleaved.
     * @param   out         Output array of VW specified by VectorWidth.
     */
    template <size_t VectorWidth>
    void InterleaveFromScalar(const TType *in, size_t dataLen, TType *out)
    {
        // TODO: SIMD this
        for (size_t idx = 0; idx < dataLen; ++idx)
        {
            for (size_t vecElem = 0; vecElem < VectorWidth; ++vecElem)
            {
                out[idx * VectorWidth + vecElem] = in[vecElem * dataLen + idx];
            }
        }
    }

    /**
     * @brief Deinterleave data from the current vector width to be
     * non-interleaved.
     * @param   in      Input array
     * @param   dataLen Length of a block of data (e.g. an element)
     * @param   out     Output array
     */
    void Deinterleave(const TType *in, size_t dataLen, TType *out)
    {
        for (size_t idx = 0; idx < dataLen; ++idx)
        {
            for (size_t vecElem = 0; vecElem < m_curVecWidth; ++vecElem)
            {
                out[vecElem * dataLen + idx] =
                    in[idx * m_curVecWidth + vecElem];
            }
        }
    }

    /**
     * @brief Construct a new Field object.
     *
     * @param blocks    Field data layout specification
     * @param num_components Number of components for a vector field.
     */
    Field(std::vector<BlockAttributes> blocks, int num_components = 1)
        : block_attributes(std::move(blocks))
    {
    }

    std::unique_ptr<MemoryRegionCPU<TType>> m_storage;
    std::vector<BlockAttributes> block_attributes;
    std::vector<std::string> component_names;

    size_t m_curVecWidth = 1;
};
