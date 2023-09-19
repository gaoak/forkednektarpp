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

namespace Nektar
{
namespace MultiRegions
{
class ExpList;
typedef std::shared_ptr<ExpList> ExpListSharedPtr;
} // namespace MultiRegions
} // namespace Nektar

/**
 * @brief Captures the structure of a block of elements of identical shape
 * and order.
 */
struct BlockAttributes
{
    // default constructor: no padding
    BlockAttributes(size_t num_elements, size_t num_pts)
        : num_elements(num_elements), num_pts(num_pts),
          block_size(num_elements * num_pts), num_padding_elements(0)
    {
    }

    size_t num_elements;
    size_t num_padding_elements;
    size_t num_pts;
    size_t block_size; // (num_elements + num_padding_elements) * num_pts
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
    FieldState state, const Nektar::MultiRegions::ExpListSharedPtr explist,
    size_t VectorWidth = 1);

/**
 * @brief A Field represents expansion data to be operated on.
 * @tparam TType  The floating-point representation used by the field.
 * @tparam TState A FieldState value representing the state of the field.
 */
template <typename TType = double, FieldState TState = DefaultState> class Field
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
     * @brief Compare this field to another field, with absolute
     * tolerance tol. Two fields must have same storage shape.
     * @return bool
     */
    bool compare(Field<TType, TState> &rhs, double tol)
    {
        const std::vector<BlockAttributes> &rhs_blocks = rhs.GetBlocks();
        TType *store     = GetStorage().GetCPUPtr();
        TType *rhs_store = rhs.GetStorage().GetCPUPtr();

        if (rhs_blocks.size() != block_attributes.size())
            return false;
        if (rhs.m_curVecWidth != m_curVecWidth)
            return false;

        for (size_t component = 0; component < GetNumComponents(); ++component)
        {
            for (size_t bl = 0; bl < block_attributes.size(); ++bl)
            {
                size_t num_pts      = block_attributes[bl].num_pts;
                size_t num_elements = block_attributes[bl].num_elements;
                size_t num_padding_elements =
                    block_attributes[bl].num_padding_elements;
                size_t num_metaBlocks =
                    (num_elements + num_padding_elements) / m_curVecWidth;

                // Check that each block have the same structure
                if (num_elements != rhs_blocks[bl].num_elements)
                    return false;
                if (num_pts != rhs_blocks[bl].num_pts)
                    return false;

                for (size_t metaBlock = 0; metaBlock < num_metaBlocks;
                     ++metaBlock)
                {
                    for (size_t coeff = 0; coeff < num_pts; ++coeff)
                    {
                        for (size_t k = 0; k < m_curVecWidth; ++k)
                        {
                            // skip padding elements
                            if (metaBlock * m_curVecWidth + k + 1 <= num_elements)
                            {
                                if (std::abs(*store - *rhs_store) > tol)
                                    return false;
                            }
                            store++;
                            rhs_store++;
                        }
                    }
                }
            }
        }
        return true;
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
    static Field<TType, TState> create(
        std::vector<BlockAttributes> blocks, int num_components = 1,
        size_t Align = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    {
        auto field = Field(std::move(blocks), num_components);

        size_t storage_size = std::accumulate(
            field.block_attributes.begin(), field.block_attributes.end(), 0,
            [](size_t acc, const BlockAttributes &block) {
                return acc + block.block_size;
            });

        // Create new TMemoryRegion and polymorphically store as MemoryRegionCPU
        field.m_storage = std::make_unique<TMemoryRegion<TType>>(
            storage_size * num_components, Align);

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
    template <size_t VectorWidth> void ReshapeStorage()
    {
        // No reshape required, early return
        if (m_curVecWidth == VectorWidth)
            return;

        ReshapeToScalar();

        // Early return if "scalar" shape is required
        if (VectorWidth == 1)
            return;

        TType *ptr = m_storage->GetCPUPtr();

        for (const auto &block : block_attributes)
        {
            const size_t numMetaBlocks =
                (block.num_elements + block.num_padding_elements) / VectorWidth;
            const size_t MetaBlockSize = VectorWidth * block.num_pts;

            Nektar::Array<Nektar::OneD, TType> temp(MetaBlockSize, 0.0);

            for (size_t metaBlock = 0; metaBlock < numMetaBlocks; ++metaBlock)
            {
                // Copy data into temporary storage because inptr and outptr
                // access the same memory location
                std::copy(ptr, ptr + MetaBlockSize, temp.get());
                InterleaveFromScalar<VectorWidth>(temp.get(), block.num_pts,
                                                  ptr);
                ptr += block.num_pts * VectorWidth;
            }
        }

        m_curVecWidth = VectorWidth;
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
    void ReshapeToScalar()
    {
        if (m_curVecWidth == 1)
            return;

        TType *inptr = m_storage->GetCPUPtr();

        for (const auto &block : block_attributes)
        {
            const size_t numMetaBlocks =
                (block.num_elements + block.num_padding_elements) /
                m_curVecWidth;
            const size_t MetaBlockSize = m_curVecWidth * block.num_pts;

            Nektar::Array<Nektar::OneD, TType> temp(MetaBlockSize, 0.0);

            for (size_t metaBlock = 0; metaBlock < numMetaBlocks; ++metaBlock)
            {
                // Copy data into temporary storage because inptr and outptr
                // access the same memory location
                std::copy(inptr, inptr + MetaBlockSize, temp.get());
                Deinterleave(temp.get(), block.num_pts, inptr);
                inptr += MetaBlockSize;
            }
        }

        m_curVecWidth = 1;
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
    std::vector<std::string> component_names = {"u"};

    size_t m_curVecWidth = 1;
};
