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
 * @brief A block means a group of elements of identical shape,
 * basis and order. BlockAttributes stores the most basic
 * information of a block.
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
 * @brief Get the BlockAttributes for a given field state from an ExpList.
 * This method basically captures identical elements that are contiguously
 * stored in the ExpList and group them into blocks. Padding elements will
 * also be set based on given vector width.
 * @param state     Field state to query.
 * @param explist   Expansion list to query.
 * @param VectorWidth Vector width to use for the field.
 * @return std::vector<BlockAttributes>
 */
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
          m_curVecWidth(rhs.m_curVecWidth), m_alignment(rhs.m_alignment)
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
        m_alignment      = std::move(rhs.m_alignment);

        return *this;
    }

    /**
     * @brief Compare this field to another field, with absolute
     * tolerance tol. Two fields must have same storage shape
     * and same components.
     * @return bool
     */
    bool compare(Field<TType, TState> &rhs, double tol)
    {
        if (rhs.GetNumComponents() != GetNumComponents())
            return false;

        const std::vector<BlockAttributes> &rhs_blocks = rhs.GetBlocks();
        if (rhs_blocks.size() != block_attributes.size())
            return false;
        if (rhs.m_curVecWidth != m_curVecWidth)
            return false;

        bool isMatched = true;

        TType *store     = GetStorage().GetCPUPtr();
        TType *rhs_store = rhs.GetStorage().GetCPUPtr();

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

                int MisMatchcnt = 0, total = 0;

                for (size_t metaBlock = 0; metaBlock < num_metaBlocks;
                     ++metaBlock)
                {
                    for (size_t coeff = 0; coeff < num_pts; ++coeff)
                    {
                        for (size_t k = 0; k < m_curVecWidth; ++k)
                        {
                            // skip padding elements
                            if (metaBlock * m_curVecWidth + k + 1 <=
                                num_elements)
                            {
                                total++;
                                if (std::abs(*store - *rhs_store) > tol)
                                {
                                    if (MisMatchcnt == 0)
                                    {
                                        std::cout
                                            << "Mismatch in block " << bl
                                            << " at metaBlock " << metaBlock
                                            << " and coefficient " << coeff
                                            << " and vector element " << k
                                            << " with value " << *store
                                            << " and " << *rhs_store
                                            << std::endl;
                                        isMatched = false;
                                    }
                                    MisMatchcnt++;
                                }
                            }
                            store++;
                            rhs_store++;
                        }
                    }
                }
                if (!isMatched)
                {
                    std::cout << "Number of mismatches in block " << bl
                              << " is " << MisMatchcnt << " out of " << total
                              << std::endl;
                }
            }
        }
        if (isMatched)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    /**
     * @brief Static templated creation method. This method create
     * new Field by giving the names of the components.
     *
     * @tparam TMemoryRegion Type of memory region to use
     * @param blocks         Field data specification.
     * @param components     Names of components for a vector field.
     * @param Align          Memory alignment to use.
     * @return Field<TType, TState>
     */
    template <template <typename> class TMemoryRegion = MemoryRegionCPU>
    static Field<TType, TState> create(
        std::vector<BlockAttributes> blocks,
        std::vector<std::string> components,
        size_t Align = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    {
        auto field         = Field(std::move(blocks), components);
        int num_components = components.size();

        size_t storage_size = std::accumulate(
            field.block_attributes.begin(), field.block_attributes.end(), 0,
            [](size_t acc, const BlockAttributes &block) {
                return acc + block.block_size;
            });

        // Create new TMemoryRegion and polymorphically store as MemoryRegionCPU
        field.m_storage = std::make_unique<TMemoryRegion<TType>>(
            storage_size * num_components, Align);
        // Record the alignment
        field.m_alignment = Align;

        return field;
    }

    /**
     * @brief Static templated creation method. This method create
     * new Field by giving the number of components.
     * @tparam TMemoryRegion Type of memory region to use
     * @param blocks         Field data specification.
     * @param num_components Number of components for a vector field.
     * @param Align          Memory alignment to use.
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
        // Record the alignment
        field.m_alignment = Align;

        return field;
    }

    /**
     * @brief Another create method that copy data from a Nektar::Array
     *
     * @tparam TMemoryRegion Type of memory region to use
     * @param array          Nektar::Array to copy from
     * @param blocks         Field storage layout.
     * @param Align          Memory alignment to use.
     * @return Field<TType, TState>
     */
    template <template <typename> class TMemoryRegion = MemoryRegionCPU>
    static Field<TType, TState> fromArray(
        Nektar::Array<Nektar::OneD, TType> const &array,
        std::vector<BlockAttributes> blocks,
        size_t Align = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    {
        auto field = create<MemoryRegionCPU>(blocks, Align);

        // given paddings, the Storage size may be larger than array size.
        // But it should not be smaller.
        ASSERTL0(field.GetStorage().size() >= array.size(),
                 "Array size should not be larger than field size!")

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

        size_t scalar_field_size = GetFieldSize();

        for (int component = 0; component < GetNumComponents(); ++component)
        {
            TType *ptr = m_storage->GetCPUPtr() + component * scalar_field_size;

            for (const auto &block : block_attributes)
            {
                const size_t numMetaBlocks =
                    (block.num_elements + block.num_padding_elements) /
                    VectorWidth;
                const size_t MetaBlockSize = VectorWidth * block.num_pts;

                Nektar::Array<Nektar::OneD, TType> temp(MetaBlockSize, 0.0);

                for (size_t metaBlock = 0; metaBlock < numMetaBlocks;
                     ++metaBlock)
                {
                    // Copy data into temporary storage because inptr and outptr
                    // access the same memory location
                    std::copy(ptr, ptr + MetaBlockSize, temp.get());
                    InterleaveFromScalar<VectorWidth>(temp.get(), block.num_pts,
                                                      ptr);
                    ptr += block.num_pts * VectorWidth;
                }
            }
        }
        m_curVecWidth = VectorWidth;
    }

    /**
     * @brief Copy data from one field/component to another. The two fields
     * must have the same storage layout.
     *
     * @param rhs Source field
     * @param rhs_component the component to load in the source field
     * @param component the component to overwrite in the destination field
     */
    void CopyDataFrom(Field<TType, TState> &rhs, size_t rhs_component = 0,
                      size_t component = 0)
    {
        ASSERTL0(rhs_component < rhs.GetNumComponents() && rhs_component > 0,
                 "rhs_component is out of range!");
        ASSERTL0(component < GetNumComponents() && component > 0,
                 "component is out of range!");

        const auto &rhs_blocks = rhs.GetBlocks();
        ASSERTL0(rhs_blocks.size() == block_attributes.size(),
                 "Number of blocks are not the same!");
        ASSERTL0(rhs.m_curVecWidth == m_curVecWidth,
                 "Vector width are not the same!");

        size_t scalar_field_size = GetFieldSize();

        const TType *rhs_ptr =
            rhs.GetStorage().GetCPUPtr() + rhs_component * scalar_field_size;
        TType *ptr = GetStorage().GetCPUPtr() + component * scalar_field_size;

        for (auto const &block : block_attributes)
        {
            for (size_t pt = 0; pt < block.block_size; ++pt)
            {
                *ptr++ = *rhs_ptr++;
            }
        }
    }

    /**
     * @brief Get BlockAttributes for the field.
     *
     * @return std::vector<BlockAttributes>
     */
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

    /**
     * @brief Gets the Field size of a single component.
     *
     * @return size_t
     */
    size_t GetFieldSize()
    {
        return m_storage->size() / GetNumComponents();
    }

    /**
     * @brief Gets the alignment of the field.
     *
     * @return size_t
     */
    size_t GetAlignment()
    {
        return m_alignment;
    }

    /**
     * @brief Get current vector width.
     *
     * @return size_t
     */
    size_t GetVecWidth()
    {
        return m_curVecWidth;
    }

    Nektar::Array<Nektar::OneD, TType> toArray() const
    {
        return Nektar::Array<Nektar::OneD, TType>(m_storage->size(),
                                                  m_storage->GetCPUPtr());
    }

private:
    /**
     * @brief Reshapes the current storage interleaving to a non-interleaved
     * arrangement. For multi-component fields, all paddings will be placed
     * at the end of each component after this operation.
     */
    void ReshapeToScalar()
    {
        if (m_curVecWidth == 1)
            return;

        size_t scalar_field_size = GetFieldSize();

        for (int component = 0; component < GetNumComponents(); ++component)
        {
            TType *inptr =
                m_storage->GetCPUPtr() + component * scalar_field_size;

            for (const auto &block : block_attributes)
            {
                const size_t numMetaBlocks =
                    (block.num_elements + block.num_padding_elements) /
                    m_curVecWidth;
                const size_t MetaBlockSize = m_curVecWidth * block.num_pts;

                Nektar::Array<Nektar::OneD, TType> temp(MetaBlockSize, 0.0);

                for (size_t metaBlock = 0; metaBlock < numMetaBlocks;
                     ++metaBlock)
                {
                    // Copy data into temporary storage because inptr and outptr
                    // access the same memory location
                    std::copy(inptr, inptr + MetaBlockSize, temp.get());
                    Deinterleave(temp.get(), block.num_pts, inptr);
                    inptr += MetaBlockSize;
                }
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
     * @param components Names of components for vector field.
     */
    Field(std::vector<BlockAttributes> blocks, int num_components = 1)
        : block_attributes(std::move(blocks)), component_names(num_components)
    {
    }

    Field(std::vector<BlockAttributes> blocks,
          std::vector<std::string> components = {"u"})
        : block_attributes(std::move(blocks)),
          component_names(std::move(components))
    {
    }

    std::unique_ptr<MemoryRegionCPU<TType>> m_storage;
    std::vector<BlockAttributes> block_attributes;
    std::vector<std::string> component_names = {"u"};

    size_t m_curVecWidth = 1;
    size_t m_alignment   = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
};
