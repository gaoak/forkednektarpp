///////////////////////////////////////////////////////////////////////////////
//
// File: AssmbScatrOpImpl.hpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description:
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AssmbScatrOpImpl.hpp
 * @brief Backend implementations of the AssmbScatr (assemble--scatter)
 * operator, including the parallel path that overlaps interior assembly
 * with inter-rank communication.
 *
 * @details
 * Assemble--scatter takes a coefficient-space field in element-local
 * storage and, for every global degree of freedom (DOF) to which several
 * local coefficients contribute, replaces each local copy by the
 * sign-weighted sum of all of them. It is the operator-library equivalent
 * of AssemblyMap::Assemble() followed by AssemblyMap::GlobalToLocal(),
 * but acts in place on the element-local data without ever forming the
 * globally-ordered array.
 *
 * The computational kernels live in AssmbScatrSerialAVXKernels.hpp
 * (Serial and AVX back-ends) and AssmbScatrDeviceKernels.hpp (device
 * back-ends); overload resolution on the ExecSpace template parameter
 * selects between them. The index and sign tables the kernels consume are
 * built once per configuration by MultiRegions::LocalToGlobalDataCreator
 * (see MultiRegions/DataWarehouse/LocalToGlobalDataWarehouse.hpp) and
 * cached in the expansion list's data warehouse, so several operator
 * instances share them.
 */

#pragma once

#include <MultiRegions/DataWarehouse/LocalToGlobalDataWarehouse.hpp>

#include "Operators/AssmbScatr/AssmbScatrOp.hpp"

#include "Operators/AssmbScatr/AssmbScatrDeviceKernels.hpp"
#include "Operators/AssmbScatr/AssmbScatrSerialAVXKernels.hpp"
#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>

using namespace Nektar;
using namespace Nektar::Operators;

namespace Nektar::Operators::detail
{

/**
 * @brief Backend implementation of AssmbScatrOp: in-place
 * assemble--scatter of a coefficient field for one execution space.
 *
 * @details
 * ### What one application does
 * Only the interior DOFs that need work are visited: those with more than one
 * local contribution (assemble followed by scatter is the identity on a DOF
 * with a single contribution, because the +-1 sign factors square to
 * one) and, when ZERODIR is set, global Dirichlet DOFs. For each visited
 * DOF the kernels accumulate the sign-weighted local coefficients and
 * write the sign-weighted sum back through the same indices.
 *
 * ### Index/sign layout and WIDTH interleaving
 * The tables address the field's coefficient storage as one flat array:
 * each index already includes the block and component offsets, and the
 * blocks of a Field are allocated contiguously (see
 * Field::AllocateFieldStorage), so block 0's pointer serves as the base
 * of the whole field. The interior tables are interleaved in groups of
 * #m_device_width (WIDTH) DOFs: the j-th index/sign entries of the WIDTH
 * DOFs of a group are adjacent in memory. On the Device back-end the
 * lanes of a warp therefore read them with unit stride (coalesced); on
 * AVX the tables are interleaved at the SIMD width, although the current
 * kernel walks the lanes with scalar loops. Each group occupies a
 * rectangular block sized by its largest valence (the group is permuted
 * so that its first DOF carries that valence); surplus slots repeat a
 * valid index with sign 0 so they are harmless if read. For the Serial
 * back-end WIDTH is 1 and the tables are simply packed.
 *
 * ### The sign table
 * #m_gsSign holds one factor per index entry:
 * - the +-1 factors provided by AssemblyMapCG::GetLocalToGlobalSign()
 *   whenever the assembly map applies sign changes and SIGNCHANGE is
 *   true;
 * - their absolute values when SIGNCHANGE is false;
 * - 0 for global Dirichlet DOFs when ZERODIR is true, so that their
 *   contributions are ignored on assembly and their local copies are
 *   zeroed on scatter;
 * - 0 in the padding slots described above.
 * Folding all of this into one table lets a single kernel serve every
 * ZERODIR/SIGNCHANGE combination without branching.
 *
 * ### Parallel operation
 * DOFs whose contributions all live on this rank ("interior") and DOFs
 * shared with other ranks ("boundary") are held in disjoint tables. One
 * application then proceeds as follows:
 * -# AssembleScatrBndKernel: per boundary DOF, form this rank's partial
 *    sum, park it (sign-weighted) in the DOF's first local coefficient
 *    and copy it into #m_send_buffer -- one slot per neighbouring rank
 *    sharing the DOF;
 * -# m_assmbCommCG->BeginComm(): start the persistent sends/receives;
 * -# AssembleScatrKernel: assemble--scatter all interior DOFs,
 *    overlapping with the message exchange;
 * -# m_assmbCommCG->EndComm(): wait for the exchange to complete;
 * -# AssembleFromBndKernel: per boundary DOF, sum the neighbours'
 *    partial sums from #m_recv_buffer together with the parked local one
 *    and scatter the total. The additions are performed in ascending
 *    rank order on every rank (#m_gsBndAssOrder gives this rank's slot),
 *    so all sharers accumulate in the same order and obtain identical
 *    floating-point values for the shared DOF.
 *
 * @tparam ExecSpace  Execution back-end (NektarSpaces::Serial, AVX or
 *                    Device); its memory_space determines where the
 *                    field data and the tables must reside.
 * @tparam TData      Floating-point type of the coefficients.
 * @tparam ZERODIR    Zero global Dirichlet DOFs instead of assembling
 *                    them (see AssmbScatrZeroDirOpImpl).
 * @tparam SIGNCHANGE Apply the assembly map's +-1 orientation factors;
 *                    if false their absolute values are used instead
 *                    (see AssmbScatrNoSignOpImpl).
 *
 * @see AssmbScatrOp for the public Apply interface and factory hooks.
 */
template <typename ExecSpace, typename TData, bool ZERODIR = false,
          bool SIGNCHANGE = true>
class AssmbScatrOpImpl : public AssmbScatrOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    /// Forwards to AssmbScatrOp; the assembly tables and, in parallel, the
    /// communication are set up lazily by SetUpMaps() on the first Apply().
    AssmbScatrOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                     const std::vector<std::string> &components)
        : AssmbScatrOp<TData>(expansionList, components)
    {
    }

    /// Operator class name; defined by the generated factory code.
    static std::string className;

    /// Creator function registered with OperatorFactory.
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<
            AssmbScatrOpImpl<ExecSpace, TData, ZERODIR, SIGNCHANGE>>(
            expansionList, components);
    }

protected:
    /// Interleave width of the interior tables: 1 (Serial), the SIMD
    /// vector width (AVX) or the device warp size (Device).
    static constexpr unsigned m_device_width =
        NektarSpaces::vector_width<ExecSpace, TData>::value;
    /// Number of local contributions (valence) per interior DOF.
    const unsigned *m_gsNumAssmb = nullptr;
    /// Interleaved local-coefficient indices of the interior DOFs.
    const unsigned *m_gsIndex = nullptr;
    /// First entry of each interior DOF within #m_gsIndex / #m_gsSign.
    const unsigned *m_gsOffset = nullptr;
    /// Factor per #m_gsIndex entry: +-1, or 0 to disable the entry.
    const int *m_gsSign = nullptr;
    /// Number of local contributions (valence) per boundary DOF.
    const unsigned *m_gsBndNumAssmb = nullptr;
    /// Number of neighbouring ranks sharing each boundary DOF.
    const unsigned *m_gsNumBndVals = nullptr;
    /// Per boundary DOF: local coefficient indices, then send/receive
    /// buffer slots in ascending neighbour-rank order (packed, width 1).
    const unsigned *m_gsBndIndex = nullptr;
    /// First entry of each boundary DOF in #m_gsBndIndex / #m_gsBndSign.
    const unsigned *m_gsBndOffset = nullptr;
    /// Slot of this rank's own partial sum in the rank-ordered boundary
    /// summation, i.e. the number of sharing ranks of lower rank.
    const unsigned *m_gsBndAssOrder = nullptr;
    /// Factors for the local entries of #m_gsBndIndex (0 in the buffer
    /// slot positions).
    const int *m_gsBndSign = nullptr;
    /// Spatial (row) communicator over which the mesh is partitioned.
    LibUtilities::CommSharedPtr m_rowComm;

    /// Number of interior assembly targets, i.e. (global DOF, component)
    /// pairs visited by AssembleScatrKernel.
    unsigned m_nGids;
    /// Number of partition-boundary assembly targets visited by the
    /// boundary kernels; set in parallel runs only.
    unsigned m_nBndGids;
    /// Number of field components the tables and buffers are set up for;
    /// 0 until the first Apply() calls SetUpMaps().
    unsigned m_numAssemblyComps = 0;

    /// True when the row communicator has more than one rank, i.e. when
    /// the boundary exchange is active; set by SetUpMaps().
    bool m_isParallel;
    /// Persistent inter-rank exchange of the partition-boundary partial
    /// sums; created by SetUpMaps() in parallel runs only.
    std::unique_ptr<MultiRegions::AssemblyComm<TData>> m_assmbCommCG;
    /// Send buffer of the boundary exchange (pinned host memory): one
    /// entry per (neighbouring rank, shared DOF, component).
    LibUtilities::MemoryRegion<TData> m_send_buffer;
    /// Receive buffer of the boundary exchange, laid out like
    /// #m_send_buffer.
    LibUtilities::MemoryRegion<TData> m_recv_buffer;

    /**
     * @brief Fetch (or build) the assembly tables for the operator's
     * component list and, in parallel, set up the boundary tables, the
     * persistent communication and the message buffers.
     *
     * The interior tables are obtained from the data warehouse keyed on
     * the operator's components, the ZERODIR flag and #m_device_width
     * (the sign table additionally on SIGNCHANGE), so they are created
     * once per configuration and shared between operator instances.
     *
     * When the row communicator has more than one rank, a
     * MultiRegions::AssemblyComm is constructed from the universal IDs
     * of the global boundary coefficients (interior coefficients are
     * masked with 0 and never communicated). Its constructor discovers,
     * collectively, which ranks share each DOF; InitSendRecvComms() then
     * creates persistent point-to-point requests bound directly to
     * #m_send_buffer / #m_recv_buffer -- to their device pointers when
     * the operator runs in the device memory space and the communicator
     * is GPU-aware, and to their pinned host pointers otherwise. Because
     * the requests capture raw pointers, the buffers must stay allocated
     * for the lifetime of #m_assmbCommCG. Each buffer holds one entry
     * per (neighbouring rank, shared DOF, component) triple, grouped
     * into one contiguous block per neighbouring rank.
     *
     * Called lazily from v_Apply() whenever the number of components
     * differs from the previous call; note that the construction is
     * collective, so all ranks must take that branch together.
     *
     * @param   numComp     Number of field components the communication
     *                      buffers are sized for; the tables themselves
     *                      are keyed on the operator's component list.
     */
    void SetUpMaps(unsigned numComp)
    {
        m_numAssemblyComps = numComp;

        // setup GS info of values interior to device
        auto GSNumAssmbKey =
            MultiRegions::DeviceLocalToGlobalNumAssembleKey<TData>(
                ZERODIR, this->m_components, m_device_width);
        m_gsNumAssmb =
            this->m_dataWarehouse->template GetData<MemSpace>(GSNumAssmbKey);

        auto GSIndexKey = MultiRegions::DeviceLocalToGlobalIndexKey<TData>(
            ZERODIR, this->m_components, m_device_width);
        m_gsIndex =
            this->m_dataWarehouse->template GetData<MemSpace>(GSIndexKey);

        auto GSOffsetKey =
            MultiRegions::DeviceLocalToGlobalIndexOffsetKey<TData>(
                ZERODIR, this->m_components, m_device_width);
        m_gsOffset =
            this->m_dataWarehouse->template GetData<MemSpace>(GSOffsetKey);

        // set up sign change array
        m_gsSign = this->m_dataWarehouse->template GetData<MemSpace>(
            MultiRegions::DeviceLocalToGlobalSignKey<TData>(
                ZERODIR, SIGNCHANGE, this->m_components, m_device_width));

        auto GSInfoKey = MultiRegions::DeviceLocalToGlobalKey<TData>(
            ZERODIR, this->m_components, m_device_width);
        // get a copy of the host to evaluate number of GIDs to assemble
        auto hostGSInfo =
            this->m_dataWarehouse->template GetData<NektarSpaces::HostSpace>(
                GSInfoKey);

        m_nGids = hostGSInfo[0];

        m_rowComm =
            this->m_expansionList->GetSession()->GetComm()->GetRowComm();
        m_isParallel = (m_rowComm->GetSize() > 1) ? true : false;

        if (m_isParallel)
        {
            auto contfield = std::dynamic_pointer_cast<MultiRegions::ContField>(
                this->m_expansionList);
            auto assemblyMap =
                std::dynamic_pointer_cast<MultiRegions::AssemblyMapCG>(
                    contfield->GetLocalToGlobalMap());
            auto globalToUniMap     = assemblyMap->GetGlobalToUniversalMap();
            auto numGlobalCoeffs    = assemblyMap->GetNumGlobalCoeffs();
            auto numGlobalBndCoeffs = assemblyMap->GetNumGlobalBndCoeffs();
            Nektar::Array<OneD, long> tmp(numGlobalCoeffs);
            Vmath::Zero(numGlobalCoeffs, tmp, 1);
            for (unsigned int i = 0; i < numGlobalBndCoeffs; ++i)
            {
                tmp[i] = globalToUniMap[i];
            }
            m_assmbCommCG = std::make_unique<MultiRegions::AssemblyComm<TData>>(
                m_rowComm, tmp);

            // setup GS info of values interior to device
            auto GSBndNumAssmbKey =
                MultiRegions::DeviceBndLocalToGlobalNumAssembleKey<TData>(
                    this->m_components);
            m_gsBndNumAssmb = this->m_dataWarehouse->template GetData<MemSpace>(
                GSBndNumAssmbKey);

            auto GSNumBndValsKey =
                MultiRegions::DeviceBndLocalToGlobalNumBndValsKey<TData>(
                    this->m_components);
            m_gsNumBndVals = this->m_dataWarehouse->template GetData<MemSpace>(
                GSNumBndValsKey);

            auto GSBndIndexKey =
                MultiRegions::DeviceBndLocalToGlobalIndexKey<TData>(
                    this->m_components);
            m_gsBndIndex = this->m_dataWarehouse->template GetData<MemSpace>(
                GSBndIndexKey);

            auto GSBndOffsetKey =
                MultiRegions::DeviceBndLocalToGlobalOffsetKey<TData>(
                    this->m_components);
            m_gsBndOffset = this->m_dataWarehouse->template GetData<MemSpace>(
                GSBndOffsetKey);

            auto GSBndAssembleOrderKey =
                MultiRegions::DeviceBndLocalToGlobalAssembleOrderKey<TData>(
                    this->m_components);
            m_gsBndAssOrder = this->m_dataWarehouse->template GetData<MemSpace>(
                GSBndAssembleOrderKey);

            // set up bnd sign change array
            m_gsBndSign = this->m_dataWarehouse->template GetData<MemSpace>(
                MultiRegions::DeviceBndLocalToGlobalSignKey<TData>(
                    ZERODIR, SIGNCHANGE, this->m_components));

            // setup GS info of values for parallal boundary of device
            auto GSBndInfoKey = MultiRegions::DeviceBndLocalToGlobalKey<TData>(
                this->m_components);

            // get a copy of the host to evaluate  number of GiDs to assemble
            auto hostGSBndInfo =
                this->m_dataWarehouse
                    ->template GetData<NektarSpaces::HostSpace>(GSBndInfoKey);

            m_nBndGids = hostGSBndInfo[0];

            auto nbuf = m_assmbCommCG->GetSREntries().size() * numComp;

            m_send_buffer =
                LibUtilities::MemoryRegion<TData>(nbuf, eHostPinned);
            m_recv_buffer =
                LibUtilities::MemoryRegion<TData>(nbuf, eHostPinned);

            // Assign pointer for future communication requests. Pointers are
            // used here directly, without memory syncronisation by accessing
            // the private member variables. Memory synchronisation will only be
            // required when BeginComm() and and EndComm() are called.
            TData *send_buffer_ptr, *recv_buffer_ptr;
            if (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace> &&
                m_rowComm->IsGPUAware())
            {
                m_send_buffer.template Initialize<NektarSpaces::DeviceSpace>(
                    0.0);
                m_recv_buffer.template Initialize<NektarSpaces::DeviceSpace>(
                    0.0);
                send_buffer_ptr =
                    (TData *)m_send_buffer
                        .template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
                recv_buffer_ptr =
                    (TData *)m_recv_buffer
                        .template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
            }
            else
            {
                m_send_buffer.template Initialize<NektarSpaces::HostSpace>(0.0);
                m_recv_buffer.template Initialize<NektarSpaces::HostSpace>(0.0);
                send_buffer_ptr =
                    (TData *)m_send_buffer
                        .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
                recv_buffer_ptr =
                    (TData *)m_recv_buffer
                        .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            }
            m_assmbCommCG->InitSendRecvComms(nbuf, send_buffer_ptr,
                                             recv_buffer_ptr, numComp);
        }
    }

    /**
     * @brief Copying overload: assemble--scatter @p in into @p out.
     *
     * @p in is copied into @p out (skipped when both refer to the same
     * field) and the in-place overload is applied to @p out; @p in is
     * otherwise left untouched.
     *
     * @param   in      Coefficient-space input field.
     * @param   out     Coefficient-space output field; must match @p in
     *                  in block size and number of components.
     */
    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &in,
                 LibUtilities::Field<TData, FieldState::Coeff> &out) override
    {
        ASSERTL1(in.GetBlocks()[0].CompSize() == out.GetBlocks()[0].CompSize(),
                 "In and out blocks are of different size");

        ASSERTL1(in.GetNumComponents() == out.GetNumComponents(),
                 "In and out have different number of components");

        if (&in != &out)
        {
            out.template Copy<MemSpace>(in);
        }

        v_Apply(out);
    }

    /**
     * @brief Assemble and scatter @p inout in place, overlapping the
     * interior kernel with inter-rank communication in parallel.
     *
     * The steps are:
     * -# Rebuild the tables and communication via SetUpMaps() if the
     *    number of components differs from the current setup.
     * -# Reshape any vector-interleaved block to interleave width 1,
     *    since the index tables address coefficients in non-interleaved
     *    layout; the original width is restored before returning.
     * -# Fetch every block's pointer in MemSpace so the whole field is
     *    resident there; the kernels then address the field's contiguous
     *    storage through block 0's base pointer alone.
     * -# Parallel only: AssembleScatrBndKernel partially assembles the
     *    partition-boundary DOFs into #m_send_buffer, parking each
     *    partial sum in its DOF's first local coefficient. The buffer is
     *    then made visible to MPI -- copied to the host or, with a
     *    GPU-aware communicator, published by synchronising the kernel
     *    stream -- and BeginComm() starts the persistent exchange.
     * -# AssembleScatrKernel assembles and scatters the interior DOFs;
     *    in parallel this overlaps with the message exchange.
     * -# Parallel only: #m_recv_buffer is marked as written in the
     *    memory space MPI delivers into, EndComm() waits for the
     *    exchange, and AssembleFromBndKernel completes the boundary DOFs
     *    from the received partial sums in ascending rank order (see the
     *    class notes on cross-rank determinism).
     *
     * @param   inout   Coefficient-space field, assembled and scattered
     *                  in place.
     */
    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &inout) override
    {
        auto numComp = inout.GetNumComponents();

        if (numComp != m_numAssemblyComps)
        {
            // set up local to global device and device boundary maps for this
            // number of components
            SetUpMaps(numComp);
        }

        // reshape data into non-interleaved if ncessary
        bool reshapeOutput = false;
        std::vector<unsigned> save_width;
        for (unsigned blk = 0; blk < inout.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;

            auto &inoutblk  = inout.GetBlocks()[blk];
            auto inoutwidth = inoutblk.GetInterleaveWidth();
            save_width.push_back(inoutwidth);

            if (inoutwidth != 1)
            {
                auto inoutPtr =
                    inoutblk.template GetPtr<MemSpace, ReadWrite>(streamID);

                LibUtilities::ReshapeStorage<ExecSpace>(
                    1u, inoutwidth,
                    inoutblk.GetNumElementsWithPadding() * numComp,
                    inoutblk.GetNumData(), inoutPtr, streamID);
                inoutblk.template SetInterleaveWidth<TData>(1);
                reshapeOutput = true;
            }
        }

        // Initialize pointer.
        const unsigned int streamID0 = 1;

        auto inoutPtr =
            inout.GetBlocks()[0].template GetPtr<MemSpace, ReadWrite>(
                streamID0);
        for (unsigned blk = 1; blk < inout.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;

            inout.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(
                streamID);
        }

        if (m_isParallel)
        {
            // setup send buffer pointer
            auto sendPtr = m_send_buffer.template GetPtr<MemSpace, WriteOnly>();

            //  assemble data into boundary send buffer
            AssembleScatrBndKernel<ExecSpace>(
                m_nBndGids, m_gsBndNumAssmb, m_gsNumBndVals, m_gsBndIndex,
                m_gsBndOffset, m_gsBndSign, inoutPtr, sendPtr);

            // Synchronize memory, if necessary.
            if (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace> &&
                !m_rowComm->IsGPUAware())
            {
                // Data must be copied to the host without GPU-aware MPI before
                // communication. No memory copy is require otherwise.
                m_send_buffer
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            }
            else if (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace> &&
                     m_rowComm->IsGPUAware())
            {
                // Synchronize stream before communication.
                nekStreamSynchronize(0);
            }

            // start  comms
            m_assmbCommCG->BeginComm();
        }

        AssembleScatrKernel<ExecSpace>(m_nGids, m_gsNumAssmb, m_gsIndex,
                                       m_gsOffset, m_gsSign, inoutPtr);

        if (m_isParallel)
        {
            // Synchronize memory, if necessary.
            if (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace> &&
                m_rowComm->IsGPUAware())
            {
                // Data is received on the device with GPU-aware MPI.
                m_recv_buffer
                    .template GetPtr<NektarSpaces::DeviceSpace, WriteOnly>();
            }
            else
            {
                // Data is received on the host without GPU-aware MPI.
                m_recv_buffer
                    .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            }

            // finish comms and syncronize
            m_assmbCommCG->EndComm();

            auto recvPtr = m_recv_buffer.template GetPtr<MemSpace, ReadOnly>();

            //  assemble data into boundary send buffer
            AssembleFromBndKernel<ExecSpace>(
                m_nBndGids, m_gsBndNumAssmb, m_gsNumBndVals, m_gsBndIndex,
                m_gsBndOffset, m_gsBndSign, m_gsBndAssOrder, recvPtr, inoutPtr);
        }

        // reshape data into non-interleaved if necessary
        if (reshapeOutput)
        {
            for (unsigned blk = 0; blk < inout.GetBlocks().size(); ++blk)
            {
                const unsigned int streamID = blk + 1;

                auto &inoutblk = inout.GetBlocks()[blk];
                auto width     = save_width[blk];

                if (inoutblk.GetInterleaveWidth() != width)
                {
                    ASSERTL1(width == m_device_width, "Unexpected width value");
                    auto inoutPtr =
                        inoutblk.template GetPtr<MemSpace, ReadWrite>(streamID);

                    LibUtilities::ReshapeStorage<ExecSpace>(
                        width, 1u,
                        inoutblk.GetNumElementsWithPadding() * numComp,
                        inoutblk.GetNumData(), inoutPtr, streamID);
                    inoutblk.template SetInterleaveWidth<TData>(width);
                }
            }
        }
    }
};

/**
 * @brief Specialised assemble--scatter with zero Dirichlet action:
 * global Dirichlet DOFs are zeroed rather than assembled.
 *
 * Equivalent to AssemblyMap::Assemble(), zeroing the global Dirichlet
 * coefficients, then AssemblyMap::GlobalToLocal(). Realised through
 * AssmbScatrOpImpl with ZERODIR set, i.e. via zero entries in the sign
 * tables rather than a separate kernel; Dirichlet DOFs with a single
 * local contribution are visited too, so their local copies are zeroed
 * as well.
 *
 * The generated factory code registers this class under
 * AssmbScatrZeroDirOp::name. The registered creator is the inherited
 * Instantiate(), which constructs an AssmbScatrOpImpl<ExecSpace, TData,
 * true, true> rather than this class; the two differ only in #className.
 */
template <typename ExecSpace, typename TData>
class AssmbScatrZeroDirOpImpl
    : public AssmbScatrOpImpl<ExecSpace, TData, true, true>
{
public:
    AssmbScatrZeroDirOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                            const std::vector<std::string> &components)
        : AssmbScatrOpImpl<ExecSpace, TData, true, true>(expansionList,
                                                         components)
    {
    }
    /// Operator class name; defined by the generated factory code.
    static std::string className;
};

/**
 * @brief Specialised assemble--scatter with no sign change: the
 * assembly map's orientation factors are replaced by their absolute
 * values, so contributions are summed without sign reconciliation.
 *
 * Not registered with the operator factory: no AssmbScatrOp-derived
 * interface class names it and no generated factory code defines
 * #className. The diagonal preconditioner (DiagPreconOpImpl) constructs
 * it directly.
 */
template <typename ExecSpace, typename TData>
class AssmbScatrNoSignOpImpl
    : public AssmbScatrOpImpl<ExecSpace, TData, false, false>
{
public:
    AssmbScatrNoSignOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                           const std::vector<std::string> &components)
        : AssmbScatrOpImpl<ExecSpace, TData, false, false>(expansionList,
                                                           components)
    {
    }
    /// Operator class name; unlike the registered implementations, no
    /// generated factory code defines it.
    static std::string className;
};
} // namespace Nektar::Operators::detail
