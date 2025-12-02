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

#pragma once

#include <MultiRegions/ContField.h>

#include "Operators/AssmbScatr/AssmbScatrOp.hpp"

#include "Operators/AssmbScatr/AssmbScatrDeviceKernels.hpp"
#include "Operators/AssmbScatr/AssmbScatrSerialAVXKernels.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::Operators;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData, bool ZERODIR = false,
          bool SIGNCHANGE = true>
class AssmbScatrOpImpl : public AssmbScatrOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    // scalar version
    AssmbScatrOpImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : AssmbScatrOp<TData>(expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);

        auto assmbMap = std::dynamic_pointer_cast<MultiRegions::AssemblyMapCG>(
            contfield->GetLocalToGlobalMap());

        m_assemblyMap.push_back(assmbMap);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            AssmbScatrOpImpl<ExecSpace, TData, ZERODIR, SIGNCHANGE>>(
            expansionList);
    }

protected:
    // #define ORIG_ASS_SCA
#ifdef ORIG_ASS_SCA
    const unsigned *m_gsInfo = nullptr;
#else
    const unsigned *m_gsNumAssmb = nullptr;
    const unsigned *m_gsIndex    = nullptr;
    const unsigned *m_gsOffset   = nullptr;
    static constexpr unsigned m_device_width =
        NektarSpaces::vector_width<ExecSpace, double>::value;
#endif
    const int *m_gsSign         = nullptr;
    const unsigned *m_gsBndInfo = nullptr;
    const int *m_gsBndSign      = nullptr;
    /// number of internal device  dofs ot assemble.
    unsigned m_nGids;
    /// number of inter device boundary dofs to assemble
    unsigned m_nBndGids;
    /// number of components mappings are setup for
    unsigned m_numAssemblyComps = 0;
    /// A vector of legacy assemblyCG maps for generation of boundary mappings
    std::vector<MultiRegions::AssemblyMapCGSharedPtr> m_assemblyMap;

    /// flag to identify when method is setup for parallel communication
    bool m_isParallel;
    /// A pointer to the assembly communication for inter device commonication
    std::unique_ptr<MultiRegions::AssemblyCommCG> m_assmbCommCG;
    /// Buffer to place send data for inter device communication
    MemoryRegion<TData> m_send_buffer;
    /// Buffer to receive data into  for inter device communication
    MemoryRegion<TData> m_recv_buffer;

    void v_SetAssemblyMap(
        std::vector<MultiRegions::AssemblyMapCGSharedPtr> &assemblyMap) override
    {
        m_assemblyMap = assemblyMap;
    }

    void SetUpMaps(unsigned numComps)
    {
        ASSERTL1(numComps == m_assemblyMap.size(),
                 "The number of assembly maps is not the same as the number of "
                 "components requested. Have you set up maps using "
                 "SetAssemblyMaps()");

        m_numAssemblyComps = numComps;

        auto blocks =
            GetBlockAttributes<TData>(FieldState::Coeff, this->m_expansionList);

#ifdef ORIG_ASS_SCA
        auto GSInfoKey = DeviceLocalToGlobalKey<TData>(m_assemblyMap, ZERODIR);
        // setup GS info of values interior to device
        m_gsInfo = this->m_dataWarehouse->template GetData<MemSpace>(GSInfoKey);
        // set up sign change array
        m_gsSign = this->m_dataWarehouse->template GetData<MemSpace>(
            DeviceLocalToGlobalSignKey<TData>(m_assemblyMap, ZERODIR,
                                              SIGNCHANGE));
#else
        auto GSInfoKey = DeviceLocalToGlobalKey<TData>(m_assemblyMap, ZERODIR,
                                                       m_device_width);
        // setup GS info of values interior to device
        auto GSNumAssmbKey = DeviceLocalToGlobalNumAssembleKey<TData>(
            m_assemblyMap, ZERODIR, m_device_width);
        m_gsNumAssmb =
            this->m_dataWarehouse->template GetData<MemSpace>(GSNumAssmbKey);

        auto GSIndexKey = DeviceLocalToGlobalIndexKey<TData>(
            m_assemblyMap, ZERODIR, m_device_width);
        m_gsIndex =
            this->m_dataWarehouse->template GetData<MemSpace>(GSIndexKey);

        auto GSOffsetKey = DeviceLocalToGlobalIndexOffsetKey<TData>(
            m_assemblyMap, ZERODIR, m_device_width);
        m_gsOffset =
            this->m_dataWarehouse->template GetData<MemSpace>(GSOffsetKey);

        // set up sign change array
        m_gsSign = this->m_dataWarehouse->template GetData<MemSpace>(
            DeviceLocalToGlobalSignKey<TData>(m_assemblyMap, ZERODIR,
                                              SIGNCHANGE, m_device_width));
#endif

        // get a copy of the host to evaluate offsets
        auto hostGSInfo =
            this->m_dataWarehouse->template GetData<NektarSpaces::HostSpace>(
                GSInfoKey);

        m_nGids = hostGSInfo[0];

        auto vCommRow =
            this->m_expansionList->GetSession()->GetComm()->GetRowComm();
        m_isParallel = (vCommRow->GetSize() > 1) ? true : false;

        if (m_isParallel)
        {
            auto globalToUniMap  = m_assemblyMap[0]->GetGlobalToUniversalMap();
            auto numGlobalCoeffs = m_assemblyMap[0]->GetNumGlobalCoeffs();
            auto numGlobalBndCoeffs = m_assemblyMap[0]->GetNumGlobalBndCoeffs();
            Nektar::Array<OneD, long> tmp(numGlobalCoeffs);
            Vmath::Zero(numGlobalCoeffs, tmp, 1);
            for (unsigned int i = 0; i < numGlobalBndCoeffs; ++i)
            {
                tmp[i] = globalToUniMap[i];
            }
            m_assmbCommCG =
                std::make_unique<MultiRegions::AssemblyCommCG>(vCommRow, tmp);

            auto numComp = m_assemblyMap.size();

            // setup GS info of values parallal boundary of device
            auto GSBndInfoKey = DeviceBndLocalToGlobalKey<TData>(numComp);

            m_gsBndInfo =
                this->m_dataWarehouse->template GetData<MemSpace>(GSBndInfoKey);

            // set up sign array
            m_gsBndSign = this->m_dataWarehouse->template GetData<MemSpace>(
                DeviceBndLocalToGlobalSignKey<TData>(m_assemblyMap, ZERODIR,
                                                     SIGNCHANGE));

            // get a copy of the host to evaluate offsets
            auto hostGSBndInfo =
                this->m_dataWarehouse
                    ->template GetData<NektarSpaces::HostSpace>(GSBndInfoKey);

            m_nBndGids = hostGSBndInfo[0];

            auto nbuf = m_assmbCommCG->GetSREntries().size() * numComp;

            m_send_buffer = MemoryRegion<TData>(nbuf, ePinned);
            m_send_buffer.template Initialize<NektarSpaces::HostSpace>(0);

            auto sendPtr =
                m_send_buffer
                    .template GetPtr<NektarSpaces::HostSpace, ReadWrite>();

            m_recv_buffer = MemoryRegion<TData>(nbuf, ePinned);
            m_recv_buffer.template Initialize<NektarSpaces::HostSpace>(0);

            auto recvPtr =
                m_recv_buffer
                    .template GetPtr<NektarSpaces::HostSpace, ReadWrite>();

            // this now seems dangerous since we are now resetting explist
            // communicator!
            m_assmbCommCG->InitSendRecvComms(nbuf, sendPtr, recvPtr, numComp);
        }
    }

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        ASSERTL1(in.GetBlocks()[0].size() == out.GetBlocks()[0].size(),
                 "In and out blocks are of different size");

        ASSERTL1(in.GetNumComponents() == out.GetNumComponents(),
                 "In and out have different number of components");

        if (&in != &out)
        {
            out.template Copy<MemSpace>(in);
            for (unsigned blk = 0; blk < in.GetBlocks().size(); ++blk)
            {
                out.GetBlocks()[blk].template SetInterleaveWidth<TData>(
                    in.GetBlocks()[blk].GetInterleaveWidth());
            }
        }

        v_Apply(out);
    }

    void v_Apply(Field<TData, FieldState::Coeff> &inout) override
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
            auto &inoutblk  = inout.GetBlocks()[blk];
            auto inoutwidth = inoutblk.GetInterleaveWidth();
            save_width.push_back(inoutwidth);

            if (inoutwidth != 1)
            {
                auto inoutPtr = inoutblk.template GetPtr<MemSpace, ReadWrite>();
                unsigned blksize = inoutblk.size();
                for (unsigned nc = 0; nc < numComp; ++nc)
                {
                    deInterleave<ExecSpace>(
                        inoutwidth,
                        inoutblk.GetNumElementsWithPadding() / inoutwidth,
                        inoutblk.GetNumData(), inoutPtr + nc * blksize);
                }
                inoutblk.template SetInterleaveWidth<TData>(1);
                reshapeOutput = true;
            }
        }

        // Initialize pointer.
        auto inoutPtr =
            inout.GetBlocks()[0].template GetPtr<MemSpace, ReadWrite>();

        if (m_isParallel)
        {
            // setup send buffer pointer
            auto sendPtr = m_send_buffer.template GetPtr<MemSpace, WriteOnly>();
            //  assemble data into boundary send buffer
            AssembleScatrBndKernel<ExecSpace>(m_nBndGids, m_gsBndInfo,
                                              m_gsBndSign, inoutPtr, sendPtr);
            // Get Pointer to ensure data on host
            m_send_buffer.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            // start  comms
            m_assmbCommCG->BeginComm();
        }

#ifdef ORIG_ASS_SCA
        AssembleScatrKernel<ExecSpace>(m_nGids, m_gsInfo, m_gsSign, inoutPtr);
#else
        AssembleScatrKernel<ExecSpace>(m_nGids, m_gsNumAssmb, m_gsIndex,
                                       m_gsOffset, m_gsSign, inoutPtr);
#endif
        if (m_isParallel)
        {
            // finish comms and syncronize
            m_assmbCommCG->EndComm();
            // Get Pointer to ensure data on device or host as required
            auto recvPtr = m_recv_buffer.template GetPtr<MemSpace, ReadOnly>();
            // Assemble from updated boundary data
            AssembleFromBndKernel<ExecSpace>(m_nBndGids, m_gsBndInfo,
                                             m_gsBndSign, recvPtr, inoutPtr);
        }

        // reshape data into non-interleaved if necessary
        if (reshapeOutput)
        {
            for (unsigned blk = 0; blk < inout.GetBlocks().size(); ++blk)
            {
                auto &inoutblk = inout.GetBlocks()[blk];
                auto width     = save_width[blk];

                if (inoutblk.GetInterleaveWidth() != width)
                {
                    ASSERTL1(width == m_device_width, "Unexpected width value");
                    auto inoutPtr =
                        inoutblk.template GetPtr<MemSpace, ReadWrite>();
                    unsigned blksize = inoutblk.size();
                    for (unsigned nc = 0; nc < numComp; ++nc)
                    {
                        interleave<ExecSpace>(
                            m_device_width,
                            inoutblk.GetNumElementsWithPadding() / width,
                            inoutblk.GetNumData(), inoutPtr + nc * blksize);
                    }
                    inoutblk.template SetInterleaveWidth<TData>(width);
                }
            }
        }
    }
};

// Specialised Assembly with Zero Dirichlet action
template <typename ExecSpace, typename TData>
class AssmbScatrZeroDirOpImpl
    : public AssmbScatrOpImpl<ExecSpace, TData, true, true>
{
public:
    AssmbScatrZeroDirOpImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : AssmbScatrOpImpl<ExecSpace, TData, true, true>(expansionList)
    {
    }
    static std::string className;
};

// Specialised Assembly with no sign change
template <typename ExecSpace, typename TData>
class AssmbScatrNoSignOpImpl
    : public AssmbScatrOpImpl<ExecSpace, TData, false, false>
{
public:
    AssmbScatrNoSignOpImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : AssmbScatrOpImpl<ExecSpace, TData, false, false>(expansionList)
    {
    }
    static std::string className;
};
} // namespace Nektar::Operators::detail
