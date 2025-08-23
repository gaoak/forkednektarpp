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
    AssmbScatrOpImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : AssmbScatrOp<TData>(expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        auto assmbMap = contfield->GetLocalToGlobalMap();

        m_nDir = assmbMap->GetNumGlobalDirBndCoeffs();

        m_numComp = 0; // initialise to zero
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
    void SetUpMaps(const unsigned numComp)
    {
        m_numComp = numComp;
        auto blocks =
            GetBlockAttributes<TData>(FieldState::Coeff, this->m_expansionList);

        auto dataWarehouse = this->m_expansionList->GetDataWarehouseSharedPtr();

        // setup GS info of values interior to device
        auto GSInfoKey = LocalToGlobalKey<TData>(m_nDir, m_numComp, ZERODIR);

        m_gsInfo = dataWarehouse->template GetData<ExecSpace>(GSInfoKey);

        // set up sign change array
        m_gsSign = dataWarehouse->template GetData<ExecSpace>(
            LocalToGlobalSignKey<TData>(m_nDir, m_numComp, ZERODIR,
                                        SIGNCHANGE));

        // get a copy of the host to evaluate offsets
        auto hostGSInfo =
            dataWarehouse->template GetData<NektarSpaces::Serial>(GSInfoKey);

        m_nGids = hostGSInfo[0];
    }

    const unsigned *m_gsInfo;
    const int *m_gsSign;
    unsigned m_nGids;
    unsigned m_numComp;
    unsigned m_nDir;

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

        auto numComp = in.GetNumComponents();

        // initialise gather scatter maps
        if (m_numComp != numComp)
        {
            SetUpMaps(numComp);
            m_numComp = numComp;
        }

        // reshape data into non-interleaved if ncessary
        bool reshapeOutput = false;
        for (unsigned blk = 0; blk < out.GetBlocks().size(); ++blk)
        {
            auto &inoutblk  = out.GetBlocks()[blk];
            auto inoutwidth = inoutblk.GetInterleaveWidth();
            if (inoutwidth != 1)
            {
                auto inoutPtr = inoutblk.template GetPtr<MemSpace, ReadWrite>();
                unsigned blksize = inoutblk.size();
                for (unsigned nc = 0; nc < m_numComp; ++nc)
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
            out.GetBlocks()[0].template GetPtr<MemSpace, ReadWrite>();

        AssembleScatrKernel<ExecSpace>(m_nGids, m_gsInfo, m_gsSign, inoutPtr);

        // reshape data into non-interleaved if ncessary
        if ((&in != &out) && reshapeOutput)
        {

            for (unsigned blk = 0; blk < out.GetBlocks().size(); ++blk)
            {
                auto &outblk   = out.GetBlocks()[blk];
                auto out_width = outblk.GetInterleaveWidth();
                auto in_width  = in.GetBlocks()[blk].GetInterleaveWidth();

                if (out_width != in_width)
                {
                    ASSERTL1(in_width ==
                                 NektarSpaces::vector_width<TData>::value,
                             "Unexpected width value");
                    auto outPtr = outblk.template GetPtr<MemSpace, ReadWrite>();
                    unsigned blksize = outblk.size();
                    for (unsigned nc = 0; nc < m_numComp; ++nc)
                    {
                        interleave<NektarSpaces::vector_width<TData>::value,
                                   ExecSpace>(
                            outblk.GetNumElementsWithPadding() / in_width,
                            outblk.GetNumData(), outPtr + nc * blksize);
                    }
                    outblk.template SetInterleaveWidth<TData>(in_width);
                }
            }
        }

        // TODO: Universal assembly on device.
        if (this->m_expansionList->GetSession()
                ->GetComm()
                ->GetRowComm()
                ->GetSize() > 1)
        {
            ASSERTL0(false, "Not set up for Multiple MPI processes");
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
