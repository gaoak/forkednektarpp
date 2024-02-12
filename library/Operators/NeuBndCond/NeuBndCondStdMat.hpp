///////////////////////////////////////////////////////////////////////////////
//
// File: NeuBndCondStdMat.hpp
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

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

#include "Operators/OperatorNeuBndCond.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{
template <typename TData>
class OperatorNeuBndCondImpl<TData, ImplStdMat>
    : public OperatorNeuBndCond<TData>
{
public:
    OperatorNeuBndCondImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorNeuBndCond<TData>(expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        m_assmbMap = contfield->GetLocalToGlobalMap();
    }

    void apply(Field<TData, FieldState::Coeff> &inout) override
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        auto &bndCondExpansions = contfield->GetBndCondExpansions();
        auto &bndConditions     = contfield->GetBndConditions();
        auto &sign   = m_assmbMap->GetBndCondCoeffsToLocalCoeffsSign();
        auto &map    = m_assmbMap->GetBndCondCoeffsToLocalCoeffsMap();
        auto ncoeffs = m_assmbMap->GetNumLocalCoeffs();

        // Copy data from input field
        Array<OneD, NekDouble> inoutarr(ncoeffs);
        auto *inarrptr = inoutarr.data();
        auto *inptr    = inout.GetStorage().GetCPUPtr();

        for (size_t block_idx = 0; block_idx < inout.GetBlocks().size();
             ++block_idx)
        {
            auto nSize  = inout.GetBlocks()[block_idx].block_size;
            auto nElmts = inout.GetBlocks()[block_idx].num_elements;
            auto nmTot  = inout.GetBlocks()[block_idx].num_pts;

            std::copy(inptr, inptr + nElmts * nmTot, inarrptr);

            inarrptr += nElmts * nmTot;
            inptr += nSize;
        }

        size_t bndcnt = 0;
        // Add weak boundary conditions to forcing
        for (size_t i = 0; i < bndCondExpansions.size(); ++i)
        {
            if (bndConditions[i]->GetBoundaryConditionType() ==
                    SpatialDomains::eNeumann ||
                bndConditions[i]->GetBoundaryConditionType() ==
                    SpatialDomains::eRobin)
            {
                auto &bndcoeff = bndCondExpansions[i]->GetCoeffs();
                if (m_assmbMap->GetSignChange())
                {
                    for (size_t j = 0; j < bndCondExpansions[i]->GetNcoeffs();
                         j++)
                    {
                        inoutarr[map[bndcnt + j]] +=
                            sign[bndcnt + j] * bndcoeff[j];
                    }
                }
                else
                {
                    for (size_t j = 0; j < bndCondExpansions[i]->GetNcoeffs();
                         j++)
                    {
                        inoutarr[map[bndcnt + j]] += bndcoeff[j];
                    }
                }
            }
            bndcnt += bndCondExpansions[i]->GetNcoeffs();
        }

        // Copy data to output field
        auto *outarrptr = inoutarr.data();
        auto *outptr    = inout.GetStorage().GetCPUPtr();

        for (size_t block_idx = 0; block_idx < inout.GetBlocks().size();
             ++block_idx)
        {
            auto nSize  = inout.GetBlocks()[block_idx].block_size;
            auto nElmts = inout.GetBlocks()[block_idx].num_elements;
            auto nmTot  = inout.GetBlocks()[block_idx].num_pts;

            std::copy(outarrptr, outarrptr + nElmts * nmTot, outptr);

            outarrptr += nElmts * nmTot;
            outptr += nSize;
        }
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorNeuBndCondImpl<TData, ImplStdMat>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    AssemblyMapCGSharedPtr m_assmbMap;
};

} // namespace Nektar::Operators::detail
