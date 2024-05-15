///////////////////////////////////////////////////////////////////////////////
//
// File: DirBndCondSerialStdMat.hpp
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

#include "Operators/DirBndCond/DirBndCondImplBase.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              std::is_same<ExecSpace, NektarSpaces::Serial>::value &&
              std::is_same<Implementation, Operators::StdMat>::value>::type>
class OperatorDirBndCondImpl
    : public OperatorDirBndCondImplBase<ExecSpace, Implementation, TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorDirBndCondImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorDirBndCondImplBase<ExecSpace, Implementation, TData>(
              expansionList)
    {
    }

    void apply(Field<TData, FieldState::Coeff> &out) override
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        auto &locToGloMap       = contfield->GetLocalToGlobalMap();
        auto &bndCondExpansions = contfield->GetBndCondExpansions();
        auto &bndConditions     = contfield->GetBndConditions();
        auto &sign = locToGloMap->GetBndCondCoeffsToLocalCoeffsSign();
        auto &map  = locToGloMap->GetBndCondCoeffsToLocalCoeffsMap();
        auto nloc  = locToGloMap->GetNumLocalCoeffs();

        size_t bndcnt = 0;
        Array<OneD, TData> outarr(nloc, 0.0);
        for (size_t i = 0; i < bndCondExpansions.size(); ++i)
        {
            if (bndConditions[i]->GetBoundaryConditionType() ==
                SpatialDomains::eDirichlet)
            {
                auto &bndcoeff = bndCondExpansions[i]->GetCoeffs();
                if (locToGloMap->GetSignChange())
                {
                    for (size_t j = 0; j < bndCondExpansions[i]->GetNcoeffs();
                         j++)
                    {
                        outarr[map[bndcnt + j]] =
                            sign[bndcnt + j] * bndcoeff[j];
                    }
                }
                else
                {
                    for (size_t j = 0; j < bndCondExpansions[i]->GetNcoeffs();
                         j++)
                    {
                        outarr[map[bndcnt + j]] = bndcoeff[j];
                    }
                }
            }
            bndcnt += bndCondExpansions[i]->GetNcoeffs();
        }

        // communicate local Dirichlet coeffs that are just
        // touching a dirichlet boundary on another partition
        auto &ParallelDirBndSign = locToGloMap->GetParallelDirBndSign();

        for (auto &it : ParallelDirBndSign)
        {
            outarr[it] *= -1;
        }

        Array<OneD, NekDouble> arr(nloc, outarr.data());
        locToGloMap->UniversalAbsMaxBnd(arr);
        std::copy(arr.get(), arr.get() + nloc, outarr.data());

        for (auto &it : ParallelDirBndSign)
        {
            outarr[it] *= -1;
        }

        auto &copyLocalDirDofs = locToGloMap->GetCopyLocalDirDofs();
        for (auto &it : copyLocalDirDofs)
        {
            outarr[std::get<0>(it)] = outarr[std::get<1>(it)] * std::get<2>(it);
        }

        // Copy the data to the output field.
        out.template copyArray<MemSpace>(outarr);
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorDirBndCondImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }
};

} // namespace Nektar::Operators::detail
