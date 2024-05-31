///////////////////////////////////////////////////////////////////////////////
//
// File: NeuBndCondSerialStdMat.hpp
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

#include "Operators/OperatorNeuBndCond.hpp"

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{
// Standard matrix implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              std::is_same<ExecSpace, NektarSpaces::Serial>::value &&
              std::is_same<Implementation, Operators::StdMat>::value>::type>
class OperatorNeuBndCondImpl : public OperatorNeuBndCond<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorNeuBndCondImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorNeuBndCond<TData>(expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        m_assmbMap = contfield->GetLocalToGlobalMap();
    }

    void apply(Field<TData, FieldState::Coeff> &inOut) override
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        auto &bndCondExpansions = contfield->GetBndCondExpansions();
        auto &bndConditions     = contfield->GetBndConditions();
        auto &sign = this->m_assmbMap->GetBndCondCoeffsToLocalCoeffsSign();
        auto &map  = this->m_assmbMap->GetBndCondCoeffsToLocalCoeffsMap();

        // Copy the data from the input field.
        Array<OneD, TData> inOutArray = inOut.toArray();

        // Add weak boundary conditions to the forcing.
        size_t bndcnt = 0;

        for (size_t i = 0; i < bndCondExpansions.size(); ++i)
        {
            auto nBndcoeff = bndCondExpansions[i]->GetNcoeffs();

            if (bndConditions[i]->GetBoundaryConditionType() ==
                    SpatialDomains::eNeumann ||
                bndConditions[i]->GetBoundaryConditionType() ==
                    SpatialDomains::eRobin)
            {
                auto &bndcoeff = bndCondExpansions[i]->GetCoeffs();

                if (this->m_assmbMap->GetSignChange())
                {
                    for (size_t j = 0; j < nBndcoeff; ++j)
                    {
                        inOutArray[map[bndcnt + j]] +=
                            sign[bndcnt + j] * bndcoeff[j];
                    }
                }
                else
                {
                    for (size_t j = 0; j < nBndcoeff; ++j)
                    {
                        inOutArray[map[bndcnt + j]] += bndcoeff[j];
                    }
                }
            }

            bndcnt += nBndcoeff;
        }

        // Copy the data to the output field.
        inOut.template copyArray<MemSpace>(inOutArray);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorNeuBndCondImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

protected:
    AssemblyMapCGSharedPtr m_assmbMap;
};

} // namespace Nektar::Operators::detail
