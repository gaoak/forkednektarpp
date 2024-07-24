///////////////////////////////////////////////////////////////////////////////
//
// File: RobBndCondSerialGeneric.hpp
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

#include "Operators/BndCondOps/OperatorRobBndCond.hpp"

#include "Operators/MathKernels/MathKernels.hpp"

#include <LocalRegions/MatrixKey.h>
#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{
// Standard matrix implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              std::is_same<ExecSpace, NektarSpaces::Serial>::value>::type>
class OperatorRobBndCondImpl : public OperatorRobBndCond<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorRobBndCondImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorRobBndCond<TData>(expansionList)
    {
        m_robin = MemoryRegion<TData>::template create<MemSpace>(
            "m_robin", expansionList->GetNcoeffs());
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out,
               const bool &negflag) override
    {
        // Copy the data from the input field.
        Array<OneD, TData> inArray = in.toArray();

        // Apply Robin boundary conditions.
        const auto *inPtr = inArray.data();
        auto *robinPtr    = m_robin.template GetPtr<MemSpace, WriteOnly>();
        m_robin.initialize(0);

        auto dimension = this->m_expansionList->GetExp(0)->GetShapeDimension();
        if (dimension == 1)
        {
            Operator1D(inPtr, robinPtr);
        }
        else if (dimension == 2)
        {
            Operator2D(inPtr, robinPtr);
        }
        else if (dimension == 3)
        {
            Operator3D(inPtr, robinPtr);
        }

        // Add Robin BC contribution to the output field.
        auto *outPtr = out.template GetPtr<MemSpace, ReadWrite>();
        for (auto const &block : out.GetBlocks())
        {
            auto nElmts = block.num_elements;
            auto nmTot  = block.num_pts;

            if (negflag)
            {
                subKernel<ExecSpace, TData>(nElmts * nmTot, outPtr, robinPtr,
                                            outPtr);
            }
            else
            {
                addKernel<ExecSpace, TData>(nElmts * nmTot, outPtr, robinPtr,
                                            outPtr);
            }

            robinPtr += nElmts * nmTot;
            outPtr += block.block_size;
        }
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorRobBndCondImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

    void Operator1D(const TData *incoeffs, TData *coeffs)
    {
        auto robinBCInfo = this->m_expansionList->GetRobinBCInfo();

        for (auto &r : robinBCInfo)
        {
            auto n      = r.first;
            auto offset = this->m_expansionList->GetCoeff_Offset(n);
            auto expPtr = this->m_expansionList->GetExp(n);

            ASSERTL1(expPtr->IsBoundaryInteriorExpansion(),
                     "Not set up for non boundary-interior expansions");

            for (auto rBC = r.second; rBC; rBC = rBC->next)
            {
                auto primCoeffs = rBC->m_robinPrimitiveCoeffs;
                auto vertid     = rBC->m_robinID;
                auto map        = expPtr->GetVertexMap(vertid);
                coeffs[offset + map] += primCoeffs[0] * incoeffs[offset + map];
            }
        }
    }

    void Operator2D(const TData *incoeffs, TData *coeffs)
    {
        auto robinBCInfo = this->m_expansionList->GetRobinBCInfo();

        for (auto &r : robinBCInfo)
        {
            auto n      = r.first;
            auto offset = this->m_expansionList->GetCoeff_Offset(n);
            auto expPtr = this->m_expansionList->GetExp(n);

            ASSERTL1(expPtr->IsBoundaryInteriorExpansion(),
                     "Not set up for non boundary-interior expansions");

            for (auto rBC = r.second; rBC; rBC = rBC->next)
            {
                auto primCoeffs = rBC->m_robinPrimitiveCoeffs;
                auto edgeid     = rBC->m_robinID;
                auto orient     = expPtr->GetTraceOrient(edgeid);
                auto edgeExp    = expPtr->GetTraceExp(edgeid);
                auto ncoeff     = edgeExp->GetNcoeffs();

                // Get map and sign.
                Array<OneD, unsigned int> map;
                Array<OneD, int> sign;
                expPtr->GetTraceToElementMap(edgeid, map, sign, orient);

                // Get mass matrix.
                StdRegions::VarCoeffMap varcoeffs;
                varcoeffs[StdRegions::eVarCoeffMass] = primCoeffs;
                LocalRegions::MatrixKey mkey(
                    StdRegions::eMass, LibUtilities::eSegment, *edgeExp,
                    StdRegions::NullConstFactorMap, varcoeffs);
                DNekScalMat &edgeMat = *edgeExp->GetLocMatrix(mkey);

                // Apply Robin boundary conditions.
                NekVector<NekDouble> vEdgeCoeffs(ncoeff);
                for (unsigned int i = 0; i < ncoeff; ++i)
                {
                    vEdgeCoeffs[i] = incoeffs[offset + map[i]] * sign[i];
                }

                vEdgeCoeffs = edgeMat * vEdgeCoeffs;
                for (unsigned int i = 0; i < ncoeff; ++i)
                {
                    coeffs[offset + map[i]] += vEdgeCoeffs[i] * sign[i];
                }
            }
        }
    }

    void Operator3D([[maybe_unused]] const TData *incoeffs,
                    [[maybe_unused]] TData *coeffs)
    {
        /*auto robinBCInfo = this->m_expansionList->GetRobinBCInfo();

        for (auto &r : robinBCInfo)
        {
            NEKERROR(ErrorUtil::efatal,
                     "OperatorRobBndCondImpl: 3D Operator not yet implemented");
        }*/
    }

protected:
    MemoryRegion<TData> m_robin;
};

} // namespace Nektar::Operators::detail
