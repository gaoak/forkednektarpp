///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseSerialSumFac.hpp
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

#include "IProductWRTBaseSerialSumFacKernels.hpp"

#include "Operators/OperatorIProductWRTBase.hpp"

#include <StdRegions/StdExpansion.h>

namespace Nektar::Operators::detail
{

// Sum-factorisation implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              std::is_same<ExecSpace, NektarSpaces::Serial>::value &&
              std::is_same<Implementation, Operators::SumFac>::value>::type>
class OperatorIProductWRTBaseImpl : public OperatorIProductWRTBase<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorIProductWRTBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTBase<TData>(expansionList)
    {
        // Initialise jacobian.
        size_t jacSize = Operator<TData>::GetGeometricFactorSize();
        m_jac          = Operator<TData>::SetJacobian(jacSize);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               [[maybe_unused]] const TData lambda = 1.0) override
    {
        auto *inPtr  = in.template GetConstPtr<MemSpace>();
        auto *outPtr = out.template GetPtr<MemSpace>();

        size_t exp_idx = 0;
        size_t jac_idx = 0;

        // Loop over the blocks.
        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            // Block dependent
            auto const &inblock  = in.GetBlocks()[block_idx];
            auto const &outblock = out.GetBlocks()[block_idx];
            auto const nElmts    = inblock.num_elements;

            // Determine shape and type of the element.
            auto const expPtr    = this->m_expansionList->GetExp(exp_idx);
            auto const shapeType = expPtr->DetShapeType();

            switch (shapeType)
            {
                // Segment
                case LibUtilities::Seg:
                    IProductWRTBaseSumFacSegKernel(inPtr, outPtr, expPtr, m_jac,
                                                   nElmts, jac_idx);
                    break;
                // Triangles
                case LibUtilities::Tri:
                    IProductWRTBaseSumFacTriKernel(inPtr, outPtr, expPtr, m_jac,
                                                   nElmts, jac_idx);
                    break;
                // Quads
                case LibUtilities::Quad:
                    IProductWRTBaseSumFacQuadKernel(inPtr, outPtr, expPtr,
                                                    m_jac, nElmts, jac_idx);
                    break;
                // Tet
                case LibUtilities::Tet:
                    IProductWRTBaseSumFacTetKernel(inPtr, outPtr, expPtr, m_jac,
                                                   nElmts, jac_idx);
                    break;
                // Pyr
                case LibUtilities::Pyr:
                    IProductWRTBaseSumFacPyrKernel(inPtr, outPtr, expPtr, m_jac,
                                                   nElmts, jac_idx);
                    break;
                // Prism
                case LibUtilities::Prism:
                    IProductWRTBaseSumFacPrismKernel(inPtr, outPtr, expPtr,
                                                     m_jac, nElmts, jac_idx);
                    break;
                // Hexes
                case LibUtilities::Hex:
                    IProductWRTBaseSumFacHexKernel(inPtr, outPtr, expPtr, m_jac,
                                                   nElmts, jac_idx);
                    break;
                default:
                    std::cout << "shapetype not implemented" << std::endl;
            }

            // Increment pointer and index for next element type.
            inPtr += inblock.block_size;
            outPtr += outblock.block_size;
            exp_idx += nElmts;
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorIProductWRTBaseImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    Array<OneD, TData> m_jac;
};

} // namespace Nektar::Operators::detail
