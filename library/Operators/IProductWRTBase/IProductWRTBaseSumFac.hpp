///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseSumFac.hpp
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

#include <StdRegions/StdExpansion.h>

#include "IProductWRTBaseSumFacKernels.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"

namespace Nektar::Operators::detail
{

// standard matrix implementation
template <typename TData>
class OperatorIProductWRTBaseImpl<TData, ImplSumFac>
    : public OperatorIProductWRTBase<TData>
{
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
        auto const *inptr = in.GetStorage().GetCPUPtr();
        auto *outptr      = out.GetStorage().GetCPUPtr();

        size_t exp_idx = 0;
        size_t jac_idx = 0;
        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            // block dependent
            auto const block    = in.GetBlocks()[block_idx];
            auto const numElmts = block.num_elements;
            auto expPtr         = this->m_expansionList->GetExp(exp_idx);
            auto shapeType      = this->m_expansionList->GetExp(exp_idx)
                                 ->GetStdExp()
                                 ->DetShapeType();

            switch (shapeType)
            {
                // Segment
                case LibUtilities::Seg:
                    IProductWRTBaseSumFacSegKernel(inptr, outptr, expPtr, m_jac,
                                                   numElmts, jac_idx);
                    break;
                // Triangles
                case LibUtilities::Tri:
                    IProductWRTBaseSumFacTriKernel(inptr, outptr, expPtr, m_jac,
                                                   numElmts, jac_idx);
                    break;
                // Quads
                case LibUtilities::Quad:
                    IProductWRTBaseSumFacQuadKernel(inptr, outptr, expPtr,
                                                    m_jac, numElmts, jac_idx);
                    break;
                // Tet
                case LibUtilities::Tet:
                    IProductWRTBaseSumFacTetKernel(inptr, outptr, expPtr, m_jac,
                                                   numElmts, jac_idx);
                    break;
                // Pyr
                case LibUtilities::Pyr:
                    IProductWRTBaseSumFacPyrKernel(inptr, outptr, expPtr, m_jac,
                                                   numElmts, jac_idx);
                    break;
                // Prism
                case LibUtilities::Prism:
                    IProductWRTBaseSumFacPrismKernel(inptr, outptr, expPtr,
                                                     m_jac, numElmts, jac_idx);
                    break;
                // Hexes
                case LibUtilities::Hex:
                    IProductWRTBaseSumFacHexKernel(inptr, outptr, expPtr, m_jac,
                                                   numElmts, jac_idx);
                    break;
                default:
                    std::cout << "shapetype not implemented" << std::endl;
            }

            inptr += in.GetBlocks()[block_idx].block_size;
            outptr += out.GetBlocks()[block_idx].block_size;
            exp_idx += numElmts;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorIProductWRTBaseImpl<TData, ImplSumFac>>(
            expansionList);
    }

    static std::string className;

private:
    Array<OneD, TData> m_jac;
};

} // namespace Nektar::Operators::detail
