///////////////////////////////////////////////////////////////////////////////
//
// File: OperatorHelmholtzImplBase.hpp
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

#include "Operators/OperatorHelmholtz.hpp"

#include "Operators/OperatorBwdTrans.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"
#include "Operators/OperatorIProductWRTDerivBase.hpp"
#include "Operators/OperatorPhysDeriv.hpp"

#include "Operators/OperatorHelper.hpp"

namespace Nektar::Operators::detail
{

// Standard matrix implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorHelmholtzImplBase : public OperatorHelmholtz<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorHelmholtzImplBase(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorHelmholtz<TData>(expansionList),
          m_bwd(Field<TData, FieldState::Phys>::template create<MemSpace>(
              "Helmholtz bwd",
              GetBlockAttributes(FieldState::Phys, expansionList))),
          m_deriv(Field<TData, FieldState::Phys>::template create<MemSpace>(
              "Helmholtz deriv",
              GetBlockAttributes(FieldState::Phys, expansionList),
              expansionList->GetCoordim(0)))
    {
        auto nCoord = this->m_expansionList->GetCoordim(0);

        m_BwdTransOp =
            BwdTrans<TData>::template create<ExecSpace, Implementation>(
                this->m_expansionList);
        m_PhysDerivOp =
            PhysDeriv<TData>::template create<ExecSpace, Implementation>(
                this->m_expansionList);
        m_IProductWRTBaseOp =
            IProductWRTBase<TData>::template create<ExecSpace, Implementation>(
                this->m_expansionList);
        m_IProductWRTDerivBaseOp = IProductWRTDerivBase<TData>::template create<
            ExecSpace, Implementation>(this->m_expansionList);

        m_diffCoeff = MemoryRegion<TData>::template create<MemSpace>(
            "Helmholtz diffCoeff", nCoord * nCoord);

        m_diffCoeff.initialize(0);

        TData *diffCoeff =
            m_diffCoeff.template GetPtr<NektarSpaces::HostSpace>();

        for (size_t d = 0; d < nCoord; d++)
        {
            diffCoeff[d * nCoord + d] = 1.0; // temporary solution
        }
    }

    // virtual void apply(Field<TData, FieldState::Coeff> &in,
    //                    Field<TData, FieldState::Coeff> &out)
    // {
    //     // Step 1: BwdTrans
    //     m_BwdTransOp->apply(in, m_bwd);

    //     // Step 2: PhysDeriv
    //     m_PhysDerivOp->apply(m_bwd, m_deriv);

    //     // Step 3: Inner product for mass matrix operation
    //     m_IProductWRTBaseOp->apply(m_bwd, out, this->m_lambda);

    //     // Step 4: Multiply by diffusion coefficient
    //     // Done by the specific implementation - CUDA vs StdMat
    //     DiffusionCoeff(m_deriv);
    //     DiffusionCoeff(this->m_deriv, m_derivcoeff);

    //     // Step 5: Inner product
    //     // Done by the specific implementation - CUDA vs StdMat
    //     this->m_IProductWRTDerivBaseOp->apply(this->m_deriv, out, true);
    //     this->m_IProductWRTDerivBaseOp->apply(m_derivcoeff, out, true);
    // }

    static std::string className;

protected:
    std::shared_ptr<OperatorBwdTrans<TData>> m_BwdTransOp;
    std::shared_ptr<OperatorPhysDeriv<TData>> m_PhysDerivOp;
    std::shared_ptr<OperatorIProductWRTBase<TData>> m_IProductWRTBaseOp;
    std::shared_ptr<OperatorIProductWRTDerivBase<TData>>
        m_IProductWRTDerivBaseOp;

    Field<TData, FieldState::Phys> m_bwd;
    Field<TData, FieldState::Phys> m_deriv;

    MemoryRegion<TData> m_diffCoeff;
};

} // namespace Nektar::Operators::detail
