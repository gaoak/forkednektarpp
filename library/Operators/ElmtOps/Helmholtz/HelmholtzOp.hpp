///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzOp.hpp
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

#include "Operators/ElmtOps/ElmtOp.hpp"

#include "Operators/ElmtOps/Helmholtz/HelmholtzBlockOp.hpp"

namespace Nektar::Operators
{

// Helmholtz base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class HelmholtzOp : public ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>
{
    friend class ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>;

public:
    static std::shared_ptr<HelmholtzOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>::
            template Create<HelmholtzOp, HelmholtzBlockOp>(
                expansionList, components, execStr, implStr);
    }

    static inline const std::string name = "Helmholtz";

    void SetLambda(TData lambda)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetLambda(lambda);
        }
        m_isSetLambda = true;
    }

    void SetDiffCoeff(std::vector<TData> &diffCoeff)
    {
        const auto coordDim      = this->m_expansionList->GetCoordim(0);
        const auto diffCoeffSize = coordDim * (coordDim + 1) / 2;
        ASSERTL0(diffCoeff.size() == diffCoeffSize,
                 "The number of diffusion coefficients must match 1, 3 or 6 "
                 "for a 1D, 2D or 3D case, respectively.")

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetDiffCoeff(diffCoeff);
        }
        m_isSetDiffCoeff = true;
    }

protected:
    bool m_isSetLambda       = false;
    bool m_isSetDiffCoeff    = false;
    bool m_isSetVarDiffCoeff = false;

    std::vector<std::shared_ptr<HelmholtzBlockOp<TData>>> m_blockOp;

    HelmholtzOp(const MultiRegions::ExpListSharedPtr &expansionList,
                const std::vector<std::string> &components)
        : ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>(expansionList,
                                                              components)
    {
    }

    ~HelmholtzOp() override = default;

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        ASSERTL1(in.GetNumComponents() == out.GetNumComponents(),
                 "Number of input and output components differ");

        ASSERTL1(in.GetNumHomoModes() == out.GetNumHomoModes(),
                 "Number of input and output homogeneous modes differ");

        ASSERTL1(m_isSetLambda,
                 "lambda has not been set."
                 "Set the value with SetLambda() before calling Apply().");

        ASSERTL1(m_isSetDiffCoeff || m_isSetVarDiffCoeff,
                 "diffusion coefficient has not been set."
                 "Set the value with SetDiffCoeff() OR SetVarDiffCoeff() "
                 "before calling Apply().");

        ASSERTL1(m_isSetDiffCoeff != m_isSetVarDiffCoeff,
                 "can't set both DiffCoeff and VarDiffCoeff."
                 "Set the value with SetDiffCoeff() OR SetVarDiffCoeff() "
                 "before calling Apply().");

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            // Block dependent.
            auto &inblock  = in.GetBlocks()[blk];
            auto &outblock = out.GetBlocks()[blk];

            this->m_blockOp[blk]->Apply(inblock, outblock);
        }
    }
};

} // namespace Nektar::Operators
