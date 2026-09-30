///////////////////////////////////////////////////////////////////////////////
//
// File: PhysGalerkinProject1DScaledBlockOp.hpp
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

#include <vector>

#include "Operators/ElmtOps/ElmtBlockOp.hpp"

namespace Nektar::Operators
{

template <typename TData>
class PhysGalerkinProject1DScaledBlockOp
    : public ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>
{
public:
    static std::shared_ptr<PhysGalerkinProject1DScaledBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        const std::string &execStr, const std::string &implStr)
    {
        return ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>::
            template Create<PhysGalerkinProject1DScaledBlockOp>(
                block_idx, exp, dataWarehouse, execStr, implStr);
    }

    static inline const std::string name = "BlockPhysGalerkinProject1DScaled";

    void SetScaleFactor(const TData &scale)
    {
        v_SetScaleFactor(scale);
    }

    void SetAppend(const bool &append)
    {
        m_append = append;
    }

protected:
    TData m_scale = -1.0; // scaling factor
    bool m_append = false;

    PhysGalerkinProject1DScaledBlockOp(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>(block_idx, exp,
                                                                 dataWarehouse)
    {
    }

    ~PhysGalerkinProject1DScaledBlockOp() override = default;

    /// \brief The per-direction point counts of the @p scale over-integrated
    /// grid this operator's interpolation and projection matrices are built
    /// for.
    ///
    /// Direction 0 scales outright, and a direction carrying one point fewer
    /// than direction 0 keeps that offset, which is what lets the switch
    /// templating see the same relationship on the over-integrated grid as on
    /// the native one.
    static std::vector<unsigned int> GetScaledNumPoints(
        const std::vector<unsigned int> &nq, const NekDouble scale)
    {
        std::vector<unsigned int> nqScaled;
        nqScaled.reserve(nq.size());

        for (size_t d = 0; d < nq.size(); ++d)
        {
            nqScaled.push_back((d != 0 && nq[0] - nq[d] == 1)
                                   ? static_cast<unsigned int>(scale * nq[0]) -
                                         1
                                   : static_cast<unsigned int>(scale * nq[d]));
        }

        return nqScaled;
    }

    virtual void v_SetScaleFactor(const TData &scale) = 0;
};

} // namespace Nektar::Operators
