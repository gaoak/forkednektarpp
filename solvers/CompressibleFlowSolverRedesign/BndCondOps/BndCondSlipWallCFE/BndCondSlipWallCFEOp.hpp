///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondSlipWallCFEOp.hpp
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

#include <SolverCore/BndCond/BndCondUpdateOp.hpp>

namespace Nektar
{

/**
 * @brief Inviscid slip wall and symmetry plane.
 *
 * Both mirror the momentum in the boundary plane,
 *
 *     m* = m - 2 (m.n) n
 *
 * leaving density and energy alone, so the *averaged* normal velocity across
 * the trace vanishes while the tangential velocity passes through untouched.
 * That is the difference from the no-slip wall of BndCondWallCFEOp, which
 * reverses the whole momentum vector and so brings the tangential velocity to
 * zero as well.
 *
 * One operator claims both `Wall` and `Symmetry` because they are the same
 * statement about the flow: no mass crosses the surface and
 * nothing is dissipated on it. This mirrors BndCondWallCFEOp holding both
 * `WallViscous` and `WallAdiabatic`.
 *
 * @see BndCondSlipWallCFEOpImpl.
 */
template <typename TData>
class BndCondSlipWallCFEOp : public SolverCore::BndCondUpdateOp<TData>
{
public:
    static std::shared_ptr<BndCondSlipWallCFEOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        return Operators::Operator<TData>::template Create<
            BndCondSlipWallCFEOp>(expansionList, components, execStr);
    }

    static inline const std::string name = "BndCondSlipWallCFEOp";

protected:
    BndCondSlipWallCFEOp(const MultiRegions::ExpListSharedPtr &expansionList,
                         const std::vector<std::string> &components)
        : SolverCore::BndCondUpdateOp<TData>(expansionList, components)
    {
    }

    ~BndCondSlipWallCFEOp() override = default;
};

} // namespace Nektar
