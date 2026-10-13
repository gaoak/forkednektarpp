///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionOpImpl.hpp
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
// Description: Execution-space implementation of the whole-field advection
// operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvectionOpImpl.hpp
 * @brief Execution-space implementation of the whole-field advection
 * operator.
 *
 * @details
 * The whole-field behaviour of the advection operator is implemented in
 * AdvectionOp (AdvectionOp.hpp); the class here registers an "Advection" +
 * execution-space key with the operator factory and, on a multi-plane 3DH1
 * expansion, owns the z-operator (AdvectionZOpImpl.hpp) that adds the
 * homogeneous term w du/dz. The plane-local numerical work is done by the
 * block operators created alongside it (see AdvectionBlockOpSerialAVXStdMat.hpp
 * and its siblings). CMake-generated translation units instantiate and
 * register this class for each enabled execution space.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, one per execution space and
 * data type; it should not normally be included by any other
 * translation unit. Other code goes through AdvectionOp.hpp and the
 * operator factory.
 */

#pragma once

#include <MultiRegions/ExpListHomogeneous1D.h>

#include <MultiRegions/ElmtOps/Advection/AdvectionOp.hpp>
#include <MultiRegions/ElmtOps/AdvectionZOpImpl.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Execution-space implementation of AdvectionOp; registers it with
 * the operator factory and implements its homogeneous-term hooks.
 *
 * @tparam ExecSpace Execution space the instance is registered for and the
 *                   z-operator runs in.
 * @tparam TData     Floating-point type of the field data.
 */
template <typename ExecSpace, typename TData>
class AdvectionOpImpl : public AdvectionOp<TData>
{
public:
    /// @brief Construct the operator for @p expansionList and
    /// @p components; forwards to AdvectionOp and builds #m_zOp on a
    /// multi-plane 3DH1 expansion.
    AdvectionOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                    const std::vector<std::string> &components)
        : AdvectionOp<TData>(expansionList, components)
    {
        // A z-op is built only for a multi-plane 3DH1 expansion; m_zOp stays
        // null for 2D/3D, 3DH2 and single-plane 3DH1, where v_Apply does the
        // xy derivatives alone.
        auto homoExpList =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                expansionList);
        // The planes this rank holds, not the homogeneous basis' point count:
        // with npz > 1 the direction is split over the column communicator
        // while the basis still reports the global total.
        const unsigned int nhomo =
            homoExpList
                ? static_cast<unsigned int>(homoExpList->GetZIDs().size())
                : 1u;

        if (nhomo > 1)
        {
            m_zOp = std::make_shared<AdvectionZOpImpl<ExecSpace, TData>>(
                expansionList);
        }
    }

    /// Registration name for the operator factory, defined by the
    /// generated registration unit.
    static std::string className;

    /// @brief Creator function registered with the operator factory;
    /// builds one operator for the given expansion list and components.
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<AdvectionOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    /// Z-operator adding the homogeneous term; null unless the expansion is
    /// 3DH1 with more than one local plane.
    std::shared_ptr<AdvectionZOpImpl<ExecSpace, TData>> m_zOp;

    /// Forward @p scale to #m_zOp, if any.
    void v_SetScaleFFT(const TData &scale) override
    {
        if (m_zOp)
        {
            m_zOp->SetScale(scale);
        }
    }

    /// Forward @p advVel to #m_zOp, if any.
    void v_SetAdvVelFFT(
        LibUtilities::Field<TData, FieldState::Phys> &advVel) override
    {
        if (m_zOp)
        {
            m_zOp->SetAdvVel(advVel);
        }
    }

    /// Add the homogeneous term to @p out through #m_zOp, if any and
    /// @p in has more than one homogeneous mode.
    void v_ApplyFFT(LibUtilities::Field<TData, FieldState::Phys> &in,
                    LibUtilities::Field<TData, FieldState::Phys> &out) override
    {
        if (m_zOp && in.GetNumHomoModes() > 1)
        {
            m_zOp->Launch(in, out);
        }
    }
};

} // namespace Nektar::MultiRegions::detail
