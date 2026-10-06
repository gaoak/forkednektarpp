///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionOpImpl.hpp
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

#include <LibUtilities/BasicUtils/Math/Math.hpp>
#include <MultiRegions/ExpListHomogeneous1D.h>

#include <MultiRegions/ElmtOps/AdvectionZOpImpl.hpp>
#include <MultiRegions/ElmtOps/BwdTrans/BwdTransOp.hpp>
#include <MultiRegions/ElmtOps/DerivZOpImpl.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp>
#include <MultiRegions/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionOp.hpp>
#include <MultiRegions/ElmtOps/Mass/MassOp.hpp>

namespace Nektar::MultiRegions::detail
{

template <typename ExecSpace, typename TData>
class LinAdvDiffReactionOpImpl : public LinAdvDiffReactionOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    LinAdvDiffReactionOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : LinAdvDiffReactionOp<TData>(expansionList, components)
    {
        // The weak z-Laplacian of a multi-plane 3DH1 field is the xy mass
        // matrix applied to minus the second z-derivative, so the z-op writes
        // that derivative into m_d2z, the mass operator takes it to m_wsp and
        // the result is subtracted from the xy part -- the minus sits here
        // rather than in the z-op, which gives the derivative itself. The
        // lambda term needs nothing here: the block operators already carry
        // it on every plane, as they do the in-plane advection. They all stay
        // uninstantiated on every other expansion.
        //
        // SetDiffCoeff sizes its tensor on the plane's coordinate dimension,
        // so a 3DH1 caller can only give the xy entries: the z term is taken
        // with a unit coefficient and no xz/yz coupling, which is the plain
        // Laplacian rather than a general anisotropic one.
        auto homoExpList =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                expansionList);
        // The planes this rank holds, not the homogeneous basis' point count:
        // with npz > 1 the direction is split over the column communicator
        // while the basis still reports the global total.
        m_nhomo = homoExpList
                      ? static_cast<unsigned int>(homoExpList->GetZIDs().size())
                      : 1u;

        if (m_nhomo > 1)
        {
            m_zOp = std::make_shared<
                DerivZOpImpl<ExecSpace, TData, DerivZLayout::Identity,
                             DerivZOrder::Second, false>>(expansionList);
            m_massOp = MassOp<TData>::Create(expansionList, components);

            auto coeffAttr =
                MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                    expansionList);
            const unsigned int nComp =
                static_cast<unsigned int>(components.size());

            m_d2z = LibUtilities::Field<TData, FieldState::Coeff>(
                "LinAdvDiffReactionD2Z", coeffAttr, nComp, m_nhomo);
            m_wsp = LibUtilities::Field<TData, FieldState::Coeff>(
                "LinAdvDiffReactionWsp", coeffAttr, nComp, m_nhomo);

            // The advection along z multiplies by a velocity that varies over
            // the plane, so it is formed at the quadrature points and taken
            // back to coefficients. Its pieces are built here and left idle
            // until SetAdvVel() says whether there is a through-plane
            // velocity at all.
            m_bwdTransOp = BwdTransOp<TData>::Create(expansionList, components);
            m_ipOp =
                IProductWRTBaseOp<TData>::Create(expansionList, components);
            m_advZOp = std::make_shared<AdvectionZOpImpl<ExecSpace, TData>>(
                expansionList);

            auto physAttr =
                MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                    expansionList);

            m_u = LibUtilities::Field<TData, FieldState::Phys>(
                "LinAdvDiffReactionUPhys", physAttr, nComp, m_nhomo);
            m_advZ = LibUtilities::Field<TData, FieldState::Phys>(
                "LinAdvDiffReactionAdvZ", physAttr, nComp, m_nhomo);
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<LinAdvDiffReactionOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    unsigned int m_nhomo = 1u;

    /// Component for component, giving the second z-derivative; the xy mass
    /// matrix and the subtraction below make it the weak z-Laplacian.
    std::shared_ptr<DerivZOpImpl<ExecSpace, TData, DerivZLayout::Identity,
                                 DerivZOrder::Second, false>>
        m_zOp;
    std::shared_ptr<MassOp<TData>> m_massOp;
    LibUtilities::Field<TData, FieldState::Coeff> m_d2z;
    LibUtilities::Field<TData, FieldState::Coeff> m_wsp;

    /// The advection along the homogeneous direction, w du/dz, accumulated
    /// at the quadrature points, then taken back to coefficients.
    std::shared_ptr<AdvectionZOpImpl<ExecSpace, TData>> m_advZOp;
    std::shared_ptr<BwdTransOp<TData>> m_bwdTransOp;
    std::shared_ptr<IProductWRTBaseOp<TData>> m_ipOp;
    LibUtilities::Field<TData, FieldState::Phys> m_u;
    LibUtilities::Field<TData, FieldState::Phys> m_advZ;

    void v_SetAdvVelFFT(
        LibUtilities::Field<TData, FieldState::Phys> &advVel) override
    {
        if (!m_advZOp)
        {
            return;
        }

        // SetAdvVel() has already checked that a 3DH1 velocity carries the
        // through-plane component the plane block operators do not count.
        // The z-op reads it as the last one, on the same planes the block
        // operators read the others.
        m_advZOp->SetAdvVel(advVel);
    }

    void v_ApplyFFT(LibUtilities::Field<TData, FieldState::Coeff> &in,
                    LibUtilities::Field<TData, FieldState::Coeff> &out) override
    {
        if (in.GetNumHomoModes() <= 1)
        {
            return;
        }

        // m_zOp is built only for a multi-plane 3DH1 expansion, so a
        // homogeneous field without one is 3DH2, which is not wired at all.
        // The guard is L0: without it the z terms would simply be missing
        // and the result would come back silently wrong in a release build.
        ASSERTL0(m_zOp, "LinAdvDiffReactionOp does not support 3DH2 "
                        "configurations.");

        // The z-op gives the second derivative itself and the weak
        // z-Laplacian takes minus it, so the mass-weighted result is
        // subtracted rather than added.
        m_zOp->Launch(in, m_d2z);
        m_massOp->Apply(m_d2z, m_wsp);
        Math::sub<ExecSpace>(out, m_wsp, out);

        // The advection along z is weak like the rest of the operator, so
        // w du/dz is formed at the quadrature points and taken back through
        // the inner product. The z-op accumulates on top of whatever it is
        // given, which here is nothing but this term.
        m_bwdTransOp->Apply(in, m_u);
        Math::zero<ExecSpace>(m_advZ);
        m_advZOp->Launch(m_u, m_advZ);
        m_ipOp->Apply(m_advZ, m_wsp);
        Math::add<ExecSpace>(out, m_wsp, out);
    }
};

} // namespace Nektar::MultiRegions::detail
