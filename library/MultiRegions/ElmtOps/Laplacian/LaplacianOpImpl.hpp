///////////////////////////////////////////////////////////////////////////////
//
// File: LaplacianOpImpl.hpp
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
// Description: Factory-registered whole-field implementation of the weak
// Laplacian operator.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file LaplacianOpImpl.hpp
 * @brief Factory-registered whole-field implementation of the weak
 * Laplacian operator.
 *
 * @details
 * The apply loop lives in LaplacianOp::v_Apply and the per-plane work
 * in the block operators. This class adds the homogeneous z term: for a
 * multi-plane 3DH1 expansion it builds a second-order DerivZOpImpl and a
 * MassOp for the execution space and runs them in v_ApplyFFT(). It also
 * supplies the registration hooks (@c className, Instantiate()) through
 * which one instantiation per execution space is entered into the
 * operator factory by the CMake-generated factory declaration code.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, one per execution space and
 * data type; it should not normally be included by any other
 * translation unit. Other code goes through LaplacianOp.hpp and the
 * operator factory.
 */

#pragma once

#include <LibUtilities/BasicUtils/Math/Math.hpp>
#include <MultiRegions/ExpListHomogeneous1D.h>

#include <MultiRegions/ElmtOps/DerivZOpImpl.hpp>
#include <MultiRegions/ElmtOps/Laplacian/LaplacianOp.hpp>
#include <MultiRegions/ElmtOps/Mass/MassOp.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Whole-field implementation of LaplacianOp for one execution
 * space, adding the z term on a multi-plane 3DH1 expansion (see the
 * file notes).
 *
 * The ExecSpace template parameter lets the CMake-generated factory
 * declaration code register a separate entry ("LaplacianSerial",
 * "LaplacianAVX", "LaplacianDevice") for each execution space enabled
 * in the build, and selects the space the z-derivative runs in.
 *
 * @tparam ExecSpace Execution space the factory entry is registered
 *                   for.
 * @tparam TData     Floating-point type of the field data.
 */
template <typename ExecSpace, typename TData>
class LaplacianOpImpl : public LaplacianOp<TData>
{
public:
    LaplacianOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                    const std::vector<std::string> &components)
        : LaplacianOp<TData>(expansionList, components)
    {
        // The weak z-Laplacian of a multi-plane 3DH1 field is the xy mass
        // matrix applied to minus the second z-derivative, so the z-op writes
        // that derivative into m_d2z, the mass operator takes it to m_wsp and
        // the result is subtracted from the xy part -- the minus sits here
        // rather than in the z-op, which gives the derivative itself. They
        // all stay uninstantiated on every other expansion.
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
        const unsigned int nhomo =
            homoExpList
                ? static_cast<unsigned int>(homoExpList->GetZIDs().size())
                : 1u;

        if (nhomo > 1)
        {
            m_zOp = std::make_shared<
                DerivZOpImpl<ExecSpace, TData, DerivZLayout::Identity,
                             DerivZOrder::Second, false>>(expansionList);
            m_massOp = MassOp<TData>::Create(expansionList, components);

            auto blockAttr =
                MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                    expansionList);
            const unsigned int nComp =
                static_cast<unsigned int>(components.size());

            m_d2z = LibUtilities::Field<TData, FieldState::Coeff>(
                "LaplacianD2Z", blockAttr, nComp, nhomo);
            m_wsp = LibUtilities::Field<TData, FieldState::Coeff>(
                "LaplacianWsp", blockAttr, nComp, nhomo);
        }
    }

    /// Operator class name; defined by the generated factory code.
    static std::string className;

    /// Creator function registered with OperatorFactory.
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<LaplacianOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    /// Component for component, giving the second z-derivative; the xy
    /// mass matrix and the subtraction below make it the weak z-Laplacian.
    std::shared_ptr<DerivZOpImpl<ExecSpace, TData, DerivZLayout::Identity,
                                 DerivZOrder::Second, false>>
        m_zOp;
    /// xy mass matrix applied to #m_d2z.
    std::shared_ptr<MassOp<TData>> m_massOp;
    /// Second z-derivative of the input, written by #m_zOp.
    LibUtilities::Field<TData, FieldState::Coeff> m_d2z;
    /// Mass matrix applied to #m_d2z, subtracted from the output.
    LibUtilities::Field<TData, FieldState::Coeff> m_wsp;

    /// Add the weak z-Laplacian to @p out through #m_zOp and #m_massOp;
    /// does nothing without them or with a single homogeneous mode.
    void v_ApplyFFT(LibUtilities::Field<TData, FieldState::Coeff> &in,
                    LibUtilities::Field<TData, FieldState::Coeff> &out) override
    {
        if (m_zOp && in.GetNumHomoModes() > 1)
        {
            m_zOp->Launch(in, m_d2z);
            m_massOp->Apply(m_d2z, m_wsp);

            // The z-op gives the second derivative itself and the weak
            // z-Laplacian takes minus it, so the mass-weighted result is
            // subtracted rather than added.
            Math::sub<ExecSpace>(out, m_wsp, out);
        }
    }
};

} // namespace Nektar::MultiRegions::detail
