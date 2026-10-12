///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseOpImpl.hpp
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

/**
 * @file IProductWRTDerivBaseOpImpl.hpp
 * @brief Factory-registered whole-field implementation of the
 * IProductWRTDerivBase operator.
 *
 * @details
 * The apply loop lives in IProductWRTDerivBaseOp::v_Apply and the
 * per-plane work in the block operators (see
 * IProductWRTDerivBaseSerialAVXStdMat.hpp and its siblings). This class
 * adds the homogeneous z term of a Coeff output: for a multi-plane 3DH1
 * expansion it builds a DerivZOpImpl and an IProductWRTBaseOp for the
 * execution space and runs them in v_ApplyFFT(). It also supplies the
 * registration hooks (@c className, Instantiate()) through which an
 * "IProductWRTDerivBase" + output-state + execution-space key
 * ("IProductWRTDerivBaseCoeffSerial", "IProductWRTDerivBasePhysDevice",
 * and so on) is entered into the operator factory by the CMake-generated
 * factory declaration code, for each enabled execution space and both
 * output states.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, one per execution space and
 * data type; it should not normally be included by any other
 * translation unit. Other code goes through IProductWRTDerivBaseOp.hpp and the
 * operator factory.
 */

#pragma once

#include <LibUtilities/BasicUtils/Math/Math.hpp>
#include <MultiRegions/ExpListHomogeneous1D.h>

#include <MultiRegions/ElmtOps/DerivZOpImpl.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseOp.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Whole-field implementation of IProductWRTDerivBaseOp for one
 * execution space, subtracting the inner product with the basis of
 * \f$\partial f_z/\partial z\f$ on a multi-plane 3DH1 Coeff field (see
 * the file notes).
 *
 * @tparam ExecSpace Execution back-end (NektarSpaces::Serial, AVX or
 *                   Device) forming part of the factory key.
 * @tparam TFieldOut Output field state of the registered operator.
 * @tparam TData     Floating-point type of the field data.
 */
template <typename ExecSpace, FieldState TFieldOut, typename TData>
class IProductWRTDerivBaseOpImpl
    : public IProductWRTDerivBaseOp<TFieldOut, TData>
{
public:
    /// @brief Construct the operator for @p expansionList and
    /// @p components; for a Coeff output on a multi-plane 3DH1 expansion
    /// also build the z term operators and workspaces.
    IProductWRTDerivBaseOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : IProductWRTDerivBaseOp<TFieldOut, TData>(expansionList, components)
    {
        // The z direction contributes -IProductWRTBase(da_z/dz): the z-op
        // writes that derivative into m_dz, the inner product takes it to
        // m_wsp and the result is subtracted from the xy part. Only the Coeff
        // variant carries it, the Phys one refusing homogeneous input, and
        // they all stay uninstantiated on every other expansion.
        if constexpr (TFieldOut == FieldState::Coeff)
        {
            auto homoExpList =
                std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                    expansionList);

            // The planes this rank holds, not the homogeneous basis' point
            // count: with npz > 1 the direction is split over the column
            // communicator while the basis still reports the global total.
            const unsigned int nhomo =
                homoExpList
                    ? static_cast<unsigned int>(homoExpList->GetZIDs().size())
                    : 1u;

            if (nhomo > 1)
            {
                m_zOp =
                    std::make_shared<DerivZOpImpl<ExecSpace, TData,
                                                  DerivZLayout::VectorZToScalar,
                                                  DerivZOrder::First, false>>(
                        expansionList);
                m_ipOp =
                    IProductWRTBaseOp<TData>::Create(expansionList, components);

                const unsigned int nComp =
                    static_cast<unsigned int>(components.size());

                m_dz = LibUtilities::Field<TData, FieldState::Phys>(
                    "IProductWRTDerivBaseDz",
                    MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                        expansionList),
                    nComp, nhomo);
                m_wsp = LibUtilities::Field<TData, FieldState::Coeff>(
                    "IProductWRTDerivBaseWsp",
                    MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                        expansionList),
                    nComp, nhomo);
            }
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
        return std::make_unique<
            IProductWRTDerivBaseOpImpl<ExecSpace, TFieldOut, TData>>(
            expansionList, components);
    }

protected:
    /// The z derivative of component 3n + 2 of the input, written to
    /// component n of #m_dz; null unless the expansion is multi-plane 3DH1
    /// and the output Coeff.
    std::shared_ptr<
        DerivZOpImpl<ExecSpace, TData, DerivZLayout::VectorZToScalar,
                     DerivZOrder::First, false>>
        m_zOp;
    /// Inner product with the basis taking #m_dz to #m_wsp; built with
    /// #m_zOp.
    std::shared_ptr<IProductWRTBaseOp<TData>> m_ipOp;
    /// Physical-space workspace holding \f$\partial f_z/\partial z\f$.
    LibUtilities::Field<TData, FieldState::Phys> m_dz;
    /// Coefficient-space workspace holding its inner product with the
    /// basis.
    LibUtilities::Field<TData, FieldState::Coeff> m_wsp;
    /// Scale factor of the operator, applied again to the z term.
    TData m_scale = 1.0;

    /// Store the scale factor for the z term.
    void v_SetScaleFFT(const TData &scale) override
    {
        m_scale = scale;
    }

    /// Subtract the scaled z term from @p out through #m_zOp and #m_ipOp;
    /// does nothing for a Phys output, without a z-op or with a single
    /// homogeneous mode.
    void v_ApplyFFT(LibUtilities::Field<TData, FieldState::Phys> &in,
                    LibUtilities::Field<TData, TFieldOut> &out) override
    {
        if constexpr (TFieldOut == FieldState::Coeff)
        {
            if (m_zOp && in.GetNumHomoModes() > 1)
            {
                m_zOp->Launch(in, m_dz);
                m_ipOp->Apply(m_dz, m_wsp);

                // One integration by parts puts the minus sign here. The
                // scale is taken again because the z term never passes
                // through the block operators that scale the xy part.
                Math::daxpy<ExecSpace>(-m_scale, m_wsp, out, out);
            }
        }
    }
};

} // namespace Nektar::MultiRegions::detail
