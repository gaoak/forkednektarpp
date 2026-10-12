///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionOp.hpp
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
// Description: Whole-field interface of the linearised
// advection-diffusion-reaction element operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file LinAdvDiffReactionOp.hpp
 * @brief Whole-field interface of the linearised
 * advection-diffusion-reaction (LinAdvDiffReaction) element operator, which
 * maps a field from coefficient space to coefficient space.
 *
 * @details
 * For every element the operator forms the weak advection-diffusion-
 * reaction residual of the expansion \f$u = \sum_n \hat{u}_n \phi_n\f$
 * against every basis function \f$\phi_p\f$ of the element,
 * \f[
 *   I_p = (\phi_p,\, \mathbf{V}\cdot\nabla u)
 *       + \lambda\,(\phi_p,\, u)
 *       + (\nabla\phi_p,\, \mathbf{D}\,\nabla u),
 * \f]
 * where the inner products are the element's quadrature sums including
 * the Jacobian. The three terms and the data they need are:
 * - **advection** \f$(\phi_p, \mathbf{V}\cdot\nabla u)\f$ -- needs the
 *   advection velocity \f$\mathbf{V}\f$ at the quadrature points, one
 *   component per coordinate direction (three on a 3DH1 expansion),
 *   supplied as a physical-space Field through SetAdvVel();
 * - **reaction** \f$\lambda (\phi_p, u)\f$ -- needs the scalar
 *   \f$\lambda\f$, supplied through SetLambda(). The implementations add
 *   \f$\lambda u\f$ to the advection term at the quadrature points before
 *   taking the inner product, so the reaction term enters with a plus
 *   sign -- the opposite sign to the eFactorLambda convention of
 *   StdRegions::eLinearAdvectionDiffusionReaction;
 * - **diffusion** \f$(\nabla\phi_p, \mathbf{D}\nabla u)\f$ -- needs the
 *   constant symmetric diffusivity tensor \f$\mathbf{D}\f$, supplied
 *   through SetDiffCoeff() in the packed form described there.
 *
 * All three setters must have been called before Apply(); v_Apply()
 * asserts as much. With \f$\mathbf{V} = 0\f$ the operator reduces to the
 * Helmholtz operator of the same sign convention. The sum-factorised
 * implementations exploit that: they include the corresponding Helmholtz
 * kernel header and reuse its diffusion-metric kernels, adding only the
 * advection term. The two StdMat implementations include no Helmholtz header
 * and instead carry their own pointwise metric kernel, the Helmholtz one with
 * the advection term added.
 *
 * On a multi-plane 3DH1 expansion the block operators apply the form
 * plane by plane, and detail::LinAdvDiffReactionOpImpl adds the
 * homogeneous-direction terms -- the weak z-Laplacian and the advection
 * \f$w\,\partial u/\partial z\f$ -- after the block loop.
 *
 * The family follows the standard element-operator layout (see
 * ElmtOp.hpp):
 * - LinAdvDiffReactionOp (this file): whole-field interface setting the
 *   Apply() field states and forwarding the coefficient setters to the
 *   block operators;
 * - detail::LinAdvDiffReactionOpImpl (LinAdvDiffReactionOpImpl.hpp): the
 *   factory-registered whole-field implementation, one entry per
 *   execution space, which owns the z-operators of the homogeneous
 *   terms;
 * - LinAdvDiffReactionBlockOp (LinAdvDiffReactionBlockOp.hpp): per-block
 *   interface, holding the coefficients the implementations read;
 * - detail::LinAdvDiffReactionBlockOpImpl: the per-block
 *   implementations, one header per implementation family --
 *   LinAdvDiffReactionSerialAVXStdMat.hpp and
 *   LinAdvDiffReactionSerialAVXSumFac.hpp each serve both the Serial and
 *   the AVX space, LinAdvDiffReactionDeviceStdMat.hpp serves
 *   Device/StdMat, and LinAdvDiffReactionDeviceSumFac.hpp serves both
 *   device sum-factorisation strategies (SumFac and SumFacTOP);
 * - the kernels in LinAdvDiffReactionSerialAVXStdMatKernels.hpp,
 *   LinAdvDiffReactionSerialAVXSumFacKernels.hpp,
 *   LinAdvDiffReactionDeviceStdMatKernels.hpp,
 *   LinAdvDiffReactionDeviceSumFacKernels.hpp and
 *   LinAdvDiffReactionDeviceSumFacTOPKernels.hpp.
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtOp.hpp>

#include <MultiRegions/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Whole-field interface of the linearised
 * advection-diffusion-reaction operator: applies the weak
 * advection-diffusion-reaction form element by element, coefficients in
 * and coefficients out.
 *
 * @details
 * As with the other element-operator families, this class exists to set
 * the Apply() parameter types at compile time -- both fields in
 * FieldState::Coeff -- and to drive the per-block operators: it holds
 * one LinAdvDiffReactionBlockOp per element block, and v_Apply() hands
 * each block of the input and output fields to the matching entry. The
 * plane-local work happens in the block operators; on a multi-plane 3DH1
 * expansion the homogeneous terms are added after the block loop through
 * the v_SetAdvVelFFT() and v_ApplyFFT() hooks, which
 * detail::LinAdvDiffReactionOpImpl implements.
 *
 * Unlike the parameter-free families, this operator carries per-term
 * coefficients that must be installed before Apply(): SetLambda(),
 * SetDiffCoeff() and SetAdvVel() each forward to every block operator
 * and record that they have been called, and v_Apply() asserts that the
 * record is complete and consistent.
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see LinAdvDiffReactionBlockOp for the per-block interface and the
 * implementation strategies available per execution space.
 */
template <typename TData>
class LinAdvDiffReactionOp
    : public ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>
{
    friend class ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>;

public:
    /**
     * @brief Create a linearised advection-diffusion-reaction operator
     * through the operator and block-operator factories.
     *
     * Builds the interface object under the factory key #name +
     * @p execStr and one block operator per element block under
     * LinAdvDiffReactionBlockOp::name + @p execStr + @p implStr (see
     * ElmtOp::Create for the two-stage construction and for the session
     * defaults used when @p execStr and @p implStr are empty).
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Names of the field components the
     *                          operator is set up for.
     * @param   execStr         Execution space ("Serial", "AVX" or
     *                          "Device"); session default if empty.
     * @param   implStr         Block implementation ("StdMat", "SumFac"
     *                          or, on Device, "SumFacTOP"); session
     *                          default if empty.
     *
     * @return The fully assembled operator. The coefficient setters
     * still have to be called before Apply().
     */
    static std::shared_ptr<LinAdvDiffReactionOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>::
            template Create<LinAdvDiffReactionOp, LinAdvDiffReactionBlockOp>(
                expansionList, components, execStr, implStr);
    }

    /// Operator base name; Create() appends the execution space to it
    /// to form the operator-factory key.
    static inline const std::string name = "LinAdvDiffReaction";

    /**
     * @brief Set the reaction coefficient on every block operator.
     *
     * The reaction term is added to the advection term at the quadrature
     * points before the inner product with the basis, so the contribution
     * to the result is \f$+\lambda(\phi_p, u)\f$ -- see the file notes on
     * the sign convention.
     *
     * @param   lambda  Reaction coefficient, the same for every element.
     */
    void SetLambda(TData lambda)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetLambda(lambda);
        }
        m_isSetLambda = true;
    }

    /**
     * @brief Set the constant diffusivity tensor on every block operator.
     *
     * The tensor is symmetric and is passed packed into its lower
     * triangle, in the order the kernels' index map
     * (LibUtilities::GetDiffCoeffMap) expects --
     * \f$n_{coord}(n_{coord}+1)/2\f$ entries:
     * - 1D: \f$(D_{00})\f$;
     * - 2D: \f$(D_{00}, D_{01}, D_{11})\f$;
     * - 3D: \f$(D_{00}, D_{01}, D_{11}, D_{02}, D_{12}, D_{22})\f$,
     *
     * with \f$n_{coord}\f$ the coordinate dimension of the elements (of
     * the planes on a 3DH1 expansion, whose z term takes a unit
     * diffusivity with no coupling), not the shape dimension;
     * LinAdvDiffReactionBlockOp::SetDiffCoeff asserts the count. The same
     * tensor applies to every element and every quadrature point; a spatially
     * varying diffusivity is not available through this interface (see
     * #m_isSetVarDiffCoeff).
     *
     * @param   diffCoeff   Packed tensor entries; copied into a host
     *                      memory region held by each block operator.
     */
    void SetDiffCoeff(std::vector<TData> &diffCoeff)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetDiffCoeff(diffCoeff);
        }
        m_isSetDiffCoeff = true;
    }

    /**
     * @brief Set the advection velocity on every block operator.
     *
     * @p advVel is a physical-space field holding the velocity at the
     * quadrature points of every plane. Each block operator stores a
     * non-owning pointer to the matching block of it, so the field must
     * outlive the operator's Apply() calls. The implementations reshape the
     * block's storage in place to their own interleave width the first time
     * Apply() finds it at another width, updating the accessor's recorded
     * width: the field's layout is modified as a side effect.
     *
     * @param   advVel  Velocity field, GetCoordim() components, or three on
     *                  a 3DH1 expansion: the plane block operators take the
     *                  first two, each plane its own, and the third is the
     *                  through-plane velocity the z pass advects with.
     */
    void SetAdvVel(LibUtilities::Field<TData, FieldState::Phys> &advVel)
    {
        // A 3DH1 expansion carries a third velocity component along the
        // homogeneous direction, which the planes themselves do not count.
        ASSERTL1(advVel.GetNumComponents() ==
                     ((advVel.GetNumHomoModes() == 1)
                          ? static_cast<unsigned int>(
                                this->m_expansionList->GetCoordim(0))
                          : 3u),
                 "Advection velocity must have coordDim components, or three "
                 "on a 3DH1 expansion");

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetAdvVel(advVel.GetBlocks()[blk]);
        }
        m_isSetAdvVel = true;

        v_SetAdvVelFFT(advVel);
    }

protected:
    /// SetLambda() has been called.
    bool m_isSetLambda = false;
    /// SetDiffCoeff() has been called.
    bool m_isSetDiffCoeff = false;
    /// A spatially varying diffusivity has been installed. Nothing in
    /// this class sets it; v_Apply() requires exactly one of it and
    /// #m_isSetDiffCoeff.
    bool m_isSetVarDiffCoeff = false;
    /// SetAdvVel() has been called.
    bool m_isSetAdvVel = false;
    /// One block operator per element block of the expansion list;
    /// populated by ElmtOp::Create().
    std::vector<std::shared_ptr<LinAdvDiffReactionBlockOp<TData>>> m_blockOp;

    /**
     * @brief Construct the interface part of a concrete implementation;
     * called by the factory-registered creator functions.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Component names the operator is set up
     *                          for.
     */
    LinAdvDiffReactionOp(const MultiRegions::ExpListSharedPtr &expansionList,
                         const std::vector<std::string> &components)
        : ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>(expansionList,
                                                              components)
    {
    }

    ~LinAdvDiffReactionOp() override = default;

    /**
     * @brief Apply the operator block by block.
     *
     * Checks that @p in and @p out conform (equal component and
     * homogeneous-mode counts) and that the coefficients of all three
     * terms have been installed -- lambda, exactly one of the constant
     * and the varying diffusivity, and the advection velocity -- then
     * hands each pair of blocks to the corresponding entry of
     * #m_blockOp, then calls v_ApplyFFT() for the homogeneous terms. The
     * output is overwritten; this family has no append mode.
     *
     * @param   in      Coefficient-space input field.
     * @param   out     Coefficient-space output field, overwritten.
     */
    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &in,
                 LibUtilities::Field<TData, FieldState::Coeff> &out) override
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

        ASSERTL1(m_isSetAdvVel,
                 "Advection velocity has not been set."
                 "Set the value with SetAdvVel() before calling Apply().");

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            // Block dependent.
            auto &inblock  = in.GetBlocks()[blk];
            auto &outblock = out.GetBlocks()[blk];

            this->m_blockOp[blk]->Apply(inblock, outblock);
        }

        // Apply FFT.
        v_ApplyFFT(in, out);
    }

    /// The z terms of a multi-plane 3DH1 field: the weak z-Laplacian, and
    /// the advection along the homogeneous direction when SetAdvVel() was
    /// given a through-plane velocity.
    virtual void v_ApplyFFT(
        LibUtilities::Field<TData, FieldState::Coeff> &in,
        LibUtilities::Field<TData, FieldState::Coeff> &out) = 0;

    /// Hand the z pass the through-plane velocity, if there is one. The
    /// plane block operators have already taken the in-plane components.
    virtual void v_SetAdvVelFFT(
        LibUtilities::Field<TData, FieldState::Phys> &advVel) = 0;
};

} // namespace Nektar::MultiRegions
