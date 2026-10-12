///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseOp.hpp
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
 * @file IProductWRTDerivBaseOp.hpp
 * @brief Whole-field interface of the IProductWRTDerivBase element
 * operator, which takes a vector-valued physical-space field to the inner
 * product of that vector with the gradient of the basis.
 *
 * @details
 * For every element the operator evaluates
 * \f[ I_p = (\nabla \phi_p, {\bf f})
 *         = \sum_k \int \frac{\partial \phi_p}{\partial x_k} f_k
 *           \, J \, \mathrm{d}\xi , \f]
 * the weak-form contribution a divergence term makes to the right-hand
 * side of a discretised equation. It is the block-operator equivalent of
 * ExpList::IProductWRTDerivBase.
 *
 * The integrand is evaluated on the reference element, so the chain rule
 * is applied first: with
 * \f$\partial\phi_p/\partial x_k =
 * \sum_d (\partial\xi_d/\partial x_k)\,\partial\phi_p/\partial\xi_d\f$
 * the sum over the coordinate directions k is folded into one
 * physical-space field per reference direction d,
 * \f[ g_d = \sum_{k} \frac{\partial \xi_d}{\partial x_k} f_k , \f]
 * and the operator is the sum over the reference directions of the inner
 * product of \f$g_d\f$ with \f$\partial\phi_p/\partial\xi_d\f$ on the
 * reference element, weighted by the quadrature weights and the Jacobian.
 * The \f$\partial\xi_d/\partial x_k\f$ are the derivative factors held per
 * block by the geometric data warehouse (DerivFactorKey; one set per
 * element on a regular block, one set per quadrature point on a deformed
 * one).
 *
 * The input field therefore carries coordDim components per output
 * component -- one per coordinate direction of the (possibly embedded)
 * element -- with the direction index running fastest: output component c
 * is formed from input components c * coordDim to
 * c * coordDim + coordDim - 1. On a multi-plane 3DH1 expansion there
 * are three per output component whatever coordDim, the block operators
 * reading the first coordDim and the z term below the third. Every
 * implementation assumes this layout; the whole-field v_Apply() checks
 * only the homogeneous-mode counts, the two fields legitimately differing
 * in their component counts.
 *
 * The output state is a template parameter, and the two instantiations
 * are separate operators with separate factory keys:
 * - FieldState::Coeff: the coefficients \f$I_p\f$ above.
 * - FieldState::Phys: the intermediate left once the transposed reference
 *   derivative has been applied but the basis has not, i.e. the value per
 *   quadrature point whose inner product with the basis, taken without
 *   the quadrature metric (IProductWRTBaseOp::SetIntegration(false)),
 *   gives the coefficient result.
 *
 * On a multi-plane 3DH1 expansion the block operators cover the x and y
 * terms plane by plane, and for a Coeff output v_ApplyFFT() then
 * subtracts the inner product with the basis of
 * \f$\partial f_z/\partial z\f$, the z term after integration by parts.
 * The Phys output does not support homogeneous expansions.
 *
 * IProductWRTDerivBaseOp is the whole-field interface class of the family
 * (see ElmtOp.hpp for the two-level element-operator structure); the
 * per-block interface is IProductWRTDerivBaseBlockOp
 * (IProductWRTDerivBaseBlockOp.hpp). The numerical work lives in the
 * detail::IProductWRTDerivBaseBlockOpImpl definitions --
 * IProductWRTDerivBaseSerialAVXStdMat.hpp and
 * IProductWRTDerivBaseSerialAVXSumFac.hpp for the Serial and AVX execution
 * spaces, IProductWRTDerivBaseDeviceStdMat.hpp and
 * IProductWRTDerivBaseDeviceSumFac.hpp for the Device execution space --
 * registered with the block-operator factory by CMake-generated
 * translation units, one per execution space, implementation and data
 * type, each registering both output states.
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtOp.hpp>

#include <MultiRegions/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Whole-field interface of the inner product with the gradient of
 * the basis: takes a physical-space field of coordDim components per
 * output component to a field in state @p TFieldOut.
 *
 * @details
 * As with the other element-operator families, this class sets the
 * Apply() field states at compile time -- input in FieldState::Phys,
 * output in @p TFieldOut -- and drives one IProductWRTDerivBaseBlockOp per
 * element block; the per-execution-space behaviour lives in the block
 * operators, not here (detail::IProductWRTDerivBaseOpImpl adds only the
 * homogeneous z term, see the file notes).
 * See ElmtOp for the block decomposition and factory scheme.
 *
 * Two settings modify what the block operators compute: an append flag
 * (accumulate onto the output instead of overwriting it) and a scale
 * factor applied to the result, both forwarded to every block operator by
 * SetAppend() and SetScale(); the scale also reaches the z term.
 *
 * @tparam TFieldOut Field state produced by Apply(): FieldState::Coeff for
 *                   the modal inner product, FieldState::Phys for the
 *                   quadrature-point intermediate (see the file notes).
 * @tparam TData     Floating-point type of the field data.
 */
template <FieldState TFieldOut, typename TData>
class IProductWRTDerivBaseOp : public ElmtOp<FieldState::Phys, TFieldOut, TData>
{
    template <typename TDataOp>
    using TIProductWRTDerivBaseOp = IProductWRTDerivBaseOp<TFieldOut, TDataOp>;
    template <typename TDataOp>
    using TIProductWRTDerivBaseBlockOp =
        IProductWRTDerivBaseBlockOp<TFieldOut, TDataOp>;
    friend class ElmtOp<FieldState::Phys, TFieldOut, TData>;

public:
    /**
     * @brief Build a complete IProductWRTDerivBase operator through the
     * operator and block-operator factories.
     *
     * The operator-factory key is #name + @p execStr and the
     * block-operator key IProductWRTDerivBaseBlockOp::name + @p execStr +
     * @p implStr, both names carrying the output state; see ElmtOp::Create
     * for the two-stage construction and the execution-space and
     * implementation lookup rules.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Names of the output field components, each
     *                          of which consumes coordDim input components.
     * @param   execStr         Execution space ("Serial", "AVX" or
     *                          "Device"); the session's setting is used
     *                          if empty.
     * @param   implStr         Implementation ("StdMat", "SumFac" or, on
     *                          Device, "SumFacTOP"); the session's
     *                          setting is used if empty.
     *
     * @return The fully assembled operator, ready to Apply().
     */
    static std::shared_ptr<IProductWRTDerivBaseOp<TFieldOut, TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Phys, TFieldOut, TData>::template Create<
            TIProductWRTDerivBaseOp, TIProductWRTDerivBaseBlockOp>(
            expansionList, components, execStr, implStr);
    }

    /// Operator base name, carrying the output state so that the Coeff and
    /// Phys instantiations register separately; Create() appends the
    /// execution space to it to form the operator-factory key.
    static inline const std::string name =
        "IProductWRTDerivBase" + FieldStateToString<TFieldOut>();

    /**
     * @brief Choose between overwriting and accumulating output.
     *
     * With @p append set, Apply() adds the inner product onto the existing
     * contents of the output field instead of overwriting it: the kernels
     * switch to accumulating stores and the matrix paths to a GEMM with
     * beta = 1. The initial state is overwrite.
     *
     * @note On the nodal shapes (NodalTri, NodalTet, NodalPrism) the flag
     * is not guaranteed to reach the output, which may be overwritten.
     *
     * @param   append  True to accumulate, false (the initial state) to
     *                  overwrite.
     */
    void SetAppend(bool append)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetAppend(append);
        }
    }

    /**
     * @brief Set the factor every block operator, and the z term,
     * multiplies its result by.
     *
     * The scale is applied to the operator's output, so with @p scale = -1
     * the operator yields \f$-(\nabla\phi_p,{\bf f})\f$, the sign the
     * weak-DG advection and interior-penalty diffusion operators use. The
     * initial value is 1.
     *
     * @param   scale   New scale factor.
     */
    void SetScale(const TData &scale)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetScale(scale);
        }
        v_SetScaleFFT(scale);
    }

protected:
    /// One block operator per element block, filled by Create().
    std::vector<std::shared_ptr<IProductWRTDerivBaseBlockOp<TFieldOut, TData>>>
        m_blockOp;

    /**
     * @brief Construct the interface part of a concrete operator; called
     * by the factory-registered creator functions.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Output component names the operator is set
     *                          up for.
     */
    IProductWRTDerivBaseOp(const MultiRegions::ExpListSharedPtr &expansionList,
                           const std::vector<std::string> &components)
        : ElmtOp<FieldState::Phys, TFieldOut, TData>(expansionList, components)
    {
    }

    ~IProductWRTDerivBaseOp() override = default;

    /**
     * @brief Apply the operator block by block.
     *
     * Checks that the two fields agree in their number of homogeneous
     * modes -- their component counts differ by the factor described in
     * the file notes, so those are not compared -- and, for a Phys output,
     * that there is a single one; then hands each input/output block pair
     * to the corresponding entry of #m_blockOp and calls v_ApplyFFT() for
     * the z term.
     *
     * @param   in      Physical-space input field holding the vector
     *                  \f${\bf f}\f$, coordDim components per output
     *                  component with the direction running fastest.
     * @param   out     Output field in state @p TFieldOut; overwritten, or
     *                  accumulated onto when append is set.
     */
    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, TFieldOut> &out) override
    {
        ASSERTL1(in.GetNumHomoModes() == out.GetNumHomoModes(),
                 "Number of input and output homogeneous modes differ");

        // The z direction's weak derivative is the plane inner product
        // weighted by the wavenumber, which lands in coefficient space, so
        // only the Coeff variant carries it. The guard is L0: without it a
        // homogeneous field would come back silently wrong in a release
        // build.
        if constexpr (TFieldOut == FieldState::Phys)
        {
            ASSERTL0(in.GetNumHomoModes() == 1,
                     "IProductWRTDerivBaseOp does not support homogeneous "
                     "(3DH1/3DH2) configurations with a Phys output; use the "
                     "Coeff variant.");
        }

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

    /// Pass the scale factor on to the z term; called by SetScale().
    virtual void v_SetScaleFFT(const TData &scale) = 0;

    /// Homogeneous z term, called by v_Apply() after the block loop;
    /// subtracts it from @p out on a multi-plane 3DH1 Coeff field and
    /// leaves any other field untouched.
    virtual void v_ApplyFFT(LibUtilities::Field<TData, FieldState::Phys> &in,
                            LibUtilities::Field<TData, TFieldOut> &out) = 0;
};

} // namespace Nektar::MultiRegions
