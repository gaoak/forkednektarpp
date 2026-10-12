///////////////////////////////////////////////////////////////////////////////
//
// File: LaplacianOp.hpp
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
// Description: Public interface of the weak Laplacian element operator,
// which maps a coefficient-space field to coefficient space.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file LaplacianOp.hpp
 * @brief Public interface of the weak Laplacian (Laplacian) element
 * operator, which maps a coefficient-space field to coefficient space.
 *
 * @details
 * For every element the operator forms the coefficient-space vector
 * \f[ L_{pq} = \int_{\Omega^e} \nabla\phi_{pq} \cdot
 *              \left( \mathbf{D}\,\nabla u \right) \, d\mathbf{x},
 *     \qquad u = \sum_{rs} \hat{u}_{rs} \phi_{rs}, \f]
 * i.e. the weak form of \f$-\nabla\cdot(\mathbf{D}\nabla u)\f$ without
 * boundary terms, applied matrix-free to the element's coefficients.
 * \f$\mathbf{D}\f$ is the constant symmetric diffusion tensor set
 * through SetDiffCoeff(); with \f$\mathbf{D} = I\f$ this is the plain
 * weak Laplacian.
 *
 * On a multi-plane 3DH1 expansion the block operators apply this to
 * every homogeneous mode, with the tensor's xy entries, and
 * v_ApplyFFT() then adds the z term
 * \f$-(\phi_{pq}, \partial^2 u/\partial z^2)\f$, with a unit
 * coefficient and no xz/yz coupling.
 *
 * ### Geometric factors and the metric contraction
 * The integral is evaluated on the reference element, so the geometry
 * enters through the Jacobian \f$J\f$ and the derivative factors
 * \f$M_{kd} = \partial\xi_d/\partial x_k\f$ of
 * SpatialDomains::GeomFactors, which convert reference derivatives into
 * physical ones,
 * \f$\partial u/\partial x_k = \sum_d M_{kd}\,\partial u/\partial\xi_d\f$.
 * Substituting that for both \f$u\f$ and \f$\phi_{pq}\f$ turns the
 * integrand into a contraction with the metric
 * \f[ G_{de} = \sum_{k,l} M_{kd} D_{kl} M_{le},
 *     \qquad L_{pq} = \sum_{d,e} \int
 *     \frac{\partial\phi_{pq}}{\partial\xi_d} \, G_{de} \,
 *     \frac{\partial u}{\partial\xi_e} \, J \, d\boldsymbol{\xi}, \f]
 * the integral being replaced by the quadrature sum. On a block of
 * regular elements \f$M\f$ and \f$J\f$ are constant per element; on a
 * deformed block they vary from point to point. Every implementation in
 * this family is instantiated with a DEFORMED template parameter that
 * selects between the two.
 *
 * ### Relation to Helmholtz
 * The weak Helmholtz operator is this operator plus a
 * \f$\lambda\f$-weighted mass term,
 * \f$(\nabla\phi_{pq}, \mathbf{D}\nabla u) + \lambda(\phi_{pq}, u)\f$.
 * Rather than calling into this family, the Helmholtz family carries
 * its own copies of these implementation and kernel headers with the
 * mass term added.
 *
 * The family follows the standard element-operator layout (see
 * ElmtOp.hpp):
 * - LaplacianOp (this file): whole-field interface setting the Apply()
 *   field states, holding the diffusion tensor and running the
 *   per-block operators, then the v_ApplyFFT() hook;
 * - detail::LaplacianOpImpl (LaplacianOpImpl.hpp): the
 *   factory-registered whole-field implementation, one entry per
 *   execution space, which builds the 3DH1 z term and runs it in
 *   v_ApplyFFT();
 * - LaplacianBlockOp (LaplacianBlockOp.hpp): per-block interface;
 * - detail::LaplacianBlockOpImpl: the per-block implementations, one
 *   header per implementation family --
 *   LaplacianSerialAVXStdMat.hpp and LaplacianSerialAVXSumFac.hpp each
 *   serve both the Serial and the AVX space,
 *   LaplacianDeviceStdMat.hpp serves Device/StdMat, and
 *   LaplacianDeviceSumFac.hpp serves both device sum-factorisation
 *   strategies (SumFac and SumFacTOP);
 * - the kernels in LaplacianSerialAVXStdMatKernels.hpp,
 *   LaplacianSerialAVXSumFacKernels.hpp,
 *   LaplacianDeviceStdMatKernels.hpp,
 *   LaplacianDeviceSumFacKernels.hpp and
 *   LaplacianDeviceSumFacTOPKernels.hpp;
 * - detail::DerivZOpImpl (DerivZOpImpl.hpp) and MassOp, shared with
 *   the other operator families: the second z-derivative and the xy
 *   mass matrix forming the z term of multi-plane 3DH1 fields.
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtOp.hpp>

#include <MultiRegions/ElmtOps/Laplacian/LaplacianBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Whole-field interface of the weak Laplacian operator: maps
 * coefficient-space fields to coefficient space.
 *
 * @details
 * As with the other element-operator families, this class sets the
 * Apply() field states at compile time -- input and output in
 * FieldState::Coeff -- and drives one LaplacianBlockOp per element block;
 * the per-execution-space behaviour lives in the block operators, not here.
 * On a multi-plane 3DH1 expansion v_ApplyFFT(), implemented by
 * detail::LaplacianOpImpl, then adds the z term. See ElmtOp for the block
 * decomposition and factory scheme.
 *
 * Apply() always overwrites the output field.
 *
 * The constant symmetric diffusion tensor is set through SetDiffCoeff()
 * and forwarded to every block operator before the first Apply().
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see LaplacianBlockOp for the per-block interface and the
 * implementation strategies available per execution space.
 */
template <typename TData>
class LaplacianOp : public ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>
{
    friend class ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>;

public:
    /**
     * @brief Create a weak-Laplacian operator through the operator and
     * block-operator factories.
     *
     * Builds the interface object under the factory key
     * `"Laplacian" + execStr` and one block operator per element block
     * under `"BlockLaplacian" + execStr + implStr` (see ElmtOp::Create
     * for the two-stage construction and for the session defaults used
     * when @p execStr / @p implStr are empty).
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Names of the field components the
     *                          operator is set up for.
     * @param   execStr         Execution space ("Serial", "AVX" or
     *                          "Device"); session default if empty.
     * @param   implStr         Block implementation ("StdMat",
     *                          "SumFac", "SumFacTOP"); session default
     *                          if empty.
     *
     * @return The assembled operator. SetDiffCoeff() must still be
     * called before Apply().
     */
    static std::shared_ptr<LaplacianOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>::
            template Create<LaplacianOp, LaplacianBlockOp>(
                expansionList, components, execStr, implStr);
    }

    /// Operator base name; Create() appends the execution space to it
    /// to form the factory key.
    static inline const std::string name = "Laplacian";

    /**
     * @brief Set the constant symmetric diffusion tensor \f$\mathbf{D}\f$ used
     * by subsequent Apply() calls.
     *
     * The tensor is passed in packed upper-triangular form, one entry
     * per independent component, so @p diffCoeff holds
     * `ncoord * (ncoord + 1) / 2` values for a coordinate dimension of
     * `ncoord`: `{D00}` in 1D, `{D00, D01, D11}` in 2D and
     * `{D00, D01, D11, D02, D12, D22}` in 3D -- the order the kernels
     * index through LibUtilities::GetDiffCoeffMap() and the one the
     * factors map onto. The values are copied down to every block
     * operator; calling this again replaces them. On a 3DH1 expansion
     * `ncoord` is the plane's coordinate dimension: the z term takes a
     * unit coefficient.
     *
     * @param   diffCoeff   Packed tensor entries, in the order above.
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

protected:
    /// Set by SetDiffCoeff(); v_Apply() asserts on it in debug builds.
    bool m_isSetDiffCoeff = false;
    /// Placeholder for a spatially varying diffusion tensor.
    bool m_isSetVarDiffCoeff = false;

    /// One block operator per element block of the expansion list;
    /// populated by ElmtOp::Create().
    std::vector<std::shared_ptr<LaplacianBlockOp<TData>>> m_blockOp;

    /**
     * @brief Construct the interface part of a concrete implementation;
     * called by the factory-registered creator functions.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Component names the operator is set up
     *                          for.
     */
    LaplacianOp(const MultiRegions::ExpListSharedPtr &expansionList,
                const std::vector<std::string> &components)
        : ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>(expansionList,
                                                              components)
    {
    }

    ~LaplacianOp() override = default;

    /**
     * @brief Apply the weak Laplacian block by block.
     *
     * Checks that @p in and @p out conform (equal component and
     * homogeneous-mode counts) and that the diffusion tensor
     * has been set, hands each pair of blocks to the
     * corresponding entry of #m_blockOp, then calls v_ApplyFFT() to add
     * the z term.
     *
     * @param   in      Coefficient-space input field.
     * @param   out     Coefficient-space output field; overwritten.
     */
    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &in,
                 LibUtilities::Field<TData, FieldState::Coeff> &out) override
    {
        ASSERTL1(in.GetNumComponents() == out.GetNumComponents(),
                 "Number of input and output components differ");

        ASSERTL1(in.GetNumHomoModes() == out.GetNumHomoModes(),
                 "Number of input and output homogeneous modes differ");

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

        // Apply FFT.
        v_ApplyFFT(in, out);
    }

    /// Add the 3DH1 z term to @p out; called by v_Apply() after the
    /// block loop.
    virtual void v_ApplyFFT(
        LibUtilities::Field<TData, FieldState::Coeff> &in,
        LibUtilities::Field<TData, FieldState::Coeff> &out) = 0;
};

} // namespace Nektar::MultiRegions
