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

/**
 * @file HelmholtzOp.hpp
 * @brief Public interface of the weak Helmholtz (Helmholtz) element
 * operator, which maps a coefficient-space field to coefficient space.
 *
 * @details
 * For every element the operator forms the coefficient-space vector
 * \f[ H_{pq} = \int_{\Omega^e} \nabla\phi_{pq} \cdot
 *              \left( \mathbf{D}\,\nabla u \right) d\mathbf{x}
 *            + \lambda \int_{\Omega^e} \phi_{pq}\, u \, d\mathbf{x},
 *     \qquad u = \sum_{rs} \hat{u}_{rs} \phi_{rs}, \f]
 * that is, the weak Laplacian of the Laplacian family plus a
 * \f$\lambda\f$-weighted mass term, applied matrix-free to the
 * element's coefficients. \f$\mathbf{D}\f$ is the constant symmetric
 * diffusion tensor set through SetDiffCoeff() and \f$\lambda\f$ the
 * scalar set through SetLambda(); both must be set before the first
 * Apply().
 *
 * On a 3D-homogeneous-1D (3DH1) field with more than one homogeneous
 * plane the block operators apply this plane by plane, lambda term
 * included, and detail::HelmholtzOpImpl adds the weak z-Laplacian in
 * v_ApplyFFT(): the xy mass matrix applied to minus the second
 * z-derivative. The diffusion tensor covers the plane's coordinate
 * dimension only, so that term has a unit coefficient and no xz/yz
 * coupling.
 *
 * ### Geometric factors and the metric contraction
 * Both integrals are evaluated on the reference element, so the
 * geometry enters through the Jacobian \f$J\f$ and the derivative
 * factors \f$M_{kd} = \partial\xi_d/\partial x_k\f$ of
 * SpatialDomains::GeomFactors, which convert reference derivatives into
 * physical ones,
 * \f$\partial u/\partial x_k = \sum_d M_{kd}\,\partial u/\partial\xi_d\f$.
 * Substituting that for both \f$u\f$ and \f$\phi_{pq}\f$ in the first
 * integral turns its integrand into a contraction with the metric
 * \f[ G_{de} = \sum_{k,l} M_{kd}\, D_{kl}\, M_{le},
 *     \qquad H_{pq} = \sum_{d,e} \int
 *     \frac{\partial\phi_{pq}}{\partial\xi_d}\, G_{de}\,
 *     \frac{\partial u}{\partial\xi_e}\, J \, d\boldsymbol{\xi}
 *     + \lambda \int \phi_{pq}\, u\, J \, d\boldsymbol{\xi}, \f]
 * the integrals being replaced by quadrature sums. On a block of
 * regular elements \f$M\f$ and \f$J\f$ are constant per element; on a
 * deformed block they vary from point to point. Every implementation in
 * this family is instantiated with a DEFORMED template parameter that
 * selects between the two.
 *
 * ### Where lambda enters and how the two terms are combined
 * No implementation assembles a Helmholtz matrix: each evaluates the
 * derivatives of \f$u\f$ at the quadrature points and combines the two
 * contributions on the way back to coefficient space. The paths differ
 * both in where those derivatives come from and in where the sum is
 * taken:
 * - the StdMat paths obtain the reference derivatives directly from the
 *   coefficients, through one standard-region matrix per direction
 *   (eDerivStdMat), and evaluate \f$u\f$ itself by backward transform
 *   only when \f$\lambda \neq 0\f$. The two terms then stay apart to the
 *   very end: the mass term is the inner product of \f$\lambda J u\f$
 *   with the basis and is written to the output first, the derivative
 *   terms being accumulated onto it by running their inner products
 *   with a GEMM beta of one. When \f$\lambda = 0\f$ the backward
 *   transform and that first inner product are skipped altogether and
 *   the first derivative inner product overwrites the output instead.
 * - the SumFac paths evaluate \f$u\f$ once by backward transform and
 *   differentiate those physical values. \f$\lambda\f$ enters at the
 *   metric stage: the physical values are scaled in place by
 *   \f$\lambda\, W J\f$ while the derivatives are scaled by
 *   \f$W J G\f$ (\f$W\f$ the quadrature weights), the transposed
 *   derivative operators then sum the derivative terms *onto* that same
 *   quadrature-point array, and one inner product with the basis --
 *   without weights, which have already been applied -- produces the
 *   result. \f$\lambda = 0\f$ takes no special path on any of the
 *   SumFac paths; it simply zeroes the mass contribution.
 * The mass term is thus always the element-operator equivalent of the
 * Mass family, and the derivative terms of the Laplacian family.
 *
 * ### Relation to the Laplacian family
 * Setting \f$\lambda = 0\f$ makes this operator the weak Laplacian.
 * Rather than calling into that family, Helmholtz carries its own
 * copies of its implementation and kernel headers with the mass term
 * added.
 *
 * The family follows the standard element-operator layout (see
 * ElmtOp.hpp):
 * - HelmholtzOp (this file): whole-field interface setting the Apply()
 *   field states, holding lambda and the diffusion tensor, and running
 *   the per-block operators, then the v_ApplyFFT() hook;
 * - detail::HelmholtzOpImpl (HelmholtzOpImpl.hpp): the
 *   factory-registered whole-field implementation, one entry per
 *   execution space, which builds the weak z-Laplacian and runs it in
 *   v_ApplyFFT();
 * - HelmholtzBlockOp (HelmholtzBlockOp.hpp): per-block interface;
 * - detail::HelmholtzBlockOpImpl: the per-block implementations, one
 *   header per implementation family --
 *   HelmholtzBlockOpSerialAVXStdMat.hpp and HelmholtzBlockOpSerialAVXSumFac.hpp
 * each serve both the Serial and the AVX space,
 *   HelmholtzBlockOpDeviceStdMat.hpp serves Device/StdMat, and
 *   HelmholtzBlockOpDeviceSumFac.hpp serves both device sum-factorisation
 *   strategies (SumFac and SumFacTOP);
 * - the kernels in HelmholtzBlockOpSerialAVXStdMatKernels.hpp,
 *   HelmholtzBlockOpSerialAVXSumFacKernels.hpp,
 *   HelmholtzBlockOpDeviceStdMatKernels.hpp,
 *   HelmholtzBlockOpDeviceSumFacKernels.hpp and
 *   HelmholtzBlockOpDeviceSumFacTOPKernels.hpp;
 * - detail::DerivZOpImpl (DerivZOpImpl.hpp) and MassOp: the second
 *   z-derivative and the xy mass matrix forming the weak z-Laplacian of
 *   multi-plane 3DH1 fields.
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtOp.hpp>

#include <MultiRegions/ElmtOps/Helmholtz/HelmholtzBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Whole-field interface of the weak Helmholtz operator: maps
 * coefficient-space fields to coefficient space.
 *
 * @details
 * As with the other element-operator families, this class sets the
 * Apply() field states at compile time -- input and output in
 * FieldState::Coeff -- and drives one HelmholtzBlockOp per element block,
 * followed by v_ApplyFFT(), which detail::HelmholtzOpImpl implements to
 * add the weak z-Laplacian of a multi-plane 3DH1 field. See ElmtOp for
 * the block decomposition and factory scheme.
 *
 * Apply() always overwrites the output field.
 *
 * Two settings specify what the block operators compute: the mass weight
 * lambda and the packed symmetric diffusion tensor, forwarded to every
 * block operator by SetLambda() and SetDiffCoeff().
 *
 * @tparam TData Floating-point type of the field data.
 */
template <typename TData>
class HelmholtzOp : public ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>
{
    friend class ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>;

public:
    /**
     * @brief Create a weak-Helmholtz operator through the operator and
     * block-operator factories.
     *
     * Builds the interface object under the factory key
     * `"Helmholtz" + execStr` and one block operator per element block
     * under `"BlockHelmholtz" + execStr + implStr` (see ElmtOp::Create
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
     * @return The assembled operator. SetLambda() and SetDiffCoeff()
     * must still be called before Apply().
     */
    static std::shared_ptr<HelmholtzOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>::
            template Create<HelmholtzOp, HelmholtzBlockOp>(
                expansionList, components, execStr, implStr);
    }

    /// Operator base name; Create() appends the execution space to it
    /// to form the factory key.
    static inline const std::string name = "Helmholtz";

    /**
     * @brief Set the scalar weight \f$\lambda\f$ of the mass term used
     * by subsequent Apply() calls.
     *
     * The value is copied down to every block operator; calling this
     * again replaces it. A lambda of zero leaves the plain weak
     * Laplacian -- the StdMat implementations then skip the mass term's
     * two matrix multiplications entirely, while the sum-factorised ones
     * take no special path: they carry the zero through as a scaling of
     * the physical values.
     *
     * @param   lambda  Weight of the \f$(\phi_{pq}, u)\f$ term.
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
     * operator; calling this again replaces them.
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
    /// Set by SetLambda(); checked by v_Apply() in debug builds.
    bool m_isSetLambda = false;
    /// Set by SetDiffCoeff(); checked by v_Apply() in debug builds.
    bool m_isSetDiffCoeff = false;
    /// Placeholder for a spatially varying diffusion tensor.
    bool m_isSetVarDiffCoeff = false;

    /// One block operator per element block of the expansion list;
    /// populated by ElmtOp::Create().
    std::vector<std::shared_ptr<HelmholtzBlockOp<TData>>> m_blockOp;

    /**
     * @brief Construct the interface part of a concrete implementation;
     * called by the factory-registered creator functions.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Component names the operator is set up
     *                          for.
     */
    HelmholtzOp(const MultiRegions::ExpListSharedPtr &expansionList,
                const std::vector<std::string> &components)
        : ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>(expansionList,
                                                              components)
    {
    }

    ~HelmholtzOp() override = default;

    /**
     * @brief Apply the weak Helmholtz operator block by block.
     *
     * Checks that @p in and @p out conform (equal component and
     * homogeneous-mode counts) and that lambda and the diffusion tensor
     * have been set, then hands each pair of blocks to the
     * corresponding entry of #m_blockOp. It then calls v_ApplyFFT(),
     * which for a field with more than one homogeneous mode adds the
     * weak z-Laplacian to @p out.
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

        // Apply FFT.
        v_ApplyFFT(in, out);
    }

    /// Weak z-Laplacian, called by v_Apply() after the block loop; adds
    /// it to every output block of a multi-plane 3DH1 field and leaves
    /// any other field untouched.
    virtual void v_ApplyFFT(
        LibUtilities::Field<TData, FieldState::Coeff> &in,
        LibUtilities::Field<TData, FieldState::Coeff> &out) = 0;
};

} // namespace Nektar::MultiRegions
