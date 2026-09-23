///////////////////////////////////////////////////////////////////////////////
//
// File: PhysInterp1DScaledOp.hpp
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
 * @file PhysInterp1DScaledOp.hpp
 * @brief Whole-field interface of the PhysInterp1DScaled element operator,
 * which re-evaluates a physical-space field on a rescaled quadrature grid
 * of the same elements.
 *
 * @details
 * ### What the operator computes
 * Each element of a block holds its values on a tensor-product grid of
 * quadrature points, \f$nm_d\f$ points in reference direction \f$d\f$.
 * The operator interpolates those values onto a grid of \f$nq_d\f$ points
 * of the same points distribution, so input and output are both in
 * FieldState::Phys and only the number of points changes. In one
 * direction that is
 * \f[ u(\xi^{new}_i) = \sum_{p} h_p(\xi^{new}_i)\, u(\xi_p), \f]
 * where \f$h_p\f$ is the Lagrange interpolant through the element's own
 * quadrature points, \f$h_p(\xi_r) = \delta_{pr}\f$; in two and three
 * directions it is the tensor product of one such sum per direction,
 * \f[ u(\xi^{new}_{i}, \eta^{new}_{j}) =
 *     \sum_{p,q} h_p(\xi^{new}_{i})\, h_q(\eta^{new}_{j})\,
 *     u(\xi_p, \eta_q). \f]
 * The element's geometry never enters -- no Jacobians or metric terms --
 * and neither do the expansion's modes: the operator maps point values to
 * point values. It is the operator-library counterpart of
 * MultiRegions::ExpList::PhysInterp1DScaled, the over-integration step
 * the solvers use to dealias nonlinear terms; PhysGalerkinProject1DScaledOp
 * maps the finer grid back. AdvectionDealiasOp performs the same
 * interpolation as a stage of its own kernels, fetching the same
 * warehouse data, rather than applying this operator.
 *
 * ### The scale factor
 * The output point counts come from the single scale factor \f$\sigma\f$
 * set through SetScaleFactor(), applied per direction to the *number of
 * quadrature points* (it is not a geometric scaling):
 * \f[ nq_0 = \lfloor \sigma\, nm_0 \rfloor, \qquad
 *     nq_d = \begin{cases}
 *       \lfloor \sigma\, nm_0 \rfloor - 1, & nm_0 - nm_d = 1 \\
 *       \lfloor \sigma\, nm_d \rfloor,     & \text{otherwise}
 *     \end{cases} \quad (d = 1, 2). \f]
 * The special case keeps a one-point offset between direction 0 and a
 * direction one point shorter -- the standard point relation of the
 * collapsed shapes -- intact in the output. It is the rule of
 * MultiRegions::ExpList::Get1DScaledTotPoints, which writes the offset
 * case as \f$\lfloor \sigma\, nm_0 - 1 \rfloor\f$, the same number for
 * \f$\sigma\, nm_0 \ge 1\f$; so the operator's output sizes match those
 * of a field the caller sized with it. Because the point counts differ
 * between input and output, the two fields do not share block
 * attributes: the output field must be built with per-element data
 * counts computed from the same \f$\sigma\f$ by this rule.
 *
 * @note These are the counts the operator interpolates onto, but not the
 * ones the sum-factorised implementations' size switch is driven by: that
 * switch recomputes per-direction output counts as
 * \f$\lfloor \sigma\, nm_d \rfloor\f$ from the scale factor and the input
 * counts, without the offset case, and reads the counts derived here only
 * for its runtime-sized fall-back (see the file notes of
 * PhysInterp1DScaledSerialAVXSumFac.hpp).
 *
 * ### Interpolation matrices
 * The per-direction matrices are the interpolation operators from the
 * direction's own points distribution to \f$nq_d\f$ points of that same
 * distribution, taken from the data warehouse under
 * LibUtilities::BasisDataKey with LibUtilities::eInterp and \f$nq_d\f$ as
 * the target count, and laid out exactly like a 1D basis table:
 * `I[p * nq_d + i]` is \f$h_p(\xi^{new}_i)\f$. That is why the
 * sum-factorised implementations can drive the backward transform's
 * tensor-product kernels unchanged, with the input points of a direction
 * playing the role of its modes. The StdMat implementations instead fetch
 * the assembled dense matrix over all directions (StdRegions::StdMatKey
 * with StdRegions::ePhysInterpStdMat or its transpose, the output counts
 * being part of the key). Either way the target point counts enter the
 * warehouse key, so one cached entry is built per basis and target
 * point count and shared between blocks and operators.
 *
 * The collapsed shapes need no special treatment: the physical points of
 * a triangle, tetrahedron, prism or pyramid form a tensor-product grid in
 * the collapsed coordinates, so the tensor-product kernels of the
 * segment, quadrilateral and hexahedron serve every shape of the matching
 * dimension.
 *
 * The family follows the standard element-operator layout (see
 * ElmtOp.hpp):
 * - PhysInterp1DScaledOp (this file): whole-field interface fixing the
 *   Apply() field states and carrying the scale factor to the blocks;
 * - detail::PhysInterp1DScaledOpImpl (PhysInterp1DScaledOpImpl.hpp): the
 *   factory-registered whole-field implementation, one entry per
 *   execution space;
 * - PhysInterp1DScaledBlockOp (PhysInterp1DScaledBlockOp.hpp): per-block
 *   interface, holding the scale factor;
 * - detail::PhysInterp1DScaledBlockOpImpl: the per-block implementations.
 *   PhysInterp1DScaledSerialAVXStdMat.hpp and
 *   PhysInterp1DScaledSerialAVXSumFac.hpp each serve both the Serial and
 *   the AVX space, PhysInterp1DScaledDeviceStdMat.hpp serves
 *   Device/StdMat, and PhysInterp1DScaledDeviceSumFac.hpp serves both
 *   device sum-factorisation strategies (SumFac and SumFacTOP);
 * - the kernel headers PhysInterp1DScaledDeviceSumFacKernels.hpp and
 *   PhysInterp1DScaledDeviceSumFacTOPKernels.hpp, which only include the
 *   backward transform's device kernels. The family has no Serial/AVX
 *   kernels header at all: the host sum-factorisation path includes
 *   BwdTransSerialAVXSumFacKernels.hpp directly, so on the Serial and AVX
 *   spaces the SumFac implementation is the backward transform's
 *   tensor-product kernels driven with interpolation matrices, with
 *   nothing of its own to specialise.
 */

#pragma once

#include "Operators/ElmtOps/ElmtOp.hpp"

#include "Operators/ElmtOps/PhysInterp1DScaled/PhysInterp1DScaledBlockOp.hpp"

namespace Nektar::Operators
{

/**
 * @brief Whole-field interface of the scaled physical-space
 * interpolation: re-evaluates a physical-space field on the rescaled
 * quadrature grid of every element.
 *
 * As with the other element-operator families, this class exists to fix
 * the Apply() parameter types at compile time -- physical space in,
 * physical space out -- and to drive the per-block operators: it holds
 * one PhysInterp1DScaledBlockOp per element block, and v_Apply() hands
 * each block of the input/output fields to the matching entry. All
 * numerical work happens in the block operators.
 *
 * SetScaleFactor() must be called before the first Apply(): it is what
 * sets the output point counts and builds or fetches the interpolation
 * data of every block operator (see the file notes for the scale-factor
 * rule). The output field must have been sized with the same factor.
 *
 * Unlike BwdTransOp this family has no append mode; Apply() always
 * overwrites the output.
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see PhysInterp1DScaledBlockOp for the per-block interface and the
 * implementation strategies available per execution space.
 */
template <typename TData>
class PhysInterp1DScaledOp
    : public ElmtOp<FieldState::Phys, FieldState::Phys, TData>
{
    friend class ElmtOp<FieldState::Phys, FieldState::Phys, TData>;

public:
    /**
     * @brief Create a scaled-interpolation operator through the operator
     * and block-operator factories.
     *
     * Builds the interface object under the factory key #name + @p execStr
     * and one block operator per element block under
     * PhysInterp1DScaledBlockOp::name + @p execStr + @p implStr (see
     * ElmtOp::Create for the two-stage construction and for the session
     * settings used when @p execStr or @p implStr is empty). The returned
     * operator still needs SetScaleFactor() before it can be applied.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Names of the field components the
     *                          operator is set up for.
     * @param   execStr         Execution space ("Serial", "AVX" or
     *                          "Device"); the session's setting if empty.
     * @param   implStr         Block implementation ("StdMat", "SumFac"
     *                          or, on the Device space, "SumFacTOP"); the
     *                          session's setting if empty.
     *
     * @return The assembled operator, ready to Apply() once its scale
     * factor is set.
     */
    static std::shared_ptr<PhysInterp1DScaledOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Phys, FieldState::Phys, TData>::
            template Create<PhysInterp1DScaledOp, PhysInterp1DScaledBlockOp>(
                expansionList, components, execStr, implStr);
    }

    /// Operator base name; Create() appends the execution space to it to
    /// form the operator-factory key.
    static inline const std::string name = "PhysInterp1DScaled";

    /**
     * @brief Set the scale factor of every block operator, fixing the
     * number of output points per direction.
     *
     * Forwarded to PhysInterp1DScaledBlockOp::SetScaleFactor on each
     * block, where it also triggers the lookup of the interpolation data
     * for the resulting point counts. Must be called before Apply().
     * Calling it again with a different factor re-targets the operator:
     * every implementation recomputes its point counts from the stored
     * input counts and fetches the matching data afresh.
     *
     * @param   scale   Multiplier applied to the per-direction
     *                  quadrature-point counts; see the file notes for
     *                  the exact rule.
     */
    void SetScaleFactor(TData scale)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetScaleFactor(scale);
        }
    }

protected:
    /// One block operator per element block of the expansion list;
    /// populated by ElmtOp::Create().
    std::vector<std::shared_ptr<PhysInterp1DScaledBlockOp<TData>>> m_blockOp;

    /**
     * @brief Construct the interface part of a concrete implementation;
     * called by the factory-registered creator functions.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Component names the operator is set up
     *                          for.
     */
    PhysInterp1DScaledOp(const MultiRegions::ExpListSharedPtr &expansionList,
                         const std::vector<std::string> &components)
        : ElmtOp<FieldState::Phys, FieldState::Phys, TData>(expansionList,
                                                            components)
    {
    }

    ~PhysInterp1DScaledOp() override = default;

    /**
     * @brief Interpolate block by block.
     *
     * Checks that @p in and @p out agree in component and
     * homogeneous-mode count -- their per-element data counts
     * deliberately differ, being the unscaled and the scaled point counts
     * -- then hands each pair of blocks to the corresponding entry of
     * #m_blockOp.
     *
     * @param   in      Physical-space input field on the elements' own
     *                  quadrature grids.
     * @param   out     Physical-space output field on the scaled grids;
     *                  overwritten.
     */
    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &out) override
    {
        ASSERTL1(in.GetNumComponents() == out.GetNumComponents(),
                 "Number of input and output components differ");

        ASSERTL1(in.GetNumHomoModes() == out.GetNumHomoModes(),
                 "Number of input and output homogeneous modes differ");

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            // Block dependent.
            auto &inblock  = in.GetBlocks()[blk];
            auto &outblock = out.GetBlocks()[blk];

            this->m_blockOp[blk]->Apply(inblock, outblock);
        }
    }
};

} // namespace Nektar::Operators
