///////////////////////////////////////////////////////////////////////////////
//
// File: DivergenceOp.hpp
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
 * @file DivergenceOp.hpp
 * @brief Public interface of the Divergence element operator, which
 * evaluates the divergence of a physical-space vector field at the
 * quadrature points of every element.
 *
 * @details
 * The operator forms
 * \f[ \nabla\cdot\mathbf{u}
 *     = \sum_{d} \frac{\partial u_d}{\partial x_d}, \f]
 * where the sum runs over the shape dimension of the expansion and
 * \f$u_d\f$ is the d-th component of the input field. Each physical
 * derivative is obtained element by element, by differentiating in the
 * element's reference directions and contracting with that element's
 * derivative factors (the metric terms
 * \f$\partial\xi_j/\partial x_d\f$), so unlike BwdTrans the geometry
 * enters through the derivative-factor tables cached in the data
 * warehouse. On a multi-plane 3DH1 expansion this is done plane by
 * plane and \f$\partial w/\partial z\f$, w being input component 2,
 * is added afterwards by the homogeneous z-derivative.
 *
 * The family follows the standard element-operator layout (see
 * ElmtOp.hpp):
 * - DivergenceOp (this file): whole-field interface setting the
 *   Apply() field states and running the per-block operators, then the
 *   v_ApplyFFT() hook;
 * - detail::DivergenceOpImpl (DivergenceOpImpl.hpp): the
 *   factory-registered whole-field implementation, one entry per
 *   execution space, which builds the homogeneous z-derivative and
 *   runs it in v_ApplyFFT();
 * - DivergenceBlockOp (DivergenceBlockOp.hpp): per-block interface;
 * - detail::DivergenceBlockOpImpl: the per-block implementations, one
 *   header per implementation family --
 *   DivergenceBlockOpSerialAVXStdMat.hpp and
 * DivergenceBlockOpSerialAVXSumFac.hpp each serve both the Serial and the AVX
 * space, DivergenceBlockOpDeviceStdMat.hpp serves Device/StdMat, and
 *   DivergenceBlockOpDeviceSumFac.hpp serves both device sum-factorisation
 *   strategies (SumFac and SumFacTOP);
 * - the kernel launchers in DivergenceBlockOpSerialAVXSumFacKernels.hpp,
 *   DivergenceBlockOpDeviceSumFacKernels.hpp and
 *   DivergenceBlockOpDeviceSumFacTOPKernels.hpp, which chain the PhysDeriv
 *   kernels of their execution space component by component. The
 *   StdMat paths have no kernels of their own: they call the PhysDeriv
 *   StdMat kernels directly;
 * - detail::DerivZOpImpl (DerivZOpImpl.hpp, shared with the other
 *   operator families): the homogeneous z-derivative used for
 *   multi-plane 3DH1 fields, invoked through v_ApplyFFT() after the
 *   per-block operators above.
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtOp.hpp>

#include <MultiRegions/ElmtOps/Divergence/DivergenceBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Whole-field interface of the divergence operator: sums the
 * physical derivatives \f$\partial u_d / \partial x_d\f$ of the first
 * shape-dimension components of a physical-space field.
 *
 * As with the other element-operator families, this class exists to
 * set the Apply() parameter types at compile time -- both input and
 * output in FieldState::Phys -- and to drive the per-block operators:
 * it holds one DivergenceBlockOp per element block, and v_Apply()
 * hands each block of the input/output fields to the matching entry.
 * The block operators form the divergence over the shape dimension; on
 * a multi-plane 3DH1 expansion v_ApplyFFT(), implemented by
 * detail::DivergenceOpImpl, then adds \f$\partial w/\partial z\f$.
 *
 * The input field must carry at least as many components as the
 * expansion's shape dimension; the first shape-dimension components
 * are interpreted as the Cartesian components of the vector field.
 * Every block operator reads only those components and accumulates
 * their derivatives into output component 0, one homogeneous plane at
 * a time; in one dimension that is
 * \f$\mathrm{d}u_0/\mathrm{d}x\f$. A 3DH1 input carries the z
 * velocity as component 2.
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see DivergenceBlockOp for the per-block interface and the
 * implementation strategies available per execution space.
 */
template <typename TData>
class DivergenceOp : public ElmtOp<FieldState::Phys, FieldState::Phys, TData>
{
    friend class ElmtOp<FieldState::Phys, FieldState::Phys, TData>;

public:
    /**
     * @brief Create a divergence operator through the operator and
     * block-operator factories.
     *
     * Builds the interface object under the factory key
     * `"Divergence" + execStr` and one block operator per element
     * block under `"BlockDivergence" + execStr + implStr` (see
     * ElmtOp::Create for the two-stage construction and for the
     * session defaults used when @p execStr / @p implStr are empty).
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Names of the field components the
     *                          operator is set up for.
     * @param   execStr         Execution space ("Serial", "AVX" or
     *                          "Device"); session default if empty.
     * @param   implStr         Implementation strategy ("StdMat",
     *                          "SumFac" or, on the Device space,
     *                          "SumFacTOP"); session default if
     *                          empty.
     *
     * @return The fully assembled operator, ready to Apply.
     */
    static std::shared_ptr<DivergenceOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Phys, FieldState::Phys, TData>::
            template Create<DivergenceOp, DivergenceBlockOp>(
                expansionList, components, execStr, implStr);
    }

    /// Operator base name; Create() appends the execution space to it
    /// to form the factory key.
    static inline const std::string name = "Divergence";

protected:
    /// One block operator per element block, filled by ElmtOp::Create
    /// and driven in order by v_Apply().
    std::vector<std::shared_ptr<DivergenceBlockOp<TData>>> m_blockOp;

    /**
     * @brief Construct the interface part of a concrete operator;
     * called by the factory-registered creator functions.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Component names the operator is set up
     *                          for.
     */
    DivergenceOp(const MultiRegions::ExpListSharedPtr &expansionList,
                 const std::vector<std::string> &components)
        : ElmtOp<FieldState::Phys, FieldState::Phys, TData>(expansionList,
                                                            components)
    {
    }

    ~DivergenceOp() override = default;

    /**
     * @brief Apply the divergence block by block.
     *
     * Checks that the homogeneous-mode counts of @p in and @p out
     * agree, hands each block pair to the corresponding block
     * operator, then calls v_ApplyFFT() to add the z derivative.
     *
     * @param   in      Physical-space vector field; its first
     *                  shape-dimension components are differentiated,
     *                  and component 2 in z on a 3DH1 expansion.
     * @param   out     Physical-space field receiving the divergence.
     */
    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &out) override
    {
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

        // Apply FFT.
        v_ApplyFFT(in, out);
    }

    /// Add \f$\partial w/\partial z\f$ to @p out; called by v_Apply()
    /// after the block loop.
    virtual void v_ApplyFFT(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &out) = 0;
};

} // namespace Nektar::MultiRegions
