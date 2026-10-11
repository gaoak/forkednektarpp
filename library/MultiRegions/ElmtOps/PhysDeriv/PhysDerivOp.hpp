///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivOp.hpp
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
 * @file PhysDerivOp.hpp
 * @brief Public interface of the physical-space derivative (PhysDeriv)
 * element operator, which maps a field at the quadrature points to its
 * gradient at the same points.
 *
 * @details
 * PhysDeriv takes \f$u\f$ sampled at an element's quadrature points and
 * returns the Cartesian derivatives
 * \f[ \frac{\partial u}{\partial x_k}(\xi_i)
 *     = \sum_{d} \frac{\partial \xi_d}{\partial x_k}(\xi_i)
 *       \frac{\partial u}{\partial \xi_d}(\xi_i), \qquad
 *     k = 0,\dots,\mathrm{coordDim}-1, \f]
 * at those same points: a tensorial derivative on the standard element
 * followed by the chain rule with the element's derivative factors
 * \f$\partial \xi_d / \partial x_k\f$ (see the notes on PhysDerivOp).
 * Both input and output are therefore in FieldState::Phys. For an
 * ordinary (non-homogeneous) field the output carries `coordDim`
 * components per input component; for a 3D-homogeneous-1D (3DH1) field
 * with more than one homogeneous plane it always carries exactly three,
 * regardless of `coordDim` -- x and y from the per-block operators, z
 * from an FFT-based homogeneous derivative (see DerivZOpImpl.hpp).
 *
 * The family follows the standard element-operator layout (see
 * ElmtOp.hpp):
 * - PhysDerivOp (this file): whole-field interface setting the Apply()
 *   field states and running the per-block operators, then the
 *   v_ApplyFFT() hook;
 * - detail::PhysDerivOpImpl (PhysDerivOpImpl.hpp): the factory-registered
 *   whole-field implementation, one entry per execution space, which
 *   builds the homogeneous z-derivative and runs it in v_ApplyFFT();
 * - PhysDerivBlockOp (PhysDerivBlockOp.hpp): per-block interface;
 * - detail::PhysDerivBlockOpImpl: the per-block implementations, one
 *   header per implementation family --
 *   PhysDerivSerialAVXStdMat.hpp and PhysDerivSerialAVXSumFac.hpp each
 *   serve both the Serial and the AVX space, PhysDerivDeviceStdMat.hpp
 *   serves Device/StdMat, and PhysDerivDeviceSumFac.hpp serves both
 *   device sum-factorisation strategies (SumFac and SumFacTOP);
 * - the kernels in PhysDerivSerialAVXStdMatKernels.hpp,
 *   PhysDerivSerialAVXSumFacKernels.hpp,
 *   PhysDerivDeviceStdMatKernels.hpp,
 *   PhysDerivDeviceSumFacKernels.hpp and
 *   PhysDerivDeviceSumFacTOPKernels.hpp;
 * - detail::DerivZOpImpl (DerivZOpImpl.hpp, shared with the other
 *   operator families): the homogeneous z-derivative used for
 *   multi-plane 3DH1 fields, invoked through v_ApplyFFT() after the
 *   per-block operators above.
 *
 * The kernel headers also define per-direction ("Dir") kernels, which
 * produce a single Cartesian derivative component and can accumulate
 * onto it. PhysDeriv itself never calls them: the sum-factorised ones
 * are the building blocks of the Divergence kernels (and, in one
 * dimension, of the CurlCurl kernels), the StdMat one,
 * MultiplyByDerivDirFactorKernel, of the Divergence StdMat
 * implementations. Conversely the all-component kernels PhysDeriv uses
 * are shared with other families: MultiplyByDerivFactorKernel with the
 * AdvectionDealias StdMat implementations, and the device
 * PhysDeriv{1,2,3}DSumFacKernel / PhysDeriv{1,2,3}DSumFacTOPKernel with
 * the AdvectionDealias, Helmholtz, Laplacian and LinAdvDiffReaction
 * device kernels.
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtOp.hpp>

#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivBlockOp.hpp>

namespace Nektar::MultiRegions
{

/// \brief Element-block operator that computes the physical-space derivative
/// \f$\nabla u\f$ over a collection of elements.
template <typename TData>
class PhysDerivOp : public ElmtOp<FieldState::Phys, FieldState::Phys, TData>
{
    friend class ElmtOp<FieldState::Phys, FieldState::Phys, TData>;

public:
    /**
     * @brief Create a physical-derivative operator, its per-block
     * operators and, for a multi-plane 3DH1 expansion, its homogeneous
     * z-derivative.
     *
     * Builds the interface object under the factory key
     * `"PhysDeriv" + execStr` and one block operator per element block
     * under `"BlockPhysDeriv" + execStr + implStr`; see ElmtOp::Create
     * for the two-stage construction and for the session defaults used
     * when @p execStr or @p implStr are empty. If @p expansionList is a
     * 3D-homogeneous-1D expansion holding more than one homogeneous
     * plane on this rank, the constructor of detail::PhysDerivOpImpl
     * additionally builds its z-derivative, detail::PhysDerivOpImpl::m_zOp;
     * for every other expansion that is left null.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Names of the field components the
     *                          operator is set up for; these name the
     *                          input components, the output field
     *                          carrying coordDim (or, for multi-plane
     *                          3DH1, three) of them each.
     * @param   execStr         Execution space ("Serial", "AVX" or
     *                          "Device"); session default if empty.
     * @param   implStr         Block implementation ("StdMat",
     *                          "SumFac" or, on the Device space,
     *                          "SumFacTOP"); session default if empty.
     *
     * @return The fully assembled operator, ready to Apply.
     */
    static std::shared_ptr<PhysDerivOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Phys, FieldState::Phys, TData>::
            template Create<PhysDerivOp, PhysDerivBlockOp>(
                expansionList, components, execStr, implStr);
    }

    /// Operator base name; ElmtOp::Create() appends the execution space
    /// to it to form the factory key, and uses it to look up the
    /// session's default block implementation.
    static inline const std::string name = "PhysDeriv";

protected:
    /// One block operator per element block of the expansion list;
    /// populated by ElmtOp::Create().
    std::vector<std::shared_ptr<PhysDerivBlockOp<TData>>> m_blockOp;

    /**
     * @brief Construct the interface part of a concrete implementation;
     * called by the factory-registered creator functions.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Component names the operator is set up
     *                          for.
     */
    PhysDerivOp(const MultiRegions::ExpListSharedPtr &expansionList,
                const std::vector<std::string> &components)
        : ElmtOp<FieldState::Phys, FieldState::Phys, TData>(expansionList,
                                                            components)
    {
    }

    ~PhysDerivOp() override = default;

    /**
     * @brief Apply the physical derivative block by block, adding the
     * homogeneous z-derivative for a multi-plane 3DH1 field.
     *
     * Requires that the fields conform -- @p out must have exactly
     * `coordDim` times as many components as @p in for a field with a
     * single homogeneous mode and exactly three times as many for one
     * with several, and the same number of homogeneous modes -- then
     * hands each pair of blocks to the corresponding entry of
     * #m_blockOp. It then calls v_ApplyFFT(), which for a field with
     * more than one homogeneous mode writes the z component of every
     * output block.
     *
     * @param   in      Physical-space input field.
     * @param   out     Physical-space output field, overwritten; see
     *                  the file notes for its component count.
     */
    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &out) override
    {
        ASSERTL1(in.GetNumComponents() ==
                     out.GetNumComponents() /
                         (in.GetNumHomoModes() == 1
                              ? static_cast<unsigned int>(
                                    this->m_expansionList->GetCoordim(0))
                              : 3u),
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

        // Apply FFT.
        v_ApplyFFT(in, out);
    }

    /// Homogeneous z-derivative, called by v_Apply() after the block
    /// loop; writes the z component of every output block of a
    /// multi-plane 3DH1 field and leaves any other field untouched.
    virtual void v_ApplyFFT(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &out) = 0;
};

} // namespace Nektar::MultiRegions
