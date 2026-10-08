///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransOp.hpp
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
// Description: Whole-field interface of the backward transform, from
// coefficient space to physical space
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file BwdTransOp.hpp
 * @brief Public interface of the backward-transform (BwdTrans) element
 * operator, which maps a field from coefficient space to physical space.
 *
 * The backward transform evaluates each element's expansion at the
 * quadrature points of its basis,
 * \f[ u(\xi_i) = \sum_n \hat{u}_n \phi_n(\xi_i), \f]
 * where \f$\hat{u}_n\f$ are the element's coefficients and \f$\phi_n\f$
 * its basis functions. It is purely element-local and independent of the
 * element geometry -- no Jacobians or metric terms enter -- and is the
 * operator-library equivalent of MultiRegions::ExpList::BwdTrans, which
 * the BwdTrans unit tests use as the reference solution.
 *
 * The family follows the standard element-operator layout (see
 * ElmtOp.hpp):
 * - BwdTransOp (this file): whole-field interface fixing the Apply()
 *   field states;
 * - detail::BwdTransOpImpl (BwdTransOpImpl.hpp): the factory-registered
 *   whole-field implementation, one entry per execution space;
 * - BwdTransBlockOp (BwdTransBlockOp.hpp): per-block interface;
 * - detail::BwdTransBlockOpImpl: the per-block implementations, one
 *   header per implementation family rather than one per registered
 *   execution-space/strategy pairing -- BwdTransSerialAVXStdMat.hpp and
 *   BwdTransSerialAVXSumFac.hpp each serve both the Serial and the AVX
 *   space, BwdTransDeviceStdMat.hpp serves Device/StdMat, and
 *   BwdTransDeviceSumFac.hpp serves both device sum-factorisation
 *   strategies (SumFac and SumFacTOP);
 * - the sum-factorised kernels in BwdTransSerialAVXSumFacKernels.hpp,
 *   BwdTransDeviceSumFacKernels.hpp and
 *   BwdTransDeviceSumFacTOPKernels.hpp.
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtOp.hpp>

#include <MultiRegions/ElmtOps/BwdTrans/BwdTransBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Whole-field interface of the backward transform: evaluates a
 * coefficient-space field at the quadrature points of every element.
 *
 * As with the other element-operator families, this class exists to fix
 * the Apply() parameter types at compile time -- input in
 * FieldState::Coeff, output in FieldState::Phys -- and to drive the
 * per-block operators: it holds one BwdTransBlockOp per element block,
 * and v_Apply() hands each block of the input and output fields to the
 * matching entry. All numerical work happens in the block operators.
 *
 * By default Apply() overwrites the output field; after SetAppend(true)
 * it instead accumulates the transform onto the output's existing
 * values.
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see BwdTransBlockOp for the per-block interface and the
 * implementation strategies available per execution space.
 */
template <typename TData>
class BwdTransOp : public ElmtOp<FieldState::Coeff, FieldState::Phys, TData>
{
    friend class ElmtOp<FieldState::Coeff, FieldState::Phys, TData>;

public:
    /**
     * @brief Create a backward-transform operator through the operator
     * and block-operator factories.
     *
     * Builds the interface object under the factory key
     * `"BwdTrans" + execStr` and one block operator per element block
     * under `"BlockBwdTrans" + execStr + implStr`; see ElmtOp::Create
     * for the two-stage construction and for the session defaults used
     * when @p execStr or @p implStr are empty.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Names of the field components the
     *                          operator is set up for.
     * @param   execStr         Execution space ("Serial", "AVX" or
     *                          "Device"); session default if empty.
     * @param   implStr         Block implementation ("StdMat",
     *                          "SumFac", "SumFacTOP" (Device only));
     *                          session default if empty.
     *
     * @return The fully assembled operator, ready to Apply().
     */
    static std::shared_ptr<BwdTransOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Coeff, FieldState::Phys, TData>::
            template Create<BwdTransOp, BwdTransBlockOp>(
                expansionList, components, execStr, implStr);
    }

    /// Operator base name; Create() appends the execution space to it
    /// to form the factory key.
    static inline const std::string name = "BwdTrans";

    /**
     * @brief Switch all block operators between overwrite and
     * accumulate mode.
     *
     * With @p append set, subsequent Apply() calls add the backward
     * transform onto the output field's existing contents instead of
     * overwriting them. The default is overwrite.
     *
     * @param   append  Accumulate into the output when true.
     */
    void SetAppend(const bool &append)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetAppend(append);
        }
    }

protected:
    /// One block operator per element block of the expansion list;
    /// populated by ElmtOp::Create().
    std::vector<std::shared_ptr<BwdTransBlockOp<TData>>> m_blockOp;

    /**
     * @brief Construct the interface part of a concrete implementation;
     * called by the factory-registered creator functions.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Component names the operator is set up
     *                          for.
     */
    BwdTransOp(const MultiRegions::ExpListSharedPtr &expansionList,
               const std::vector<std::string> &components)
        : ElmtOp<FieldState::Coeff, FieldState::Phys, TData>(expansionList,
                                                             components)
    {
    }

    ~BwdTransOp() override = default;

    /**
     * @brief Apply the backward transform block by block.
     *
     * Checks that @p in and @p out conform -- equal component and
     * homogeneous-mode counts -- then hands each pair of blocks to the
     * corresponding entry of #m_blockOp.
     *
     * @param   in      Coefficient-space input field.
     * @param   out     Physical-space output field; overwritten, or
     *                  accumulated into when append mode is set.
     */
    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &in,
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

} // namespace Nektar::MultiRegions
