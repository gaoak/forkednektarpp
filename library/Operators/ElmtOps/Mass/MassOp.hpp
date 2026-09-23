///////////////////////////////////////////////////////////////////////////////
//
// File: MassOp.hpp
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
// Description: Whole-field interface of the mass element operator.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file MassOp.hpp
 * @brief Whole-field interface of the Mass element operator, which applies
 * every element's mass matrix to a coefficient-space field.
 *
 * @details
 * For every element the operator integrates the expansion against each of
 * that element's own basis functions,
 * \f[ I_n = \int_{\Omega^e} \phi_n\,u \,\mathrm{d}x
 *         = \sum_m \hat{u}_m \int_{\Omega^e} \phi_n\,\phi_m \,\mathrm{d}x
 *        \approx \sum_i w_i\,J_i\,\phi_n(\xi_i)
 *                 \sum_m \hat{u}_m\,\phi_m(\xi_i), \f]
 * that is, it applies the element mass matrix to the input coefficients.
 * Written that way the operator is the composition of the two families it
 * is built from: BwdTrans evaluates
 * \f$u(\xi_i) = \sum_m \hat{u}_m \phi_m(\xi_i)\f$ at the quadrature points,
 * and IProductWRTBase integrates that result against the basis, supplying
 * the quadrature metric -- the tensor product of the expansion's
 * per-direction weight tables \f$w_i\f$ times the element Jacobian
 * \f$J_i\f$ (one value per element for a regular geometry, one per
 * quadrature point for a deformed one). Every implementation in this
 * family reuses those two families' kernels or standard-region matrices
 * for the two stages; they differ in where the quadrature metric is
 * applied, set out below, and in that the regular-geometry StdMat path
 * collapses the pair into the single precomputed standard-region mass
 * matrix.
 *
 * Where the metric enters depends on the implementation:
 * - the sum-factorised paths, on the host and on the device, hand the
 *   expansion's per-direction weight tables and the block's Jacobian table
 *   to the inner-product kernel, which applies them inside its
 *   contraction; the exception is the device SumFacTOP path, whose kernels
 *   multiply the staged physical values by the metric between the two
 *   stages and then call the inner-product shape kernels in their
 *   metric-free form;
 * - the StdMat paths inherit the weights from the standard-region matrices
 *   they fetch from the data warehouse (which are built by applying the
 *   corresponding StdExpansion operation to unit vectors) and apply the
 *   Jacobian separately -- to the intermediate physical values for a
 *   deformed block, and, since the Jacobian is then one constant per
 *   element, to the coefficient-space result for a regular one.
 *
 * MassOp is the whole-field interface class of the family (see ElmtOp.hpp
 * for the two-level element-operator structure); the per-block interface
 * is MassBlockOp (MassBlockOp.hpp) and the factory shell
 * detail::MassOpImpl (MassOpImpl.hpp). The numerical work lives in the
 * detail::MassBlockOpImpl definitions -- MassSerialAVXStdMat.hpp and
 * MassSerialAVXSumFac.hpp for the Serial and AVX execution spaces,
 * MassDeviceStdMat.hpp for the Device StdMat implementation and
 * MassDeviceSumFac.hpp for both Device sum-factorisation strategies
 * (SumFac and SumFacTOP) -- registered with the block-operator factory by
 * CMake-generated translation units, one per execution space,
 * implementation and data type. The sum-factorised kernel launchers are in
 * MassSerialAVXSumFacKernels.hpp, MassDeviceSumFacKernels.hpp and
 * MassDeviceSumFacTOPKernels.hpp.
 *
 * @see BwdTransOp.hpp and IProductWRTBaseOp.hpp for the two operators
 * composed here, in particular the note in the latter on the weight tables
 * of the collapsed shapes.
 */

#pragma once

#include "Operators/ElmtOps/ElmtOp.hpp"

#include "Operators/ElmtOps/Mass/MassBlockOp.hpp"

namespace Nektar::Operators
{

/**
 * @brief Whole-field interface of the mass operator: applies each
 * element's mass matrix to a coefficient-space field.
 *
 * @details
 * As with the other element-operator families, this class sets the
 * Apply() field states at compile time -- coefficient space in and out,
 * BwdTransOp's input state composed with IProductWRTBaseOp's output state
 * -- and drives one MassBlockOp per element block; the per-execution-space
 * behaviour lives in the block operators, not here (the factory shell
 * detail::MassOpImpl adds nothing beyond registration). See ElmtOp for the
 * block decomposition and factory scheme.
 *
 * Apply() always overwrites the output field: the family has neither an
 * append mode nor a scale factor, and instantiates the underlying BwdTrans
 * kernels with APPEND false and the IProductWRTBase kernels with SCALE and
 * APPEND false throughout. The transform kernels have no SCALE parameter
 * of their own.
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see MassBlockOp for the per-block interface.
 */
template <typename TData>
class MassOp : public ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>
{
    friend class ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>;

public:
    /**
     * @brief Build a complete Mass operator through the operator and
     * block-operator factories.
     *
     * The operator-factory key is #name + @p execStr and the block-operator
     * key MassBlockOp::name + @p execStr + @p implStr; see ElmtOp::Create
     * for the two-stage construction and the execution-space and
     * implementation lookup rules.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Names of the field components the operator
     *                          is set up for.
     * @param   execStr         Execution space ("Serial", "AVX" or
     *                          "Device"); the session's setting is used
     *                          if empty.
     * @param   implStr         Implementation ("StdMat", "SumFac" or, on
     *                          Device, "SumFacTOP"); the session's
     *                          setting is used if empty.
     *
     * @return The fully assembled operator, ready to Apply().
     */
    static std::shared_ptr<MassOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>::
            template Create<MassOp, MassBlockOp>(expansionList, components,
                                                 execStr, implStr);
    }

    /// Operator base name; Create() appends the execution space to it to
    /// form the operator-factory key.
    static inline const std::string name = "Mass";

protected:
    /// One block operator per element block, filled by Create().
    std::vector<std::shared_ptr<MassBlockOp<TData>>> m_blockOp;

    /**
     * @brief Construct the interface part of a concrete operator; called by
     * the factory-registered creator functions.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Field component names the operator is set
     *                          up for.
     */
    MassOp(const MultiRegions::ExpListSharedPtr &expansionList,
           const std::vector<std::string> &components)
        : ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>(expansionList,
                                                              components)
    {
    }

    ~MassOp() override = default;

    /**
     * @brief Apply the operator block by block.
     *
     * Checks that the two fields agree in their component and
     * homogeneous-mode counts, then hands each input/output block pair to
     * the corresponding entry of #m_blockOp. Input and output are both in
     * coefficient space, so the two fields carry the same number of data
     * values per element.
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
