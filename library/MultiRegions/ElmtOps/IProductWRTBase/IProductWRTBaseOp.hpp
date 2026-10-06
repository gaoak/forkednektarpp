///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseOp.hpp
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
// Description: Whole-field interface of the inner product with the
// basis
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTBaseOp.hpp
 * @brief Whole-field interface of the IProductWRTBase element operator,
 * which takes a physical-space field to its inner product with every basis
 * function of every element.
 *
 * @details
 * For every element the operator evaluates
 * \f[ I_n = \int_{\Omega^e} u\,\phi_n \,\mathrm{d}x
 *         \approx \sum_i w_i\,J_i\,\phi_n(\xi_i)\,u(\xi_i), \f]
 * that is, it applies the transpose of the element's backward-transform
 * matrix to the input after weighting the input by the quadrature metric:
 * the tensor product of the expansion's per-direction weight tables
 * \f$w_i\f$ times the element Jacobian \f$J_i\f$ (one value per element
 * for a regular geometry, one per quadrature point for a deformed one).
 * Through that Jacobian, and unlike BwdTrans, whose matrix it transposes,
 * the operator does depend on the element geometry.
 *
 * The per-direction weight tables are the expansion's own
 * (LibUtilities::BasisDataKey with eWeights): for the collapsed shapes --
 * triangle, tetrahedron, prism and pyramid -- they are not the raw
 * one-dimensional quadrature weights but already carry the factors of the
 * collapsed coordinate map, which is why \f$w_i\f$ above is their tensor
 * product rather than the tensor product of the Gaussian weights. The
 * standard-region matrices of the StdMat implementations carry the same
 * factors, being built by applying StdExpansion::IProductWRTBase to the
 * physical unit vectors.
 *
 * Two settings modify that default behaviour, both forwarded to every
 * block operator of the family:
 * - SetIntegration(false) drops the quadrature metric, leaving the plain
 *   basis transpose \f$\sum_i \phi_n(\xi_i)\,u(\xi_i)\f$ for input that
 *   already carries \f$w J\f$. That is how the physical-output
 *   instantiation of IProductWRTDerivBase is completed into coefficients
 *   (see IProductWRTDerivBaseOp.hpp).
 * - SetScale(s) multiplies the result by the scalar s.
 *
 * IProductWRTBaseOp is the whole-field interface class of the family (see
 * ElmtOp.hpp for the two-level element-operator structure); the per-block
 * interface is IProductWRTBaseBlockOp (IProductWRTBaseBlockOp.hpp). The
 * numerical work lives in the detail::IProductWRTBaseBlockOpImpl
 * definitions -- IProductWRTBaseSerialAVXStdMat.hpp and
 * IProductWRTBaseSerialAVXSumFac.hpp for the Serial and AVX execution
 * spaces, IProductWRTBaseDeviceStdMat.hpp for the Device StdMat
 * implementation and IProductWRTBaseDeviceSumFac.hpp for both Device
 * sum-factorisation strategies (SumFac and SumFacTOP) -- registered with
 * the block-operator factory by CMake-generated translation units, one per
 * execution space, implementation and data type. The sum-factorised
 * kernels are in IProductWRTBaseSerialAVXSumFacKernels.hpp,
 * IProductWRTBaseDeviceSumFacKernels.hpp and
 * IProductWRTBaseDeviceSumFacTOPKernels.hpp.
 *
 * @see BwdTransOp.hpp for the operator this one transposes.
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtOp.hpp>

#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Whole-field interface of the inner product with the basis:
 * integrates a physical-space field against every basis function of every
 * element.
 *
 * @details
 * As with the other element-operator families, this class sets the
 * Apply() field states at compile time -- input in FieldState::Phys,
 * output in FieldState::Coeff, the reverse of BwdTransOp -- and drives one
 * IProductWRTBaseBlockOp per element block; the per-execution-space
 * behaviour lives in the block operators, not here (the factory shell
 * detail::IProductWRTBaseOpImpl adds nothing beyond registration). See
 * ElmtOp for the block decomposition and factory scheme.
 *
 * Apply() always overwrites the output field; the family has no append
 * mode (the APPEND template parameter of the kernels is instantiated false
 * throughout).
 *
 * Two settings modify what the block operators compute: the scale factor
 * applied to the result and the quadrature-metric switch, both forwarded
 * to every block operator by SetScale() and SetIntegration().
 *
 * @tparam TData Floating-point type of the field data.
 */
template <typename TData>
class IProductWRTBaseOp
    : public ElmtOp<FieldState::Phys, FieldState::Coeff, TData>
{
    friend class ElmtOp<FieldState::Phys, FieldState::Coeff, TData>;

public:
    /**
     * @brief Build a complete IProductWRTBase operator through the operator
     * and block-operator factories.
     *
     * The operator-factory key is #name + @p execStr and the block-operator
     * key IProductWRTBaseBlockOp::name + @p execStr + @p implStr; see
     * ElmtOp::Create for the two-stage construction and the execution-space
     * and implementation lookup rules.
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
    static std::shared_ptr<IProductWRTBaseOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Phys, FieldState::Coeff, TData>::
            template Create<IProductWRTBaseOp, IProductWRTBaseBlockOp>(
                expansionList, components, execStr, implStr);
    }

    /// Operator base name; Create() appends the execution space to it to
    /// form the operator-factory key.
    static inline const std::string name = "IProductWRTBase";

    /**
     * @brief Set the factor every block operator multiplies its result by.
     *
     * The factor applies to the coefficient-space result whether or not
     * the quadrature metric is in use. The initial value is 1; the
     * sum-factorised implementations then instantiate their kernels with
     * SCALE = false, dropping the final multiplication, while the StdMat
     * implementations have no such kernel variant and pass the factor on
     * whatever its value -- as the scale argument of their Jacobian kernel
     * or as the alpha of their GEMM. The one exception is the Serial/AVX
     * StdMat path with the quadrature metric off, which scales its GEMM
     * result in a separate loop and skips that loop at 1.
     *
     * @param   scale   Factor applied to the result of subsequent Apply()
     *                  calls.
     */
    void SetScale(const TData &scale)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetScale(scale);
        }
    }

    /**
     * @brief Switch the quadrature metric on (the initial state) or off for
     * every block operator.
     *
     * With @p value false the operator applies the basis transpose alone,
     * without the quadrature weights and the Jacobian, which is what an
     * input already carrying \f$wJ\f$ needs (see the file notes). With
     * @p value true it performs the full inner product.
     *
     * @param   value   Apply the quadrature metric when true.
     */
    void SetIntegration(bool value)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetIntegration(value);
        }
    }

protected:
    /// One block operator per element block, filled by Create().
    std::vector<std::shared_ptr<IProductWRTBaseBlockOp<TData>>> m_blockOp;

    /**
     * @brief Construct the interface part of a concrete operator; called by
     * the factory-registered creator functions.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Field component names the operator is set
     *                          up for.
     */
    IProductWRTBaseOp(const MultiRegions::ExpListSharedPtr &expansionList,
                      const std::vector<std::string> &components)
        : ElmtOp<FieldState::Phys, FieldState::Coeff, TData>(expansionList,
                                                             components)
    {
    }

    ~IProductWRTBaseOp() override = default;

    /**
     * @brief Apply the operator block by block.
     *
     * Checks that the two fields agree in their component and
     * homogeneous-mode counts, then hands each input/output block pair to
     * the corresponding entry of #m_blockOp.
     *
     * @param   in      Physical-space input field.
     * @param   out     Coefficient-space output field; overwritten.
     */
    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
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

} // namespace Nektar::MultiRegions
