///////////////////////////////////////////////////////////////////////////////
//
// File: AvgAssmbScatrOpImpl.hpp
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
// Description: Implementation of the averaging assemble-scatter operator for
// one execution space, which weights a field by the cached inverse
// multiplicity and assembles and scatters it.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <LibUtilities/BasicUtils/Math/Math.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>
#include <MultiRegions/AssmbScatr/AssmbScatrOp.hpp>
#include <MultiRegions/AvgAssmbScatr/AvgAssmbScatrKernels.hpp>
#include <MultiRegions/AvgAssmbScatr/AvgAssmbScatrOp.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Implementation of AvgAssmbScatrOp for one execution space.
 *
 * @details
 * The constructor builds the inverse multiplicity
 * #m_invMultiplicity, the weight \f$1/m\f$ of every local coefficient, and
 * the AssmbScatrOp #m_assmbScatrOp. Apply() multiplies the input by the
 * weight and assembles and scatters the product, which forms
 * \f$s_i \sum_j s_j x_j / m\f$.
 *
 * The multiplicity is the sign-free assemble-scatter, AssmbScatrNoSignOp,
 * of a field of ones, which leaves \f$|s_i| m\f$ in every local
 * coefficient. That is a whole number, zero exactly where the sign is
 * zero, so its reciprocal is taken where it is at least one and the weight
 * is zero elsewhere, which sets the sign-zero coefficients to zero.
 *
 * The weight is built at interleave width 1. Apply() reshapes a block of
 * it to the interleave width of the matching block of the input when the
 * two differ, and keeps it at that width, so a caller applying the
 * operator repeatedly at one width reshapes it once. The input is not
 * reshaped by this operator, and AssmbScatrOp restores the width of the
 * output, so the output has the interleave width of the input on return.
 *
 * Construction applies AssmbScatrNoSignOp, which communicates in parallel
 * runs, so every rank must construct the operator together.
 *
 * @tparam ExecSpace  Execution back-end (NektarSpaces::Serial, AVX or
 *                    Device).
 * @tparam TData      Floating-point type of the coefficients.
 */
template <typename ExecSpace, typename TData>
class AvgAssmbScatrOpImpl : public AvgAssmbScatrOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    /// Build the inverse multiplicity and the assemble-scatter operator.
    AvgAssmbScatrOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                        const std::vector<std::string> &components)
        : AvgAssmbScatrOp<TData>(expansionList, components),
          m_assmbScatrOp(AssmbScatrOp<TData>::Create(expansionList, components,
                                                     ExecSpace::name)),
          m_invMultiplicity(
              "Inverse multiplicity",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)
    {
        m_invMultiplicity.template Initialize<MemSpace>(1.0);
        AssmbScatrNoSignOp<TData>::Create(expansionList, components,
                                          ExecSpace::name)
            ->Apply(m_invMultiplicity);

        for (unsigned int blk = 0; blk < m_invMultiplicity.GetBlocks().size();
             ++blk)
        {
            const unsigned int streamID = blk + 1;

            auto &block = m_invMultiplicity.GetBlocks()[blk];
            InverseMultiplicityKernel<ExecSpace>(
                block.CompSize() * m_invMultiplicity.GetNumComponents(),
                block.template GetPtr<MemSpace, ReadWrite>(streamID), streamID);
        }
    }

    /// Operator class name; defined by the generated factory code.
    static std::string className;

    /// Creator function registered with OperatorFactory.
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<AvgAssmbScatrOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    /// Assemble-scatter of the weighted field.
    std::shared_ptr<AssmbScatrOp<TData>> m_assmbScatrOp;
    /// Inverse multiplicity of every local coefficient, zero where the
    /// orientation sign is zero; each block at the interleave width of the
    /// last input it was applied to.
    LibUtilities::Field<TData, FieldState::Coeff> m_invMultiplicity;

    /// Average @p inout in place.
    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &inout) override
    {
        v_Apply(inout, inout);
    }

    /**
     * @brief Average @p in into @p out.
     *
     * Every block of #m_invMultiplicity is first reshaped to the interleave
     * width of the matching block of @p in where the two differ. @p in is
     * then multiplied by it into @p out, which takes the interleave width
     * of @p in, and @p out is assembled and scattered in place.
     *
     * @param   in      Coefficient-space input field; may be @p out.
     * @param   out     Coefficient-space output field.
     */
    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &in,
                 LibUtilities::Field<TData, FieldState::Coeff> &out) override
    {
        for (unsigned int blk = 0; blk < m_invMultiplicity.GetBlocks().size();
             ++blk)
        {
            const unsigned int streamID = blk + 1;

            auto &invMultiplicityBlk = m_invMultiplicity.GetBlocks()[blk];
            const unsigned int inWidth =
                in.GetBlocks()[blk].GetInterleaveWidth();
            const unsigned int invMultiplicityWidth =
                invMultiplicityBlk.GetInterleaveWidth();

            if (inWidth != invMultiplicityWidth)
            {
                auto invMultiplicityPtr =
                    invMultiplicityBlk.template GetPtr<MemSpace, ReadWrite>(
                        streamID);
                LibUtilities::ReshapeStorage<ExecSpace>(
                    inWidth, invMultiplicityWidth,
                    invMultiplicityBlk.GetNumElementsWithPadding() *
                        m_invMultiplicity.GetNumComponents(),
                    invMultiplicityBlk.GetNumData(), invMultiplicityPtr,
                    streamID);
                invMultiplicityBlk.template SetInterleaveWidth<TData>(inWidth);
            }
        }

        Math::mul<ExecSpace>(m_invMultiplicity, in, out);
        m_assmbScatrOp->Apply(out);
    }
};

} // namespace Nektar::MultiRegions::detail
