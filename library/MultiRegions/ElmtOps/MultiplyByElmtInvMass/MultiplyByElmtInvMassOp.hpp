///////////////////////////////////////////////////////////////////////////////
//
// File: MultiplyByElmtInvMassOp.hpp
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
 * @file MultiplyByElmtInvMassOp.hpp
 * @brief Whole-field interface of the MultiplyByElmtInvMass element
 * operator, which multiplies a coefficient-space field by the inverse of
 * each element's own mass matrix.
 *
 * @details
 * For every element the operator applies the inverse of that element's
 * mass matrix
 * \f[ M^e_{mn} = \int_{\Omega^e} \phi_m \phi_n \,\mathrm{d}\boldsymbol{x} \f]
 * to that element's coefficients. It maps coefficient space to coefficient
 * space and couples no degrees of freedom between elements; it is the
 * block-operator counterpart of ExpList::MultiplyByElmtInvMass. Applied to
 * the output of IProductWRTBaseOp it gives the element-local
 * (discontinuous Galerkin) forward transform, which is how
 * SolverCore::EquationSystem::ProjectPhysToCoeffs uses it. The weak-DG
 * advection and diffusion operators
 * (SolverCore::detail::AdvectionWeakDGOpImpl,
 * SolverCore::detail::DiffusionIPOpImpl and
 * SolverCore::detail::AdvWeakDGDiffusionIPOpImpl) apply it to their
 * assembled residual -- the volume inner product with respect to the
 * derivative of the basis plus the trace integral, and for interior-penalty
 * diffusion the symmetric penalty term -- to invert the element mass matrix
 * of the semi-discrete system.
 *
 * Unlike the transform-type families there is no StdMat/SumFac split: a
 * single implementation is registered per execution space under the
 * Generic tag (see ElmtBlockOp.hpp), and the only case distinction it makes
 * is between blocks of regular and of deformed geometry (see
 * MultiplyByElmtInvMassBlockOp.hpp). The family is laid out as
 * - MultiplyByElmtInvMassOp (this file): whole-field interface, which also
 *   assembles the per-element inverse mass matrices that blocks of deformed
 *   geometry need;
 * - detail::MultiplyByElmtInvMassOpImpl (MultiplyByElmtInvMassOpImpl.hpp):
 *   the factory-registered whole-field shell, one entry per execution
 *   space;
 * - MultiplyByElmtInvMassBlockOp (MultiplyByElmtInvMassBlockOp.hpp):
 *   per-block interface;
 * - detail::MultiplyByElmtInvMassBlockOpImpl, defined for the Serial and
 *   AVX execution spaces in MultiplyByElmtInvMassBlockOpSerialAVXGeneric.hpp
 * and for the Device execution space in
 *   MultiplyByElmtInvMassBlockOpDeviceGeneric.hpp.
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtOp.hpp>

#include <MultiRegions/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Whole-field interface of the element-wise inverse mass operator:
 * replaces every element's coefficients by the solution of that element's
 * mass system.
 *
 * @details
 * As with the other element-operator families this class sets the Apply()
 * field states at compile time -- coefficient space in, coefficient space
 * out -- and drives one MultiplyByElmtInvMassBlockOp per element block:
 * v_Apply() hands each block of the input and output fields to the matching
 * entry of #m_blockOp, and all numerical work happens in the block
 * operators (the factory shell detail::MultiplyByElmtInvMassOpImpl adds
 * nothing beyond registration). What this class adds on top of the
 * standard ElmtOp::Create machinery is the assembly of the per-element
 * inverse mass matrices that blocks of deformed geometry need; see
 * Create().
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see MultiplyByElmtInvMassBlockOp for the per-block interface and the
 * regular/deformed case distinction the implementations make.
 */
template <typename TData>
class MultiplyByElmtInvMassOp
    : public ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>
{
    friend class ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>;

public:
    /**
     * @brief Build a complete element-wise inverse mass operator through
     * the operator and block-operator factories and hand each block
     * operator the inverse mass data it needs.
     *
     * The interface object and one block operator per element block are
     * built by ElmtOp::Create under the keys #name + @p execStr and
     * MultiplyByElmtInvMassBlockOp::name + @p execStr + @p implStr. Only
     * the "Generic" implementation is registered for this family, so
     * ElmtBlockOp::Create falls back to it whatever @p implStr asks for.
     *
     * Every block is then handed its inverse mass matrices through
     * MultiplyByElmtInvMassBlockOp::SetInvMassMatrix:
     * - a block of deformed elements receives one dense nmTot x nmTot
     *   matrix per element, copied element by element from the block's
     *   Collection -- Expansion::GetLocMatrix with StdRegions::eInvMass,
     *   which for deformed geometry builds and inverts the element's own
     *   mass matrix -- into a vector in element order. The vector is sized
     *   for the block's padded element count, so the trailing padding
     *   elements keep the zero matrices it was allocated with;
     * - a block of regular elements receives an empty vector: its
     *   elements' mass matrices are the standard-region mass matrix scaled
     *   by the element's constant Jacobian, so the implementations combine
     *   the cached standard-region inverse with a division by the Jacobian
     *   instead of storing one matrix per element.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Names of the field components the operator
     *                          is set up for.
     * @param   execStr         Execution space ("Serial", "AVX" or
     *                          "Device"); the session's setting is used
     *                          if empty.
     * @param   implStr         Implementation; the session's setting is
     *                          used if empty, and either way the lookup
     *                          resolves to "Generic" as described above.
     *
     * @return The fully assembled operator, ready to Apply().
     *
     * @note The copy takes the raw storage of the matrix GetLocMatrix()
     * returns and does not apply its scale factor; for deformed geometry
     * LocalRegions returns the inverted mass matrix with a unit scale.
     */
    static std::shared_ptr<MultiplyByElmtInvMassOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        auto op = ElmtOp<FieldState::Coeff, FieldState::Coeff,
                         TData>::template Create<MultiplyByElmtInvMassOp,
                                                 MultiplyByElmtInvMassBlockOp>(
            expansionList, components, execStr, implStr);

        // Loop over the blocks.
        auto blockAttr =
            MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                expansionList);

        for (unsigned int blk = 0; blk < op->m_blockOp.size(); ++blk)
        {
            std::vector<TData> dmat;
            const auto coll =
                MultiRegions::GetCollection(expansionList, blk).GetExpVector();
            const auto exp   = coll[0];
            const auto nmTot = exp->GetNcoeffs();
            const auto deformed =
                exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
            const auto nelmt    = blockAttr[blk].GetNumElements();
            const auto nelmtPad = blockAttr[blk].GetNumElementsWithPadding();

            if (deformed)
            {
                dmat.resize(nelmtPad * nmTot * nmTot);
                auto dmatptr = dmat.data();
                for (size_t e = 0; e < nelmt; ++e)
                {
                    const auto exp = coll[e];
                    const auto &InvMass =
                        exp->GetLocMatrix(StdRegions::eInvMass);
                    std::copy_n(InvMass->GetRawPtr(), nmTot * nmTot, dmatptr);
                    dmatptr += nmTot * nmTot;
                }
            }

            op->m_blockOp[blk]->SetInvMassMatrix(dmat);
        }

        return op;
    }

    /// Operator base name; Create() appends the execution space to it to
    /// form the operator-factory key.
    static inline const std::string name = "MultiplyByElmtInvMass";

protected:
    /// One block operator per element block, filled by Create().
    std::vector<std::shared_ptr<MultiplyByElmtInvMassBlockOp<TData>>> m_blockOp;

    /**
     * @brief Construct the interface part of a concrete operator; called by
     * the factory-registered creator functions.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Field component names the operator is set
     *                          up for.
     */
    MultiplyByElmtInvMassOp(const MultiRegions::ExpListSharedPtr &expansionList,
                            const std::vector<std::string> &components)
        : ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>(expansionList,
                                                              components)
    {
    }

    ~MultiplyByElmtInvMassOp() override = default;

    /**
     * @brief Apply the operator block by block.
     *
     * Checks that the two fields agree in their component and
     * homogeneous-mode counts, then hands each input/output block pair to
     * the corresponding entry of #m_blockOp.
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

} // namespace Nektar::MultiRegions
