///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTPhysNormalDerivTraceOp.hpp
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
 * @file IProductWRTPhysNormalDerivTraceOp.hpp
 * @brief Public interface of the IProductWRTPhysNormalDerivTrace element
 * operator, the lift against the normal derivative of the test function,
 * which maps a field living on an element's traces into an array shaped
 * like the element's volume physical space.
 *
 * @details
 * Given a field evaluated at the quadrature points of an element's
 * traces, this operator accumulates
 * \f[ \Lambda_{\boldsymbol p} \mathrel{+}= \sum_{F \in \partial E}
 *     \int_F g \; \partial_n h_{\boldsymbol p}\big|_F \, \mathrm{d}s ,
 * \f]
 * where \f$h_{\boldsymbol p}\f$ is the tensor product of the cardinal
 * (hat) functions of the @em volume quadrature grid. The entries of
 * \f$\Lambda\f$ are therefore surface inner products of the trace field
 * against the normal derivative of a single cardinal function of the
 * volume grid, in the same pre-\f$B^{\mathsf{T}}\f$ convention as the
 * plain trace lift: they are not modal coefficients, and the modal
 * right-hand side is recovered downstream exactly as it is there.
 *
 * The derivative along the outward normal is expressed through the
 * element local derivatives \f$\partial_{\eta_d} h_{\boldsymbol p}\f$
 * and the per-point factors \f$J_F\, n \cdot \nabla \eta_d\f$ that
 * the data warehouse supplies under
 * LocalRegions::JacNormGeomFactorLocTraceKey. On a direction-\f$a\f$
 * trace the tensor product factorises as it does for the plain lift, so
 * that each term of the sum over \f$d\f$ is a sum-factorised inner
 * product in which one direction reads the derivative of its
 * interpolation table and the rest read the table itself. That is the
 * structure the kernels exploit, term by term.
 *
 * This is the term the symmetric interior-penalty diffusion needs beside
 * the plain trace lift of IProductWRTPhysTraceOp, and its input and
 * output layouts are those of that operator: a packed trace field in,
 * laid out by GetLocTraceBlockAttributes(), and a volume field out.
 *
 * The family follows the standard element-operator layout (see
 * ElmtOp.hpp):
 * - IProductWRTPhysNormalDerivTraceOp (this file): whole-field interface
 *   fixing the Apply() field states, plus the vector-input entry points;
 * - detail::IProductWRTPhysNormalDerivTraceOpImpl
 *   (IProductWRTPhysNormalDerivTraceOpImpl.hpp): the factory-registered
 *   whole-field implementation, one entry per execution space, which
 *   also supplies the per-direction staging copy ApplyVector() needs;
 * - IProductWRTPhysNormalDerivTraceBlockOp
 *   (IProductWRTPhysNormalDerivTraceBlockOp.hpp): per-block interface,
 *   which also carries the append flag, the tangential collocation flags
 *   and the active-direction selector;
 * - detail::IProductWRTPhysNormalDerivTraceBlockOpImpl: the per-block
 *   implementations, in
 *   IProductWRTPhysNormalDerivTraceSerialAVXSumFac.hpp (serving both
 *   the Serial and the AVX space) and
 *   IProductWRTPhysNormalDerivTraceDeviceSumFac.hpp (Device);
 * - the kernels in the matching *SumFacKernels.hpp headers.
 *
 * The per-shape entry points, the ShapeBlock() specialisations, come from the
 * generated translation-unit template
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in. It expands the
 * switch template
 * LibUtilities/BasicUtils/Switch/BlockOpSwitchPhysTraceExtract.h.in, whose
 * per-shape template arguments carry trace point counts per normal direction.
 *
 * @note There is one algorithm here, registered under Operators::SumFac
 * and, on Device, under SumFacTOP from the same header. A StdMat request
 * is mapped to SumFac by IProductWRTPhysNormalDerivTraceBlockOp::Create(),
 * so a whole-field selection of StdMat, SumFac or SumFacTOP can include
 * this operator.
 *
 * @see IProductWRTPhysTraceOp.hpp for the plain trace lift this operator
 * is built on, whose layouts and conventions it follows.
 */

#pragma once

#include "Operators/ElmtOps/ElmtOp.hpp"

#include "Operators/ElmtOps/IProductWRTPhysNormalDerivTrace/IProductWRTPhysNormalDerivTraceBlockOp.hpp"

namespace Nektar::Operators
{

/**
 * @brief Whole-field interface of the normal-derivative trace lift: for
 * every element, lifts trace data into a volume physical-space shaped
 * array tested against \f$\partial_n h_{\boldsymbol p}\f$ rather than
 * \f$h_{\boldsymbol p}\f$.
 *
 * @details
 * As with the other element-operator families, this class exists to fix
 * the Apply() parameter types at compile time and to drive the per-block
 * operators: it holds one IProductWRTPhysNormalDerivTraceBlockOp per
 * element block, and v_Apply() hands each block of the input/output
 * fields to the matching entry. All numerical work happens in the block
 * operators.
 *
 * ### Field layouts
 * Both fields are in FieldState::Phys, but they are not the same shape
 * and the type system does not tell them apart:
 * - @em in is a packed trace field, laid out by
 *   GetLocTraceBlockAttributes(). Within an element the traces come in
 *   pair order, the \f$N_0\f$ pair first, then the \f$N_1\f$ pair or
 *   single, then \f$N_2\f$, face-major within a pair.
 * - @em out is a volume field, laid out by GetBlockAttributes(), with
 *   one entry per volume quadrature point.
 *
 * The conformance checks in v_Apply() therefore compare only component
 * counts and homogeneous-mode counts. Passing the two layouts the wrong
 * way round is not caught here.
 *
 * ### Entry points
 * - Apply() (inherited from ElmtOp) lifts one scalar trace field against
 *   the normal derivative of the basis, the normal folded into the
 *   geometric factors. Whether the volume array is zeroed first or
 *   accumulated onto is selected by SetAppend().
 * - ApplyVector() lifts one trace field per Cartesian direction against
 *   that direction's derivative alone, in two overloads differing only
 *   in how the caller hands the per-direction data over.
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see IProductWRTPhysNormalDerivTraceBlockOp for the per-block
 * interface and the implementations available per execution space.
 */
template <typename TData>
class IProductWRTPhysNormalDerivTraceOp
    : public ElmtOp<FieldState::Phys, FieldState::Phys, TData>
{
    friend class ElmtOp<FieldState::Phys, FieldState::Phys, TData>;

public:
    /**
     * @brief Create a normal-derivative trace lift operator through the
     * operator and block-operator factories.
     *
     * Builds the interface object under the factory key
     * `"IProductWRTPhysNormalDerivTrace" + execStr` and one block
     * operator per element block under
     * `"BlockIProductWRTPhysNormalDerivTrace" + execStr + implStr` (see
     * ElmtOp::Create for the two-stage construction and for the session
     * defaults used when @p execStr / @p implStr are empty).
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Names of the field components the
     *                          operator is set up for.
     * @param   execStr         Execution space ("Serial", "AVX" or
     *                          "Device"); session default if empty.
     * @param   implStr         Block implementation; StdMat resolves to
     *                          the SumFac registration, see the file
     *                          notes.
     *
     * @return The fully assembled operator, ready to Apply.
     */
    static std::shared_ptr<IProductWRTPhysNormalDerivTraceOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Phys, FieldState::Phys, TData>::
            template Create<IProductWRTPhysNormalDerivTraceOp,
                            IProductWRTPhysNormalDerivTraceBlockOp>(
                expansionList, components, execStr, implStr);
    }

    /// Operator base name; Create() appends the execution space to it
    /// to form the factory key.
    static inline const std::string name = "IProductWRTPhysNormalDerivTrace";

    /**
     * @brief Set the tangential collocation flags of every block
     * operator, passing false to force the fast path off.
     *
     * Where a trace slot's points coincide with the volume points the
     * tangential contraction collapses to a pointwise multiply by the trace
     * weights and the factors. Clearing the flags selects the general
     * contraction instead, which is how the unit tests reach that code
     * path on meshes whose points do coincide. The setter is wholesale, so
     * passing true does not restore the flags a block started with.
     *
     * @param   isCollocated    Value written to every flag of every block.
     */
    void SetIsCollocated(const bool isCollocated)
    {
        // Loop over the blocks.
        for (unsigned blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetIsCollocated(isCollocated);
        }
    }

    /**
     * @brief Set the append flag of this operator and of every block
     * operator.
     *
     * When set, the output field is accumulated onto instead of being
     * zeroed first. Every entry point consults it: Apply() through the
     * block flags, and both ApplyVector() overloads through #m_append,
     * whose value they restore on the blocks once their per-direction
     * passes are done.
     *
     * @param   append  New value of the flag.
     */
    void SetAppend(bool append)
    {
        m_append = append;

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetAppend(append);
        }
    }

    /**
     * @brief The vector-input lift: sum_k < dphi/dx_k, g_k >.
     *
     * Where Apply() lifts a single scalar trace field against the normal
     * derivative of the basis - the normal folded into the geometric
     * factors - this lifts one trace field per Cartesian direction against
     * that direction's derivative alone. It is the lift the compressible
     * SIP adjoint term needs, whose per-direction data (A^T n)_k [u] is not
     * a scalar times the normal.
     *
     * @p in carries dim * ncomp components, direction-minor: component
     * n * dim + k holds g_k of solution component n. @p out carries ncomp
     * components in the same pre-B^T physical convention as Apply().
     *
     * Implemented as dim passes of the scalar machinery, each against one
     * direction's uncontracted factors (SetActiveDir on the block ops),
     * with the input slice for that direction copied contiguously into
     * a trace field of its own; passes after the first append, and the
     * caller's own append setting governs the first.
     *
     * Available on every backend: SetActiveDir() is part of the block
     * operator interface.
     *
     * @param   in      Packed trace field of dim * ncomp components,
     *                  direction-minor.
     * @param   out     Volume-shaped output field of ncomp components,
     *                  zeroed and written or accumulated onto according
     *                  to SetAppend().
     */
    void ApplyVector(LibUtilities::Field<TData, FieldState::Phys> &in,
                     LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        const unsigned dim = this->m_expansionList->GetShapeDimension();

        ASSERTL1(in.GetNumComponents() == out.GetNumComponents() * dim,
                 "ApplyVector: input must carry dim * ncomp components");
        ASSERTL1(in.GetNumHomoModes() == 1 && out.GetNumHomoModes() == 1,
                 "ApplyVector: homogeneous modes are not supported");

        const unsigned ncomp = out.GetNumComponents();

        if (!m_activeDirTrace)
        {
            auto attrs = MultiRegions::GetLocTraceBlockAttributes<
                TData, FieldState::Phys>(this->m_expansionList);
            m_activeDirTrace =
                std::make_unique<LibUtilities::Field<TData, FieldState::Phys>>(
                    "f_dirtrace", attrs, ncomp, 1);
        }

        // Loop over directions.
        for (unsigned k = 0; k < dim; ++k)
        {
            // Loop over the blocks.
            for (unsigned blk = 0; blk < this->m_blockOp.size(); ++blk)
            {
                v_CopyDirection(in.GetBlocks()[blk],
                                m_activeDirTrace->GetBlocks()[blk], k, dim,
                                blk);

                this->m_blockOp[blk]->SetActiveDir(static_cast<int>(k));
                this->m_blockOp[blk]->SetAppend(m_append || k > 0);
                this->m_blockOp[blk]->Apply(m_activeDirTrace->GetBlocks()[blk],
                                            out.GetBlocks()[blk]);
            }
        }

        // Back to the scalar contract and the caller's append flag for
        // any later Apply().
        for (unsigned blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetActiveDir(-1);
            this->m_blockOp[blk]->SetAppend(m_append);
        }
    }

    /**
     * @brief ApplyVector() taking one field per direction.
     *
     * The same lift with @p in holding dim separate ncomp-component fields,
     * in[k] carrying g_k - the natural product of a trace-flux
     * implementation's per-direction scatters (see
     * SolverCore::TraceFluxOp::SetSymmTensorOutput). No staging copies: each
     * field is already the contiguous slice the per-direction pass wants.
     *
     * @param   in      One packed trace field per direction, each of
     *                  ncomp components.
     * @param   out     Volume-shaped output field of ncomp components,
     *                  zeroed and written or accumulated onto according
     *                  to SetAppend().
     */
    void ApplyVector(
        std::vector<LibUtilities::Field<TData, FieldState::Phys>> &in,
        LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        const unsigned dim = this->m_expansionList->GetShapeDimension();

        ASSERTL1(in.size() == dim,
                 "ApplyVector: one input field per direction");

        // Loop over directions.
        for (unsigned k = 0; k < dim; ++k)
        {
            ASSERTL1(in[k].GetNumComponents() == out.GetNumComponents(),
                     "ApplyVector: component count mismatch");

            // Loop over the blocks.
            for (unsigned blk = 0; blk < this->m_blockOp.size(); ++blk)
            {
                this->m_blockOp[blk]->SetActiveDir(static_cast<int>(k));
                this->m_blockOp[blk]->SetAppend(m_append || k > 0);
                this->m_blockOp[blk]->Apply(in[k].GetBlocks()[blk],
                                            out.GetBlocks()[blk]);
            }
        }

        // Back to the scalar contract and the caller's append flag for
        // any later Apply().
        for (unsigned blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetActiveDir(-1);
            this->m_blockOp[blk]->SetAppend(m_append);
        }
    }

protected:
    /// One block operator per element block of the expansion list;
    /// populated by ElmtOp::Create().
    std::vector<std::shared_ptr<IProductWRTPhysNormalDerivTraceBlockOp<TData>>>
        m_blockOp;

    /// Trace data of the direction ApplyVector() has active on the block
    /// operators, its slices of the input gathered into ncomp components.
    std::unique_ptr<LibUtilities::Field<TData, FieldState::Phys>>
        m_activeDirTrace;

    /// The caller's append flag, as last set by SetAppend().
    /// ApplyVector() overrides the block flags direction by direction, so
    /// it keeps this copy to reset them to when it is done. Default false.
    bool m_append = false;

    /**
     * @brief Copy direction @p dir of one block of the input into the
     * matching block of #m_activeDirTrace: component n of @p outblock
     * takes component n * @p dim + @p dir of @p inblock.
     *
     * A component slice is CompSize values whatever the interleave, so the
     * copy is layout-blind; @p outblock is then given @p inblock's
     * interleave. The implementation issues the copy in the memory space
     * its execution space names, on the stream block @p blk applies on, so
     * device data is never brought back to the host and the copy is
     * ordered ahead of the apply that reads it.
     *
     * @param   inblock     Direction-minor input, dim * ncomp components.
     * @param   outblock    Destination block, ncomp components.
     * @param   dir         Direction whose slices are copied.
     * @param   dim         Shape dimension, the stride between the slices.
     * @param   blk         Index of the block being copied.
     */
    virtual void v_CopyDirection(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        const unsigned int dir, const unsigned int dim,
        const unsigned int blk) = 0;

    /**
     * @brief Construct the interface part of a concrete implementation;
     * called by the factory-registered creator functions.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Component names the operator is set up
     *                          for.
     */
    IProductWRTPhysNormalDerivTraceOp(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : ElmtOp<FieldState::Phys, FieldState::Phys, TData>(expansionList,
                                                            components)
    {
    }

    ~IProductWRTPhysNormalDerivTraceOp() override = default;

    /**
     * @brief Lift the whole element boundary, block by block.
     *
     * Checks that the fields agree in component count and in number of
     * homogeneous modes -- not that their block layouts are the trace
     * and volume ones described in the class notes -- then hands each
     * pair of blocks to the corresponding entry of #m_blockOp.
     *
     * @param   in      Packed trace input field.
     * @param   out     Volume-shaped output field, zeroed and written or
     *                  accumulated onto according to SetAppend().
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
