///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTPhysTraceOp.hpp
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
 * @file IProductWRTPhysTraceOp.hpp
 * @brief Public interface of the IProductWRTPhysTrace element operator,
 * the discontinuous Galerkin lifting term, which maps a field living on
 * an element's traces into an array shaped like the element's volume
 * physical space.
 *
 * @details
 * Given a field already evaluated at the quadrature points of an
 * element's traces, typically a numerical flux \f$\hat f\f$, this
 * operator accumulates
 * \f[ \Lambda_{\boldsymbol p} \mathrel{+}=
 *     \sum_{F \in \partial E} \int_F \hat f \;
 *     h_{\boldsymbol p}\big|_F \; J_F \, \mathrm{d}s , \f]
 * where \f$h_{\boldsymbol p}(\boldsymbol\xi) = \prod_a h_{p_a}(\xi_a)\f$
 * is the tensor product of the cardinal (hat) functions of the
 * @em volume quadrature grid, \f$h_p(\xi_q) = \delta_{pq}\f$, \f$J_F\f$
 * is the trace Jacobian, and the traces \f$F\f$ of an element \f$E\f$
 * are its endpoint vertices in one dimension, its edges in two and its
 * faces in three.
 *
 * The entries of \f$\Lambda\f$ are therefore surface inner products of
 * the trace field against a single cardinal function of the volume grid.
 * They are @em not modal coefficients and not point values of any
 * function. The modal right-hand side is recovered downstream by a
 * transposed basis contraction carrying neither quadrature weights nor
 * volume Jacobian -- in the operator pipeline that is IProductWRTBase
 * created with `SetIntegration(false)`, i.e. \f$B^{\mathsf{T}}\f$ rather
 * than \f$B^{\mathsf{T}} W\f$ -- because the surface rule's weights and
 * the face Jacobian are already inside \f$\Lambda\f$.
 *
 * On a direction-\f$a\f$ trace at \f$\xi_a = \pm 1\f$ the tensor product
 * factorises as
 * \f[ h_{\boldsymbol p}\big|_{\xi_a = \pm 1} = h_{p_a}(\pm 1)
 *     \prod_{b \neq a} h_{p_b}(\xi_b) , \f]
 * the normal-direction value times the hat functions of the in-trace
 * directions. That is the structure the kernels exploit: a
 * sum-factorised inner product over the trace quadrature points followed
 * by a rank-1 lift along the normal direction.
 *
 * The family follows the standard element-operator layout (see
 * ElmtOp.hpp):
 * - IProductWRTPhysTraceOp (this file): whole-field interface fixing the
 *   Apply() field states, plus the per-trace entry point;
 * - detail::IProductWRTPhysTraceOpImpl
 *   (IProductWRTPhysTraceOpImpl.hpp): the factory-registered whole-field
 *   implementation, one entry per execution space;
 * - IProductWRTPhysTraceBlockOp (IProductWRTPhysTraceBlockOp.hpp):
 *   per-block interface, which also carries the append flag;
 * - detail::IProductWRTPhysTraceBlockOpImpl: the per-block
 *   implementations, in IProductWRTPhysTraceSerialAVXGeneric.hpp (serving
 *   both the Serial and the AVX space) and
 *   IProductWRTPhysTraceDeviceGeneric.hpp (Device);
 * - the kernels in IProductWRTPhysTraceSerialAVXGenericKernels.hpp and
 *   IProductWRTPhysTraceDeviceGenericKernels.hpp.
 *
 * The per-shape entry points, the ShapeBlock() specialisations, come from the
 * generated translation-unit template
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in, so the
 * size-templated OperatorND() instantiations of the SumFac families are kept
 * even though this operator registers a single implementation. They expand the
 * switch template
 * LibUtilities/BasicUtils/Switch/BlockOpSwitchPhysTraceExtract.h.in, whose
 * per-shape template arguments carry trace point counts per normal direction.
 *
 * @note There is one trace inner-product algorithm and one
 * registration, under Operators::Generic. A StdMat, SumFac or SumFacTOP
 * request does not fail the factory lookup and does not need a shim
 * header per implementation: ElmtBlockOp::Create() falls back to the
 * `"Generic"` key whenever the requested one is absent, so a
 * whole-field selection of any implementation can include this
 * operator. SumFacTOP is the case that matters numerically -- the block
 * class instantiated with that tag would have run kernels indexing at
 * the warp size against a width-one interleave -- and there is now no
 * such instantiation to make.
 *
 * @warning A nodal expansion gets a result in the modal space. The
 * operator integrates against the volume basis and stops, whereas a
 * nodal expansion's coefficients are the ones
 * StdNodalTriExp::v_IProductWRTBase reaches by following that integral
 * with NodalToModalTranspose. That transform is applied nowhere here, so
 * every coefficient a NodalTri, NodalTet or NodalPrism produces is
 * wrong. The nodal meshes under UnitTests/Operators/run are registered
 * as fixtures but no test case uses them for this operator, the four
 * that would be written failing on this.
 *
 * @see PhysTraceExtractOp.hpp for the adjoint operator, which
 * interpolates a volume field onto the trace points and applies the same
 * interpolation tables untransposed. It maps physical values to physical
 * values and so is not affected by the nodal coefficient space.
 */

#pragma once

#include "Operators/ElmtOps/ElmtOp.hpp"

#include "Operators/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceBlockOp.hpp"

namespace Nektar::Operators
{

// IProductWRTPhysTrace base class
// Defines the apply operator to enforce apply parameter types
/**
 * @brief Whole-field interface of the DG lifting term: lifts trace data
 * into a volume physical-space shaped array for every element.
 *
 * @details
 * As with the other element-operator families, this class exists to fix
 * the Apply() parameter types at compile time and to drive the per-block
 * operators: it holds one IProductWRTPhysTraceBlockOp per element block,
 * and v_Apply() hands each block of the input/output fields to the
 * matching entry. All numerical work happens in the block operators.
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
 * The conformance checks in v_Apply() and IProductWRTPhysTrace()
 * therefore compare only component counts and homogeneous-mode counts.
 * Passing the two layouts the wrong way round is not caught here.
 *
 * ### Entry points
 * - Apply() (inherited from ElmtOp) lifts the whole element boundary in
 *   one call, every trace of the element contributing to the volume
 *   array. Whether that array is zeroed first or accumulated onto is
 *   selected by SetAppend(), on the paths that honour it (see
 *   IProductWRTPhysTraceBlockOp::m_append).
 * - IProductWRTPhysTrace() lifts one trace of every element, zeroing
 *   the volume array or accumulating onto it according to SetAppend()
 *   just as Apply() does. Every trace contributes to the same array, so
 *   a caller assembling the whole boundary clears on the first trace
 *   and appends on the rest.
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see IProductWRTPhysTraceBlockOp for the per-block interface and the
 * implementations available per execution space.
 */
template <typename TData>
class IProductWRTPhysTraceOp
    : public ElmtOp<FieldState::Phys, FieldState::Phys, TData>
{
    friend class ElmtOp<FieldState::Phys, FieldState::Phys, TData>;

public:
    /**
     * @brief Create a trace inner-product operator through the operator
     * and block-operator factories.
     *
     * Builds the interface object under the factory key
     * `"IProductWRTPhysTrace" + execStr` and one block operator per
     * element block under
     * `"BlockIProductWRTPhysTrace" + execStr + implStr` (see
     * ElmtOp::Create for the two-stage construction and for the session
     * defaults used when @p execStr / @p implStr are empty).
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Names of the field components the
     *                          operator is set up for.
     * @param   execStr         Execution space ("Serial", "AVX" or
     *                          "Device"); session default if empty.
     * @param   implStr         Block implementation; every value
     *                          resolves to the single "Generic"
     *                          registration, see the file notes.
     *
     * @return The fully assembled operator, ready to Apply.
     */
    static std::shared_ptr<IProductWRTPhysTraceOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Phys, FieldState::Phys,
                      TData>::template Create<IProductWRTPhysTraceOp,
                                              IProductWRTPhysTraceBlockOp>(
            expansionList, components, execStr, implStr);
    }

    /// Operator base name; Create() appends the execution space to it
    /// to form the factory key.
    static inline const std::string name = "IProductWRTPhysTrace";

    /**
     * @brief Lift a single trace of every element, accumulating into
     * @p out.
     *
     * @details
     * The per-trace counterpart of Apply(): rather than walking the
     * whole element boundary it processes trace @p traceid alone,
     * reading that trace's slice of the packed input. Whether @p out is
     * zeroed first or accumulated onto follows SetAppend(), as it does
     * for Apply().
     *
     * Every trace of an element contributes to the same volume-shaped
     * @p out, so clearing it is only meaningful on the first trace of a
     * sequence. A caller assembling the whole boundary trace by trace
     * therefore issues one call per trace, advancing @p inOffset in the
     * packed order of the traces rather than in trace-id order, and
     * either calls SetAppend(false) before the first of them and
     * SetAppend(true) before the rest, or clears @p out itself and
     * leaves SetAppend(true) throughout.
     *
     * @param   traceid     Local trace index within the element's shape,
     *                      in the Nektar edge/face numbering; the
     *                      implementations map it to a (normal
     *                      direction, position) pair and, for regular
     *                      geometry, to a Jacobian slot.
     * @param   in          Packed trace field; see the class notes for
     *                      its layout.
     * @param   inOffset    Offset of this trace's data within one
     *                      element's packed trace entries, counted in
     *                      entries per element. The implementations
     *                      scale it by the interleave width. For regular
     *                      geometry the Jacobian slot is not derived
     *                      from it but looked up per trace id.
     * @param   out         Volume-shaped output field, accumulated into.
     *
     * @note Not available for segments: both implementations raise a
     * fatal error for a segment block, the per-trace segment path
     * being unimplemented.
     */
    void IProductWRTPhysTrace(const unsigned traceid,
                              LibUtilities::Field<TData, FieldState::Phys> &in,
                              const unsigned inOffset,
                              LibUtilities::Field<TData, FieldState::Phys> &out)

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

            this->m_blockOp[blk]->IProductWRTPhysTrace(traceid, inblock,
                                                       inOffset, outblock);
        }
    }

    /**
     * @brief Set the tangential collocation flags of every block
     * operator, passing false to force the fast path off.
     *
     * @details
     * The implementations record one collocation flag per (normal
     * direction, tangential direction) at setup, set when the trace
     * quadrature points of that direction coincide with the volume
     * points of the corresponding direction. Where the flag is set the
     * tangential contraction collapses to a pointwise multiply by the
     * trace weights and the Jacobian. Clearing the flags selects the
     * general contraction instead, which is how the unit tests reach
     * that code path on meshes whose points do coincide; it is a
     * different route to the same mathematical operation.
     *
     * @param   isCollocated    Value written to every flag of every
     *                          block.
     *
     * @note The flags live on IProductWRTPhysTraceBlockOp itself, so this
     * reaches every implementation without a virtual hook. The setter
     * is wholesale, so passing true does not restore the flags a block
     * started with; it asserts collocation the constructor may not have
     * found.
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
     * @brief Set the append flag of every block operator.
     *
     * When set, the output field is accumulated onto instead of being
     * zeroed first. Both entry points consult the flag: the bulk
     * Apply() path and the per-trace IProductWRTPhysTrace() alike.
     *
     * @param   append  New value of the flag.
     */
    void SetAppend(bool append)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetAppend(append);
        }
    }

protected:
    /// One block operator per element block of the expansion list;
    /// populated by ElmtOp::Create().
    std::vector<std::shared_ptr<IProductWRTPhysTraceBlockOp<TData>>> m_blockOp;

    /**
     * @brief Construct the interface part of a concrete implementation;
     * called by the factory-registered creator functions.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Component names the operator is set up
     *                          for.
     */
    IProductWRTPhysTraceOp(const MultiRegions::ExpListSharedPtr &expansionList,
                           const std::vector<std::string> &components)
        : ElmtOp<FieldState::Phys, FieldState::Phys, TData>(expansionList,
                                                            components)
    {
    }

    ~IProductWRTPhysTraceOp() override = default;

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
     *                  accumulated onto according to SetAppend(), where
     *                  that path honours it (see
     *                  IProductWRTPhysTraceBlockOp::m_append).
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
