///////////////////////////////////////////////////////////////////////////////
//
// File: PhysTraceExtractOp.hpp
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
 * @file PhysTraceExtractOp.hpp
 * @brief Public interface of the PhysTraceExtract element operator, which
 * interpolates a volume physical-space field onto the quadrature points
 * of the element's traces.
 *
 * @details
 * Given a field at the volume quadrature points of an element, this
 * operator evaluates it at the trace quadrature points,
 * \f[ u(\boldsymbol\xi^{tr}) = \sum_{\boldsymbol p} u_{\boldsymbol p}\,
 *     h_{\boldsymbol p}(\boldsymbol\xi^{tr}) , \f]
 * where \f$h_{\boldsymbol p}(\boldsymbol\xi) = \prod_a h_{p_a}(\xi_a)\f$
 * is the tensor product of the cardinal (hat) functions of the @em volume
 * quadrature grid, \f$h_p(\xi_q) = \delta_{pq}\f$, the \f$u_{\boldsymbol
 * p}\f$ are the field values at the volume quadrature points, and the
 * traces of an element are its endpoint vertices in one dimension, its
 * edges in two and its faces in three.
 *
 * On a direction-\f$a\f$ trace at \f$\xi_a = \pm 1\f$ the tensor product
 * factorises as
 * \f[ h_{\boldsymbol p}\big|_{\xi_a = \pm 1} = h_{p_a}(\pm 1)
 *     \prod_{b \neq a} h_{p_b}(\xi_b) , \f]
 * so the evaluation splits into a normal-direction interpolation with
 * \f$h_{p_a}(\pm 1)\f$, which collapses the volume array onto the trace's
 * plane, followed by a tangential interpolation of that plane onto the
 * trace quadrature points with \f$h_{p_b}(\xi_b^{tr})\f$. That is the
 * structure the kernels exploit.
 *
 * Both stages use `eInterp` tables taken from the data warehouse at
 * setup: the normal-direction table \f$h_p(\pm 1)\f$ (called `ntbasis` in
 * this operator's kernels, `nbasis` in IProductWRTPhysTrace's) and the
 * trace table \f$h_p(\xi^{tr})\f$ (`tbasis`). Neither is an expansion
 * basis evaluated anywhere, and `nm` throughout this family holds
 * `GetNumPoints`, that is volume quadrature counts and not modes. The
 * operator is a pure interpolation: no quadrature weights and no
 * Jacobian enter it.
 *
 * It is the adjoint of IProductWRTPhysTrace in the hat basis. Both build
 * their `eInterp` BasisDataKeys from the same basis key, point count and
 * points type, the Serial/AVX adjoint storing its copy broadcast across
 * a SIMD vector. This operator applies the tables untransposed where
 * IProductWRTPhysTrace applies them transposed and additionally weights
 * them with the trace quadrature weights and the trace Jacobian.
 *
 * The operator only ever interpolates point values. Nothing in this
 * repository calls it outside its unit tests.
 *
 * The family follows the standard element-operator layout (see
 * ElmtOp.hpp):
 * - PhysTraceExtractOp (this file): whole-field interface fixing the
 *   Apply() field states, plus the per-trace entry point;
 * - detail::PhysTraceExtractOpImpl (PhysTraceExtractOpImpl.hpp): the
 *   factory-registered whole-field implementation, one entry per
 *   execution space;
 * - PhysTraceExtractBlockOp (PhysTraceExtractBlockOp.hpp): per-block
 *   interface;
 * - detail::PhysTraceExtractBlockOpImpl: the per-block implementations,
 *   in PhysTraceExtractSerialAVXGeneric.hpp (serving both the Serial and
 *   the AVX space) and PhysTraceExtractDeviceGeneric.hpp (Device);
 * - the kernels in PhysTraceExtractSerialAVXGenericKernels.hpp and
 *   PhysTraceExtractDeviceGenericKernels.hpp.
 *
 * The per-shape entry points are generated from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in, one translation unit
 * per shape, execution space and data type, so the size-templated OperatorND
 * instantiations of the SumFac families are kept even though this operator
 * registers a single implementation. Those units include the switch template
 * LibUtilities/BasicUtils/Switch/BlockOpSwitchPhysTraceExtract.h.in, which is
 * named after this operator: its per-shape template arguments carry trace point
 * counts per normal direction.
 *
 * @note There is one trace-extraction algorithm and one registration,
 * under Operators::Generic. A StdMat, SumFac or SumFacTOP request does
 * not fail the factory lookup and does not need a shim header per
 * implementation: ElmtBlockOp::Create() falls back to the `"Generic"`
 * key whenever the requested one is absent, so a whole-field selection
 * of any implementation can include this operator. SumFacTOP is the
 * case that matters numerically -- the block class instantiated with
 * that tag would have run kernels indexing at the warp size against a
 * width-one interleave -- and there is now no such instantiation to
 * make.
 *
 * @warning For three-dimensional shapes the general normal-direction
 * interpolation, the arm taken when the volume rule of the normal
 * direction contains no domain endpoints (Gauss-type rules), produces
 * wrong values on the bulk path. Its location has not been established:
 * the `ntbasis` row strides the per-trace path dispatches are correct,
 * that path taking its stride as an explicit `tstride` argument rather
 * than from the face-loop bound. No three-dimensional Gauss fixture
 * exists, so the unit-test suites stay green regardless.
 *
 * @see IProductWRTPhysTraceOp.hpp for the adjoint operator, the
 * discontinuous Galerkin lifting term, which maps trace data back into a
 * volume-shaped array.
 */

#pragma once

#include "Operators/ElmtOps/ElmtOp.hpp"

#include "Operators/ElmtOps/PhysTraceExtract/PhysTraceExtractBlockOp.hpp"

namespace Nektar::Operators
{

// PhysTraceExtract base class
// Defines the apply operator to enforce apply parameter types
/**
 * @brief Whole-field interface of the trace extraction: interpolates the
 * volume field of every element onto that element's trace quadrature
 * points.
 *
 * @details
 * As with the other element-operator families, this class exists to fix
 * the Apply() parameter types at compile time and to drive the per-block
 * operators: it holds one PhysTraceExtractBlockOp per element block, and
 * v_Apply() hands each block of the input/output fields to the matching
 * entry. All numerical work happens in the block operators.
 *
 * ### Field layouts
 * Both fields are in FieldState::Phys, but they are not the same shape
 * and the type system does not tell them apart:
 * - @em in is a volume field, laid out by GetBlockAttributes(), with one
 *   entry per volume quadrature point;
 * - @em out is a packed trace field, laid out by
 *   GetLocTraceBlockAttributes(). Within an element the traces come in
 *   pair order, the \f$N_0\f$ pair first, then the \f$N_1\f$ pair or
 *   single, then \f$N_2\f$, face-major within a pair.
 *
 * This is the mirror image of IProductWRTPhysTrace, which reads the
 * packed trace layout and writes the volume one. The conformance checks
 * in v_Apply() and ExtractTrace() compare only component counts and
 * homogeneous-mode counts, so passing the two layouts the wrong way round
 * is not caught here.
 *
 * ### Entry points
 * - Apply() (inherited from ElmtOp) extracts the whole element boundary
 *   in one call, filling every trace of the packed record.
 * - ExtractTrace() extracts one trace of every element into a
 *   caller-chosen offset of the packed record.
 *
 * This operator has no append flag: both paths write the entries they
 * are responsible for.
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see PhysTraceExtractBlockOp for the per-block interface and the
 * implementations available per execution space.
 */
template <typename TData>
class PhysTraceExtractOp
    : public ElmtOp<FieldState::Phys, FieldState::Phys, TData>
{
    friend class ElmtOp<FieldState::Phys, FieldState::Phys, TData>;

public:
    /**
     * @brief Create a trace extraction operator through the operator and
     * block-operator factories.
     *
     * Builds the interface object under the factory key
     * `"PhysTraceExtract" + execStr` and one block operator per element
     * block under `"BlockPhysTraceExtract" + execStr + implStr` (see
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
    static std::shared_ptr<PhysTraceExtractOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Phys, FieldState::Phys, TData>::
            template Create<PhysTraceExtractOp, PhysTraceExtractBlockOp>(
                expansionList, components, execStr, implStr);
    }

    /// Operator base name; Create() appends the execution space to it
    /// to form the factory key.
    static inline const std::string name = "PhysTraceExtract";

    /**
     * @brief Extract a single trace of every element into @p out at
     * @p outOffset.
     *
     * @details
     * The per-trace counterpart of Apply(): rather than filling the whole
     * packed record it interpolates onto trace @p traceid alone and
     * writes that trace's entries, leaving the rest of the record
     * untouched by this call. A caller assembling the whole boundary
     * therefore issues one call per trace, advancing @p outOffset in the
     * packed order of the traces rather than in trace-id order; that is
     * how the operator-level unit tests drive it, for instance
     * `3, 1, 0, 2` on a quadrilateral and `4, 2, 1, 3, 0, 5` on a
     * hexahedron.
     *
     * @param   traceid     Local trace index within the element's shape,
     *                      in the Nektar edge/face numbering; the
     *                      implementations map it to a (normal
     *                      direction, position) pair.
     * @param   in          Volume-shaped input field; see the class notes
     *                      for its layout.
     * @param   out         Packed trace field, written at @p outOffset.
     * @param   outOffset   Offset of this trace's data within one
     *                      element's packed trace entries, counted in
     *                      entries per element. Each implementation
     *                      addresses the output at that entry index in
     *                      its own interleave width: the device
     *                      launchers multiply by the warp size, the
     *                      Serial/AVX path indexes a simd_t array.
     *
     * @note Not available for every shape. On Serial/AVX a segment block
     * raises a fatal error, the per-trace segment path being
     * unimplemented there, while the Device path has a segment launcher.
     * Either implementation reports a shape outside its list on standard
     * output and leaves @p out unwritten.
     *
     * @note In three dimensions the general arm of the normal-direction
     * interpolation this reaches takes the `ntbasis` row stride as an
     * explicit `tstride` argument rather than from the face-loop bound,
     * which is one on this path; the defect the file notes describe is
     * on the bulk path.
     */
    void ExtractTrace(const unsigned traceid,
                      LibUtilities::Field<TData, FieldState::Phys> &in,
                      LibUtilities::Field<TData, FieldState::Phys> &out,
                      const unsigned outOffset)
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

            this->m_blockOp[blk]->ExtractTrace(traceid, inblock, outblock,
                                               outOffset);
        }
    }

    /**
     * @brief Set the tangential collocation flags of every block
     * operator, passing false to force the fast path off.
     *
     * @details
     * The implementations record one collocation flag per (normal
     * direction, tangential direction) at setup, six of them in three
     * dimensions, set when the trace quadrature points of that direction
     * coincide with the volume quadrature points of the corresponding
     * direction. Where the flag is set the tangential interpolation
     * collapses to a straight copy of the plane. Clearing the flags
     * selects the general contraction instead, which is how the unit
     * tests reach that code path on meshes whose points do coincide; it
     * is a different route to the same mathematical operation.
     *
     * @param   isCollocated    Value written to every flag of every
     *                          block.
     *
     * @note The flags live on PhysTraceExtractBlockOp itself, so this
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

protected:
    /// One block operator per element block of the expansion list;
    /// populated by ElmtOp::Create().
    std::vector<std::shared_ptr<PhysTraceExtractBlockOp<TData>>> m_blockOp;

    /**
     * @brief Construct the interface part of a concrete implementation;
     * called by the factory-registered creator functions.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Component names the operator is set up
     *                          for.
     */
    PhysTraceExtractOp(const MultiRegions::ExpListSharedPtr &expansionList,
                       const std::vector<std::string> &components)
        : ElmtOp<FieldState::Phys, FieldState::Phys, TData>(expansionList,
                                                            components)
    {
    }

    ~PhysTraceExtractOp() override = default;

    /**
     * @brief Extract every trace of every element, block by block.
     *
     * Checks that the fields agree in component count and in number of
     * homogeneous modes -- not that their block layouts are the volume
     * and trace ones described in the class notes -- then hands each pair
     * of blocks to the corresponding entry of #m_blockOp, which fills the
     * whole packed trace record of each element.
     *
     * @param   in      Volume-shaped input field.
     * @param   out     Packed trace output field.
     *
     * @note This is the path the nodal-shape guard note in the file
     * documentation applies to.
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
