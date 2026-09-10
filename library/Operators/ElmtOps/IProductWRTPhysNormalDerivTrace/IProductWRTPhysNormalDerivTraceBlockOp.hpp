///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTPhysNormalDerivTraceBlockOp.hpp
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
// Description: Per-block interface of the lift against the normal derivative
// of the test function
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTPhysNormalDerivTraceBlockOp.hpp
 * @brief Per-block interface class of the lift against the normal
 * derivative of the test function, IProductWRTPhysNormalDerivTrace.
 */

#pragma once

#include <vector>

#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include "Operators/ElmtOps/ElmtBlockOp.hpp"

namespace Nektar::Operators::detail
{

/**
 * @brief Directions in which a shape is collapsed: those whose chain rule
 * from the element local coordinate to the collapsed one carries a
 * \f$2/(1 - \eta)\f$ factor. Direction 0 is never collapsed.
 *
 * The collapsed directions always use a Gauss-Radau distribution that
 * excludes \f$\eta = +1\f$, so \f$2/(1 - \eta)\f$ is finite at the
 * element's own points. It is not finite at the trace's points, which is
 * why the factor is applied to the volume result rather than folded into
 * the trace data.
 */
constexpr bool IsCollapsedDir(LibUtilities::ShapeType shape, unsigned dir)
{
    switch (shape)
    {
        case LibUtilities::Tri:
        case LibUtilities::NodalTri:
            return dir == 1;
        case LibUtilities::Tet:
        case LibUtilities::NodalTet:
            return dir == 1 || dir == 2;
        case LibUtilities::Prism:
        case LibUtilities::NodalPrism:
        case LibUtilities::Pyr:
            return dir == 2;
        default:
            return false;
    }
}

/**
 * @brief Whether component @p d of the trace normal derivative factors,
 * which multiplies \f$\partial/\partial\eta_d\f$, needs the
 * \f$2/(1 - \eta)\f$ factor of collapsed direction @p dir.
 *
 * Reading off the chain rules, with \f$f_d\f$ the Cartesian to element
 * local factors:
 *
 *   Tri    d0: (f0 + (1+eta0)/2 f1) 2/(1-eta1)
 *          d1:  f1
 *   Prism  d0: (f0 + (1+eta0)/2 f2) 2/(1-eta2)
 *   Pyr    d0: (f0 + (1+eta0)/2 f2) 2/(1-eta2)
 *          d1: (f1 + (1+eta1)/2 f2) 2/(1-eta2)
 *   Tet    d0: (f0 + (1+eta0)/2 (f1+f2)) 2/(1-eta1) 2/(1-eta2)
 *          d1: (f1 + (1+eta1)/2 f2) 2/(1-eta2)
 *
 * The bracketed part is regular and stays in the trace data; this returns
 * which of the singular multipliers the component still needs.
 */
constexpr bool CompNeedsCollapsedFac(LibUtilities::ShapeType shape, unsigned d,
                                     unsigned dir)
{
    if (!IsCollapsedDir(shape, dir))
    {
        return false;
    }

    switch (shape)
    {
        case LibUtilities::Tri:
        case LibUtilities::NodalTri:
            return d == 0; // dir == 1
        case LibUtilities::Prism:
        case LibUtilities::NodalPrism:
            return d == 0; // dir == 2
        case LibUtilities::Pyr:
            return d == 0 || d == 1; // dir == 2
        case LibUtilities::Tet:
        case LibUtilities::NodalTet:
            return (dir == 1) ? (d == 0) : (d == 0 || d == 1);
        default:
            return false;
    }
}

} // namespace Nektar::Operators::detail

namespace Nektar::Operators
{

/**
 * @brief Per-block interface of the normal-derivative trace lift: lifts
 * the traces of every element of one block against the normal derivative
 * of the volume cardinal functions into that block's volume physical-space
 * storage.
 *
 * @details
 * Fixes the BlockAccessor states accepted by Apply() -- physical space in,
 * physical space out -- with the input block holding the element's packed
 * trace values and the output block its volume quadrature points. See
 * IProductWRTPhysNormalDerivTraceOp for both layouts and for the
 * mathematics.
 *
 * On top of ElmtBlockOp this class adds:
 * - #m_append, the accumulate flag, with SetAppend();
 * - #m_isCollocated, the tangential collocation flags the implementations
 *   fill, with SetIsCollocated() to overwrite them;
 * - SetActiveDir(), which selects the derivative-factor set the next
 *   Apply() consumes, for the vector-input lift.
 *
 * The concrete implementations are the
 * detail::IProductWRTPhysNormalDerivTraceBlockOpImpl definitions in
 * IProductWRTPhysNormalDerivTraceSerialAVXGeneric.hpp (Serial and AVX) and
 * IProductWRTPhysNormalDerivTraceDeviceGeneric.hpp (Device). Their
 * per-shape entry points, the ShapeBlock() specialisations, are generated
 * from LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in,
 * one translation unit per shape, execution space and data type.
 *
 * There is a single implementation, registered under Operators::Generic,
 * as for IProductWRTPhysTrace: a caller asking for StdMat, SumFac or
 * SumFacTOP is served that registration by ElmtBlockOp::Create(), which
 * falls back to the `"Generic"` factory key whenever the requested key is
 * absent.
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see ElmtBlockOp for the Apply()/v_Apply() contract;
 * IProductWRTPhysNormalDerivTraceOp for the whole-field interface driving
 * this class; IProductWRTPhysTraceBlockOp for the plain trace lift this
 * operator is built on.
 */
template <typename TData>
class IProductWRTPhysNormalDerivTraceBlockOp
    : public ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>
{
public:
    /**
     * @brief Create one block operator through the block-operator
     * factory, under the key
     * `"BlockIProductWRTPhysNormalDerivTrace" + execStr + implStr`.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     * @param   execStr         Execution-space part of the factory key.
     * @param   implStr         Implementation part of the factory key.
     *
     * @return The newly created block operator.
     */
    static std::shared_ptr<IProductWRTPhysNormalDerivTraceBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        std::string execStr, std::string implStr)
    {
        return ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>::
            template Create<IProductWRTPhysNormalDerivTraceBlockOp>(
                block_idx, exp, dataWarehouse, execStr, implStr);
    }

    /// Block-operator base name; Create() appends the execution space
    /// and implementation to it to form the factory key.
    static inline const std::string name =
        "BlockIProductWRTPhysNormalDerivTrace";

    /**
     * @brief Set every one of this block's tangential collocation flags,
     * so that passing false runs the general contraction where the fast
     * path would otherwise be taken.
     *
     * The setter is wholesale: it overwrites every flag, discarding what
     * the constructor worked out from the point distributions, so passing
     * true does not restore the flags of a block whose points do not in
     * fact coincide.
     *
     * @param   isCollocated    Value written to every flag.
     */
    void SetIsCollocated(const bool isCollocated)
    {
        for (unsigned i = 0; i < m_isCollocated.size(); ++i)
        {
            m_isCollocated[i] = isCollocated;
        }
    }

    /**
     * @brief Set the accumulate flag.
     *
     * @param   append  New value of #m_append.
     */
    void SetAppend(bool append)
    {
        m_append = append;
    }

    /**
     * @brief Select which derivative-factor set the next Apply() uses.
     *
     * -1 (the default) is the scalar form, contracted with the trace
     * normal; 0 to dim-1 lifts the input against
     * \f$\partial\phi/\partial x_{dir}\f$ alone, the per-direction pass of
     * the vector-input variant. See
     * IProductWRTPhysNormalDerivTraceOp::ApplyVector().
     *
     * @param   dir     Cartesian direction, or -1 for the contracted set.
     */
    void SetActiveDir(const int dir)
    {
        v_SetActiveDir(dir);
    }

protected:
    /// Whether Apply() accumulates onto the output block rather than
    /// zeroing it and writing. Default false.
    bool m_append = false;

    /// One flag per (normal direction, tangential direction), filled by the
    /// implementation's constructor: the trace points of that direction
    /// coincide with the volume points of the direction they run along, so
    /// that the tangential contraction reduces to a pointwise multiply by
    /// the trace weights and the factors. Six entries in three dimensions,
    /// two in two, one in one. A slot carrying the derivative basis is never
    /// treated as collocated, whatever its flag. Cleared wholesale by
    /// SetIsCollocated().
    std::vector<bool> m_isCollocated;

    /**
     * @brief Construct the interface part of a concrete block operator;
     * called by the factory-registered creator functions.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    IProductWRTPhysNormalDerivTraceBlockOp(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>(block_idx, exp,
                                                                 dataWarehouse)
    {
    }

    ~IProductWRTPhysNormalDerivTraceBlockOp() override = default;

    /// @brief Hook behind SetActiveDir(); both implementations swap the
    /// factor pointer the kernels read.
    virtual void v_SetActiveDir(const int dir) = 0;
};

} // namespace Nektar::Operators
