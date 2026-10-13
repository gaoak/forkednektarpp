///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionOp.hpp
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
// Description: Strong-form (non-conservative) advection: advVel \dot \grad(u)
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvectionOp.hpp
 * @brief Whole-field interface of the advection element operator, which
 * evaluates the convective term V . grad(u) at the quadrature points.
 *
 * @details
 * Advection is a physical-space to physical-space element operator: for
 * each component u of the input field it computes scale * (V . grad u) at
 * every quadrature point, where V is a user-supplied advection velocity
 * field with one component per coordinate dimension (three on a 3DH1
 * expansion). This is the non-conservative (convective) form of the
 * advection term.
 *
 * AdvectionOp is the whole-field interface class of the family (see
 * ElmtOp.hpp for the two-level element-operator structure); the per-block
 * interface is AdvectionBlockOp (AdvectionBlockOp.hpp). The plane-local
 * numerical work lives in the detail::AdvectionBlockOpImpl definitions --
 * AdvectionBlockOpSerialAVXStdMat.hpp and AdvectionBlockOpSerialAVXSumFac.hpp
 * for the Serial and AVX execution spaces, AdvectionBlockOpDeviceStdMat.hpp and
 * AdvectionBlockOpDeviceSumFac.hpp for the Device execution space -- registered
 * with the block-operator factory by CMake-generated translation units. On
 * a multi-plane 3DH1 expansion the homogeneous term w du/dz is added by
 * the z-operator detail::AdvectionZOpImpl (AdvectionZOpImpl.hpp), owned by
 * detail::AdvectionOpImpl (AdvectionOpImpl.hpp).
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtOp.hpp>

#include <MultiRegions/ElmtOps/Advection/AdvectionBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Whole-field interface of the advection element operator: writes
 * scale * (V . grad u) at every quadrature point of every component u of
 * the input field.
 *
 * @details
 * The operator maps physical space to physical space. Before the first
 * Apply() the advection velocity V must be supplied through SetAdvVel();
 * Apply() then evaluates, element by element,
 * \f$ s \sum_k V_k \, \partial u / \partial x_k \f$ at each quadrature
 * point -- overwriting the output, or accumulating onto it when
 * SetAppend(true) is set -- with s the factor given by SetScale()
 * (default 1). The same velocity is applied to every component of the
 * input field. On a multi-plane 3DH1 expansion the sum includes the
 * homogeneous term w du/dz, added after the block loop through the
 * v_SetScaleFFT(), v_SetAdvVelFFT() and v_ApplyFFT() hooks.
 *
 * As with the other element-operator families, this class sets the
 * Apply() field states at compile time and drives one AdvectionBlockOp per
 * element block; the per-execution-space behaviour lives in the block
 * operators and, for the homogeneous term, in the z-operator owned by
 * detail::AdvectionOpImpl, which implements the hooks. See ElmtOp for the
 * block decomposition and factory scheme.
 *
 * @tparam TData Floating-point type of the field data.
 */
template <typename TData>
class AdvectionOp : public ElmtOp<FieldState::Phys, FieldState::Phys, TData>
{
    friend class ElmtOp<FieldState::Phys, FieldState::Phys, TData>;

public:
    /**
     * @brief Build a complete advection operator through the operator and
     * block-operator factories.
     *
     * See ElmtOp::Create for the two-stage construction and the
     * execution-space/implementation lookup rules.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Names of the field components.
     * @param   execStr         Execution space ("Serial", "AVX" or
     *                          "Device"); the session's setting is used
     *                          if empty.
     * @param   implStr         Implementation ("StdMat", "SumFac" or, on
     *                          Device, "SumFacTOP"); the session's
     *                          setting is used if empty.
     *
     * @return The fully assembled operator. SetAdvVel() must still be
     * called before the first Apply().
     */
    static std::shared_ptr<AdvectionOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Phys, FieldState::Phys, TData>::
            template Create<AdvectionOp, AdvectionBlockOp>(
                expansionList, components, execStr, implStr);
    }

    /// Operator base name; Create() appends the execution space to it
    /// to form the operator-factory key.
    static inline const std::string name = "Advection";

    /**
     * @brief Set the scalar factor applied to the advection term on every
     * block and on the homogeneous term, so that Apply() produces
     * scale * (V . grad u).
     *
     * @param   scale   New scale factor (initially 1).
     */
    void SetScale(const TData &scale)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetScale(scale);
        }
        v_SetScaleFFT(scale);
    }

    /**
     * @brief Choose between overwriting and accumulating output.
     *
     * With append set, Apply() adds scale * (V . grad u) onto the existing
     * contents of the output field instead of overwriting it -- useful
     * when the advection term is one of several contributions to a
     * right-hand side.
     *
     * @param   append  True to accumulate, false (the initial state) to
     *                  overwrite.
     */
    void SetAppend(const bool &append)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetAppend(append);
        }
    }

    /**
     * @brief Supply the advection velocity field V; must be called before
     * the first Apply().
     *
     * @p advVel must be built on the operator's expansion list and hold
     * one component per coordinate dimension of the mesh, or three on a
     * 3DH1 expansion, each a physical-space scalar field; the component
     * count is checked with a level-1 assertion. Each block accessor is
     * handed to the corresponding block operator, which keeps a
     * non-owning pointer to it, and the whole field to v_SetAdvVelFFT(),
     * so the field must outlive the operator's use. The block
     * implementations reshape the velocity storage in place to their own
     * interleave width the first time Apply() finds it at another width,
     * updating the accessor's recorded width: the field's interleave
     * width is modified as a side effect.
     *
     * @param   advVel  Advection velocity field, one component per
     *                  coordinate dimension (three on 3DH1).
     */
    void SetAdvVel(LibUtilities::Field<TData, FieldState::Phys> &advVel)
    {
        // A 3DH1 expansion carries a third velocity component along the
        // homogeneous direction, which the planes themselves do not count.
        ASSERTL1(advVel.GetNumComponents() ==
                     ((advVel.GetNumHomoModes() == 1)
                          ? static_cast<unsigned int>(
                                this->m_expansionList->GetCoordim(0))
                          : 3u),
                 "Advection velocity must have coordDim components, or three "
                 "on a 3DH1 expansion");

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetAdvVel(advVel.GetBlocks()[blk]);
        }
        v_SetAdvVelFFT(advVel);
        m_isSetAdvVel = true;
    }

protected:
    /// Whether SetAdvVel() has been called; checked by v_Apply().
    bool m_isSetAdvVel = false;
    /// One block operator per element block, filled by Create().
    std::vector<std::shared_ptr<AdvectionBlockOp<TData>>> m_blockOp;

    /**
     * @brief Construct the interface part of a concrete operator; called
     * by the factory-registered creator functions.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Component names the operator is set up
     *                          for.
     */
    AdvectionOp(const MultiRegions::ExpListSharedPtr &expansionList,
                const std::vector<std::string> &components)
        : ElmtOp<FieldState::Phys, FieldState::Phys, TData>(expansionList,
                                                            components)
    {
    }

    ~AdvectionOp() override = default;

    /**
     * @brief Apply the advection operator block by block.
     *
     * Checks that the fields conform (component and homogeneous-mode
     * counts) and that the advection velocity has been set, then hands
     * each input/output block pair to the corresponding entry of
     * #m_blockOp, and finally calls v_ApplyFFT() to add the homogeneous
     * term.
     *
     * @param   in      Physical-space input field.
     * @param   out     Physical-space output field; overwritten, or
     *                  accumulated onto when append is set.
     */
    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &out) override
    {
        ASSERTL1(in.GetNumComponents() == out.GetNumComponents(),
                 "Number of input and output components differ");

        ASSERTL1(in.GetNumHomoModes() == out.GetNumHomoModes(),
                 "Number of input and output homogeneous modes differ");

        ASSERTL1(m_isSetAdvVel,
                 "Advection velocity has not been set."
                 "Set the value with SetAdvVel() before calling Apply().");

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

    /// Pass the scale factor on to the homogeneous term; called by
    /// SetScale().
    virtual void v_SetScaleFFT(const TData &scale) = 0;

    /// Pass the advection velocity on to the homogeneous term; called by
    /// SetAdvVel().
    virtual void v_SetAdvVelFFT(
        LibUtilities::Field<TData, FieldState::Phys> &advVel) = 0;

    /// Add the homogeneous term to @p out; called by v_Apply() after the
    /// block loop.
    virtual void v_ApplyFFT(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &out) = 0;
};

} // namespace Nektar::MultiRegions
