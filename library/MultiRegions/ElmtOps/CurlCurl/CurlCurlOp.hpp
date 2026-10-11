///////////////////////////////////////////////////////////////////////////////
//
// File: CurlCurlOp.hpp
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
 * @file CurlCurlOp.hpp
 * @brief Whole-field interface of the curl-curl element operator, which
 * evaluates curl(curl u) at the quadrature points.
 *
 * @details
 * CurlCurl is a physical-space to physical-space element operator: it maps
 * a vector field given by its values at the quadrature points to the
 * values of curl(curl u) at the same points, element by element, by
 * differentiating the elemental interpolant twice. It is the strong,
 * collocation form of the operator; no integration against test functions
 * is involved.
 *
 * CurlCurlOp is the whole-field interface class of the family (see
 * ElmtOp.hpp for the two-level element-operator structure); the per-block
 * interface is CurlCurlBlockOp (CurlCurlBlockOp.hpp). The numerical work
 * lives in the detail::CurlCurlBlockOpImpl definitions --
 * CurlCurlSerialAVXStdMat.hpp and CurlCurlSerialAVXSumFac.hpp for the
 * Serial and AVX execution spaces, CurlCurlDeviceStdMat.hpp and
 * CurlCurlDeviceSumFac.hpp for the Device execution space -- registered
 * with the block-operator factory by CMake-generated translation units.
 * On a multi-plane 3DH1 expansion the block operators take only the plane
 * part of one curl, and detail::CurlCurlOpImpl (CurlCurlOpImpl.hpp)
 * completes the double curl through v_ApplyFFT().
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtOp.hpp>

#include <MultiRegions/ElmtOps/CurlCurl/CurlCurlBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Whole-field interface of the curl-curl element operator: replaces
 * the quadrature-point values of a vector field u by those of
 * curl(curl u), element by element.
 *
 * @details
 * Writing the input components as (u0, u1, u2) along the Cartesian
 * directions (x0, x1, x2):
 * - In three dimensions the vorticity w = curl u is formed first,
 *       w0 = d(u2)/dx1 - d(u1)/dx2,
 *       w1 = d(u0)/dx2 - d(u2)/dx0,
 *       w2 = d(u1)/dx0 - d(u0)/dx1,
 *   and the output is q = curl w by the same formulae.
 * - In two dimensions the field is treated as (u0, u1, 0), whose curl has
 *   the single out-of-plane component
 *       w = d(u1)/dx0 - d(u0)/dx1,
 *   and the output is the in-plane vector
 *       q = ( d(w)/dx1, -d(w)/dx0 ).
 * Every output component therefore couples all input components. All
 * derivatives are collocation derivatives of the elemental interpolant,
 * formed with the same standard-region derivative matrices, or
 * one-dimensional derivative matrices and sum factorisation, as
 * PhysDeriv.
 *
 * The operator is defined for two- and three-dimensional expansions whose
 * coordinate dimension equals the shape dimension (both asserted by the
 * block-implementation constructors); embedded manifolds are not
 * supported. The input field must carry at least as many components as
 * the shape dimension: the block implementations read the first two (2D)
 * or three (3D) and write the same number of output components. A
 * multi-plane 3DH1 field must carry exactly three components (asserted).
 *
 * As with the other element-operator families, this class sets the
 * Apply() field states at compile time -- physical space in and out --
 * and drives one CurlCurlBlockOp per element block, followed by
 * v_ApplyFFT(). On a multi-plane 3DH1 field each block operator forms
 * only the plane part of one curl, and detail::CurlCurlOpImpl adds the
 * z parts and runs the block operators again for the second curl;
 * everywhere else the block operators take both curls and v_ApplyFFT()
 * does nothing. See ElmtOp for the block decomposition and factory
 * scheme.
 *
 * @tparam TData Floating-point type of the field data.
 */
template <typename TData>
class CurlCurlOp : public ElmtOp<FieldState::Phys, FieldState::Phys, TData>
{
    friend class ElmtOp<FieldState::Phys, FieldState::Phys, TData>;

public:
    /**
     * @brief Build a complete curl-curl operator through the operator and
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
     * @return The fully assembled operator, ready to Apply().
     */
    static std::shared_ptr<CurlCurlOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Phys, FieldState::Phys, TData>::
            template Create<CurlCurlOp, CurlCurlBlockOp>(
                expansionList, components, execStr, implStr);
    }

    /// Operator base name; Create() appends the execution space to it
    /// to form the operator-factory key.
    static inline const std::string name = "CurlCurl";

protected:
    /// One block operator per element block, filled by Create().
    std::vector<std::shared_ptr<CurlCurlBlockOp<TData>>> m_blockOp;

    /**
     * @brief Construct the interface part of a concrete operator; called
     * by the factory-registered creator functions.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Component names the operator is set up
     *                          for.
     */
    CurlCurlOp(const MultiRegions::ExpListSharedPtr &expansionList,
               const std::vector<std::string> &components)
        : ElmtOp<FieldState::Phys, FieldState::Phys, TData>(expansionList,
                                                            components)
    {
    }

    ~CurlCurlOp() override = default;

    /**
     * @brief Apply the curl-curl operator block by block.
     *
     * Checks that the two fields agree in their number of homogeneous
     * modes, hands each input/output block pair to the corresponding
     * entry of #m_blockOp, then calls v_ApplyFFT().
     *
     * @param   in      Physical-space input field holding the components
     *                  of u.
     * @param   out     Physical-space output field receiving the
     *                  components of curl(curl u).
     */
    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &out) override
    {
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

        // Apply FFT.
        v_ApplyFFT(in, out);
    }

    /// Complete the double curl of a multi-plane 3DH1 field, of which the
    /// block operators have left the plane part of the first curl in
    /// @p out; does nothing on any other expansion.
    virtual void v_ApplyFFT(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &out) = 0;
};

} // namespace Nektar::MultiRegions
