///////////////////////////////////////////////////////////////////////////////
//
// File: MaxStdVelocityOp.hpp
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
// Description: MaxStdVelocity operator base class.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/Common/Operator.hpp"
#include "SolverCore/CFL/MaxStdVelocity/MaxStdVelocityBlockOp.hpp"
#include "SolverCore/SolverCore.hpp"

namespace Nektar::SolverCore
{

/**
 * @brief Reciprocal of the advective time scale over the whole mesh.
 *
 * Returns
 *
 * \f[ \max_{\text{points}} \; \sigma \, (P-1)^2 \f]
 *
 * where \f$\sigma\f$ is the standard element velocity computed by
 * MaxStdVelocityBlockOp and \f$P\f$ the polynomial order of the element the
 * point belongs to. The standard velocity already carries the element size
 * through the derivative factors, so the result has dimensions of one over
 * time and a timestep follows by dividing a stability constant by it:
 *
 * \f[ \Delta t = \frac{\text{CFL} \; \alpha}{c_\lambda \, \text{invTimeScale}}
 * \f]
 *
 * with \f$c_\lambda = 0.2\f$ and \f$\alpha\f$ the stability limit of the
 * time integration scheme.
 *
 * Reducing to one number rather than an array per element is what the
 * blocking makes natural: elements are grouped by shape and order, so the
 * order weight is constant within a block and can be folded in before the
 * blocks are combined. Variable polynomial order needs no special handling,
 * since it produces separate blocks.
 *
 * The input field carries the velocity components, optionally followed by a
 * scalar wave speed when SetSoundSpeedFactor() has been given a non-zero
 * weight. A scalar advection or incompressible problem passes velocity alone;
 * a compressible one appends the speed of sound.
 */
template <typename TData>
class MaxStdVelocityOp : public Operators::Operator<TData>
{
public:
    ~MaxStdVelocityOp() override = default;

    static std::shared_ptr<MaxStdVelocityOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        // Force a genuine cross-library symbol reference into libSolverCore so
        // the MaxStdVelocity*OpImpl static factory registrations are not
        // dropped by the linker/loader -- see EnsureLinked() in SolverCore.hpp.
        EnsureLinked();

        auto session = expansionList->GetSession();

        std::string execStr0 =
            (execStr == "")
                ? Operators::Operator<TData>::GetOpExecSpace(session)
                : execStr;

        auto op = Operators::Operator<TData>::template Create<MaxStdVelocityOp>(
            expansionList, components, execStr0);

        auto blockAttr =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                expansionList);

        for (unsigned int block_idx = 0; block_idx < blockAttr.size();
             block_idx++)
        {
            // The block's representative expansion.
            const auto exp =
                MultiRegions::GetCollection(expansionList, block_idx)
                    .GetExpVector()[0];
            op->m_blockOp.push_back(MaxStdVelocityBlockOp<TData>::Create(
                block_idx, exp, expansionList->GetDataWarehouseSharedPtr(),
                execStr0));

            // Every element of a block shares its order, so this is the
            // block's weight, kept at least one so that a first order
            // expansion does not report an infinite permissible timestep.
            const TData order = std::max(
                TData(exp->EvalBasisNumModesMax()) - TData(1), TData(1));
            op->m_orderWeight.push_back(order * order);
        }

        op->m_data = LibUtilities::MemoryRegion<TData>(
            "MaxStdVelocityReduce", blockAttr.size(), eHostPinned);

        return op;
    }

    static inline const std::string name = "MaxStdVelocity";

    TData Apply(LibUtilities::Field<TData, FieldState::Phys> &in)
    {
        return v_Apply(in);
    }

    TData operator()(LibUtilities::Field<TData, FieldState::Phys> &in)
    {
        return v_Apply(in);
    }

    /**
     * @brief Weight the wave speed carried as the field's last component.
     *
     * Zero, the default, means the field is velocity alone. One is the
     * compressible case, where the last component is the speed of sound.
     */
    void SetSoundSpeedFactor(const TData &soundSpeedFactor)
    {
        for (unsigned int blk = 0; blk < m_blockOp.size(); ++blk)
        {
            m_blockOp[blk]->SetSoundSpeedFactor(soundSpeedFactor);
        }
    }

protected:
    std::vector<std::shared_ptr<MaxStdVelocityBlockOp<TData>>> m_blockOp;
    /// \f$(P-1)^2\f$ for each block, applied before the blocks are combined.
    std::vector<TData> m_orderWeight;
    LibUtilities::MemoryRegion<TData> m_data;

    MaxStdVelocityOp(const MultiRegions::ExpListSharedPtr &expansionList,
                     const std::vector<std::string> &components)
        : Operators::Operator<TData>(expansionList, components)
    {
    }

    virtual TData v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in) = 0;
};

} // namespace Nektar::SolverCore
