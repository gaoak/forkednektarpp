///////////////////////////////////////////////////////////////////////////////
//
// File: MaxStdVelocityBlockOp.hpp
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
// Description: MaxStdVelocity block operator base class.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/Common/BlockOperator.hpp"

namespace Nektar::SolverCore
{

/**
 * @brief Largest standard element velocity in one block.
 *
 * The transport speed a CFL condition is written against is not the physical
 * velocity but its image on the reference element, which is what carries the
 * element's size and shape: a small or stretched element sees a larger
 * standard velocity for the same flow. Per quadrature point this operator
 * forms
 *
 * \f[ \sqrt{ \sum_i \left( \left| \sum_j g_{ji} u_j \right| +
 *                          f \left| \sum_j g_{ji} c \right| \right)^2 } \f]
 *
 * with \f$g\f$ the derivative factors, and reduces it to a maximum over the
 * block.
 *
 * The optional \f$c\f$ is a scalar wave speed - the speed of sound for a
 * compressible flow - transformed as though it acted equally in every
 * direction. That overestimates the true acoustic contribution, so a
 * timestep derived from it errs towards being too small.
 *
 * Only the velocity is needed for an incompressible or scalar advection
 * problem, in which case @p f is zero and the second term vanishes.
 *
 * @see MaxStdVelocityOp for the whole-field reduction and the order weighting.
 */
template <typename TData>
class MaxStdVelocityBlockOp : public Operators::BlockOperator<TData>
{
public:
    ~MaxStdVelocityBlockOp() override = default;

    static std::shared_ptr<MaxStdVelocityBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        const std::string &execStr)
    {
        return Operators::BlockOperator<TData>::template Create<
            MaxStdVelocityBlockOp>(block_idx, exp, dataWarehouse, execStr);
    }

    static inline const std::string name = "BlockMaxStdVelocity";

    /**
     * @brief Reduce @p inblock to its largest standard velocity.
     *
     * @param inblock         Velocity components, followed by the wave speed
     *                        when the sound speed factor is non-zero.
     * @param data            Receives the block's maximum at the operator's
     *                        block index, so the caller can weight each block
     *                        by its own polynomial order before reducing.
     */
    void Apply(LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
               LibUtilities::MemoryRegion<TData> &data)
    {
        this->v_Apply(inblock, data);
    }

    void operator()(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::MemoryRegion<TData> &data)
    {
        this->v_Apply(inblock, data);
    }

    /// Weight on the wave speed carried as the block's last component; zero,
    /// the default, means the block carries velocity alone.
    void SetSoundSpeedFactor(const TData &soundSpeedFactor)
    {
        m_soundSpeedFactor = soundSpeedFactor;
    }

protected:
    TData m_soundSpeedFactor = TData(0);

    MaxStdVelocityBlockOp(const unsigned int block_idx,
                          const LocalRegions::ExpansionSharedPtr &exp,
                          LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : Operators::BlockOperator<TData>(block_idx, exp, dataWarehouse)
    {
    }

    virtual void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::MemoryRegion<TData> &data) = 0;
};

} // namespace Nektar::SolverCore
