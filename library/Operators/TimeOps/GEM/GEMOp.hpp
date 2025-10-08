///////////////////////////////////////////////////////////////////////////////
//
// File: GEMOp.hpp
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

#pragma once

#include <Operators/TimeOps/TimeOp.hpp>

namespace Nektar::Operators
{

// GEM base class
// Defines the apply operator to enforce apply parameter types
template <typename TData> class GEMOp : public TimeOp<TData>
{
protected:
    bool m_initialized{false};
    std::deque<Field<TData, FieldState::Phys>> m_T;
    std::deque<Field<TData, FieldState::Phys>> m_T0;

    GEMOp(const MultiRegions::ExpListSharedPtr &expansionList)
        : TimeOp<TData>(expansionList)
    {
    }

    ~GEMOp() override = default;

    template <typename ExecSpace>
    void ExtrapolateSolution(Field<TData, FieldState::Phys> &inout)
    {
        using MemSpace = typename ExecSpace::memory_space;

        if (this->m_variant == "")
        {
            // No extrapolation required for first-order
            if (this->m_order == 1)
            {
                inout.template Copy<MemSpace>(this->m_T0[0]);
            }
            else
            {
                // Extrapolate solution.
                for (unsigned int m = 1; m < this->m_order; ++m)
                {
                    // Aitken - Neville formula.
                    for (unsigned int k = m; k < this->m_order; ++k)
                    {
                        sub<ExecSpace>(this->m_T0[k], this->m_T0[k - 1],
                                       this->m_T[k]);
                        daxpy<ExecSpace>(
                            (TData)(k - m + 1) / ((k + 1) - (k - m + 1)),
                            this->m_T[k], this->m_T0[k], this->m_T[k]);
                    }

                    // Copy new values to old values.
                    for (unsigned int k = m; k < this->m_order; ++k)
                    {
                        this->m_T0[k].template Copy<MemSpace>(this->m_T[k]);
                    }
                }

                // Copy final solution.
                inout.template Copy<MemSpace>(this->m_T.back());
            }
        }
        else if (this->m_variant == "Midpoint")
        {
            // No extrapolation required for second-order
            if (this->m_order == 2)
            {
                inout.template Copy<MemSpace>(this->m_T0[0]);
            }
            else
            {
                // Extrapolate solution
                for (unsigned int m = 1; m < this->m_order / 2; ++m)
                {
                    // Aitken - Neville formula
                    for (unsigned int k = m; k < this->m_order / 2; ++k)
                    {
                        sub<ExecSpace>(this->m_T0[k], this->m_T0[k - 1],
                                       this->m_T[k]);
                        daxpy<ExecSpace>(
                            (TData)std::pow(k - m + 1, 2) /
                                (std::pow(k + 1, 2) - std::pow(k - m + 1, 2)),
                            this->m_T[k], this->m_T0[k], this->m_T[k]);
                    }

                    // Copy new values to old values
                    for (unsigned int k = m; k < this->m_order / 2; ++k)
                    {
                        this->m_T0[k].template Copy<MemSpace>(this->m_T[k]);
                    }
                }

                // Copy final solution.
                inout.template Copy<MemSpace>(this->m_T.back());
            }
        }
    }
};

} // namespace Nektar::Operators
