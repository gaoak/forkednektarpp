///////////////////////////////////////////////////////////////////////////////
//
// File: IMEXBaseOp.hpp
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
// Description: Base class for IMEXOp and IMEXdirkOp schemes.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <SolverCore/TimeOps/TimeOp.hpp>

#include <functional>
#include <vector>

namespace Nektar::SolverCore
{

// Base class of the schemes whose implicit solves combine explicit
// contributions, which pass them to solver functions that take them.
template <typename TData> class IMEXBaseOp : public TimeOp<TData>
{
public:
    /// The explicit contributions an implicit solve combines: their
    /// coefficients, latest first, and the id and length of the history of
    /// the scheme that combines them. The coefficients are only set in an
    /// implicit solve.
    struct ExplicitWeights
    {
        std::vector<TData> coefficients;
        unsigned int historyId   = 0;
        unsigned int historySize = 0;
    };

    typedef std::function<void(LibUtilities::Field<TData, FieldState::Phys> &,
                               LibUtilities::Field<TData, FieldState::Phys> &,
                               const TData &, const TData &,
                               const ExplicitWeights &)>
        weightsFunctorType;

    /// DefineExplicitRhs() for a solver that keeps its own histories of
    /// quantities other than the solution, e.g. the velocity for the
    /// higher-order pressure BCs, and combines them with the weights of
    /// the explicit terms. The function also takes ExplicitWeights, whose
    /// history id and size tell it which history to update.
    template <typename FuncPointerT, typename ObjectPointerT>
    void DefineExplicitRhsWithWeights(FuncPointerT func, ObjectPointerT obj)
    {
        Bind(func, obj, this->m_explicitRhsFunctor,
             m_explicitRhsWeightsFunctor);
    }

    /// DefineImplicit() for such a solver. The function also takes
    /// ExplicitWeights, whose coefficients are those with which the implicit
    /// solve combines the explicit terms, to apply to its histories. Not
    /// needed by a solver that only uses the solution.
    template <typename FuncPointerT, typename ObjectPointerT>
    void DefineImplicitWithWeights(FuncPointerT func, ObjectPointerT obj)
    {
        Bind(func, obj, this->m_implicitFunctor, m_implicitWeightsFunctor);
    }

    /// Coefficients with which implicit solve @p stage of a step combines
    /// the explicit terms, latest first; IMEX ignores @p stage.
    std::vector<TData> GetExplicitCoefficients(
        const unsigned int stage = 1) const
    {
        return v_GetExplicitCoefficients(stage);
    }

    using TimeOp<TData>::CopyFunctors;

    void CopyFunctors(const IMEXBaseOp<TData> &src)
    {
        TimeOp<TData>::CopyFunctors(src);
        m_explicitRhsWeightsFunctor = src.m_explicitRhsWeightsFunctor;
        m_implicitWeightsFunctor    = src.m_implicitWeightsFunctor;
    }

protected:
    // The solver's functions that take ExplicitWeights, which the
    // explicit RHS and implicit functors of TimeOp call without one.
    weightsFunctorType m_explicitRhsWeightsFunctor;
    weightsFunctorType m_implicitWeightsFunctor;

    // Id of the history of this scheme, distinct from that of the start-up
    // scheme of a multistep scheme.
    unsigned int m_historyId;
    inline static unsigned int m_nextHistoryId = 0;
    // Most explicit contributions an implicit solve of this scheme combines,
    // set by the scheme.
    unsigned int m_historySize = 0;

    IMEXBaseOp(const MultiRegions::ExpListSharedPtr &expansionList,
               const std::vector<std::string> &components)
        : TimeOp<TData>(expansionList, components),
          m_historyId(++m_nextHistoryId)
    {
    }

    ~IMEXBaseOp() override = default;

    /// DoExplicitRhs() that passes the history to a solver function that
    /// takes ExplicitWeights.
    void DoExplicitRhsWithWeights(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &out, const TData &time,
        const TData &factor) const
    {
        if (m_explicitRhsWeightsFunctor)
        {
            m_explicitRhsWeightsFunctor(in, out, time, factor,
                                        {{}, m_historyId, m_historySize});
        }
        else
        {
            TimeOp<TData>::DoExplicitRhs(in, out, time, factor);
        }
    }

    /// DoImplicit() for implicit solve @p stage of a step, which passes its
    /// coefficients and the history to a solver function that takes
    /// ExplicitWeights.
    void DoImplicitWithWeights(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &out, const TData &time,
        const TData &lambda, const unsigned int stage) const
    {
        if (m_implicitWeightsFunctor)
        {
            m_implicitWeightsFunctor(
                in, out, time, lambda,
                {v_GetExplicitCoefficients(stage), m_historyId, m_historySize});
        }
        else
        {
            TimeOp<TData>::DoImplicit(in, out, time, lambda);
        }
    }

    /// Coefficients of the explicit contributions combined by implicit
    /// solve @p stage of a step, latest first.
    virtual std::vector<TData> v_GetExplicitCoefficients(
        const unsigned int stage) const = 0;

    template <typename FuncPointerT, typename ObjectPointerT>
    static void Bind(FuncPointerT func, ObjectPointerT obj,
                     typename TimeOp<TData>::functorType2 &functor,
                     weightsFunctorType &weightsFunctor)
    {
        weightsFunctor =
            std::bind(func, obj, std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4,
                      std::placeholders::_5);
        functor =
            [weightsFunctor](LibUtilities::Field<TData, FieldState::Phys> &in,
                             LibUtilities::Field<TData, FieldState::Phys> &out,
                             const TData &a, const TData &b) {
                weightsFunctor(in, out, a, b, ExplicitWeights{});
            };
    }
};

} // namespace Nektar::SolverCore
