///////////////////////////////////////////////////////////////////////////////
//
// File: TimeOp.hpp
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

#include "Operators/Common/Operator.hpp"
#include "Operators/MathKernels/MathKernels.hpp"

namespace Nektar::Operators
{

// TimeIntegration base class
template <typename TData> class TimeOp : public Operator<TData>
{
public:
    void Apply(Field<TData, FieldState::Phys> &inout)
    {
        this->v_Apply(inout);
    }

    void operator()(Field<TData, FieldState::Phys> &inout)
    {
        this->v_Apply(inout);
    }

    void Initialise(Field<TData, FieldState::Phys> &initial, TData &time,
                    size_t &step)
    {
        this->v_Initialise(initial, time, step);
    }

    /// Functor definitions and generic handles for projection, and explicit
    /// and implicit evaluation.
    // Functor typedefs
    typedef std::function<void(Field<TData, FieldState::Phys> &, TData &)>
        functorType;
    typedef std::function<void(Field<TData, FieldState::Phys> &,
                               Field<TData, FieldState::Phys> &, TData &)>
        functorExplicitType;

    template <typename FuncPointerT, typename ObjectPointerT>
    void DefineExplicit(FuncPointerT func, ObjectPointerT obj)
    {
        m_explicitFunctor =
            std::bind(func, obj, std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3);
    }

    template <typename FuncPointerT, typename ObjectPointerT>
    void DefineProjection(FuncPointerT func, ObjectPointerT obj)
    {
        m_projectionFunctor =
            std::bind(func, obj, std::placeholders::_1, std::placeholders::_2);
    }

    template <typename FuncPointerT, typename ObjectPointerT>
    void DefineImplicit(FuncPointerT func, ObjectPointerT obj)
    {
        m_implicitFunctor =
            std::bind(func, obj, std::placeholders::_1, std::placeholders::_2);
    }

    void DoExplicit(Field<TData, FieldState::Phys> &in,
                    Field<TData, FieldState::Phys> &out, TData &dt_gamma) const
    {
        ASSERTL1(m_explicitFunctor,
                 "Explicit functor should be defined for this time "
                 "integration scheme. Use DefineExplicit() within "
                 "solver definition.");
        m_explicitFunctor(in, out, dt_gamma);
    }

    void DoProjection(Field<TData, FieldState::Phys> &in, TData &gamma) const
    {
        ASSERTL1(m_projectionFunctor,
                 "Projection functor should be defined for this time "
                 "integration scheme. Use DefineProjection() within "
                 "solver definition.");
        m_projectionFunctor(in, gamma);
    }

    void DoImplicit(Field<TData, FieldState::Phys> &in, TData &dt_gamma) const
    {
        ASSERTL1(m_implicitFunctor,
                 "Implicit functor should be defined for this time "
                 "integration scheme. Use DefineImplicit() within "
                 "solver definition.");
        m_implicitFunctor(in, dt_gamma);
    }

    void CopyFunctorsFrom(const TimeOp<TData> &src)
    {
        // functors and relevant booleans
        this->m_implicitFunctor   = src.m_implicitFunctor;
        this->m_explicitFunctor   = src.m_explicitFunctor;
        this->m_projectionFunctor = src.m_projectionFunctor;
    }

protected:
    // General parameters
    unsigned int m_intOrder;
    TData m_timestep;
    std::string m_intMethod;
    std::string m_intVariant;

    // Functors to explicit, projection and implicit part of time integration
    functorExplicitType m_explicitFunctor;
    functorType m_projectionFunctor;
    functorType m_implicitFunctor;

    TimeOp(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
        // Get TimeIntegration meta data
        m_intOrder   = expansionList->GetSession()->GetTimeIntScheme().order;
        m_intMethod  = expansionList->GetSession()->GetTimeIntScheme().method;
        m_intVariant = expansionList->GetSession()->GetTimeIntScheme().variant;
        m_timestep   = expansionList->GetSession()->GetParameter("TimeStep");
    }

    ~TimeOp() override = default;

    virtual void v_Apply(Field<TData, FieldState::Phys> &inout) = 0;

    virtual void v_Initialise(Field<TData, FieldState::Phys> &initial,
                              TData &time, size_t &step) = 0;
};

} // namespace Nektar::Operators
