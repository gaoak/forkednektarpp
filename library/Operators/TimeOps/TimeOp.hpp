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
    static std::shared_ptr<TimeOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::string &method = "", const unsigned int &order = 0,
        const std::string &variant = "", const std::string &execStr = "")
    {
        auto session = expansionList->GetSession();

        std::string method0 =
            (method == "")
                ? expansionList->GetSession()->GetTimeIntScheme().method
                : method;

        unsigned int order0 =
            (order == 0) ? expansionList->GetSession()->GetTimeIntScheme().order
                         : order;

        std::string variant0 =
            (variant == "")
                ? expansionList->GetSession()->GetTimeIntScheme().variant
                : variant;

        std::string execStr0 =
            (execStr == "")
                ? session->GetCmdLineArgument<std::string>("opExecSpace")
                : execStr;

        // Set key.
        std::string requestedKey =
            method0 + variant0 + std::to_string(order0) + execStr0;

        // Get operator factory.
        OperatorFactory<TData> &factory = GetOperatorFactory<TData>();

        // No suitable operator was found.
        if (!factory.ModuleExists(requestedKey))
        {
            std::stringstream msg;
            msg << "No such operator: " << requestedKey << std::endl;
            factory.PrintAvailableClasses(msg);
            NEKERROR(ErrorUtil::efatal, msg.str());
        }

        auto op = std::static_pointer_cast<TimeOp<TData>>(
            factory.CreateInstance(requestedKey, expansionList));

        // Set operator meta data
        op->m_timestep = expansionList->GetSession()->GetParameter("TimeStep");

        return op;
    }

    void Apply(Field<TData, FieldState::Phys> &inout)
    {
        this->v_Apply(inout);
    }

    void operator()(Field<TData, FieldState::Phys> &inout)
    {
        this->v_Apply(inout);
    }

    /// Functor definitions and generic handles for projection, and explicit
    /// and implicit evaluation.
    // Functor typedefs
    typedef std::function<void(Field<TData, FieldState::Phys> &,
                               Field<TData, FieldState::Phys> &, const TData &)>
        functorType1;
    typedef std::function<void(Field<TData, FieldState::Phys> &,
                               Field<TData, FieldState::Phys> &, const TData &,
                               const TData &)>
        functorType2;

    template <typename FuncPointerT, typename ObjectPointerT>
    void DefineExplicit(FuncPointerT func, ObjectPointerT obj)
    {
        m_explicitFunctor =
            std::bind(func, obj, std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4);
    }

    template <typename FuncPointerT, typename ObjectPointerT>
    void DefineProjection(FuncPointerT func, ObjectPointerT obj)
    {
        m_projectionFunctor =
            std::bind(func, obj, std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3);
    }

    template <typename FuncPointerT, typename ObjectPointerT>
    void DefineImplicit(FuncPointerT func, ObjectPointerT obj)
    {
        m_implicitFunctor =
            std::bind(func, obj, std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4);
    }

    void DoExplicit(Field<TData, FieldState::Phys> &in,
                    Field<TData, FieldState::Phys> &out, const TData &time,
                    const TData &factor) const
    {
        ASSERTL1(m_explicitFunctor,
                 "Explicit functor should be defined for this time "
                 "integration scheme. Use DefineExplicit() within "
                 "solver definition.");
        m_explicitFunctor(in, out, time, factor);
    }

    void DoProjection(Field<TData, FieldState::Phys> &in,
                      Field<TData, FieldState::Phys> &out,
                      const TData &time) const
    {
        ASSERTL1(m_projectionFunctor,
                 "Projection functor should be defined for this time "
                 "integration scheme. Use DefineProjection() within "
                 "solver definition.");
        m_projectionFunctor(in, out, time);
    }

    void DoImplicit(Field<TData, FieldState::Phys> &in,
                    Field<TData, FieldState::Phys> &out, const TData &time,
                    const TData &lambda) const
    {
        ASSERTL1(m_implicitFunctor,
                 "Implicit functor should be defined for this time "
                 "integration scheme. Use DefineImplicit() within "
                 "solver definition.");
        m_implicitFunctor(in, out, time, lambda);
    }

    void CopyFunctorsFrom(const TimeOp<TData> &src)
    {
        // functors and relevant booleans
        this->m_implicitFunctor   = src.m_implicitFunctor;
        this->m_explicitFunctor   = src.m_explicitFunctor;
        this->m_projectionFunctor = src.m_projectionFunctor;
    }

    // Move‐out
    std::deque<Field<TData, FieldState::Phys>> TakeSolutions()
    {
        return std::move(m_solutions);
    }

    std::deque<Field<TData, FieldState::Phys>> TakeExplicits()
    {
        return std::move(m_explicits);
    }

    std::deque<Field<TData, FieldState::Phys>> TakeImplicits()
    {
        return std::move(m_implicits);
    }

    // Move‐in
    void SetSolutions(std::deque<Field<TData, FieldState::Phys>> &&solutions)
    {
        ASSERTL0(m_solutions.empty(),
                 "Do not call SetSolutions() if m_solutions is "
                 "already defined in TimeOp operator.")
        m_solutions = std::move(solutions);
    }

    void SetExplicits(std::deque<Field<TData, FieldState::Phys>> &&explicits)
    {
        ASSERTL0(m_explicits.empty(),
                 "Do not call SetExplicits() if m_solutions is "
                 "already defined in TimeOp operator.")
        m_explicits = std::move(explicits);
    }

    void SetImplicits(std::deque<Field<TData, FieldState::Phys>> &&implicits)
    {
        ASSERTL0(m_implicits.empty(),
                 "Do not call SetImplicits() if m_solutions is "
                 "already defined in TimeOp operator.")
        m_implicits = std::move(implicits);
    }

    void SetTimeStep(const TData &timestep)
    {
        m_timestep = timestep;
    }

    void SetTime(const TData &time)
    {
        m_time = time;
    }

    void SetStep(const unsigned int &step)
    {
        m_step = step;
    }

    TData GetTimeStep(void) const
    {
        return m_timestep;
    }

    TData GetTime(void) const
    {
        return m_time;
    }

    unsigned int GetStep(void) const
    {
        return m_step;
    }

protected:
    // General parameters
    unsigned int m_step = 0;
    TData m_time        = 0.0;
    TData m_timestep    = 0.0;

    // Storage for previous solutions in Fields
    // and memory region for pointer access on device
    std::deque<Field<TData, FieldState::Phys>> m_solutions;

    // Storage for previous explicit parts in Fields
    // and memory region for pointer access on device
    std::deque<Field<TData, FieldState::Phys>> m_explicits;

    // Storage for previous implicit parts in Fields
    // and memory region for pointer access on device
    std::deque<Field<TData, FieldState::Phys>> m_implicits;

    // Functors to explicit, projection and implicit part of time integration
    functorType1 m_projectionFunctor;
    functorType2 m_explicitFunctor;
    functorType2 m_implicitFunctor;

    TimeOp(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
    }

    ~TimeOp() override = default;

    virtual void v_Apply(Field<TData, FieldState::Phys> &inout) = 0;

    /*
     *  Roll over solutions for next time step.
     *  The operation for 3rd order works as follows:
     *  Upon input:
     *  param inout:        u^n
     *  param m_solutions: [u^{n-1}, u^{n-2}]
     *  Upon output:
     *  param inout:        u^{n-2}
     *  param m_solutions: [u^{n}, u^{n-1}]
     *  Note that the first order schemes do not use this.
     */
    void RollOver(Field<TData, FieldState::Phys> &inout,
                  std::deque<Field<TData, FieldState::Phys>> &fieldDeque)
    {
        // Save last solution
        auto tmp = std::move(fieldDeque.back());

        // Remove last solution from deque
        fieldDeque.pop_back();

        // Add current solution to front
        fieldDeque.push_front(std::move(inout));

        // Move saved solution to inout
        // (use inout as temporary storage of oldest solution)
        inout = std::move(tmp);
    }

    /*
     *  Roll over explicit parts for next time step.
     *  The operation for 3rd order works as follows:
     *  Upon input:
     *  param fieldDeque: [u^{n}, u^{n-1}, u^{n-2}]
     *  Upon output:
     *  param fieldDeque: [u^{n-2}, u^{n}, u^{n-1}]
     *  Note that the first order schemes do not use this.
     */
    void RollOver(std::deque<Field<TData, FieldState::Phys>> &fieldDeque)
    {
        // Save last solution
        auto tmp = std::move(fieldDeque.back());

        // Remove last solution from back
        fieldDeque.pop_back();

        // Add last solution to front
        fieldDeque.push_front(std::move(tmp));
    }
};

} // namespace Nektar::Operators
