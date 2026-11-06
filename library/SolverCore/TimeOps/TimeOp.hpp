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

#include <LibUtilities/BasicUtils/Math/Math.hpp>
#include <LibUtilities/BasicUtils/Math/MathHelper.hpp>
#include <MultiRegions/Common/Operator.hpp>

#include <deque>
#include <functional>
#include <memory>
#include <vector>

namespace Nektar::SolverCore
{

template <typename TData> class TimeOp;

// Typename alias for the factory
template <typename TData>
using TimeOpFactory =
    Nektar::LibUtilities::NekFactory<std::string, TimeOp<TData>,
                                     const MultiRegions::ExpListSharedPtr &,
                                     const std::vector<std::string> &>;
// Operator factory singleton
template <typename TData> TimeOpFactory<TData> &GetTimeOpFactory();

// TimeIntegration base class
template <typename TData> class TimeOp : public MultiRegions::Operator<TData>
{
public:
    /// Create the scheme from @p method, @p order and @p variant, which
    /// default to the session's TimeIntegrationScheme.
    static std::shared_ptr<TimeOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &method = "", const unsigned int &order = 0,
        const std::string &variant           = "",
        const std::vector<TData> &freeParams = std::vector<TData>{},
        const std::string &execStr           = "")
    {
        auto session = expansionList->GetSession();

        std::string method0 =
            (method == "") ? session->GetTimeIntScheme().method : method;

        unsigned int order0 =
            (order == 0) ? session->GetTimeIntScheme().order : order;

        std::string variant0 =
            (variant == "") ? session->GetTimeIntScheme().variant : variant;

        std::string execStr0 =
            (execStr == "")
                ? MultiRegions::Operator<TData>::GetOpExecSpace(session)
                : execStr;

        // Set key.
        std::string requestedKey;

        if (method0 == "ExplicitSDC" || method0 == "ImplicitSDC" ||
            method0 == "IMEXSDC" || method0 == "ExplicitGEM" ||
            method0 == "ImplicitGEM" || method0 == "IMEXGEM")
        {
            // Specialization for SDC and GEM.
            // Note: SDC and GEM are not order templated.
            requestedKey = method0 + execStr0;

            // Get operator factory.
            TimeOpFactory<TData> &factory = GetTimeOpFactory<TData>();

            // No suitable operator was found.
            if (!factory.ModuleExists(requestedKey))
            {
                std::stringstream msg;
                msg << "No such operator: " << requestedKey << std::endl;
                factory.PrintAvailableClasses(msg);
                NEKERROR(ErrorUtil::efatal, msg.str());
            }

            auto op =
                std::static_pointer_cast<TimeOp<TData>>(factory.CreateInstance(
                    requestedKey, expansionList, components));
            op->v_SetScheme(order0, variant0, freeParams);
            return op;
        }
        else
        {
            // Other schemes are order and variant templated.
            requestedKey =
                method0 + variant0 + std::to_string(order0) + execStr0;

            // Get operator factory.
            TimeOpFactory<TData> &factory = GetTimeOpFactory<TData>();

            // No suitable operator was found.
            if (!factory.ModuleExists(requestedKey))
            {
                std::stringstream msg;
                msg << "No such operator: " << requestedKey << std::endl;
                factory.PrintAvailableClasses(msg);
                NEKERROR(ErrorUtil::efatal, msg.str());
            }

            return std::static_pointer_cast<TimeOp<TData>>(
                factory.CreateInstance(requestedKey, expansionList,
                                       components));
        }
    }

    /// Advance @p inout by one time step.
    void Apply(LibUtilities::Field<TData, FieldState::Phys> &inout)
    {
        this->v_Apply(inout);
    }

    void operator()(LibUtilities::Field<TData, FieldState::Phys> &inout)
    {
        this->v_Apply(inout);
    }

    /// Projection: (in, out, time).
    typedef std::function<void(LibUtilities::Field<TData, FieldState::Phys> &,
                               LibUtilities::Field<TData, FieldState::Phys> &,
                               const TData &)>
        functorType1;
    /// RHS evaluations: (in, out, time, factor), out = factor * f(in);
    /// implicit solve: (in, out, time, lambda), (1 - lambda L) out = in.
    typedef std::function<void(LibUtilities::Field<TData, FieldState::Phys> &,
                               LibUtilities::Field<TData, FieldState::Phys> &,
                               const TData &, const TData &)>
        functorType2;

    /// Register the solver's explicit RHS.
    template <typename FuncPointerT, typename ObjectPointerT>
    void DefineExplicitRhs(FuncPointerT func, ObjectPointerT obj)
    {
        m_explicitRhsFunctor =
            std::bind(func, obj, std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4);
    }

    /// Register the solver's implicit RHS.
    template <typename FuncPointerT, typename ObjectPointerT>
    void DefineImplicitRhs(FuncPointerT func, ObjectPointerT obj)
    {
        m_implicitRhsFunctor =
            std::bind(func, obj, std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4);
    }

    /// Register the solver's projection.
    template <typename FuncPointerT, typename ObjectPointerT>
    void DefineProjection(FuncPointerT func, ObjectPointerT obj)
    {
        m_projectionFunctor =
            std::bind(func, obj, std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3);
    }

    /// Register the solver's implicit solve.
    template <typename FuncPointerT, typename ObjectPointerT>
    void DefineImplicit(FuncPointerT func, ObjectPointerT obj)
    {
        m_implicitFunctor =
            std::bind(func, obj, std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4);
    }

    /// Register the solver functions of @p src, e.g. on a start-up scheme.
    void CopyFunctors(const TimeOp<TData> &src)
    {
        this->m_implicitFunctor    = src.m_implicitFunctor;
        this->m_explicitRhsFunctor = src.m_explicitRhsFunctor;
        this->m_implicitRhsFunctor = src.m_implicitRhsFunctor;
        this->m_projectionFunctor  = src.m_projectionFunctor;
    }

    // Move the stored levels out of the operator, e.g. to hand
    // them between schemes.
    std::deque<LibUtilities::Field<TData, FieldState::Phys>> TakeSolutions()
    {
        return std::move(m_solutions);
    }

    std::deque<LibUtilities::Field<TData, FieldState::Phys>> TakeExplicits()
    {
        return std::move(m_explicits);
    }

    std::deque<LibUtilities::Field<TData, FieldState::Phys>> TakeImplicits()
    {
        return std::move(m_implicits);
    }

    // Move the stored levels into the operator.
    void SetSolutions(
        std::deque<LibUtilities::Field<TData, FieldState::Phys>> &&solutions)
    {
        ASSERTL0(m_solutions.empty(),
                 "Do not call SetSolutions() if m_solutions is "
                 "already defined in TimeOp operator.")
        m_solutions = std::move(solutions);
    }

    void SetExplicits(
        std::deque<LibUtilities::Field<TData, FieldState::Phys>> &&explicits)
    {
        ASSERTL0(m_explicits.empty(),
                 "Do not call SetExplicits() if m_solutions is "
                 "already defined in TimeOp operator.")
        m_explicits = std::move(explicits);
    }

    void SetImplicits(
        std::deque<LibUtilities::Field<TData, FieldState::Phys>> &&implicits)
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

    /// Extent of the stability region along the negative real axis.
    TData GetTimeStability(void) const
    {
        return v_GetTimeStability();
    }

protected:
    // Steps taken, current time and time step
    unsigned int m_step = 0;
    TData m_time        = 0.0;
    TData m_timestep    = 0.0;

    // Block attributes of the fields the operator allocates
    std::vector<LibUtilities::BlockAttributes<FieldState::Phys>> m_blockAttr;

    // Storage for previous solutions in Fields
    // and memory region for pointer access on device
    std::deque<LibUtilities::Field<TData, FieldState::Phys>> m_solutions;

    // Storage for previous explicit parts in Fields
    // and memory region for pointer access on device
    std::deque<LibUtilities::Field<TData, FieldState::Phys>> m_explicits;

    // Storage for previous implicit parts in Fields
    // and memory region for pointer access on device
    std::deque<LibUtilities::Field<TData, FieldState::Phys>> m_implicits;

    // Functors to explicit, projection and implicit part of time integration
    functorType1 m_projectionFunctor;
    functorType2 m_explicitRhsFunctor;
    functorType2 m_implicitRhsFunctor;
    functorType2 m_implicitFunctor;

    // Takes the time step from the session parameter TimeStep.
    TimeOp(const MultiRegions::ExpListSharedPtr &expansionList,
           const std::vector<std::string> components)
        : MultiRegions::Operator<TData>(expansionList, components)
    {
        this->m_timestep =
            expansionList->GetSession()->GetParameter("TimeStep");
        this->m_blockAttr =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                expansionList,
                MultiRegions::Operator<TData>::GetDefaultInterleaveWidth(
                    expansionList->GetSession()));
    }

    ~TimeOp() override = default;

    /// The creators return a std::unique_ptr to this class, which the
    /// factory converts to the shared_ptr it hands out; their deleter
    /// destroys the operator through this destructor.
    friend struct std::default_delete<TimeOp>;

    // For SDC and GEM: set the order, variant and free parameters and
    // allocate the workspace. Called once, by Create(), straight after the
    // operator is constructed.
    virtual void v_SetScheme(
        [[maybe_unused]] const unsigned int &order,
        [[maybe_unused]] const std::string &variant,
        [[maybe_unused]] const std::vector<TData> &freeParams)
    {
    }

    // One time step of the scheme.
    virtual void v_Apply(
        LibUtilities::Field<TData, FieldState::Phys> &inout) = 0;

    // See GetTimeStability(); 1 unless the scheme overrides it.
    virtual TData v_GetTimeStability(void) const
    {
        return 1.0;
    }

    // out = factor * explicit RHS of in.
    void DoExplicitRhs(LibUtilities::Field<TData, FieldState::Phys> &in,
                       LibUtilities::Field<TData, FieldState::Phys> &out,
                       const TData &time, const TData &factor) const
    {
        ASSERTL1(m_explicitRhsFunctor,
                 "DoExplicitRhs functor should be defined for this time "
                 "integration scheme. Use DefineExplicitRhs() within "
                 "solver definition.");
        m_explicitRhsFunctor(in, out, time, factor);
    }

    // out = factor * implicit RHS of in.
    void DoImplicitRhs(LibUtilities::Field<TData, FieldState::Phys> &in,
                       LibUtilities::Field<TData, FieldState::Phys> &out,
                       const TData &time, const TData &factor) const
    {
        ASSERTL1(m_implicitRhsFunctor,
                 "DoImplicitRhs functor should be defined for this time "
                 "integration scheme. Use DefineImplicitRhs() within "
                 "solver definition.");
        m_implicitRhsFunctor(in, out, time, factor);
    }

    // Project in onto out.
    void DoProjection(LibUtilities::Field<TData, FieldState::Phys> &in,
                      LibUtilities::Field<TData, FieldState::Phys> &out,
                      const TData &time) const
    {
        ASSERTL1(m_projectionFunctor,
                 "Projection functor should be defined for this time "
                 "integration scheme. Use DefineProjection() within "
                 "solver definition.");
        m_projectionFunctor(in, out, time);
    }

    // Solve (1 - lambda L) out = in.
    void DoImplicit(LibUtilities::Field<TData, FieldState::Phys> &in,
                    LibUtilities::Field<TData, FieldState::Phys> &out,
                    const TData &time, const TData &lambda) const
    {
        ASSERTL1(m_implicitFunctor,
                 "Implicit functor should be defined for this time "
                 "integration scheme. Use DefineImplicit() within "
                 "solver definition.");
        m_implicitFunctor(in, out, time, lambda);
    }

    /*
     *  Roll over solutions for next time step.
     *  The operation for 3rd order works as follows:
     *  Upon input:
     *  param inout:        u^n
     *  param solutions: [u^{n-1}, u^{n-2}]
     *  Upon output:
     *  param inout:        u^{n-2}
     *  param solutions: [u^{n}, u^{n-1}]
     *  Note that the first order schemes do not use this.
     */
    void RollOver(
        LibUtilities::Field<TData, FieldState::Phys> &inout,
        std::deque<LibUtilities::Field<TData, FieldState::Phys>> &solutions)
    {
        // Save last solution
        auto tmp = std::move(solutions.back());

        // Remove last solution from deque
        solutions.pop_back();

        // Add current solution to front
        solutions.push_front(std::move(inout));

        // Move saved solution to inout
        // (use inout as temporary storage of oldest solution)
        inout = std::move(tmp);
    }

    /*
     *  Roll over explicit parts for next time step.
     *  The operation for 3rd order works as follows:
     *  Upon input:
     *  param solutions: [u^{n}, u^{n-1}, u^{n-2}]
     *  Upon output:
     *  param solutions: [u^{n-2}, u^{n}, u^{n-1}]
     *  Note that the first order schemes do not use this.
     */
    void RollOver(
        std::deque<LibUtilities::Field<TData, FieldState::Phys>> &solutions)
    {
        // Save last solution
        auto tmp = std::move(solutions.back());

        // Remove last solution from back
        solutions.pop_back();

        // Add last solution to front
        solutions.push_front(std::move(tmp));
    }
};

} // namespace Nektar::SolverCore
