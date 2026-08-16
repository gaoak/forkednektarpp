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

#include "LibUtilities/BasicUtils/Math/Math.hpp"
#include "LibUtilities/BasicUtils/Math/MathHelper.hpp"
#include "Operators/Common/Operator.hpp"

#include <deque>
#include <memory>
#include <vector>

namespace Nektar::SolverCore
{

template <typename TData> class TimeOp;

enum class TimeOpExtrapolationType
{
    StateExtrapolation,
    BdfHistory,
    ExplicitContribution
};

enum class TimeOpExtrapolationMode
{
    Assign,
    Append
};

// Typename alias for the factory
template <typename TData>
using TimeOpFactory =
    Nektar::LibUtilities::NekFactory<std::string, TimeOp<TData>,
                                     const MultiRegions::ExpListSharedPtr &,
                                     const std::vector<std::string> &>;
template <typename TData>
using GEMOpFactory =
    Nektar::LibUtilities::NekFactory<std::string, TimeOp<TData>,
                                     const MultiRegions::ExpListSharedPtr &,
                                     const std::vector<std::string> &,
                                     const unsigned int &, const std::string &>;
template <typename TData>
using SDCOpFactory = Nektar::LibUtilities::NekFactory<
    std::string, TimeOp<TData>, const MultiRegions::ExpListSharedPtr &,
    const std::vector<std::string> &, const unsigned int &, const std::string &,
    const std::vector<TData> &>;

// Operator factory singleton
template <typename TData> TimeOpFactory<TData> &GetTimeOpFactory();
template <typename TData> GEMOpFactory<TData> &GetGEMOpFactory();
template <typename TData> SDCOpFactory<TData> &GetSDCOpFactory();

// TimeIntegration base class
template <typename TData> class TimeOp : public Operators::Operator<TData>
{
public:
    ~TimeOp() override = default;

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
                ? Operators::Operator<TData>::GetOpExecSpace(session)
                : execStr;

        // Set key.
        std::string requestedKey;

        if (method0 == "ExplicitSDC" || method0 == "ImplicitSDC" ||
            method0 == "IMEXSDC")
        {
            // Specialization for SDC
            requestedKey = method0 + execStr0;

            // Get operator factory.
            SDCOpFactory<TData> &factory = GetSDCOpFactory<TData>();

            // No suitable operator was found.
            if (!factory.ModuleExists(requestedKey))
            {
                std::stringstream msg;
                msg << "No such operator: " << requestedKey << std::endl;
                factory.PrintAvailableClasses(msg);
                NEKERROR(ErrorUtil::efatal, msg.str());
            }

            return std::static_pointer_cast<TimeOp<TData>>(
                factory.CreateInstance(requestedKey, expansionList, components,
                                       order0, variant0, freeParams));
        }
        else if (method0 == "ExplicitGEM" || method0 == "ImplicitGEM" ||
                 method0 == "IMEXGEM")
        {
            // Specialization for GEM
            requestedKey = method0 + execStr0;

            // Get operator factory.
            GEMOpFactory<TData> &factory = GetGEMOpFactory<TData>();

            // No suitable operator was found.
            if (!factory.ModuleExists(requestedKey))
            {
                std::stringstream msg;
                msg << "No such operator: " << requestedKey << std::endl;
                factory.PrintAvailableClasses(msg);
                NEKERROR(ErrorUtil::efatal, msg.str());
            }

            return std::static_pointer_cast<TimeOp<TData>>(
                factory.CreateInstance(requestedKey, expansionList, components,
                                       order0, variant0));
        }
        else
        {
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

    void Apply(LibUtilities::Field<TData, FieldState::Phys> &inout)
    {
        this->v_Apply(inout);
    }

    void operator()(LibUtilities::Field<TData, FieldState::Phys> &inout)
    {
        this->v_Apply(inout);
    }

    /// Functor definitions and generic handles for projection, and explicit
    /// and implicit evaluation.
    // Functor typedefs
    typedef std::function<void(LibUtilities::Field<TData, FieldState::Phys> &,
                               LibUtilities::Field<TData, FieldState::Phys> &,
                               const TData &)>
        functorType1;
    typedef std::function<void(LibUtilities::Field<TData, FieldState::Phys> &,
                               LibUtilities::Field<TData, FieldState::Phys> &,
                               const TData &, const TData &)>
        functorType2;

    template <typename FuncPointerT, typename ObjectPointerT>
    void DefineExplicitRhs(FuncPointerT func, ObjectPointerT obj)
    {
        m_explicitRhsFunctor =
            std::bind(func, obj, std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4);
    }

    template <typename FuncPointerT, typename ObjectPointerT>
    void DefineImplicitRhs(FuncPointerT func, ObjectPointerT obj)
    {
        m_implicitRhsFunctor =
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

    void DoExplicitRhs(LibUtilities::Field<TData, FieldState::Phys> &in,
                       LibUtilities::Field<TData, FieldState::Phys> &out,
                       const TData &time, const TData &factor) const
    {
        ASSERTL1(m_explicitRhsFunctor,
                 "DoExplicitRhs functor should be defined for this time "
                 "integration scheme. Use DefineExplicitRhs() within "
                 "solver definition.");
        if (m_useExplicitContributionExtrapolation)
        {
            GetExplicitContributionContext().historyId =
                m_explicitContributionHistoryId;
        }
        m_explicitRhsFunctor(in, out, time, factor);
    }

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

    void CopyFunctorsFrom(TimeOp<TData> &src)
    {
        // functors and relevant booleans
        this->m_implicitFunctor    = src.m_implicitFunctor;
        this->m_explicitRhsFunctor = src.m_explicitRhsFunctor;
        this->m_implicitRhsFunctor = src.m_implicitRhsFunctor;
        this->m_projectionFunctor  = src.m_projectionFunctor;
        this->m_useExplicitContributionExtrapolation =
            src.m_useExplicitContributionExtrapolation;
        this->m_explicitContributionParentContext =
            src.m_useExplicitContributionExtrapolation
                ? &src.GetExplicitContributionContext()
                : nullptr;
    }

    void EnableExplicitContributionExtrapolation()
    {
        m_useExplicitContributionExtrapolation = true;
        m_explicitContributionParentContext    = nullptr;
    }

    std::vector<TData> GetExtrapolationCoefficients(
        const TimeOpExtrapolationType type, const unsigned int historySize)
    {
        return v_GetExtrapolationCoefficients(type, historySize);
    }

    void ExtrapolateHistory(
        std::deque<LibUtilities::Field<TData, FieldState::Phys>> &history,
        LibUtilities::Field<TData, FieldState::Phys> &out,
        const TimeOpExtrapolationType type,
        const TimeOpExtrapolationMode mode = TimeOpExtrapolationMode::Assign,
        const TData scale                  = 1.0)
    {
        ExtrapolateHistory(history, out,
                           v_GetExtrapolationCoefficients(
                               type, static_cast<unsigned int>(history.size())),
                           mode, scale);
    }

    void ExtrapolateHistory(
        std::deque<LibUtilities::Field<TData, FieldState::Phys>> &history,
        LibUtilities::Field<TData, FieldState::Phys> &out,
        const std::vector<TData> &coeffs,
        const TimeOpExtrapolationMode mode = TimeOpExtrapolationMode::Assign,
        const TData scale                  = 1.0)
    {
        v_ExtrapolateHistory(history, out, coeffs, mode, scale);
    }

    void ExtrapolateExplicitContribution(
        std::deque<LibUtilities::Field<TData, FieldState::Phys>> &history,
        LibUtilities::Field<TData, FieldState::Phys> &out,
        const TimeOpExtrapolationMode mode = TimeOpExtrapolationMode::Assign,
        const TData scale                  = 1.0)
    {
        ASSERTL0(m_useExplicitContributionExtrapolation,
                 "Explicit contribution extrapolation has not been enabled "
                 "for this time integration operator.");
        const auto &coeffs = GetExplicitContributionContext().coefficients;
        ASSERTL0(!coeffs.empty(),
                 "No explicit contribution coefficients are available for "
                 "this implicit solve.");
        ASSERTL0(history.size() >= coeffs.size(),
                 "Explicit contribution history is shorter than the current "
                 "coefficient list.");

        const auto execSpace = Operators::Operator<TData>::GetOpExecSpace(
            this->m_expansionList->GetSession());
        Math::MathHelper math(execSpace);

        if (mode == TimeOpExtrapolationMode::Assign)
        {
            math.zero(out);
        }

        for (unsigned int i = 0; i < coeffs.size(); ++i)
        {
            const TData coeff = coeffs[i];
            if (coeff != (TData)0.0)
            {
                math.daxpy(scale * coeff, history[i], out, out);
            }
        }
    }

    unsigned int GetExplicitContributionHistoryId() const
    {
        return GetExplicitContributionContext().historyId;
    }

    // Move‐out
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

    // Move‐in
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

protected:
    // General parameters
    unsigned int m_step = 0;
    TData m_time        = 0.0;
    TData m_timestep    = 0.0;

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

    struct ExplicitContributionContext
    {
        std::vector<TData> coefficients;
        unsigned int historyId = 0;
    };

    mutable ExplicitContributionContext m_explicitContributionContextStorage;
    mutable ExplicitContributionContext *m_explicitContributionParentContext =
        nullptr;
    bool m_useExplicitContributionExtrapolation                    = false;
    unsigned int m_explicitContributionHistoryId                   = 0;
    inline static unsigned int m_nextExplicitContributionHistoryId = 0;

    TimeOp(const MultiRegions::ExpListSharedPtr &expansionList,
           const std::vector<std::string> components)
        : Operators::Operator<TData>(expansionList, components),
          m_explicitContributionHistoryId(++m_nextExplicitContributionHistoryId)
    {
        this->m_timestep =
            expansionList->GetSession()->GetParameter("TimeStep");
    }

    // Use for GEM
    TimeOp(const MultiRegions::ExpListSharedPtr &expansionList,
           const std::vector<std::string> components,
           [[maybe_unused]] const unsigned int &order,
           [[maybe_unused]] const std::string &variant)
        : Operators::Operator<TData>(expansionList, components),
          m_explicitContributionHistoryId(++m_nextExplicitContributionHistoryId)
    {
        this->m_timestep =
            expansionList->GetSession()->GetParameter("TimeStep");
    }

    // Use for SDC
    TimeOp(const MultiRegions::ExpListSharedPtr &expansionList,
           const std::vector<std::string> components,
           [[maybe_unused]] const unsigned int &order,
           [[maybe_unused]] const std::string &variant,
           [[maybe_unused]] const std::vector<TData> &freeParams)
        : Operators::Operator<TData>(expansionList, components),
          m_explicitContributionHistoryId(++m_nextExplicitContributionHistoryId)
    {
        this->m_timestep =
            expansionList->GetSession()->GetParameter("TimeStep");
    }

    virtual void v_Apply(
        LibUtilities::Field<TData, FieldState::Phys> &inout) = 0;

    void SetExplicitContributionCoefficients(std::vector<TData> coeffs)
    {
        if (!m_useExplicitContributionExtrapolation)
        {
            return;
        }
        auto &context        = GetExplicitContributionContext();
        context.historyId    = m_explicitContributionHistoryId;
        context.coefficients = std::move(coeffs);
    }

    void ClearExplicitContributionCoefficients()
    {
        if (!m_useExplicitContributionExtrapolation)
        {
            return;
        }
        GetExplicitContributionContext().coefficients.clear();
    }

    ExplicitContributionContext &GetExplicitContributionContext() const
    {
        return m_explicitContributionParentContext
                   ? *m_explicitContributionParentContext
                   : m_explicitContributionContextStorage;
    }

    virtual std::vector<TData> v_GetExtrapolationCoefficients(
        [[maybe_unused]] const TimeOpExtrapolationType type,
        [[maybe_unused]] const unsigned int historySize)
    {
        ASSERTL0(false, "This time integration operator does not provide "
                        "extrapolation coefficients.");
        return {};
    }

    virtual void v_ExtrapolateHistory(
        std::deque<LibUtilities::Field<TData, FieldState::Phys>> &history,
        LibUtilities::Field<TData, FieldState::Phys> &out,
        const std::vector<TData> &coeffs, const TimeOpExtrapolationMode mode,
        const TData scale)
    {
        ASSERTL0(history.size() == coeffs.size(),
                 "History size must match extrapolation coefficient count.");

        const auto execSpace = Operators::Operator<TData>::GetOpExecSpace(
            this->m_expansionList->GetSession());
        Math::MathHelper math(execSpace);

        if (mode == TimeOpExtrapolationMode::Assign)
        {
            math.zero(out);
        }

        for (unsigned int i = 0; i < history.size(); ++i)
        {
            if (coeffs[i] != (TData)0.0)
            {
                math.daxpy(scale * coeffs[i], history[i], out, out);
            }
        }
    }

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
    void RollOver(
        LibUtilities::Field<TData, FieldState::Phys> &inout,
        std::deque<LibUtilities::Field<TData, FieldState::Phys>> &fieldDeque)
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
    void RollOver(
        std::deque<LibUtilities::Field<TData, FieldState::Phys>> &fieldDeque)
    {
        // Save last solution
        auto tmp = std::move(fieldDeque.back());

        // Remove last solution from back
        fieldDeque.pop_back();

        // Add last solution to front
        fieldDeque.push_front(std::move(tmp));
    }
};

} // namespace Nektar::SolverCore
