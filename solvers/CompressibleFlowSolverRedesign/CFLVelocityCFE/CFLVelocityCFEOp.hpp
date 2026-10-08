///////////////////////////////////////////////////////////////////////////////
//
// File: CFLVelocityCFEOp.hpp
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
// Description: Velocity and sound speed for the Courant estimate.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <boost/algorithm/string/case_conv.hpp>

#include <MultiRegions/Common/Operator.hpp>

#include "EquationOfState/SupportedEoS.hpp"

namespace Nektar
{

/**
 * @brief Convert conserved variables to the wave speeds the Courant
 * estimate needs.
 *
 * Writes the @p ndim velocity components followed by the speed of sound,
 * which is what MaxStdVelocityOp consumes with its sound speed factor set
 * to one. The conversion is kept out of MaxStdVelocityOp so that the
 * latter stays free of any equation of state and remains usable by the
 * incompressible and advection-diffusion solvers, which have no sound
 * speed to supply.
 */
template <typename TData>
class CFLVelocityCFEOp : public MultiRegions::Operator<TData>
{
public:
    static std::shared_ptr<CFLVelocityCFEOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        auto session = expansionList->GetSession();

        std::string EoSName;
        if (session->DefinesEquationOfState())
        {
            EoSName = boost::to_upper_copy(session->GetEquationOfState().type);
        }
        else
        {
            NEKERROR(ErrorUtil::efatal,
                     "No EquationOfState section defined in session file");
        }

        std::string execStr0 =
            (execStr == "")
                ? MultiRegions::Operator<TData>::GetOpExecSpace(session)
                : execStr;

        // The key carries the equation of state between the name and the
        // execution space, so the factory is asked directly rather than
        // through Operator::Create, which would prepend the name a second
        // time.
        const std::string requestedKey = name + EoSName + execStr0;

        MultiRegions::OperatorFactory<TData> &factory =
            MultiRegions::GetOperatorFactory<TData>();

        if (!factory.ModuleExists(requestedKey))
        {
            std::stringstream msg;
            msg << "No such operator: " << requestedKey << std::endl;
            factory.PrintAvailableClasses(msg);
            NEKERROR(ErrorUtil::efatal, msg.str());
        }

        return std::static_pointer_cast<CFLVelocityCFEOp<TData>>(
            factory.CreateInstance(requestedKey, expansionList, components));
    }

    static inline const std::string name = "CFLVelocityCFE";

    void Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
               LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        v_Apply(in, out);
    }

    void operator()(LibUtilities::Field<TData, FieldState::Phys> &in,
                    LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        v_Apply(in, out);
    }

protected:
    CFLVelocityCFEOp(const MultiRegions::ExpListSharedPtr &expansionList,
                     const std::vector<std::string> &components)
        : MultiRegions::Operator<TData>(expansionList, components)
    {
    }

    ~CFLVelocityCFEOp() override = default;

    virtual void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                         LibUtilities::Field<TData, FieldState::Phys> &out) = 0;
};

} // namespace Nektar
