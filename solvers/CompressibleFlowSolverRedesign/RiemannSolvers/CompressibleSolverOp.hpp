///////////////////////////////////////////////////////////////////////////////
//
// File: CompressibleSolverOp.hpp
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
// Description: CompressibleSolver operator base class.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/SolverUtilsOps/RiemannSolvers/RiemannSolverOp.hpp"

namespace Nektar::Operators
{

// CompressibleSolver operator base class
template <typename TData>
class CompressibleSolverOp : public RiemannSolverOp<TData>
{
public:
    static std::shared_ptr<CompressibleSolverOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components, const std::string &method,
        const std::string &execStr)
    {
        return std::static_pointer_cast<CompressibleSolverOp<TData>>(
            RiemannSolverOp<TData>::Create(expansionList, components,
                                           name + method, execStr));
    }

    static inline const std::string name = "CompressibleSolver";

protected:
    CompressibleSolverOp(const MultiRegions::ExpListSharedPtr &expansionList,
                         const std::vector<std::string> &components)
        : RiemannSolverOp<TData>(expansionList, components)
    {
    }

    ~CompressibleSolverOp() override = default;
};

// Helper function
template <typename TData>
NEK_DEVICE_INLINE TData GetPressure(const TData &rho, const TData &e)
{
    // Ideal gas law: P = (gamma - 1) * rho * e
    const TData gamma = 1.4; // Specific heat ratio for air
    return (gamma - 1) * rho * e;
}

template <typename TData>
NEK_DEVICE_INLINE TData GetSoundSpeed(const TData &rho, const TData &e)
{
    using std::sqrt;

    // Ideal gas law: P = (gamma - 1) * rho * e
    const TData gamma = 1.4; // Specific heat ratio for air
    NekDouble p       = GetPressure(rho, e);
    return std::sqrt(gamma * p / rho);
}

template <typename TData>
NEK_DEVICE_INLINE TData GetRoeSoundSpeed(
    [[maybe_unused]] const TData &rhoL, [[maybe_unused]] const TData &pL,
    [[maybe_unused]] const TData &eL, [[maybe_unused]] const TData &HL,
    [[maybe_unused]] const TData &srL, [[maybe_unused]] const TData &rhoR,
    [[maybe_unused]] const TData &pR, [[maybe_unused]] const TData &eR,
    [[maybe_unused]] const TData &HR, [[maybe_unused]] const TData &srR,
    const TData &HRoe, const TData &URoe2, [[maybe_unused]] const TData &srLR)
{
    using std::sqrt;

    // Specific heat ratio for air
    const TData gamma = 1.4;
    // Calculate sound speed using ideal gas relation
    return sqrt((gamma - 1.0) * (HRoe - 0.5 * URoe2));
}

} // namespace Nektar::Operators
