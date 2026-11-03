///////////////////////////////////////////////////////////////////////////////
//
// File: AeroForcesOp.hpp
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
// Description: Operator interface evaluating pressure and viscous forces on
// boundaries.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <array>
#include <vector>

#include <MultiRegions/Common/Operator.hpp>
#include <SolverCore/SolverCore.hpp>

namespace Nektar::SolverCore
{

/**
 * @brief Integrate pressure and viscous forces over selected boundaries.
 *
 * Volume-to-boundary restriction, traction evaluation, quadrature and the
 * local reduction execute in the selected operator execution space, and the
 * forces and moments are reduced to rank zero in one packed buffer.
 *
 * After Apply(), the getters return Cartesian 3-vectors, with component
 * @c d (0 = x, 1 = y, 2 = z) at index @c d:
 *
 * - GetPressureForce(): the pressure force
 *   \f$ \int_\Gamma \rho p \mathbf{n} \, d\Gamma \f$;
 * - GetViscousForce(): the viscous force
 *   \f$ -\int_\Gamma \mu (\nabla \mathbf{u} + \nabla \mathbf{u}^T)
 *   \mathbf{n} \, d\Gamma \f$;
 * - GetPressureMoment(): the moment of the pressure traction about the
 *   pivot, \f$ \int_\Gamma \mathbf{r} \times \rho p \mathbf{n} \,
 *   d\Gamma \f$;
 * - GetViscousMoment(): the moment of the viscous traction about the pivot.
 *
 * \f$ \Gamma \f$ is the union of the selected boundary regions,
 * \f$ \mathbf{n} \f$ the unit normal pointing out of the domain, so that the
 * forces are those the fluid exerts on the boundary, and \f$ \mathbf{r} \f$
 * the position relative to the pivot. In 2D the z force components and the x
 * and y moment components are zero, and the z moment components hold the
 * in-plane moments.
 *
 * Apply() is collective: every rank must call it. The getters read the
 * reduced values on the host and are valid on rank zero only, the other
 * ranks holding their own local contributions. Internally the four 3-vectors
 * are packed in m_result, starting at the offsets pressureForce,
 * viscousForce, pressureMoment and viscousMoment.
 */
template <typename TData>
class AeroForcesOp : public MultiRegions::Operator<TData>
{
public:
    /// @p expansionList and @p components are those of the velocity.
    static std::shared_ptr<AeroForcesOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        EnsureLinked();
        return MultiRegions::Operator<TData>::template Create<AeroForcesOp>(
            expansionList, components, execStr);
    }

    static inline const std::string name = "AeroForces";

    /// @p velocityComponents and @p pressureComponent are the indices of the
    /// velocity and pressure in the fields whose gradient and values Apply()
    /// receives, which may hold further components.
    void Configure(const std::vector<unsigned int> &velocityComponents,
                   const MultiRegions::ExpListSharedPtr &pressureExpansion,
                   const unsigned int pressureComponent,
                   const std::vector<bool> &selectedBoundaryRegions,
                   const TData rho, const TData mu,
                   const std::array<TData, 3> &pivot)
    {
        v_Configure(velocityComponents, pressureExpansion, pressureComponent,
                    selectedBoundaryRegions, rho, mu, pivot);
    }

    /// Collective: every rank must call it.
    void Apply(LibUtilities::Field<TData, FieldState::Phys> &pressure,
               LibUtilities::Field<TData, FieldState::Phys> &velocityGradient)
    {
        v_Apply(pressure, velocityGradient);
    }

    /// Collective: as Apply().
    void operator()(
        LibUtilities::Field<TData, FieldState::Phys> &pressure,
        LibUtilities::Field<TData, FieldState::Phys> &velocityGradient)
    {
        v_Apply(pressure, velocityGradient);
    }

    /// Pressure force from the last Apply(), valid on rank zero only.
    std::array<TData, 3> GetPressureForce()
    {
        return GetResultVector(pressureForce);
    }

    /// Viscous force from the last Apply(), valid on rank zero only.
    std::array<TData, 3> GetViscousForce()
    {
        return GetResultVector(viscousForce);
    }

    /// Moment of the pressure traction about the pivot from the last
    /// Apply(), valid on rank zero only.
    std::array<TData, 3> GetPressureMoment()
    {
        return GetResultVector(pressureMoment);
    }

    /// Moment of the viscous traction about the pivot from the last Apply(),
    /// valid on rank zero only.
    std::array<TData, 3> GetViscousMoment()
    {
        return GetResultVector(viscousMoment);
    }

protected:
    /// Offsets of the 3-vectors in m_result.
    static constexpr unsigned int pressureForce  = 0;
    static constexpr unsigned int viscousForce   = 3;
    static constexpr unsigned int pressureMoment = 6;
    static constexpr unsigned int viscousMoment  = 9;
    /// Number of entries in m_result.
    static constexpr unsigned int resultSize = 12;

    /// Reduced result of Apply(), of size resultSize, allocated by
    /// Configure().
    LibUtilities::MemoryRegion<TData> m_result;

    AeroForcesOp(const MultiRegions::ExpListSharedPtr &expansionList,
                 const std::vector<std::string> &components)
        : MultiRegions::Operator<TData>(expansionList, components)
    {
    }

    ~AeroForcesOp() override = default;

    virtual void v_Configure(
        const std::vector<unsigned int> &velocityComponents,
        const MultiRegions::ExpListSharedPtr &pressureExpansion,
        const unsigned int pressureComponent,
        const std::vector<bool> &selectedBoundaryRegions, const TData rho,
        const TData mu, const std::array<TData, 3> &pivot) = 0;

    virtual void v_Apply(
        LibUtilities::Field<TData, FieldState::Phys> &pressure,
        LibUtilities::Field<TData, FieldState::Phys> &velocityGradient) = 0;

    /// The 3-vector at @p offset of the reduced result, read on the host.
    std::array<TData, 3> GetResultVector(const unsigned int offset)
    {
        const auto result =
            m_result.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        return {result[offset], result[offset + 1], result[offset + 2]};
    }
};

} // namespace Nektar::SolverCore
