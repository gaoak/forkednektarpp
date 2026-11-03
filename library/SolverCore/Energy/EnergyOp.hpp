///////////////////////////////////////////////////////////////////////////////
//
// File: EnergyOp.hpp
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
// Description: Device-portable kinetic-energy and enstrophy operator.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <vector>

#include <MultiRegions/Common/Operator.hpp>
#include <SolverCore/SolverCore.hpp>

namespace Nektar::SolverCore
{

/**
 * @brief Integrate the squared velocity and vorticity magnitudes and the
 * domain volume.
 *
 * The quadrature and the local reduction execute in the selected operator
 * execution space, and the three integrals are reduced to rank zero in one
 * packed buffer. The velocity gradient is an input, so that it can be shared
 * with other consumers.
 *
 * After Apply(), the getters return:
 *
 * - GetKineticEnergy(): \f$ \int_\Omega |\mathbf{u}|^2 \, d\Omega \f$;
 * - GetEnstrophy(): \f$ \int_\Omega |\boldsymbol{\omega}|^2 \, d\Omega
 *   \f$, with \f$ \boldsymbol{\omega} = \nabla \times \mathbf{u} \f$,
 *   which in 2D is the scalar \f$ \partial_x v - \partial_y u \f$;
 * - GetVolume(): the domain volume \f$ \int_\Omega 1 \, d\Omega \f$ (the
 *   area in 2D).
 *
 * The two integrals carry neither the factor one half nor the division by the
 * volume; FilterEnergy applies both.
 *
 * Apply() is collective: every rank must call it. The getters read the
 * reduced values on the host and are valid on rank zero only, the other
 * ranks holding their own local contributions. Internally the three values
 * are packed in m_result, at the indices kineticEnergy, enstrophy and volume.
 */
template <typename TData> class EnergyOp : public MultiRegions::Operator<TData>
{
public:
    static std::shared_ptr<EnergyOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        EnsureLinked();
        return MultiRegions::Operator<TData>::template Create<EnergyOp>(
            expansionList, components, execStr);
    }

    static inline const std::string name = "Energy";

    /// @p velocityComponents are the indices of the velocity components in
    /// the velocity field, which may hold further components.
    void Configure(const std::vector<unsigned int> &velocityComponents)
    {
        v_Configure(velocityComponents);
    }

    /// Collective: every rank must call it.
    void Apply(LibUtilities::Field<TData, FieldState::Phys> &velocity,
               LibUtilities::Field<TData, FieldState::Phys> &velocityGradient)
    {
        v_Apply(velocity, velocityGradient);
    }

    /// Collective: as Apply().
    void operator()(
        LibUtilities::Field<TData, FieldState::Phys> &velocity,
        LibUtilities::Field<TData, FieldState::Phys> &velocityGradient)
    {
        v_Apply(velocity, velocityGradient);
    }

    /// Integral of the squared velocity magnitude from the last Apply(),
    /// valid on rank zero only.
    TData GetKineticEnergy()
    {
        return GetResult(kineticEnergy);
    }

    /// Integral of the squared vorticity magnitude from the last Apply(),
    /// valid on rank zero only.
    TData GetEnstrophy()
    {
        return GetResult(enstrophy);
    }

    /// Domain volume from the last Apply(), valid on rank zero only.
    TData GetVolume()
    {
        return GetResult(volume);
    }

protected:
    /// Indices of the integrals in m_result.
    static constexpr unsigned int kineticEnergy = 0;
    static constexpr unsigned int enstrophy     = 1;
    static constexpr unsigned int volume        = 2;
    /// Number of entries in m_result.
    static constexpr unsigned int resultSize = 3;

    /// Reduced result of Apply(), of size resultSize, allocated by the
    /// implementation's constructor.
    LibUtilities::MemoryRegion<TData> m_result;

    EnergyOp(const MultiRegions::ExpListSharedPtr &expansionList,
             const std::vector<std::string> &components)
        : MultiRegions::Operator<TData>(expansionList, components)
    {
    }

    ~EnergyOp() override = default;

    virtual void v_Configure(
        const std::vector<unsigned int> &velocityComponents) = 0;
    virtual void v_Apply(
        LibUtilities::Field<TData, FieldState::Phys> &velocity,
        LibUtilities::Field<TData, FieldState::Phys> &velocityGradient) = 0;

    /// Entry @p index of the reduced result, read on the host.
    TData GetResult(const unsigned int index)
    {
        return m_result
            .template GetPtr<NektarSpaces::HostSpace, ReadOnly>()[index];
    }
};

} // namespace Nektar::SolverCore
