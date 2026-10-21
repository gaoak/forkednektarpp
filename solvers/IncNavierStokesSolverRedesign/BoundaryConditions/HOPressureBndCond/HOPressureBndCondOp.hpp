///////////////////////////////////////////////////////////////////////////////
//
// File: HOPressureBndCondOp.hpp
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
// Description: Higher-order pressure boundary condition operator.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <MultiRegions/BndCondOps/NeuBndCond/NeuBndCondOp.hpp>

namespace Nektar
{

/**
 * @brief Higher-order pressure boundary condition, added to the pressure
 * system as a Neumann contribution.
 *
 * @details
 * On every pressure boundary region with a Neumann condition and
 * USERDEFINEDTYPE="H", the pressure flux is the normal component of
 * scale * (rhs - kinvis * CurlCurl(velocity)), less the acceleration needed
 * to reach the prescribed boundary velocity, in the velocity components that
 * are Dirichlet on the region. Update() computes it from the
 * current fields, with the kinematic viscosity set by SetKinvis(), the scale
 * set by SetScale() and at the time set by SetTime(); Apply() adds its inner
 * product to the pressure forcing. It is added to the pressure system with
 * LinearSystemOp::AddNeumannBCOp(op, userDefinedType), so that the default
 * Neumann operator of the system leaves its regions out.
 */
template <typename TData>
class HOPressureBndCondOp : public MultiRegions::NeuBndCondOp<TData>
{
public:
    static std::shared_ptr<HOPressureBndCondOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        return MultiRegions::Operator<TData>::template Create<
            HOPressureBndCondOp>(expansionList, components, execStr);
    }

    static inline const std::string name = "HOPressureBndCond";

    /// USERDEFINEDTYPE of the pressure Neumann conditions the operator
    /// applies.
    static inline const std::string userDefinedType = "H";

    /**
     * @brief Set the velocity expansion list and the names of the velocity
     * components, from which the boundary-adjacent elements are found.
     */
    void SetVelocityContext(
        const MultiRegions::ExpListSharedPtr &velocityExpansionList,
        const std::vector<std::string> &velocityComponents)
    {
        v_SetVelocityContext(velocityExpansionList, velocityComponents);
    }

    /**
     * @brief Set the kinematic viscosity kinvis Update() uses.
     */
    void SetKinvis(const TData &kinvis)
    {
        m_kinvis      = kinvis;
        m_isSetKinvis = true;
    }

    /**
     * @brief Set the time at which UpdateWithAcceleration() evaluates the
     * prescribed boundary velocity (default 0).
     */
    void SetTime(const TData &time)
    {
        m_time = time;
    }

    /**
     * @brief Set the scale by which Update() multiplies the fields it is
     * passed (default 1).
     */
    void SetScale(const TData &scale)
    {
        m_scale = scale;
    }

    /**
     * @brief Compute the pressure flux scale * (rhs - kinvis *
     * CurlCurl(velocity)) . n on the boundary.
     */
    void Update(LibUtilities::Field<TData, FieldState::Phys> &rhs,
                LibUtilities::Field<TData, FieldState::Phys> &velocity)
    {
        v_Update(rhs, velocity);
    }

    /**
     * @brief Compute the pressure flux, adding the acceleration needed to
     * reach the prescribed boundary velocity at the time set by SetTime().
     *
     * @details
     * @p predictor is the complete right-hand side of the implicit stage,
     * which holds the explicit contribution @p rhs of Update(), so rhs
     * cancels from the flux scale * (rhs - kinvis * CurlCurl(velocity)) . n
     * + scale * (predictor - rhs) . n - scale * u_b(time) . n, which is
     * scale * (predictor - kinvis * CurlCurl(velocity)) . n
     * - scale * u_b(time) . n, with u_b the prescribed boundary velocity.
     *
     * A velocity component that is not Dirichlet on a region has no
     * prescribed boundary velocity and takes no acceleration: its flux there
     * is scale * (rhs - kinvis * CurlCurl(velocity)) n_d. @p rhs is read
     * only when HasNonDirichletVelocity(), and may be uninstantiated
     * otherwise.
     */
    void UpdateWithAcceleration(
        LibUtilities::Field<TData, FieldState::Phys> &rhs,
        LibUtilities::Field<TData, FieldState::Phys> &velocity,
        LibUtilities::Field<TData, FieldState::Phys> &predictor)
    {
        v_UpdateWithAcceleration(rhs, velocity, predictor);
    }

    /**
     * @brief Whether a region on this rank has a velocity component that is
     * not Dirichlet, so that UpdateWithAcceleration() reads rhs.
     */
    bool HasNonDirichletVelocity() const
    {
        return m_hasNonDirichletVelocity;
    }

protected:
    /// Set by SetVelocityContext(); read by HasNonDirichletVelocity().
    bool m_hasNonDirichletVelocity = false;
    /// Set by SetKinvis(); checked by Update() in debug builds.
    bool m_isSetKinvis = false;
    /// Kinematic viscosity, set by SetKinvis().
    TData m_kinvis = 0.0;
    /// Time of the prescribed boundary velocity, set by SetTime().
    TData m_time = 0.0;
    /// Scale of the fields passed to Update(), set by SetScale().
    TData m_scale = 1.0;

    HOPressureBndCondOp(const MultiRegions::ExpListSharedPtr &expansionList,
                        const std::vector<std::string> &components)
        : MultiRegions::NeuBndCondOp<TData>(expansionList, components)
    {
    }

    ~HOPressureBndCondOp() override = default;

    virtual void v_SetVelocityContext(
        const MultiRegions::ExpListSharedPtr &velocityExpansionList,
        const std::vector<std::string> &velocityComponents) = 0;

    virtual void v_Update(
        LibUtilities::Field<TData, FieldState::Phys> &rhs,
        LibUtilities::Field<TData, FieldState::Phys> &velocity) = 0;

    virtual void v_UpdateWithAcceleration(
        LibUtilities::Field<TData, FieldState::Phys> &rhs,
        LibUtilities::Field<TData, FieldState::Phys> &velocity,
        LibUtilities::Field<TData, FieldState::Phys> &predictor) = 0;
};

} // namespace Nektar
