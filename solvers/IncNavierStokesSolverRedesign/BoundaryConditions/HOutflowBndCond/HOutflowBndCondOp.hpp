///////////////////////////////////////////////////////////////////////////////
//
// File: HOutflowBndCondOp.hpp
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
// Description: Coupled Dong high-order outflow boundary condition.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <MultiRegions/BndCondOps/DirBndCond/DirBndCondOp.hpp>
#include <MultiRegions/BndCondOps/NeuBndCond/NeuBndCondOp.hpp>
#include <MultiRegions/Common/Operator.hpp>

namespace Nektar
{

/**
 * @brief Dong's high-order outflow boundary condition, a pressure Dirichlet
 * condition coupled with a velocity Neumann condition.
 *
 * @details
 * It applies to the boundary regions whose pressure condition is Dirichlet
 * and whose velocity conditions are Neumann, all with
 * USERDEFINEDTYPE="HOutflow". The operator holds the boundary data both
 * conditions share and is not itself a boundary condition operator:
 * ApplyPressure() sets the Dirichlet pressure coefficients and
 * ApplyVelocity() adds the velocity flux. GetPressureDirichletOp() and
 * GetVelocityNeumannOp() return the Dirichlet condition of the pressure
 * system and the Neumann condition of the velocity system, which forward
 * their Apply() to them. UpdatePressure() sets the pressure from the
 * extrapolated velocity before the pressure solve. ApplyVelocity() computes
 * the velocity flux from the solved pressure set by SetPressure() and the
 * boundary data of the last UpdatePressure(), so the flux always reflects
 * the pressure as it stands when the velocity system applies it. Both
 * UpdatePressure() and ApplyVelocity() take the kinematic viscosity set by
 * SetKinvis(), and UpdatePressure() the time set by SetTime() and the
 * velocity scale set by SetScale().
 */
template <typename TData>
class HOutflowBndCondOp : public MultiRegions::Operator<TData>
{
public:
    static std::shared_ptr<HOutflowBndCondOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &pressureExpansionList,
        const std::vector<std::string> &pressureComponents,
        const std::string &execStr = "")
    {
        return MultiRegions::Operator<TData>::template Create<
            HOutflowBndCondOp>(pressureExpansionList, pressureComponents,
                               execStr);
    }

    static inline const std::string name = "HOutflowBndCond";

    /// USERDEFINEDTYPE of the pressure Dirichlet and velocity Neumann
    /// conditions the operator applies.
    static inline const std::string userDefinedType = "HOutflow";

    /**
     * @brief Set the velocity expansion list and the names of the velocity
     * components, and build the velocity boundary data.
     */
    void SetVelocityContext(
        const MultiRegions::ExpListSharedPtr &velocityExpansionList,
        const std::vector<std::string> &velocityComponents)
    {
        v_SetVelocityContext(velocityExpansionList, velocityComponents);
    }

    /**
     * @brief Set the kinematic viscosity UpdatePressure() and the velocity
     * Neumann condition use.
     */
    void SetKinvis(const TData &kinvis)
    {
        m_kinvis = kinvis;
    }

    /**
     * @brief Set the time at which UpdatePressure() evaluates the Neumann
     * value f_b of the velocity conditions (default 0).
     */
    void SetTime(const TData &time)
    {
        m_time = time;
    }

    /**
     * @brief Set the scale by which UpdatePressure() multiplies the velocity
     * it is passed, and its gradient, to give the velocity u* (default 1).
     */
    void SetScale(const TData &scale)
    {
        m_scale = scale;
    }

    /**
     * @brief Set the pressure boundary values from the extrapolated velocity.
     *
     * @param velocity IMEX-extrapolated velocity at the scale of the time
     *                 integrator; the scale set by SetScale() converts it,
     *                 and its gradient, to the velocity u*.
     */
    void UpdatePressure(LibUtilities::Field<TData, FieldState::Phys> &velocity)
    {
        v_UpdatePressure(velocity);
    }

    /**
     * @brief Set the solved pressure, in physical space, from which the
     * velocity Neumann condition computes the velocity flux when it is
     * applied.
     *
     * @details
     * The operator does not take ownership of @p pressure, which must
     * outlive it. The flux is computed from the values @p pressure holds
     * when the velocity Neumann condition is applied, so a pressure
     * refilled in place need not be set again.
     */
    void SetPressure(LibUtilities::Field<TData, FieldState::Phys> &pressure)
    {
        m_pressure = &pressure;
    }

    /**
     * @brief Set the Dirichlet pressure coefficients of the regions in
     * @p out, from the pressure boundary values of the last
     * UpdatePressure().
     */
    void ApplyPressure(LibUtilities::Field<TData, FieldState::Coeff> &out)
    {
        v_ApplyPressure(out);
    }

    /**
     * @brief Compute the velocity flux of the regions from the solved
     * pressure set by SetPressure(), and add its inner product with the
     * boundary basis to the velocity forcing @p inout.
     */
    void ApplyVelocity(LibUtilities::Field<TData, FieldState::Coeff> &inout)
    {
        v_ApplyVelocity(inout);
    }

    /**
     * @brief The Dirichlet condition of the pressure system, whose Apply()
     * is ApplyPressure().
     *
     * @details
     * It sets only the pressure coefficients of the regions, and is added
     * to the pressure system with LinearSystemOp::AddDirichletBCOp(op),
     * after the default Dirichlet operator of the system, whose values for
     * the regions it overwrites.
     */
    std::shared_ptr<MultiRegions::DirBndCondOp<TData>> GetPressureDirichletOp()
    {
        return v_GetPressureDirichletOp();
    }

    /**
     * @brief The Neumann condition of the velocity system, whose Apply() is
     * ApplyVelocity(), available once SetVelocityContext() has been called.
     *
     * @details
     * It applies the whole velocity flux of the regions, and is added to
     * the velocity system with LinearSystemOp::AddNeumannBCOp(op,
     * userDefinedType), so that the default Neumann operator of the system
     * leaves them out.
     */
    std::shared_ptr<MultiRegions::NeuBndCondOp<TData>> GetVelocityNeumannOp()
    {
        return v_GetVelocityNeumannOp();
    }

protected:
    /// Kinematic viscosity, set by SetKinvis().
    TData m_kinvis = 0.0;
    /// Time of the Neumann value f_b, set by SetTime().
    TData m_time = 0.0;
    /// Scale of the velocity passed to UpdatePressure(), set by SetScale().
    TData m_scale = 1.0;
    /// Solved pressure, set by SetPressure(); not owned.
    LibUtilities::Field<TData, FieldState::Phys> *m_pressure = nullptr;

    HOutflowBndCondOp(
        const MultiRegions::ExpListSharedPtr &pressureExpansionList,
        const std::vector<std::string> &pressureComponents)
        : MultiRegions::Operator<TData>(pressureExpansionList,
                                        pressureComponents)
    {
    }

    ~HOutflowBndCondOp() override = default;

    virtual void v_SetVelocityContext(
        const MultiRegions::ExpListSharedPtr &velocityExpansionList,
        const std::vector<std::string> &velocityComponents) = 0;

    virtual void v_UpdatePressure(
        LibUtilities::Field<TData, FieldState::Phys> &velocity) = 0;

    virtual void v_ApplyPressure(
        LibUtilities::Field<TData, FieldState::Coeff> &out) = 0;

    virtual void v_ApplyVelocity(
        LibUtilities::Field<TData, FieldState::Coeff> &inout) = 0;

    virtual std::shared_ptr<MultiRegions::DirBndCondOp<TData>>
    v_GetPressureDirichletOp() = 0;

    virtual std::shared_ptr<MultiRegions::NeuBndCondOp<TData>>
    v_GetVelocityNeumannOp() = 0;
};

} // namespace Nektar
