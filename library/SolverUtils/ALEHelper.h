///////////////////////////////////////////////////////////////////////////////
//
// File: ALEHelper.h
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
// Description: Helper class for ALE process
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SOLVERUTILS_ALEHELPER_H
#define NEKTAR_SOLVERUTILS_ALEHELPER_H

#include <MultiRegions/ExpList.h>
#include <SolverUtils/SolverUtilsDeclspec.h>
#include <SpatialDomains/Movement/Movement.h>
#include <SpatialDomains/Movement/Zones.h>
#include <boost/core/ignore_unused.hpp>

namespace Nektar::SolverUtils
{

struct ALEBase;
typedef std::shared_ptr<ALEBase> ALEBaseShPtr;

class ALEHelper
{
public:
    virtual ~ALEHelper() = default;
    SOLVER_UTILS_EXPORT virtual void v_ALEInitObject(
        int spaceDim, Array<OneD, MultiRegions::ExpListSharedPtr> &fields);
    SOLVER_UTILS_EXPORT void InitObject(
        int spaceDim, Array<OneD, MultiRegions::ExpListSharedPtr> &fields);

    SOLVER_UTILS_EXPORT virtual void v_UpdateGridVelocity(
        const NekDouble &time);

    SOLVER_UTILS_EXPORT virtual void v_ALEPreMultiplyMass(
        Array<OneD, Array<OneD, NekDouble>> &fields);

    SOLVER_UTILS_EXPORT void ALEDoElmtInvMass(
        Array<OneD, Array<OneD, NekDouble>> &traceNormals,
        Array<OneD, Array<OneD, NekDouble>> &fields, NekDouble time);

    SOLVER_UTILS_EXPORT void ALEDoElmtInvMassBwdTrans(
        const Array<OneD, const Array<OneD, NekDouble>> &inarray,
        Array<OneD, Array<OneD, NekDouble>> &outarray);

    SOLVER_UTILS_EXPORT void MoveMesh(
        const NekDouble &time,
        Array<OneD, Array<OneD, NekDouble>> &traceNormals);

    inline const Array<OneD, const Array<OneD, NekDouble>> &GetGridVelocity()
    {
        return m_gridVelocity;
    }

    SOLVER_UTILS_EXPORT const Array<OneD, const Array<OneD, NekDouble>> &
    GetGridVelocityTrace();
    SOLVER_UTILS_EXPORT void ExtraFldOutputGridVelocity(
        std::vector<Array<OneD, NekDouble>> &fieldcoeffs,
        std::vector<std::string> &variables);

protected:
    Array<OneD, MultiRegions::ExpListSharedPtr> m_fieldsALE;
    Array<OneD, Array<OneD, NekDouble>> m_gridVelocity;
    Array<OneD, Array<OneD, NekDouble>> m_gridVelocityTrace;
    std::vector<ALEBaseShPtr> m_ALEs;
    bool m_ALESolver          = false;
    bool m_ImplicitALESolver  = false;
    NekDouble m_prevStageTime = 0.0;
    int m_spaceDim;
};

struct ALEBase
{
    virtual ~ALEBase() = default;

    inline void UpdateGridVel(
        const NekDouble time,
        Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
        Array<OneD, Array<OneD, NekDouble>> &gridVelocity)
    {
        v_UpdateGridVel(time, fields, gridVelocity);
    }

private:
    virtual void v_UpdateGridVel(
        const NekDouble time,
        Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
        Array<OneD, Array<OneD, NekDouble>> &gridVelocity) = 0;
};

struct ALEFixed final : public ALEBase
{
    ALEFixed(SpatialDomains::ZoneBaseShPtr zone);

    void v_UpdateGridVel(
        const NekDouble time,
        Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
        Array<OneD, Array<OneD, NekDouble>> &gridVelocity) final;

private:
    SpatialDomains::ZoneFixedShPtr m_zone;
};

struct ALETranslate final : public ALEBase
{
    ALETranslate(SpatialDomains::ZoneBaseShPtr zone);

    void v_UpdateGridVel(
        const NekDouble time,
        Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
        Array<OneD, Array<OneD, NekDouble>> &gridVelocity) final;

private:
    SpatialDomains::ZoneTranslateShPtr m_zone;
};

struct ALERotate final : public ALEBase
{
    ALERotate(SpatialDomains::ZoneBaseShPtr zone);

    void v_UpdateGridVel(
        const NekDouble time,
        Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
        Array<OneD, Array<OneD, NekDouble>> &gridVelocity) final;

private:
    SpatialDomains::ZoneRotateShPtr m_zone;
};

typedef std::shared_ptr<ALEFixed> ALEFixedShPtr;
typedef std::shared_ptr<ALETranslate> ALETranslateShPtr;
typedef std::shared_ptr<ALERotate> ALERotateShPtr;

} // namespace Nektar::SolverUtils
#endif // NEKTAR_SOLVERUTILS_ALEHELPER_H
