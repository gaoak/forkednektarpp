///////////////////////////////////////////////////////////////////////////////
//
// File: FilterAeroForces.cpp
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
// Description: Outputs aerodynamic forces during time-stepping.
//
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <set>

#include <LibUtilities/BasicUtils/Equation.h>
#include <LibUtilities/BasicUtils/ParseUtils.h>
#include <LibUtilities/BasicUtils/StringUtils.hpp>
#include <SpatialDomains/Conditions.h>

#include <SolverCore/Filters/FilterAeroForces.h>

namespace Nektar::SolverCore
{

std::string FilterAeroForces::className =
    GetFilterFactory().RegisterCreatorFunction("AeroForces",
                                               FilterAeroForces::create);

FilterAeroForces::FilterAeroForces(
    const LibUtilities::SessionReaderSharedPtr &session, const ParamMap &params)
    : Filter(session)
{
    m_outputFile = Filter::SetupOutput(".fce", params);
    FilterParameters parameters(session, params, "FilterAeroForces");
    parameters.Consume("OutputFile");
    m_outputFrequency = parameters.PositiveFrequency("OutputFrequency", 1);
    m_schedule = StepSchedule(m_outputFrequency, true, parameters.StartTime());
    m_boundaryString = parameters.RequiredString("Boundary");

    auto parseVector = [&](const std::string &name,
                           std::array<double, 3> defaultValue) {
        if (!parameters.Has(name))
        {
            return defaultValue;
        }
        std::vector<std::string> values;
        StringUtils::Split(
            values, StringUtils::TrimCopy(parameters.RequiredString(name)),
            StringUtils::IsAnyOf(" \t\r\n"), StringUtils::eTokenCompressOn);
        ASSERTL0(values.size() == 3, "AeroForces parameter '" + name +
                                         "' requires exactly three values.");
        std::array<double, 3> result{};
        for (unsigned int i = 0; i < 3; ++i)
        {
            LibUtilities::Equation equation(m_session->GetInterpreter(),
                                            values[i]);
            result[i] = equation.Evaluate();
            ASSERTL0(std::isfinite(result[i]), "AeroForces parameter '" + name +
                                                   "' values must be finite.");
        }
        return result;
    };

    for (unsigned int i = 0; i < 3; ++i)
    {
        m_directions[i][i]     = 1.0;
        const std::string name = "Direction" + std::to_string(i + 1);
        m_directions[i]        = parseVector(name, m_directions[i]);
        double norm            = 0.0;
        for (unsigned int j = 0; j < 3; ++j)
        {
            norm += m_directions[i][j] * m_directions[i][j];
        }
        ASSERTL0(norm > NekConstants::kNekZeroTol,
                 "AeroForces direction must have non-zero length.");
        norm = std::sqrt(norm);
        for (unsigned int j = 0; j < 3; ++j)
        {
            m_directions[i][j] /= norm;
        }
    }

    ASSERTL0(!(parameters.Has("PivotPoint") && parameters.Has("MomentPoint")),
             "AeroForces cannot specify both PivotPoint and MomentPoint.");
    if (parameters.Has("PivotPoint"))
    {
        m_pivotPoint = parseVector("PivotPoint", m_pivotPoint);
    }
    else if (parameters.Has("MomentPoint"))
    {
        m_pivotPoint = parseVector("MomentPoint", m_pivotPoint);
    }

    if (m_session->DefinesParameter("Kinvis"))
    {
        m_rho = m_session->DefinesParameter("rho")
                    ? m_session->GetParameter("rho")
                    : 1.0;
        m_mu  = m_rho * m_session->GetParameter("Kinvis");
    }
    else
    {
        ASSERTL0(m_session->DefinesParameter("mu"),
                 "AeroForces requires either 'Kinvis' or 'mu'.");
        m_mu = m_session->GetParameter("mu");
    }
    ASSERTL0(std::isfinite(m_rho) && std::isfinite(m_mu),
             "AeroForces density and viscosity must be finite.");
    parameters.ValidateAllConsumed();
    m_executionSpace = MultiRegions::Operator<double>::GetOpExecSpace(session);
}

void FilterAeroForces::v_Initialise(FilterContext &context)
{
    auto velocityView = context.fields.GetRole(FieldRoles::Velocity);
    auto pressureView = context.fields.GetRole(FieldRoles::Pressure);
    ASSERTL0(velocityView && pressureView,
             "AeroForces requires velocity and pressure field views.");

    m_velocityExp = velocityView->expansion;
    m_pressureExp = pressureView->expansion;
    m_spaceDim    = velocityView->spaceDimension;
    m_momentDim   = m_spaceDim == 2 ? 1 : 3;

    ASSERTL0(m_spaceDim == 2 || m_spaceDim == 3,
             "AeroForces supports only 2D and 3D expansions.");
    ASSERTL0(m_velocityExp->GetShapeDimension() == m_spaceDim,
             "AeroForces does not support embedded-manifold expansions.");
    ASSERTL0(velocityView->field.get().GetNumHomoModes() == 1 &&
                 pressureView->field.get().GetNumHomoModes() == 1,
             "AeroForces does not support homogeneous expansions.");

    const auto first = m_boundaryString.find_first_of('[');
    const auto last  = m_boundaryString.find_last_of(']');
    ASSERTL0(first != std::string::npos && last != std::string::npos &&
                 first < last,
             "Error reading boundary region definition: " + m_boundaryString);
    const std::string boundaryIds =
        m_boundaryString.substr(first + 1, last - first - 1);
    const bool parseGood =
        ParseUtils::GenerateSeqVector(boundaryIds, m_boundaryRegions);
    ASSERTL0(parseGood && !m_boundaryRegions.empty(),
             "Unable to read boundary regions for FilterAeroForces: " +
                 boundaryIds);
    std::set<unsigned int> uniqueBoundaries(m_boundaryRegions.begin(),
                                            m_boundaryRegions.end());
    ASSERTL0(uniqueBoundaries.size() == m_boundaryRegions.size(),
             "AeroForces boundary selection contains duplicate IDs.");

    const auto &bndConditions = m_velocityExp->GetBndConditions();
    m_boundaryRegionIsInList.assign(bndConditions.size(), false);
    SpatialDomains::BoundaryConditions bcs(m_session,
                                           m_velocityExp->GetGraph());
    const auto &regions = bcs.GetBoundaryRegions();
    ASSERTL0(regions.size() == bndConditions.size(),
             "Boundary region and boundary condition counts differ.");
    // The regions are those of this rank's partition, which need not hold
    // every selected region, so a selected identifier is checked to exist on
    // some rank.
    std::vector<int> found(m_boundaryRegions.size(), 0);
    unsigned int region = 0;
    for (const auto &[id, regionDefinition] : regions)
    {
        static_cast<void>(regionDefinition);
        const auto selected =
            std::find(m_boundaryRegions.begin(), m_boundaryRegions.end(), id);
        m_boundaryRegionIsInList[region++] =
            selected != m_boundaryRegions.end();
        if (selected != m_boundaryRegions.end())
        {
            found[selected - m_boundaryRegions.begin()] = 1;
        }
    }
    m_session->GetComm()->AllReduce(found, LibUtilities::ReduceMax);
    for (size_t i = 0; i < m_boundaryRegions.size(); ++i)
    {
        ASSERTL0(found[i] != 0, "AeroForces boundary region " +
                                    std::to_string(m_boundaryRegions[i]) +
                                    " does not exist.");
    }

    ASSERTL0(velocityView->components.size() == m_spaceDim,
             "AeroForces requires one velocity component per dimension.");
    ASSERTL0(pressureView->components.size() == 1,
             "AeroForces requires exactly one pressure component.");
    m_aeroForcesOp = AeroForcesOp<double>::Create(
        m_velocityExp, ComponentNames(velocityView->components));
    m_aeroForcesOp->Configure(
        ComponentIndices(velocityView->components), m_pressureExp,
        pressureView->components.front().index, m_boundaryRegionIsInList, m_rho,
        m_mu, m_pivotPoint);

    m_pressureForce.assign(m_spaceDim, 0.0);
    m_viscousForce.assign(m_spaceDim, 0.0);
    m_pressureMoment.assign(m_momentDim, 0.0);
    m_viscousMoment.assign(m_momentDim, 0.0);

    m_writer = std::make_unique<RootTableWriter>(m_session->GetComm());
    m_writer->Open(m_outputFile);
    WriteHeader(boundaryIds);
    v_Apply(context);
}

void FilterAeroForces::v_Apply(FilterContext &context)
{
    if (!m_schedule.Due(context))
    {
        return;
    }

    CalculateForces(context);
    WriteForces(context.time);
}

void FilterAeroForces::CalculateForces(FilterContext &context)
{
    auto velocityView = context.fields.GetRole(FieldRoles::Velocity);
    auto pressureView = context.fields.GetRole(FieldRoles::Pressure);
    ASSERTL0(velocityView && pressureView,
             "AeroForces velocity or pressure field view is unavailable.");

    auto &gradient =
        context.derived.GetGradient(*velocityView, m_executionSpace);
    (*m_aeroForcesOp)(pressureView->field.get(), gradient);
    if (!m_writer->IsWriter())
    {
        return;
    }
    const auto pressureForce  = m_aeroForcesOp->GetPressureForce();
    const auto viscousForce   = m_aeroForcesOp->GetViscousForce();
    const auto pressureMoment = m_aeroForcesOp->GetPressureMoment();
    const auto viscousMoment  = m_aeroForcesOp->GetViscousMoment();
    std::copy_n(pressureForce.begin(), m_spaceDim, m_pressureForce.begin());
    std::copy_n(viscousForce.begin(), m_spaceDim, m_viscousForce.begin());
    // In 2D the moment is the z component, the only nonzero one.
    const unsigned int firstMoment = m_spaceDim == 2 ? 2 : 0;
    std::copy_n(pressureMoment.begin() + firstMoment, m_momentDim,
                m_pressureMoment.begin());
    std::copy_n(viscousMoment.begin() + firstMoment, m_momentDim,
                m_viscousMoment.begin());

    std::vector<double> projectedPressure(m_spaceDim, 0.0);
    std::vector<double> projectedViscous(m_spaceDim, 0.0);
    for (unsigned int i = 0; i < m_spaceDim; ++i)
    {
        for (unsigned int j = 0; j < m_spaceDim; ++j)
        {
            projectedPressure[i] += m_pressureForce[j] * m_directions[i][j];
            projectedViscous[i] += m_viscousForce[j] * m_directions[i][j];
        }
    }
    m_pressureForce = std::move(projectedPressure);
    m_viscousForce  = std::move(projectedViscous);

    if (m_spaceDim == 3)
    {
        projectedPressure.assign(3, 0.0);
        projectedViscous.assign(3, 0.0);
        for (unsigned int i = 0; i < 3; ++i)
        {
            for (unsigned int j = 0; j < 3; ++j)
            {
                projectedPressure[i] +=
                    m_pressureMoment[j] * m_directions[i][j];
                projectedViscous[i] += m_viscousMoment[j] * m_directions[i][j];
            }
        }
        m_pressureMoment = std::move(projectedPressure);
        m_viscousMoment  = std::move(projectedViscous);
    }
}

// The column-name row is not commented, so that the table reads directly with
// "df = pd.read_csv(filename, sep=r"\s+", comment="#")".
void FilterAeroForces::WriteHeader(const std::string &boundaryString)
{
    if (!m_writer->IsWriter())
    {
        return;
    }
    auto &stream = m_writer->Stream();
    stream << "# Forces and moments acting on bodies";
    m_writer->EndRow();

    for (unsigned int i = 0; i < m_spaceDim; ++i)
    {
        stream << "# Direction" << i + 1 << " = (";

        for (unsigned int j = 0; j < 3; ++j)
        {
            if (j > 0)
            {
                stream << " ";
            }

            stream << std::setprecision(4) << m_directions[i][j];
        }

        stream << ")";
        m_writer->EndRow();
    }

    stream << "# Moments around (";

    for (unsigned int i = 0; i < m_pivotPoint.size(); ++i)
    {
        if (i > 0)
        {
            stream << " ";
        }

        stream << std::setprecision(4) << m_pivotPoint[i];
    }

    stream << ")";
    m_writer->EndRow();
    stream << "# Boundary regions: " << boundaryString;
    m_writer->EndRow();

    // Column names: deliberately NOT prefixed with '#'
    stream << "Time";

    for (unsigned int i = 1; i <= m_spaceDim; ++i)
    {
        stream << " F" << i << "-press" << " F" << i << "-visc" << " F" << i
               << "-total";
    }

    if (m_spaceDim == 2)
    {
        stream << " M3-press" << " M3-visc" << " M3-total";
    }
    else
    {
        for (unsigned int i = 1; i <= 3; ++i)
        {
            stream << " M" << i << "-press" << " M" << i << "-visc" << " M" << i
                   << "-total";
        }
    }

    m_writer->EndRow();
    stream << std::scientific << std::setprecision(16);
}

void FilterAeroForces::WriteForces(const double time)
{
    if (!m_writer->IsWriter())
    {
        return;
    }

    // Set scientific notation for numeric force outputs
    auto &stream = m_writer->Stream();
    stream << std::scientific << std::setprecision(16) << time;

    for (unsigned int i = 0; i < m_spaceDim; ++i)
    {
        stream << " " << m_pressureForce[i] << " " << m_viscousForce[i] << " "
               << m_pressureForce[i] + m_viscousForce[i];
    }

    for (unsigned int i = 0; i < m_momentDim; ++i)
    {
        stream << " " << m_pressureMoment[i] << " " << m_viscousMoment[i] << " "
               << m_pressureMoment[i] + m_viscousMoment[i];
    }

    m_writer->EndRow();
}

void FilterAeroForces::v_Finalise([[maybe_unused]] FilterContext &context)
{
    if (m_writer)
    {
        m_writer->Close();
    }
}

} // namespace Nektar::SolverCore
