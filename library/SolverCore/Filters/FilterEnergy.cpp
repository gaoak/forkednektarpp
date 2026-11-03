///////////////////////////////////////////////////////////////////////////////
//
// File: FilterEnergy.cpp
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
// Description: Outputs kinetic energy and enstrophy during time-stepping.
//
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <iomanip>

#include <LibUtilities/BasicUtils/Equation.h>
#include <SolverCore/Filters/FilterEnergy.h>

namespace Nektar::SolverCore
{

std::string FilterEnergy::className =
    GetFilterFactory().RegisterCreatorFunction("Energy", FilterEnergy::create);

FilterEnergy::FilterEnergy(const LibUtilities::SessionReaderSharedPtr &session,
                           const ParamMap &params)
    : Filter(session)
{
    m_outputFile = Filter::SetupOutput(".eny", params);
    FilterParameters parameters(session, params, "FilterEnergy");
    parameters.Consume("OutputFile");
    m_outputFrequency = parameters.PositiveFrequency("OutputFrequency", 0);
    m_schedule = StepSchedule(m_outputFrequency, true, parameters.StartTime());
    parameters.ValidateAllConsumed();
    m_executionSpace = MultiRegions::Operator<double>::GetOpExecSpace(session);
}

void FilterEnergy::v_Initialise(FilterContext &context)
{
    auto velocityView = context.fields.GetRole(FieldRoles::Velocity);
    ASSERTL0(velocityView, "Energy requires a velocity field view.");
    ASSERTL0(velocityView->spaceDimension == 2 ||
                 velocityView->spaceDimension == 3,
             "Energy supports only 2D and 3D velocity fields.");
    ASSERTL0(velocityView->expansion->GetShapeDimension() ==
                 velocityView->spaceDimension,
             "Energy does not support embedded-manifold expansions.");
    ASSERTL0(velocityView->components.size() == velocityView->spaceDimension,
             "Energy requires one velocity component per dimension.");
    ASSERTL0(velocityView->field.get().GetNumHomoModes() == 1,
             "Energy does not support homogeneous expansions.");

    m_energyOp = EnergyOp<double>::Create(
        velocityView->expansion, ComponentNames(velocityView->components));
    m_energyOp->Configure(ComponentIndices(velocityView->components));

    m_writer = std::make_unique<RootTableWriter>(m_session->GetComm());
    m_writer->Open(m_outputFile);
    WriteHeader();
    v_Apply(context);
}

void FilterEnergy::v_Apply(FilterContext &context)
{
    if (!m_schedule.Due(context))
    {
        return;
    }

    auto velocityView = context.fields.GetRole(FieldRoles::Velocity);
    ASSERTL0(velocityView, "Energy velocity field view is unavailable.");

    auto &gradient =
        context.derived.GetGradient(*velocityView, m_executionSpace);
    (*m_energyOp)(velocityView->field.get(), gradient);
    if (!m_writer->IsWriter())
    {
        return;
    }
    // The operator returns the raw integrals of |u|^2 and |omega|^2; the
    // table holds half of each, averaged over the domain.
    const double volume = m_energyOp->GetVolume();
    ASSERTL0(volume > 0.0, "Energy encountered a non-positive domain volume.");
    const double kineticEnergy =
        m_energyOp->GetKineticEnergy() / (2.0 * volume);
    const double enstrophy = m_energyOp->GetEnstrophy() / (2.0 * volume);

    WriteEnergy(context.time, kineticEnergy, enstrophy);
}

// The uncommented column-name row makes the output directly readable with:
// "df = pd.read_csv(filename, sep=r"\s+", comment="#")"
void FilterEnergy::WriteHeader()
{
    m_writer->Metadata("Volume-averaged kinetic energy and enstrophy (2D/3D)");
    m_writer->Columns({"Time", "KineticEnergy", "Enstrophy"});
}

void FilterEnergy::WriteEnergy(const double time, const double kineticEnergy,
                               const double enstrophy)
{
    if (!m_writer->IsWriter())
    {
        return;
    }

    m_writer->Stream() << time << " " << kineticEnergy << " " << enstrophy;
    m_writer->EndRow();
}

void FilterEnergy::v_Finalise([[maybe_unused]] FilterContext &context)
{
    if (m_writer)
    {
        m_writer->Close();
    }
}

} // namespace Nektar::SolverCore
