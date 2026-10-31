///////////////////////////////////////////////////////////////////////////////
//
// File: FilterMean.cpp
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
// Description: Outputs the domain average of every variable during
// time-stepping.
//
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <iomanip>

#include <SolverCore/Filters/FilterMean.h>

namespace Nektar::SolverCore
{

std::string FilterMean::className =
    GetFilterFactory().RegisterCreatorFunction("Mean", FilterMean::create);

/**
 * @brief Configure the Mean filter.
 *
 * Recognised parameters are `OutputFile` (default: the session name, with
 * extension `.avg`), the required `OutputFrequency` in steps, and `StartTime`.
 */
FilterMean::FilterMean(const LibUtilities::SessionReaderSharedPtr &session,
                       const ParamMap &params)
    : Filter(session)
{
    m_outputFile = Filter::SetupOutput(".avg", params);
    FilterParameters parameters(session, params, "FilterMean");
    parameters.Consume("OutputFile");
    m_outputFrequency = parameters.PositiveFrequency("OutputFrequency", 0);
    m_schedule = StepSchedule(m_outputFrequency, true, parameters.StartTime());
    parameters.ValidateAllConsumed();
}

/**
 * @brief Create one MeanOp per canonical physical group, record the domain
 * volume and write the table header and the initial averages.
 *
 * The physical groups visit every solver variable exactly once, including
 * pressure for the incompressible solver, and each group is integrated with
 * its own expansion.
 */
void FilterMean::v_Initialise(FilterContext &context)
{
    ASSERTL0(!m_session->DefinesSolverInfo("HOMOGENEOUS"),
             "Mean does not support homogeneous expansions.");

    const auto &groups = context.fields.GetPhysicalGroups();
    ASSERTL0(!groups.empty(), "Mean found no physical fields.");

    m_meanOps.clear();
    m_componentCounts.clear();
    m_variables.clear();
    for (const auto &view : groups)
    {
        ASSERTL0(view.spaceDimension == 2 || view.spaceDimension == 3,
                 "Mean supports only 2D and 3D fields.");
        ASSERTL0(view.expansion->GetShapeDimension() == view.spaceDimension,
                 "Mean does not support embedded-manifold expansions.");
        ASSERTL0(view.field.get().GetNumHomoModes() == 1,
                 "Mean does not support homogeneous expansions.");
        ASSERTL0(view.field.get().GetNumComponents() == view.components.size(),
                 "Mean physical field component mismatch.");

        const auto names = ComponentNames(view.components);
        m_meanOps.push_back(MeanOp<double>::Create(view.expansion, names));
        m_componentCounts.push_back(names.size());
        m_variables.insert(m_variables.end(), names.begin(), names.end());
    }
    m_means.assign(m_variables.size(), 0.0);

    // ComputeMeans() reads the averages on the writer rank only.
    m_writer = std::make_unique<RootTableWriter>(m_session->GetComm());
    m_writer->Open(m_outputFile);
    ComputeMeans(context);
    WriteHeader(groups.front().spaceDimension);

    if (m_schedule.Due(context))
    {
        WriteMeans(context.time);
    }
}

void FilterMean::v_Apply(FilterContext &context)
{
    if (!m_schedule.Due(context))
    {
        return;
    }

    ComputeMeans(context);
    WriteMeans(context.time);
}

/**
 * @brief Evaluate the domain average of every variable.
 *
 * Collective: every rank must call this so each operator's reduction
 * completes, although only rank zero receives the averages and the volume.
 */
void FilterMean::ComputeMeans(FilterContext &context)
{
    const auto &groups = context.fields.GetPhysicalGroups();
    ASSERTL0(groups.size() == m_meanOps.size(),
             "Mean physical field count changed.");

    size_t offset = 0;
    for (size_t group = 0; group < groups.size(); ++group)
    {
        ASSERTL0(groups[group].components.size() == m_componentCounts[group],
                 "Mean physical field components changed.");
        auto &meanOp = *m_meanOps[group];
        meanOp(groups[group].field.get());
        if (m_writer->IsWriter())
        {
            const auto means = meanOp.GetMeans();
            std::copy(means.begin(), means.end(), m_means.begin() + offset);
            // Every group spans the same domain, so the volume is the first's.
            if (group == 0)
            {
                m_volume = meanOp.GetVolume();
            }
        }
        offset += m_componentCounts[group];
    }
}

// The header and column widths keep the table interchangeable with the
// legacy `.avg` format: `# Time <variables...> <measure> <value>`.
void FilterMean::WriteHeader(const unsigned int spaceDimension)
{
    if (!m_writer->IsWriter())
    {
        return;
    }

    auto &stream = m_writer->Stream();
    stream << "# Time";
    for (const auto &variable : m_variables)
    {
        stream << std::setw(22) << variable;
    }
    stream << std::setw(22) << (spaceDimension == 2 ? "area" : "volume") << " "
           << std::setprecision(11) << m_volume;
    m_writer->EndRow();
}

void FilterMean::WriteMeans(const double time)
{
    if (!m_writer->IsWriter())
    {
        return;
    }

    auto &stream = m_writer->Stream();
    stream << std::setw(17) << std::setprecision(8) << time;
    for (const auto mean : m_means)
    {
        stream << std::setw(22) << std::setprecision(11) << mean;
    }
    m_writer->EndRow();
}

void FilterMean::v_Finalise([[maybe_unused]] FilterContext &context)
{
    if (m_writer)
    {
        m_writer->Close();
    }
}

} // namespace Nektar::SolverCore
