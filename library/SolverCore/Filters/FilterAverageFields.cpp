///////////////////////////////////////////////////////////////////////////////
//
// File: FilterAverageFields.cpp
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
// Description: Outputs time-averaged solution fields during time-stepping.
//
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <charconv>
#include <cmath>
#include <exception>
#include <iostream>
#include <system_error>
#include <unordered_map>

#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <MultiRegions/Common/Operator.hpp>
#include <SolverCore/Filters/FilterAverageFields.h>

namespace Nektar::SolverCore
{

namespace
{
/// Parse @p value, the text of the quantity @p what, as an unsigned integer.
unsigned int ParseUnsigned(const std::string &value, const std::string &what)
{
    unsigned int result  = 0;
    const char *last     = value.data() + value.size();
    const auto [ptr, ec] = std::from_chars(value.data(), last, result);
    ASSERTL0(!value.empty() && ec == std::errc() && ptr == last,
             "AverageFields " + what + " must be an unsigned integer.");
    return result;
}

void ValidateRestartTime(const LibUtilities::FieldMetaDataMap &metadata,
                         const std::string &name)
{
    const auto it = metadata.find(name);
    if (it == metadata.end())
    {
        return;
    }
    size_t used  = 0;
    double value = 0.0;
    try
    {
        value = std::stod(it->second, &used);
    }
    catch (const std::exception &)
    {
        NEKERROR(ErrorUtil::efatal, "AverageFields restart metadata '" + name +
                                        "' must be finite.");
    }
    ASSERTL0(used == it->second.size() && std::isfinite(value),
             "AverageFields restart metadata '" + name + "' must be finite.");
}

std::vector<unsigned int> ElementModes(
    const LibUtilities::FieldDefinitionsSharedPtr &definition,
    const size_t element)
{
    const size_t dimension = definition->m_basis.size();
    const size_t offset    = definition->m_uniOrder ? 0 : element * dimension;
    ASSERTL0(offset + dimension <= definition->m_numModes.size(),
             "AverageFields restart has malformed mode metadata.");
    return {definition->m_numModes.begin() + offset,
            definition->m_numModes.begin() + offset + dimension};
}

bool FieldLayoutsMatch(const LibUtilities::FieldDefinitionsSharedPtr &expected,
                       const LibUtilities::FieldDefinitionsSharedPtr &actual)
{
    if (expected->m_shapeType != actual->m_shapeType ||
        expected->m_basis != actual->m_basis ||
        expected->m_numHomogeneousDir != actual->m_numHomogeneousDir ||
        expected->m_elementIDs.size() != actual->m_elementIDs.size())
    {
        return false;
    }

    std::unordered_map<unsigned int, size_t> actualElements;
    for (size_t i = 0; i < actual->m_elementIDs.size(); ++i)
    {
        actualElements.emplace(actual->m_elementIDs[i], i);
    }
    for (size_t i = 0; i < expected->m_elementIDs.size(); ++i)
    {
        const auto found = actualElements.find(expected->m_elementIDs[i]);
        if (found == actualElements.end() ||
            ElementModes(expected, i) != ElementModes(actual, found->second))
        {
            return false;
        }
    }
    return true;
}
} // namespace

std::string FilterAverageFields::cmdSetStartFilterFileNum =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "set-filter-averagefield-start-number", "",
        "Set the starting number of the average field filter file number.");

std::string FilterAverageFields::className =
    GetFilterFactory().RegisterCreatorFunction("AverageFields",
                                               FilterAverageFields::create);

/**
 * @class FilterAverageFields
 *
 * @brief Time averages of all solver variables.
 *
 * Every sampled step adds the output coefficients of each coefficient group
 * to an unscaled accumulator in the operator execution space. A written file
 * holds the accumulators scaled by one over the number of samples.
 */

/**
 * @brief Configure the AverageFields filter.
 *
 * Recognised parameters are `OutputFile` (default: the session name),
 * `OutputFrequency` in steps (default: `NumSteps`), `SampleFrequency` in
 * steps (default: one), `StartTime`, `RestartFile`, and `PhaseAverage` with
 * `PhaseAveragePeriod` and `PhaseAveragePhase`. The final file is
 * `<OutputFile>_avg.fld` and periodic files are `<OutputFile>_<k>_avg.fld`,
 * k = 0, 1, ...
 */
FilterAverageFields::FilterAverageFields(
    const LibUtilities::SessionReaderSharedPtr &session, const ParamMap &params)
    : Filter(session),
      m_executionSpace(MultiRegions::Operator<double>::GetOpExecSpace(session)),
      m_math(m_executionSpace)
{
    m_outputFile = Filter::SetupOutput(".fld", params);
    FilterParameters parameters(session, params, "FilterAverageFields");
    parameters.Consume("OutputFile");
    unsigned int defaultOutputFrequency = 0;
    if (!parameters.Has("OutputFrequency"))
    {
        m_session->LoadParameter("NumSteps", defaultOutputFrequency);
    }
    m_outputFrequency =
        parameters.PositiveFrequency("OutputFrequency", defaultOutputFrequency);
    m_sampleFrequency      = parameters.PositiveFrequency("SampleFrequency", 1);
    const double startTime = parameters.StartTime();
    m_sampleSchedule       = StepSchedule(m_sampleFrequency, false, startTime);
    m_outputSchedule       = StepSchedule(m_outputFrequency, false, startTime);

    m_restartFile = parameters.OptionalString("RestartFile");
    if (!m_restartFile.empty() && !fs::path(m_restartFile).has_extension())
    {
        m_restartFile += ".fld";
    }

    m_phaseSample = parameters.StrictBoolean("PhaseAverage", false);

    if (m_phaseSample)
    {
        ASSERTL0(parameters.Has("PhaseAveragePeriod") &&
                     parameters.Has("PhaseAveragePhase"),
                 "Phase averaging requires PhaseAveragePeriod and "
                 "PhaseAveragePhase.");
        m_phaseSamplePeriod = parameters.Finite("PhaseAveragePeriod", 0.0);
        m_phaseSamplePhase  = parameters.Finite("PhaseAveragePhase", 0.0);
        ASSERTL0(m_phaseSamplePeriod > 0.0,
                 "PhaseAveragePeriod must be greater than 0.");
        ASSERTL0(m_phaseSamplePhase >= 0.0 && m_phaseSamplePhase <= 1.0,
                 "PhaseAveragePhase must be between 0 and 1.");

        const double dt  = m_session->GetParameter("TimeStep");
        m_phaseTolerance = dt * m_sampleFrequency / (2.0 * m_phaseSamplePeriod);
    }
    else
    {
        ASSERTL0(!parameters.Has("PhaseAveragePeriod") &&
                     !parameters.Has("PhaseAveragePhase"),
                 "PhaseAveragePeriod and PhaseAveragePhase require "
                 "PhaseAverage=true.");
    }
    parameters.ValidateAllConsumed();
    m_series = std::make_unique<OutputSeries>(m_outputFile);
}

/**
 * @brief Allocate an accumulator for every coefficient group, load the
 * restart file if one is given and set the first file number.
 */
void FilterAverageFields::v_Initialise(FilterContext &context)
{
    m_numSamples = 0;

    m_sourceFields = context.fields.GetCoefficientGroups();
    ASSERTL0(!m_sourceFields.empty(),
             "FilterAverageFields found no fields to average.");

    m_accumulators.clear();
    m_accumulators.reserve(m_sourceFields.size());
    for (const auto &view : m_sourceFields)
    {
        auto &source = view.field.get();
        ASSERTL0(source.GetNumHomoModes() == 1,
                 "AverageFields does not support homogeneous expansions.");
        auto blocks =
            MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
                view.expansion);
        LibUtilities::Field<double, FieldState::Coeff> accumulator(
            "average " + source.GetName(), blocks, source.GetNumComponents(),
            source.GetNumHomoModes());
        accumulator.SetInterleaveWidth(source);
        m_math.zero(accumulator);
        m_accumulators.emplace_back(std::move(accumulator));
    }

    m_fieldMetaData = context.output.getMetadata();
    m_fieldMetaData.erase("FilterFileNum");
    m_fieldMetaData["InitialTime"] = std::to_string(context.time);

    if (!m_restartFile.empty())
    {
        LoadRestartFile();
    }

    if (m_session->DefinesCmdLineArgument(
            "set-filter-averagefield-start-number"))
    {
        m_series->SetIndex(
            ParseUnsigned(m_session->GetCmdLineArgument<std::string>(
                              "set-filter-averagefield-start-number"),
                          "argument set-filter-averagefield-start-number"));
    }
}

/**
 * @brief On a step due for sampling, add a sample when it is in phase, or
 * always without phase averaging, and, on a step that is also due for output,
 * write the numbered file once there is a sample.
 */
void FilterAverageFields::v_Apply(FilterContext &context)
{
    if (!m_sampleSchedule.Due(context))
    {
        return;
    }

    if (!m_phaseSample || IsPhaseSample(context.time))
    {
        AccumulateSample(context);
        ++m_numSamples;
    }

    if (m_outputSchedule.Due(context) && m_numSamples > 0)
    {
        WriteAverage(context, true);
    }
}

/**
 * @brief Write the final average, zero-valued with a warning if no sample was
 * taken.
 */
void FilterAverageFields::v_Finalise(FilterContext &context)
{
    if (m_numSamples == 0)
    {
        if (m_session->GetComm()->GetRank() == 0)
        {
            std::cerr << "WARNING: AverageFields produced no samples; "
                         "writing a zero-valued final average."
                      << std::endl;
        }
    }
    WriteAverage(context);
}

/**
 * @brief Whether @p time lies within the phase tolerance of the requested
 * phase of the period.
 */
bool FilterAverageFields::IsPhaseSample(const double time) const
{
    const double currentPhase =
        time / m_phaseSamplePeriod - std::floor(time / m_phaseSamplePeriod);
    const double relativePhase = std::abs(m_phaseSamplePhase - currentPhase);
    return relativePhase < m_phaseTolerance ||
           std::abs(relativePhase - 1.0) < m_phaseTolerance;
}

/**
 * @brief Add the current output coefficients to the accumulators.
 *
 * An accumulator block at another interleave width than its source is first
 * reshaped to the source's, since the sum is formed entry by entry.
 */
void FilterAverageFields::AccumulateSample(FilterContext &context)
{
    // Taking the views again lets the equation system bring its output
    // coefficients up to date first.
    auto sourceFields = context.fields.GetCoefficientGroups();
    ASSERTL1(sourceFields.size() == m_accumulators.size(),
             "AverageFields output field count changed during integration.");

    for (size_t i = 0; i < sourceFields.size(); ++i)
    {
        auto &source      = sourceFields[i].field.get();
        auto &accumulator = m_accumulators[i];
        ASSERTL1(source.size() == accumulator.size(),
                 "AverageFields output field layout changed during "
                 "integration.");
        MatchInterleaveWidths(accumulator, source, m_executionSpace);
        m_math.daxpy(1.0, source, accumulator, accumulator);
    }
}

/**
 * @brief Write the accumulators scaled by one over the number of samples, to
 * the next numbered file if @p numbered, else to the final file.
 */
void FilterAverageFields::WriteAverage(FilterContext &context,
                                       const bool numbered)
{
    CoefficientFieldViews fields;
    fields.reserve(m_accumulators.size());
    for (size_t i = 0; i < m_accumulators.size(); ++i)
    {
        fields.push_back({m_accumulators[i], m_sourceFields[i].expansion,
                          m_sourceFields[i].spaceDimension,
                          m_sourceFields[i].components});
    }

    m_fieldMetaData["FinalTime"]          = std::to_string(context.time);
    m_fieldMetaData["Time"]               = std::to_string(context.time);
    m_fieldMetaData["NumberOfFieldDumps"] = std::to_string(m_numSamples);

    const fs::path outputPath(m_outputFile);
    const std::string ext = outputPath.extension().string();
    fs::path filename;
    if (!numbered)
    {
        filename = outputPath.parent_path() /
                   (outputPath.stem().string() + "_avg" + ext);
    }
    else
    {
        const auto index                 = m_series->GetIndex();
        m_fieldMetaData["FilterFileNum"] = std::to_string(index);
        fs::path numberedPath(m_series->Next());
        filename = numberedPath.parent_path() /
                   (numberedPath.stem().string() + "_avg" + ext);
    }

    const double scale =
        m_numSamples == 0 ? 1.0 : 1.0 / static_cast<double>(m_numSamples);
    context.output.writeOutputFields(SetupOutput(ext, filename.string()),
                                     fields, m_fieldMetaData, scale);
}

/**
 * @brief Load an earlier average into the accumulators, multiplied by its
 * sample count, after checking it against the solver's fields.
 */
void FilterAverageFields::LoadRestartFile()
{
    std::vector<LibUtilities::FieldDefinitionsSharedPtr> fieldDef;
    std::vector<std::vector<double>> fieldData;
    LibUtilities::FieldMetaDataMap restartMetadata;
    auto fieldIo =
        LibUtilities::FieldIO::CreateForFile(m_session, m_restartFile);
    fieldIo->Import(m_restartFile, fieldDef, fieldData, restartMetadata);
    ASSERTL0(fieldDef.size() == fieldData.size() && !fieldDef.empty(),
             "AverageFields restart file has inconsistent field data.");
    ValidateRestartTime(restartMetadata, "InitialTime");
    ValidateRestartTime(restartMetadata, "FinalTime");
    ValidateRestartTime(restartMetadata, "Time");

    if (restartMetadata.count("NumberOfFieldDumps") > 0)
    {
        m_numSamples = ParseUnsigned(restartMetadata.at("NumberOfFieldDumps"),
                                     "restart metadata 'NumberOfFieldDumps'");
    }
    else
    {
        m_numSamples = 1;
    }
    ASSERTL0(m_numSamples > 0,
             "AverageFields restart file contains no field samples.");

    if (restartMetadata.count("InitialTime") > 0)
    {
        m_fieldMetaData["InitialTime"] = restartMetadata.at("InitialTime");
    }
    if (restartMetadata.count("FilterFileNum") > 0)
    {
        const auto restartIndex =
            ParseUnsigned(restartMetadata.at("FilterFileNum"),
                          "restart metadata 'FilterFileNum'");
        m_series->SetIndex(restartIndex + 1);
    }

    for (size_t i = 0; i < m_accumulators.size(); ++i)
    {
        const auto variables = ComponentNames(m_sourceFields[i].components);
        const auto ncoeffs =
            static_cast<size_t>(m_sourceFields[i].expansion->GetNcoeffs());
        std::vector<double> allCoeffs(ncoeffs * variables.size(), 0.0);

        for (size_t nc = 0; nc < variables.size(); ++nc)
        {
            std::vector<double> component(ncoeffs, 0.0);
            std::string variable = variables[nc];
            bool foundVariable   = false;
            const auto expectedDefinitions =
                m_sourceFields[i].expansion->GetFieldDefinitions();
            for (size_t j = 0; j < fieldDef.size(); ++j)
            {
                const bool containsVariable =
                    std::find(fieldDef[j]->m_fields.begin(),
                              fieldDef[j]->m_fields.end(),
                              variable) != fieldDef[j]->m_fields.end();
                if (containsVariable)
                {
                    const bool layoutMatches = std::any_of(
                        expectedDefinitions.begin(), expectedDefinitions.end(),
                        [&](const auto &expected) {
                            return FieldLayoutsMatch(expected, fieldDef[j]);
                        });
                    ASSERTL0(layoutMatches,
                             "AverageFields restart layout for variable '" +
                                 variable + "' does not match the solver.");
                    foundVariable = true;
                    m_sourceFields[i].expansion->ExtractDataToCoeffs(
                        fieldDef[j], fieldData[j], variable, component);
                }
            }
            ASSERTL0(foundVariable,
                     "AverageFields restart file is missing variable '" +
                         variable + "'.");
            std::transform(
                component.begin(), component.end(),
                allCoeffs.begin() + nc * ncoeffs,
                [this](const double value) { return m_numSamples * value; });
        }

        m_accumulators[i].CopyVector<NektarSpaces::HostSpace>(allCoeffs);
    }
}

} // namespace Nektar::SolverCore
