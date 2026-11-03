///////////////////////////////////////////////////////////////////////////////
//
// File: FilterReynoldsStresses.cpp
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
// Description: Outputs time-averaged fields and Reynolds stresses.
//
///////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <MultiRegions/Common/Operator.hpp>
#include <SolverCore/Filters/FilterReynoldsStresses.h>

namespace Nektar::SolverCore
{

std::string FilterReynoldsStresses::className =
    GetFilterFactory().RegisterCreatorFunction("ReynoldsStresses",
                                               FilterReynoldsStresses::create);

/**
 * @class FilterReynoldsStresses
 *
 * @brief Time averages of all solver variables and the covariances
 * \f$ \overline{u_a' u_b'} \f$ of the velocity components.
 *
 * Every sampled step adds the coefficients of each coefficient group to an
 * accumulator, as AverageFields does, and adds the pointwise products
 * \f$ u_a u_b \f$, \f$ a \le b \f$, of the velocity at the quadrature points
 * to a physical-space accumulator \f$ S_{ab} \f$. When a file is written,
 * the summed velocity \f$ U_a = \sum_n u_a \f$ is transformed back to
 * physical space and the covariance is formed from the two moments,
 *
 * \f[ \overline{u_a' u_b'} = \frac{1}{N} \left( S_{ab} -
 *     \frac{U_a U_b}{N} \right), \f]
 *
 * then projected element by element onto the velocity expansion. The written
 * variables are the solver variables followed by uu, uv, uw, vv, vw, ww
 * (uu, uv, vv in two dimensions).
 */

/**
 * @brief Configure the ReynoldsStresses filter.
 *
 * Recognised parameters are `OutputFile` (default: the session name),
 * `OutputFrequency` in steps (default: `NumSteps`), `SampleFrequency` in
 * steps (default: one) and `StartTime`. The final file is
 * `<OutputFile>_stress.fld` and periodic files are
 * `<OutputFile>_<k>_stress.fld`, k = 0, 1, ...
 */
FilterReynoldsStresses::FilterReynoldsStresses(
    const LibUtilities::SessionReaderSharedPtr &session, const ParamMap &params)
    : Filter(session),
      m_executionSpace(MultiRegions::Operator<double>::GetOpExecSpace(session)),
      m_math(m_executionSpace)
{
    const std::vector<std::pair<std::string, std::string>> unsupported = {
        {"RestartFile", "restarting from an earlier average"},
        {"MovingAverage", "exponential moving averages"},
        {"alpha", "exponential moving averages"},
        {"tau", "exponential moving averages"},
        {"ScaleNumModes", "sampling on a scaled expansion"},
        {"PhaseAverage", "phase averaging"},
        {"PhaseAveragePeriod", "phase averaging"},
        {"PhaseAveragePhase", "phase averaging"},
        {"Modules", "FieldConvert modules"},
        {"options", "FieldConvert options"}};
    for (const auto &[name, feature] : unsupported)
    {
        ASSERTL0(params.count(name) == 0,
                 "FilterReynoldsStresses does not support " + feature +
                     " (parameter '" + name + "').");
    }

    m_outputFile = Filter::SetupOutput(".fld", params);
    FilterParameters parameters(session, params, "FilterReynoldsStresses");
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
    parameters.ValidateAllConsumed();
    m_series = std::make_unique<OutputSeries>(m_outputFile);
}

/**
 * @brief Allocate the accumulators, the output workspace and the operators.
 *
 * The coefficient accumulators follow the registry's coefficient groups. The
 * velocity is taken from the velocity role, whose components must appear at
 * the same indices in one coefficient group on the same expansion, so that
 * the summed coefficients of that group transform back to the summed
 * velocity. The output workspace is allocated here, so that writing a file
 * neither allocates nor frees memory.
 */
void FilterReynoldsStresses::v_Initialise(FilterContext &context)
{
    ASSERTL0(!m_session->DefinesSolverInfo("HOMOGENEOUS"),
             "ReynoldsStresses does not support homogeneous expansions.");

    auto velocityView = context.fields.GetRole(FieldRoles::Velocity);
    ASSERTL0(velocityView, "ReynoldsStresses requires a velocity field view.");
    ASSERTL0(velocityView->spaceDimension == 2 ||
                 velocityView->spaceDimension == 3,
             "ReynoldsStresses supports only 2D and 3D velocity fields.");
    ASSERTL0(velocityView->expansion->GetShapeDimension() ==
                 velocityView->spaceDimension,
             "ReynoldsStresses does not support embedded-manifold "
             "expansions.");
    ASSERTL0(velocityView->components.size() == velocityView->spaceDimension,
             "ReynoldsStresses requires one velocity component per "
             "dimension.");
    ASSERTL0(velocityView->field.get().GetNumHomoModes() == 1,
             "ReynoldsStresses does not support homogeneous expansions.");

    m_numSamples = 0;

    m_sourceFields = context.fields.GetCoefficientGroups();
    ASSERTL0(!m_sourceFields.empty(),
             "ReynoldsStresses found no fields to average.");

    // Locate the coefficient group holding the velocity, with the velocity
    // at the same component indices as in the physical field.
    const auto velocityNames   = ComponentNames(velocityView->components);
    const auto velocityIndices = ComponentIndices(velocityView->components);
    bool foundVelocity         = false;
    for (unsigned int i = 0; i < m_sourceFields.size() && !foundVelocity; ++i)
    {
        if (m_sourceFields[i].expansion != velocityView->expansion)
        {
            continue;
        }
        const auto names = ComponentNames(m_sourceFields[i].components);
        foundVelocity    = true;
        for (size_t c = 0; c < velocityNames.size(); ++c)
        {
            foundVelocity = foundVelocity &&
                            velocityIndices[c] < names.size() &&
                            names[velocityIndices[c]] == velocityNames[c];
        }
        m_velocityGroup = i;
    }
    ASSERTL0(foundVelocity, "ReynoldsStresses found no coefficient group "
                            "matching the velocity field layout.");

    m_accumulators.clear();
    m_accumulators.reserve(m_sourceFields.size());
    for (const auto &view : m_sourceFields)
    {
        auto &source = view.field.get();
        ASSERTL0(source.GetNumHomoModes() == 1,
                 "ReynoldsStresses does not support homogeneous expansions.");
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

    const auto &expansion     = velocityView->expansion;
    const auto &velocityGroup = m_sourceFields[m_velocityGroup];
    m_velocityExpansion       = expansion;
    m_productNames = ReynoldsStressOp<double>::ProductNames(velocityNames);
    const auto physBlocks =
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(expansion);
    const auto coeffBlocks =
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(expansion);
    m_products = LibUtilities::Field<double, FieldState::Phys>(
        "velocity products", physBlocks, m_productNames, 1);
    m_math.zero(m_products);
    m_velocitySum = LibUtilities::Field<double, FieldState::Phys>(
        "velocity sum",
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
            velocityGroup.expansion),
        m_accumulators[m_velocityGroup].GetNumComponents(), 1);
    m_covarianceSum = LibUtilities::Field<double, FieldState::Phys>(
        "covariance sum", physBlocks, m_productNames, 1);
    m_covarianceInnerProduct = LibUtilities::Field<double, FieldState::Coeff>(
        "covariance inner product", coeffBlocks, m_productNames, 1);
    m_stressSum = LibUtilities::Field<double, FieldState::Coeff>(
        "Reynolds stress sum", coeffBlocks, m_productNames, 1);
    // A write without samples writes the stresses as they are, zero.
    m_math.zero(m_stressSum);

    m_reynoldsStressOp =
        ReynoldsStressOp<double>::Create(expansion, velocityNames);
    m_reynoldsStressOp->Configure(velocityIndices);
    m_bwdTransOp = MultiRegions::BwdTransOp<double>::Create(
        velocityGroup.expansion, ComponentNames(velocityGroup.components));
    m_iProductWRTBaseOp = MultiRegions::IProductWRTBaseOp<double>::Create(
        expansion, m_productNames);
    m_multiplyByElmtInvMassOp =
        MultiRegions::MultiplyByElmtInvMassOp<double>::Create(expansion,
                                                              m_productNames);

    m_fieldMetaData = context.output.getMetadata();
    m_fieldMetaData.erase("FilterFileNum");
    m_fieldMetaData["InitialTime"] = std::to_string(context.time);
}

/**
 * @brief Sample on due steps and, on a sampled step that is also an output
 * step, write the numbered file.
 */
void FilterReynoldsStresses::v_Apply(FilterContext &context)
{
    if (!m_sampleSchedule.Due(context))
    {
        return;
    }

    AccumulateSample(context);
    ++m_numSamples;

    if (m_outputSchedule.Due(context))
    {
        WriteStresses(context, true);
    }
}

/**
 * @brief Write the final statistics, zero-valued with a warning if no sample
 * was taken.
 */
void FilterReynoldsStresses::v_Finalise(FilterContext &context)
{
    if (m_numSamples == 0 && m_session->GetComm()->GetRank() == 0)
    {
        std::cerr << "WARNING: ReynoldsStresses produced no samples; "
                     "writing zero-valued final statistics."
                  << std::endl;
    }
    WriteStresses(context);
}

/**
 * @brief Add the current coefficients and velocity products to the
 * accumulators.
 *
 * An accumulator block at another interleave width than its source is first
 * reshaped to the source's, since the sum is formed entry by entry;
 * ReynoldsStressOp lays the products out like the velocity itself.
 */
void FilterReynoldsStresses::AccumulateSample(FilterContext &context)
{
    auto sourceFields = context.fields.GetCoefficientGroups();
    ASSERTL1(sourceFields.size() == m_accumulators.size(),
             "ReynoldsStresses field count changed during integration.");
    auto velocityView = context.fields.GetRole(FieldRoles::Velocity);
    ASSERTL0(velocityView,
             "ReynoldsStresses velocity field view is unavailable.");

    for (size_t i = 0; i < sourceFields.size(); ++i)
    {
        auto &source      = sourceFields[i].field.get();
        auto &accumulator = m_accumulators[i];
        ASSERTL1(source.size() == accumulator.size(),
                 "ReynoldsStresses field layout changed during integration.");
        MatchInterleaveWidths(accumulator, source, m_executionSpace);
        m_math.daxpy(1.0, source, accumulator, accumulator);
    }

    m_reynoldsStressOp->SetScale(1.0);
    (*m_reynoldsStressOp)(velocityView->field.get(), m_products);
}

/**
 * @brief Write the averages and the Reynolds stresses, to the next numbered
 * file if @p numbered, else to the final file.
 *
 * The coefficient accumulators and the covariance sum
 * \f$ S_{ab} - U_a U_b / N \f$ are written with the common scale
 * \f$ 1/N \f$. The covariance is projected with the element-local
 * \f$ M_e^{-1} (\cdot, \phi) \f$, which needs no boundary conditions for the
 * product variables. The workspace is refilled in place.
 */
void FilterReynoldsStresses::WriteStresses(FilterContext &context,
                                           const bool numbered)
{
    const auto &velocityGroup = m_sourceFields[m_velocityGroup];
    const auto &expansion     = m_velocityExpansion;

    if (m_numSamples > 0)
    {
        (*m_bwdTransOp)(m_accumulators[m_velocityGroup], m_velocitySum);
        m_math.copy(m_products, m_covarianceSum);
        m_reynoldsStressOp->SetScale(-1.0 / static_cast<double>(m_numSamples));
        (*m_reynoldsStressOp)(m_velocitySum, m_covarianceSum);
        (*m_iProductWRTBaseOp)(m_covarianceSum, m_covarianceInnerProduct);
        (*m_multiplyByElmtInvMassOp)(m_covarianceInnerProduct, m_stressSum);
    }

    CoefficientFieldViews fields;
    fields.reserve(m_accumulators.size() + 1);
    for (size_t i = 0; i < m_accumulators.size(); ++i)
    {
        fields.push_back({m_accumulators[i], m_sourceFields[i].expansion,
                          m_sourceFields[i].spaceDimension,
                          m_sourceFields[i].components});
    }
    fields.push_back({m_stressSum, expansion, velocityGroup.spaceDimension,
                      MakeFieldComponents(m_productNames)});

    m_fieldMetaData["FinalTime"]          = std::to_string(context.time);
    m_fieldMetaData["Time"]               = std::to_string(context.time);
    m_fieldMetaData["NumberOfFieldDumps"] = std::to_string(m_numSamples);

    fs::path outputPath(m_outputFile);
    if (numbered)
    {
        m_fieldMetaData["FilterFileNum"] = std::to_string(m_series->GetIndex());
        outputPath                       = m_series->Next();
    }
    const std::string ext   = outputPath.extension().string();
    const fs::path filename = outputPath.parent_path() /
                              (outputPath.stem().string() + "_stress" + ext);

    const double scale =
        m_numSamples == 0 ? 1.0 : 1.0 / static_cast<double>(m_numSamples);
    context.output.writeOutputFields(SetupOutput(ext, filename.string()),
                                     fields, m_fieldMetaData, scale);
}

} // namespace Nektar::SolverCore
