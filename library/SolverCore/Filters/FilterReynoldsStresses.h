///////////////////////////////////////////////////////////////////////////////
//
// File: FilterReynoldsStresses.h
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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <LibUtilities/BasicUtils/Math/MathHelper.hpp>
#include <MultiRegions/ElmtOps/BwdTrans/BwdTransOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp>
#include <MultiRegions/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassOp.hpp>
#include <SolverCore/Filters/Filter.h>
#include <SolverCore/Filters/FilterUtils.h>
#include <SolverCore/ReynoldsStress/ReynoldsStressOp.hpp>

namespace Nektar::SolverCore
{

class FilterReynoldsStresses : public Filter
{
public:
    friend class MemoryManager<FilterReynoldsStresses>;

    static FilterSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &session,
        const ParamMap &params)
    {
        return MemoryManager<FilterReynoldsStresses>::AllocateSharedPtr(session,
                                                                        params);
    }

    static std::string className;

protected:
    SOLVER_CORE_EXPORT FilterReynoldsStresses(
        const LibUtilities::SessionReaderSharedPtr &session,
        const ParamMap &params);
    SOLVER_CORE_EXPORT ~FilterReynoldsStresses() override = default;

    SOLVER_CORE_EXPORT void v_Initialise(FilterContext &context) override;
    SOLVER_CORE_EXPORT void v_Apply(FilterContext &context) override;
    SOLVER_CORE_EXPORT void v_Finalise(FilterContext &context) override;

private:
    std::string m_executionSpace;
    Math::MathHelper m_math;
    CoefficientFieldViews m_sourceFields;
    std::vector<LibUtilities::Field<double, FieldState::Coeff>> m_accumulators;
    LibUtilities::Field<double, FieldState::Phys> m_products;
    /// Output workspace, allocated at initialisation and refilled at each
    /// write: the summed velocity at the quadrature points, the covariance
    /// sum \f$ S_{ab} - U_a U_b / N \f$, its inner products with the basis
    /// and its element-local projection.
    LibUtilities::Field<double, FieldState::Phys> m_velocitySum;
    LibUtilities::Field<double, FieldState::Phys> m_covarianceSum;
    LibUtilities::Field<double, FieldState::Coeff> m_covarianceInnerProduct;
    LibUtilities::Field<double, FieldState::Coeff> m_stressSum;
    LibUtilities::FieldMetaDataMap m_fieldMetaData;

    MultiRegions::ExpListSharedPtr m_velocityExpansion;
    unsigned int m_velocityGroup = 0;
    std::vector<std::string> m_productNames;
    std::shared_ptr<ReynoldsStressOp<double>> m_reynoldsStressOp;
    std::shared_ptr<MultiRegions::BwdTransOp<double>> m_bwdTransOp;
    std::shared_ptr<MultiRegions::IProductWRTBaseOp<double>>
        m_iProductWRTBaseOp;
    std::shared_ptr<MultiRegions::MultiplyByElmtInvMassOp<double>>
        m_multiplyByElmtInvMassOp;

    unsigned int m_numSamples      = 0;
    unsigned int m_outputFrequency = 0;
    unsigned int m_sampleFrequency = 1;

    std::string m_outputFile;
    StepSchedule m_sampleSchedule;
    StepSchedule m_outputSchedule;
    std::unique_ptr<OutputSeries> m_series;

    void AccumulateSample(FilterContext &context);
    void WriteStresses(FilterContext &context, bool numbered = false);
};

} // namespace Nektar::SolverCore
