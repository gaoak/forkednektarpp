///////////////////////////////////////////////////////////////////////////////
//
// File: TestSVVLaplacianFilter.cpp
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
// Description: Unit tests for StdExpansion::SVVLaplacianFilter.
//
///////////////////////////////////////////////////////////////////////////////

#include <StdRegions/StdHexExp.h>
#include <StdRegions/StdPrismExp.h>
#include <StdRegions/StdPyrExp.h>
#include <StdRegions/StdQuadExp.h>
#include <StdRegions/StdSegExp.h>
#include <StdRegions/StdTetExp.h>
#include <StdRegions/StdTriExp.h>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace Nektar::StdRegionsUnitTests
{

using namespace LibUtilities;

/// Mode indices (p, q, r) of an orthogonal expansion, in storage order.
using ModeLabels = std::vector<std::array<int, 3>>;

/**
 * @brief Check SVVLaplacianFilter mode by mode.
 *
 * Each mode of the orthogonal expansion @p ortho is transformed to physical
 * space and filtered with @p factors. The filter acts diagonally on the
 * orthogonal coefficients, so the result must be the same mode scaled by
 * @p weight, and every other coefficient must vanish.
 */
void CheckSVVLaplacianFilter(const StdRegions::StdExpansionSharedPtr &ortho,
                             const StdRegions::ConstFactorMap &factors,
                             const std::vector<NekDouble> &weight)
{
    const int nm = ortho->GetNcoeffs();
    const int nq = ortho->GetTotPoints();

    BOOST_REQUIRE_EQUAL(static_cast<int>(weight.size()), nm);

    const StdRegions::StdMatrixKey mkey(StdRegions::eLaplacian,
                                        ortho->DetShapeType(), *ortho, factors);
    const NekDouble tol = 1.0e-10;

    for (int mode = 0; mode < nm; ++mode)
    {
        Array<OneD, NekDouble> orthoCoeffs(nm, 0.0);
        orthoCoeffs[mode] = 1.0;

        Array<OneD, NekDouble> phys(nq), filtered(nm);
        ortho->BwdTrans(orthoCoeffs, phys);
        ortho->SVVLaplacianFilter(phys, mkey);
        ortho->FwdTrans(phys, filtered);

        NekDouble diff = 0.0;
        for (int n = 0; n < nm; ++n)
        {
            const NekDouble expected = n == mode ? weight[mode] : 0.0;
            diff = std::max(diff, std::abs(filtered[n] - expected));
        }
        BOOST_CHECK_MESSAGE(diff < tol, "mode " << mode << " of weight "
                                                << weight[mode] << " off by "
                                                << diff);
    }
}

/// DG kernel weight of the mode @p label of an expansion with @p nmodes.
NekDouble DGKernelWeight(const std::vector<int> &nmodes,
                         const std::array<int, 3> &label,
                         const NekDouble diffCoeff)
{
    int maxModes = 0, maxLabel = 0;
    for (size_t d = 0; d < nmodes.size(); ++d)
    {
        maxModes = std::max(maxModes, nmodes[d]);
        maxLabel = std::max(maxLabel, label[d]);
    }
    const int row = std::clamp(maxModes - StdRegions::kSVVDGFiltermodesmin, 0,
                               StdRegions::kSVVDGFiltermodesmax -
                                   StdRegions::kSVVDGFiltermodesmin);
    const int col = std::min(maxLabel, StdRegions::kSVVDGFiltermodesmax - 1);
    return diffCoeff * StdRegions::kSVVDGFilter[row][col];
}

/// Power kernel weight of the mode @p label of an expansion with @p nmodes.
NekDouble PowerKernelWeight(const std::vector<int> &nmodes,
                            const std::array<int, 3> &label,
                            const NekDouble cutoff, const NekDouble diffCoeff)
{
    NekDouble fac = 0.0;
    for (size_t d = 0; d < nmodes.size(); ++d)
    {
        fac = std::max(fac, std::pow((1.0 * label[d]) / (nmodes[d] - 1),
                                     cutoff * nmodes[d]));
    }
    return diffCoeff * fac;
}

/**
 * @brief Check the DG and power kernels of SVVLaplacianFilter.
 *
 * The weight of a mode depends only on its indices (p, q, r) and the mode
 * counts, so a mode filtered under the wrong indices, or not at all, shows
 * as a coefficient scaled by the wrong weight.
 */
void CheckKernels(const StdRegions::StdExpansionSharedPtr &ortho,
                  const std::vector<int> &nmodes, const ModeLabels &labels)
{
    BOOST_REQUIRE_EQUAL(static_cast<int>(labels.size()), ortho->GetNcoeffs());

    const NekDouble svvDiffCoeff = 0.5, kerDiffCoeff = 3.0, cutoff = 0.7;
    std::vector<NekDouble> weight(labels.size());

    StdRegions::ConstFactorMap dgFactors;
    dgFactors[StdRegions::eFactorSVVCutoffRatio]    = cutoff;
    dgFactors[StdRegions::eFactorSVVDiffCoeff]      = svvDiffCoeff;
    dgFactors[StdRegions::eFactorSVVDGKerDiffCoeff] = kerDiffCoeff;
    for (size_t n = 0; n < labels.size(); ++n)
    {
        weight[n] =
            DGKernelWeight(nmodes, labels[n], svvDiffCoeff * kerDiffCoeff);
    }
    CheckSVVLaplacianFilter(ortho, dgFactors, weight);

    StdRegions::ConstFactorMap powerFactors;
    powerFactors[StdRegions::eFactorSVVCutoffRatio]       = cutoff;
    powerFactors[StdRegions::eFactorSVVDiffCoeff]         = svvDiffCoeff;
    powerFactors[StdRegions::eFactorSVVPowerKerDiffCoeff] = kerDiffCoeff;
    for (size_t n = 0; n < labels.size(); ++n)
    {
        weight[n] = PowerKernelWeight(nmodes, labels[n], cutoff,
                                      svvDiffCoeff * kerDiffCoeff);
    }
    CheckSVVLaplacianFilter(ortho, powerFactors, weight);
}

/// Orthogonal quadrilateral; modes stored with p fastest.
StdRegions::StdExpansionSharedPtr OrthoQuad(const int nm0, const int nm1,
                                            ModeLabels &labels)
{
    const BasisKey b0(eOrtho_A, nm0, PointsKey(nm0 + 1, eGaussLobattoLegendre));
    const BasisKey b1(eOrtho_A, nm1, PointsKey(nm1 + 1, eGaussLobattoLegendre));

    labels.clear();
    for (int q = 0; q < nm1; ++q)
    {
        for (int p = 0; p < nm0; ++p)
        {
            labels.push_back({p, q, 0});
        }
    }
    return MemoryManager<StdRegions::StdQuadExp>::AllocateSharedPtr(b0, b1);
}

/// Orthogonal hexahedron; modes stored with p fastest, then q, then r.
StdRegions::StdExpansionSharedPtr OrthoHex(const int nm0, const int nm1,
                                           const int nm2, ModeLabels &labels)
{
    const BasisKey b0(eOrtho_A, nm0, PointsKey(nm0 + 1, eGaussLobattoLegendre));
    const BasisKey b1(eOrtho_A, nm1, PointsKey(nm1 + 1, eGaussLobattoLegendre));
    const BasisKey b2(eOrtho_A, nm2, PointsKey(nm2 + 1, eGaussLobattoLegendre));

    labels.clear();
    for (int r = 0; r < nm2; ++r)
    {
        for (int q = 0; q < nm1; ++q)
        {
            for (int p = 0; p < nm0; ++p)
            {
                labels.push_back({p, q, r});
            }
        }
    }
    return MemoryManager<StdRegions::StdHexExp>::AllocateSharedPtr(b0, b1, b2);
}

BOOST_AUTO_TEST_CASE(TestSegSVVLaplacianFilter)
{
    const int nm = 6;
    const BasisKey b0(eOrtho_A, nm, PointsKey(nm + 1, eGaussLobattoLegendre));
    auto ortho = MemoryManager<StdRegions::StdSegExp>::AllocateSharedPtr(b0);

    // Exponential kernel, adding no diffusion to the modes below the cutoff.
    const NekDouble svvDiffCoeff = 0.5, cutoffRatio = 0.5;
    StdRegions::ConstFactorMap factors;
    factors[StdRegions::eFactorSVVCutoffRatio] = cutoffRatio;
    factors[StdRegions::eFactorSVVDiffCoeff]   = svvDiffCoeff;

    const int cutoff = static_cast<int>(cutoffRatio * nm);
    std::vector<NekDouble> weight(nm);
    for (int p = 0; p < nm; ++p)
    {
        weight[p] = p >= cutoff
                        ? svvDiffCoeff *
                              std::exp(-(p - nm) * (p - nm) /
                                       static_cast<NekDouble>((p - cutoff + 1) *
                                                              (p - cutoff + 1)))
                        : 0.0;
    }
    CheckSVVLaplacianFilter(ortho, factors, weight);
}

BOOST_AUTO_TEST_CASE(TestQuadSVVLaplacianFilter)
{
    // Anisotropic orders tell a mode's p and q apart.
    for (auto [nm0, nm1] :
         {std::array{5, 5}, std::array{4, 6}, std::array{6, 4}})
    {
        ModeLabels labels;
        auto ortho = OrthoQuad(nm0, nm1, labels);
        CheckKernels(ortho, {nm0, nm1}, labels);

        // Exponential kernel, acting on the total degree p + q.
        const NekDouble svvDiffCoeff = 0.5, cutoffRatio = 0.5;
        StdRegions::ConstFactorMap factors;
        factors[StdRegions::eFactorSVVCutoffRatio] = cutoffRatio;
        factors[StdRegions::eFactorSVVDiffCoeff]   = svvDiffCoeff;

        const int nmodes = std::min(nm0, nm1);
        const int cutoff = static_cast<int>(cutoffRatio * nmodes);
        std::vector<NekDouble> weight(labels.size());
        for (size_t n = 0; n < labels.size(); ++n)
        {
            const int deg = labels[n][0] + labels[n][1];
            weight[n] =
                deg >= cutoff
                    ? svvDiffCoeff *
                          std::exp(-(deg - nmodes) * (deg - nmodes) /
                                   static_cast<NekDouble>((deg - cutoff + 1) *
                                                          (deg - cutoff + 1)))
                    : 0.0;
        }
        CheckSVVLaplacianFilter(ortho, factors, weight);
    }
}

BOOST_AUTO_TEST_CASE(TestHexSVVLaplacianFilter)
{
    // Anisotropic orders tell a mode's p, q and r apart.
    for (auto [nm0, nm1, nm2] :
         {std::array{4, 4, 4}, std::array{3, 4, 5}, std::array{5, 4, 3}})
    {
        ModeLabels labels;
        auto ortho = OrthoHex(nm0, nm1, nm2, labels);
        CheckKernels(ortho, {nm0, nm1, nm2}, labels);

        // Exponential kernel, acting on each index separately.
        const NekDouble svvDiffCoeff = 0.5, cutoffRatio = 0.5;
        StdRegions::ConstFactorMap factors;
        factors[StdRegions::eFactorSVVCutoffRatio] = cutoffRatio;
        factors[StdRegions::eFactorSVVDiffCoeff]   = svvDiffCoeff;

        const int cutoff = static_cast<int>(cutoffRatio * std::min(nm0, nm1));
        const int nmodes = std::max({nm0, nm1, nm2});
        auto fac         = [&](const int j) {
            NekDouble f = 1.0;
            if (j >= cutoff)
            {
                f = std::abs((j - nmodes) / (j - cutoff + 1.0));
                f *= f;
            }
            return f;
        };

        std::vector<NekDouble> weight(labels.size());
        for (size_t n = 0; n < labels.size(); ++n)
        {
            const auto [p, q, r] = labels[n];
            weight[n] =
                (p >= cutoff || q >= cutoff || r >= cutoff)
                    ? svvDiffCoeff * std::exp(-(fac(p) + fac(q) + fac(r)))
                    : 0.0;
        }
        CheckSVVLaplacianFilter(ortho, factors, weight);
    }
}

BOOST_AUTO_TEST_CASE(TestTriSVVLaplacianFilter)
{
    const int nm = 5;
    const BasisKey b0(eOrtho_A, nm, PointsKey(nm + 1, eGaussLobattoLegendre));
    const BasisKey b1(eOrtho_B, nm, PointsKey(nm, eGaussRadauMAlpha1Beta0));
    auto ortho =
        MemoryManager<StdRegions::StdTriExp>::AllocateSharedPtr(b0, b1);

    // Modes are stored with p outermost and q < nm - p.
    ModeLabels labels;
    for (int p = 0; p < nm; ++p)
    {
        for (int q = 0; q < nm - p; ++q)
        {
            labels.push_back({p, q, 0});
        }
    }
    CheckKernels(ortho, {nm, nm}, labels);
}

BOOST_AUTO_TEST_CASE(TestTetSVVLaplacianFilter)
{
    for (auto [nm0, nm1, nm2] : {std::array{4, 4, 4}, std::array{6, 6, 6}})
    {
        const BasisKey b0(eOrtho_A, nm0,
                          PointsKey(nm0 + 1, eGaussLobattoLegendre));
        const BasisKey b1(eOrtho_B, nm1,
                          PointsKey(nm1, eGaussRadauMAlpha1Beta0));
        const BasisKey b2(eOrtho_C, nm2,
                          PointsKey(nm2, eGaussRadauMAlpha2Beta0));
        auto ortho =
            MemoryManager<StdRegions::StdTetExp>::AllocateSharedPtr(b0, b1, b2);

        // Modes are stored with p outermost, then q < nm1 - p, then
        // r < nm2 - p - q.
        ModeLabels labels;
        for (int p = 0; p < nm0; ++p)
        {
            for (int q = 0; q < nm1 - p; ++q)
            {
                for (int r = 0; r < nm2 - p - q; ++r)
                {
                    labels.push_back({p, q, r});
                }
            }
        }
        CheckKernels(ortho, {nm0, nm1, nm2}, labels);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismSVVLaplacianFilter)
{
    const int nm = 4;
    const BasisKey b0(eOrtho_A, nm, PointsKey(nm + 1, eGaussLobattoLegendre));
    const BasisKey b2(eOrtho_B, nm, PointsKey(nm, eGaussRadauMAlpha1Beta0));
    auto ortho =
        MemoryManager<StdRegions::StdPrismExp>::AllocateSharedPtr(b0, b0, b2);

    // Modes are stored with p outermost, then q, then r < nm - p.
    ModeLabels labels;
    for (int p = 0; p < nm; ++p)
    {
        for (int q = 0; q < nm; ++q)
        {
            for (int r = 0; r < nm - p; ++r)
            {
                labels.push_back({p, q, r});
            }
        }
    }
    CheckKernels(ortho, {nm, nm, nm}, labels);
}

BOOST_AUTO_TEST_CASE(TestPyrSVVLaplacianFilter)
{
    const int nm = 4;
    const BasisKey b0(eOrtho_A, nm, PointsKey(nm + 1, eGaussLobattoLegendre));
    const BasisKey b2(eOrthoPyr_C, nm, PointsKey(nm, eGaussRadauMAlpha2Beta0));
    auto ortho =
        MemoryManager<StdRegions::StdPyrExp>::AllocateSharedPtr(b0, b0, b2);

    // Modes are stored with p outermost, then q, then r < nm - max(p, q).
    ModeLabels labels;
    for (int p = 0; p < nm; ++p)
    {
        for (int q = 0; q < nm; ++q)
        {
            for (int r = 0; r < nm - std::max(p, q); ++r)
            {
                labels.push_back({p, q, r});
            }
        }
    }
    CheckKernels(ortho, {nm, nm, nm}, labels);
}

} // namespace Nektar::StdRegionsUnitTests
