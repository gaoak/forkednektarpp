///////////////////////////////////////////////////////////////////////////////
//
// File: VariableDiffusion.hpp
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <MultiRegions/ExpList.h>

/** Shared reference data for a spatially varying, fully coupled SPD tensor. */
template <typename TData> struct VariableDiffusion
{
    std::unique_ptr<
        Nektar::LibUtilities::Field<TData, Nektar::FieldState::Phys>>
        field;
    std::vector<Nektar::Array<Nektar::OneD, double>> values;

    void Initialise(const Nektar::MultiRegions::ExpListSharedPtr &exp,
                    unsigned int modes)
    {
        using namespace Nektar;
        const auto dim                = exp->GetCoordim(0);
        const unsigned int nDiffCoeff = dim * (dim + 1) / 2;
        const auto nphys              = exp->GetTotPoints() / modes;
        values.resize(nDiffCoeff);
        for (auto &component : values)
        {
            component = Array<OneD, double>(nphys);
        }
        // D = k_perp I + (k_parallel-k_perp) b b^T, with a varying
        // direction and transport rates. Every off-diagonal entry is nonzero.
        for (size_t p = 0; p < nphys; ++p)
        {
            const double b[3] = {1.0 + 0.01 * p, 0.7 + 0.003 * p,
                                 0.3 + 0.002 * p};
            double norm       = 0.0;
            for (int d = 0; d < dim; ++d)
            {
                norm += b[d] * b[d];
            }
            for (int j = 0, c = 0; j < dim; ++j)
            {
                for (int i = 0; i <= j; ++i, ++c)
                {
                    values[c][p] = (i == j ? 0.8 + 0.001 * p : 0.0) +
                                   (2.0 + 0.002 * p) * b[i] * b[j] / norm;
                }
            }
        }
        field = std::make_unique<LibUtilities::Field<TData, FieldState::Phys>>(
            "diffusion",
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                exp, MultiRegions::UnitInterleaveWidths(exp)),
            nDiffCoeff, modes);
        CopyToField();
    }

    void CopyToField()
    {
        using namespace Nektar;
        const auto nphys = values[0].size();
        const auto modes = field->GetNumHomoModes();
        Array<OneD, TData> data(values.size() * modes * nphys);
        for (unsigned int c = 0; c < values.size(); ++c)
        {
            for (unsigned int m = 0; m < modes; ++m)
            {
                for (size_t p = 0; p < nphys; ++p)
                {
                    data[(c * modes + m) * nphys + p] = values[c][p];
                }
            }
        }
        field->template CopyArray<NektarSpaces::HostSpace>(data);
    }

    // Legacy SegExp's assembled Laplacian ignores tensor varcoeffs on
    // regular geometry. Assemble the weak form from its legacy transforms
    // and physical derivatives to cover embedded segments as well.
    void ApplySegment(const Nektar::MultiRegions::ExpListSharedPtr &exp,
                      size_t element,
                      const Nektar::Array<Nektar::OneD, const double> &in,
                      Nektar::Array<Nektar::OneD, double> &out, double lambda,
                      const Nektar::StdRegions::VarCoeffMap &vel = {}) const
    {
        using namespace Nektar;
        auto expansion    = exp->GetExp(element);
        const auto nq     = expansion->GetTotPoints();
        const auto nm     = expansion->GetNcoeffs();
        const auto dim    = expansion->GetCoordim();
        const auto offset = exp->GetPhys_Offset(element);
        Array<OneD, double> phys(nq), source(nq), flux(nq), tmp(nm);
        std::vector<Array<OneD, double>> gradient(dim);
        expansion->BwdTrans(in, phys);
        const StdRegions::VarCoeffType velocities[] = {
            StdRegions::eVarCoeffVelX, StdRegions::eVarCoeffVelY,
            StdRegions::eVarCoeffVelZ};
        for (int p = 0; p < nq; ++p)
        {
            source[p] = lambda * phys[p];
        }
        for (int d = 0; d < dim; ++d)
        {
            gradient[d] = Array<OneD, double>(nq);
            expansion->PhysDeriv(d, phys, gradient[d]);
            auto velocity = vel.find(velocities[d]);
            if (velocity != vel.end())
            {
                for (int p = 0; p < nq; ++p)
                {
                    source[p] += velocity->second[p] * gradient[d][p];
                }
            }
        }
        expansion->IProductWRTBase(source, out);
        for (int d = 0; d < dim; ++d)
        {
            Vmath::Zero(nq, flux, 1);
            for (int k = 0; k < dim; ++k)
            {
                const int i = std::min(d, k), j = std::max(d, k);
                const auto &coefficient = values[j * (j + 1) / 2 + i];
                for (int p = 0; p < nq; ++p)
                {
                    flux[p] += coefficient[offset + p] * gradient[k][p];
                }
            }
            expansion->IProductWRTDerivBase(d, flux, tmp);
            Vmath::Vadd(nm, tmp, 1, out, 1, out, 1);
        }
    }

    Nektar::StdRegions::VarCoeffMap ForElement(
        const Nektar::MultiRegions::ExpListSharedPtr &exp, size_t element) const
    {
        using namespace Nektar;
        StdRegions::VarCoeffMap result;
        const StdRegions::VarCoeffType types[] = {
            StdRegions::eVarCoeffD00, StdRegions::eVarCoeffD01,
            StdRegions::eVarCoeffD11, StdRegions::eVarCoeffD02,
            StdRegions::eVarCoeffD12, StdRegions::eVarCoeffD22};
        if (field)
        {
            for (unsigned int c = 0; c < values.size(); ++c)
            {
                result[types[c]] = values[c] + exp->GetPhys_Offset(element);
            }
        }
        return result;
    }
};
