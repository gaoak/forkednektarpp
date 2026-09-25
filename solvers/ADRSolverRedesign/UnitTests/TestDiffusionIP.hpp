///////////////////////////////////////////////////////////////////////////////
//
// File: TestDiffusionIP.hpp
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
// Description: Fixture for the interior penalty diffusion operator, comparing
// it against the legacy SolverUtils InteriorPenalty diffusion.
//
///////////////////////////////////////////////////////////////////////////////

#include "ADRSolverRedesign/DiffusionScalarVolFlux/DiffusionScalarVolFluxOp.hpp"
#include "SolverCore/Diffusion/DiffusionIP/DiffusionIPOp.hpp"
#include "SolverCore/TraceFlux/TraceFluxOp.hpp"

#include <SolverUtils/Diffusion/Diffusion.h>
#include <SolverUtils/Diffusion/DiffusionIP.h>

#include "UnitTests/Operators/TestOp.hpp"

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;
using namespace Nektar::Operators;
using namespace Nektar::SolverCore;

/**
 * @brief Interior penalty diffusion, redesign against legacy.
 *
 * Mirrors the AdvectionWeakDG fixture. The value of having it is that
 * diffusion previously had no operator-level test at all: its only cover was
 * the solver regression cases, which report an L2 error against an exact
 * solution and so say only that something is wrong, not where. This compares
 * the operator's output pointwise against the legacy operator applied to the
 * same field, which is what makes a discrepancy locatable.
 */
template <typename TData>
class TestDiffusionIP : public TestOp<TData, FieldState::Phys, FieldState::Phys,
                                      MultiRegions::DisContField>
{
public:
    TestDiffusionIP() = default;

    void SetTestCase()
    {
        // Set initial conditions from the quadrature point coordinates, which
        // do not move when the mesh is partitioned, so serial and parallel
        // runs start from the same field. Quadratic, so the second derivative
        // the operator takes is not identically zero.
        for (unsigned blk = 0, eid = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto &block = this->fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            const auto ndim = this->fixt_explist->GetCoordim(0);

            for (unsigned n = 0; n < this->fixt_in->GetNumComponents() *
                                         this->fixt_in->GetNumHomoModes();
                 ++n)
            {
                for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
                {
                    auto coords =
                        this->fixt_explist->GetExp(eid + el)->GetCoords();

                    for (unsigned phys = 0; phys < block.GetNumData();
                         ++phys, ++cnt)
                    {
                        TData v = 1.0 + TData(n);
                        for (unsigned d = 0; d < ndim; ++d)
                        {
                            const TData x = coords[d][phys];
                            v += TData(d + 1) * x + 0.5 * x * x;
                        }
                        inptr[cnt] = v;
                    }
                }
                inptr += block.CompSize();
            }

            eid += block.GetNumElements();
        }

        // Isotropic unit diffusivity, which is what the legacy flux vector
        // below applies too.
        const unsigned nDim = this->fixt_explist->GetCoordim(0);
        m_diffCoeff.assign(nDim * (nDim + 1) / 2, TData(0.0));
        for (unsigned d = 0; d < nDim; ++d)
        {
            m_diffCoeff[d * (d + 3) / 2] = TData(1.0);
        }

        ExpectedSolution();
    }

    void RunTestCase()
    {
        // Legacy DiffusionIP loads IPPenaltyCoeff and never applies it - its
        // production penalty is p^2 times the reciprocal length and nothing
        // else - where the redesign multiplies the coefficient in. The
        // redesign's default is 1 precisely so the two coincide, but this
        // test should not depend on a default staying put: pin it, so the
        // comparison means the same operator whatever the defaults become.
        TData one = 1.0;
        this->session->SetParameter("IPPenaltyCoeff", one);

        this->fixt_explist->SetDataWarehouse();
        this->fixt_explist->GetTrace()->SetDataWarehouse();

        auto DiffusionOp = DiffusionIPOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());

        auto volFluxOp = DiffusionScalarVolFluxOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());

        auto traceFluxOp = SolverCore::TraceFluxOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables(),
            "DiffusionScalarIPTraceFlux");

        volFluxOp->SetDiffCoeff(m_diffCoeff);
        traceFluxOp->SetDiffCoeff(m_diffCoeff);

        DiffusionOp->SetVolumeFluxOp(volFluxOp);
        DiffusionOp->SetTraceFluxOp(traceFluxOp);

        DiffusionOp->Apply(*this->fixt_in, *this->fixt_out);
    }

    void ExpectedSolution()
    {
        unsigned numComp = this->fixt_in->GetNumComponents();
        size_t nphys     = this->fixt_explist->GetTotPoints();
        auto diffObject  = SolverUtils::GetDiffusionFactory().CreateInstance(
            "InteriorPenalty", "InteriorPenalty");
        // DiffusionIP does not use m_fluxVector at all. It calls three
        // separate functors, and leaving any of them unset is a
        // bad_function_call at apply time rather than at setup. The third is
        // the symmetric flux, which is where the symmetrising terms hook in.
        diffObject->SetDiffusionFluxCons(&TestDiffusionIP::GetFluxCons, this);
        diffObject->SetDiffusionFluxConsTrace(
            &TestDiffusionIP::GetFluxConsTrace, this);
        diffObject->SetDiffusionSymmFluxCons(&TestDiffusionIP::GetSymmFluxCons,
                                             this);
        diffObject->SetSpecialBndTreat(&TestDiffusionIP::SpecialBndTreat, this);

        // Scalars, not conservative variables; identical to what
        // UnsteadyDiffusion now does, so this fixture and the legacy solver
        // agree on what the reference semantics are.
        std::static_pointer_cast<SolverUtils::DiffusionIP>(diffObject)
            ->SetScalarAveraging(true);

        Nektar::Array<Nektar::OneD, Nektar::MultiRegions::ExpListSharedPtr> exp(
            numComp);

        std::vector<std::string> variables = this->session->GetVariables();
        auto graph = SpatialDomains::MeshGraphIO::Read(this->session);
        ASSERTL0(variables.size() >= numComp,
                 "numComp is larger than the number of variables");

        for (unsigned i = 0; i < numComp; ++i)
        {
            // in case of orthogonal expansions reset expansion info for legacy
            // code to work
            if (!this->fixt_explist->GetExp(0)->IsBoundaryInteriorExpansion())
            {
                graph->ResetExpansionInfoToModified(variables[i]);
            }
            exp[i] =
                MemoryManager<MultiRegions::DisContField>::AllocateSharedPtr(
                    this->session, graph, variables[i], true, true,
                    Collections::eNoCollection);
        }

        diffObject->InitObject(this->session, exp);

        Array<OneD, TData> in = this->fixt_in->ToArray();
        Array<OneD, TData> out(nphys * numComp, 0.0);

        Array<OneD, Array<OneD, TData>> inarray(numComp);
        Array<OneD, Array<OneD, TData>> outarray(numComp);

        for (unsigned i = 0; i < numComp; ++i)
        {
            inarray[i] = Array<OneD, TData>(nphys, 0.0);
            for (size_t j = 0; j < nphys; j++)
            {
                inarray[i][j] = in[i * nphys + j];
            }
            outarray[i] = Array<OneD, TData>(nphys, 0.0);
        }

        diffObject->Diffuse(numComp, exp, inarray, outarray);

        for (unsigned i = 0; i < numComp; ++i)
        {
            for (size_t j = 0; j < nphys; ++j)
            {
                out[i * nphys + j] = outarray[i][j];
            }
        }

        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(out);
    }

    /// Mark every variable as carrying a non-zero flux.
    static void SetAllNonZero(const unsigned n, Array<OneD, int> &nonZeroIndex)
    {
        nonZeroIndex = Array<OneD, int>(n);
        for (unsigned i = 0; i < n; ++i)
        {
            nonZeroIndex[i] = i;
        }
    }

    /**
     * @brief Isotropic unit viscous tensor, matching #m_diffCoeff.
     *
     * Serves both the volume and the trace hook: with a constant unit
     * diffusivity the flux is the gradient itself either side of a trace, so
     * there is nothing to distinguish them.
     */
    void GetFluxCons(
        const int nDim,
        [[maybe_unused]] const Array<OneD, Array<OneD, TData>> &inarray,
        const TensorOfArray3D<TData> &qfield,
        TensorOfArray3D<TData> &viscousTensor, Array<OneD, int> &nonZeroIndex,
        [[maybe_unused]] const Array<OneD, Array<OneD, TData>> &normal)
    {
        const unsigned nConvectiveFields = qfield[0].size();
        const unsigned nPts              = qfield[0][0].size();

        // Which variables have a non-zero viscous flux. For a scalar it is
        // all of them; the compressible solver leaves density out, which is
        // why the hook exists. Leaving it empty is not neutral - DiffusionIP
        // loops over it to decide what to write, so an unset one produces a
        // silent zero rather than an error.
        SetAllNonZero(nConvectiveFields, nonZeroIndex);

        for (int j = 0; j < nDim; ++j)
        {
            for (unsigned i = 0; i < nConvectiveFields; ++i)
            {
                Vmath::Vcopy(nPts, qfield[j][i], 1, viscousTensor[j][i], 1);
            }
        }
    }

    /**
     * @brief The same flux on a trace, where it is already contracted with the
     * normal.
     *
     * So the output has one direction rather than nDim - which is the whole
     * difference between this and GetFluxCons(), and why the reference
     * implementation in NavierStokesCFE binds a different instantiation of
     * its flux function here rather than the same one. Binding the volume
     * form in this slot indexes the output past its first direction, and the
     * failure surfaces well inside SolverUtils rather than at the binding.
     */
    void GetFluxConsTrace(
        const int nDim,
        [[maybe_unused]] const Array<OneD, Array<OneD, TData>> &inarray,
        const TensorOfArray3D<TData> &qfield, TensorOfArray3D<TData> &outarray,
        Array<OneD, int> &nonZeroIndex,
        const Array<OneD, Array<OneD, TData>> &normal)
    {
        const unsigned nConvectiveFields = qfield[0].size();
        const unsigned nPts              = qfield[0][0].size();

        SetAllNonZero(nConvectiveFields, nonZeroIndex);

        for (unsigned i = 0; i < nConvectiveFields; ++i)
        {
            Vmath::Zero(nPts, outarray[0][i], 1);
            for (int j = 0; j < nDim; ++j)
            {
                Vmath::Vvtvp(nPts, qfield[j][i], 1, normal[j], 1,
                             outarray[0][i], 1, outarray[0][i], 1);
            }
        }
    }

    /**
     * @brief Symmetric flux for the interior penalty form.
     *
     * With unit isotropic diffusivity this is the jump carried into each
     * coordinate direction weighted by the trace normal. Kept separate from
     * GetFluxCons() because it is the term the symmetrising work varies, and
     * having it as its own hook is what lets that be exercised here.
     */
    void GetSymmFluxCons(
        const int nDim,
        [[maybe_unused]] const Array<OneD, Array<OneD, TData>> &inaverg,
        const Array<OneD, Array<OneD, TData>> &inarray,
        TensorOfArray3D<TData> &outarray, Array<OneD, int> &nonZeroIndex,
        const Array<OneD, Array<OneD, TData>> &normal)
    {
        const unsigned nConvectiveFields = inarray.size();
        const unsigned nPts              = inarray[0].size();

        SetAllNonZero(nConvectiveFields, nonZeroIndex);

        for (int j = 0; j < nDim; ++j)
        {
            for (unsigned i = 0; i < nConvectiveFields; ++i)
            {
                Vmath::Vmul(nPts, inarray[i], 1, normal[j], 1, outarray[j][i],
                            1);
            }
        }
    }

    /// Boundary fix-up hook. A scalar diffusion has nothing to correct - it
    /// exists for the compressible solver, which resets the energy at a wall -
    /// but DiffusionIP calls it unconditionally, so it has to be bound.
    void SpecialBndTreat(
        [[maybe_unused]] Array<OneD, Array<OneD, TData>> &consvar)
    {
    }

public:
    std::vector<TData> m_diffCoeff;
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public TestDiffusionIP<float>                          \
    {                                                                          \
    public:                                                                    \
        type##float()                                                          \
        {                                                                      \
            std::string dir = "./";                                            \
            meshName = dir + filename;                                         \
        }                                                                      \
    };
#else
#define TESTFLOAT(type, filename)
#endif
#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
#define TESTDOUBLE(type, filename)                                             \
    class type : public TestDiffusionIP<double>                                \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            std::string dir = "./";                                            \
            meshName = dir + filename;                                         \
        }                                                                      \
    };
#else
#define TESTDOUBLE(type, filename)
#endif
#define TEST(type, filename)                                                   \
    TESTFLOAT(type, filename)                                                  \
    TESTDOUBLE(type, filename)
// clang-format on

TEST(Quad, "run/Quad_Dir.xml")

TEST(Tri, "run/Tri_Dir.xml")

TEST(TriQuad_VarP, "run/TriQuad_varP_Dir.xml")

TEST(Hex, "run/Hex_Dir.xml")

TEST(Prism, "run/Prism_Dir.xml")

TEST(Tet, "run/Tet_Dir.xml")

TEST(CubeAllElements, "run/cube_all_elements_dir.xml")
