///////////////////////////////////////////////////////////////////////////////
//
// File: TestFwdTransBC.hpp
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

#include "TestOp.hpp"

#include "Operators/BndCondOps/FwdTransBC/FwdTransBCOp.hpp"

#include <SpatialDomains/Conditions.h>

#include <cmath>
#include <cstdlib>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::Operators;

template <typename TData>
class TestFwdTransBC : public TestOp<TData, FieldState::Phys, FieldState::Coeff>
{
public:
    TestFwdTransBC() = default;

    void SetFixture(const unsigned int nhomo) override
    {
        const auto &components = this->session->GetVariables();
        auto inblockAttr       = GetBoundaryBlockAttributes<FieldState::Phys>();
        auto outblockAttr = GetBoundaryBlockAttributes<FieldState::Coeff>();

        auto f_in       = Field<TData, FieldState::Phys>("f_in", inblockAttr,
                                                         components, nhomo);
        auto f_out      = Field<TData, FieldState::Coeff>("f_out", outblockAttr,
                                                          components, nhomo);
        auto f_expected = Field<TData, FieldState::Coeff>(
            "f_expected", outblockAttr, components, nhomo);
        this->fixt_in  = new Field<TData, FieldState::Phys>(std::move(f_in));
        this->fixt_out = new Field<TData, FieldState::Coeff>(std::move(f_out));
        this->fixt_expected =
            new Field<TData, FieldState::Coeff>(std::move(f_expected));
    }

    void SetTestCase()
    {
        this->fixt_in->template CopyVector<NektarSpaces::HostSpace>(
            GetBoundaryPhysValues());

        this->fixt_out->template Initialize<NektarSpaces::HostSpace>(0.0);

        // Compute expected solution.
        ExpectedSolution();
    }

    void RunTestCase()
    {
        const auto &components = this->session->GetVariables();
        const auto nComp       = components.size();
        const auto boundaryExp = GetBoundaryConditionExpansions();
        const auto totalPhys   = GetTotalBoundaryPoints(boundaryExp);
        const auto totalCoeffs = GetTotalBoundaryCoeffs(boundaryExp);
        auto inPhys            = this->fixt_in->template ToArray<TData>();
        std::vector<TData> result(totalCoeffs * nComp, 0.0);
        size_t physOffset  = 0;
        size_t coeffOffset = 0;

        for (const auto &expList : boundaryExp)
        {
            expList->SetDataWarehouse();

            auto physBlocks =
                GetBlockAttributes<TData, FieldState::Phys>(expList);
            auto coeffBlocks =
                GetBlockAttributes<TData, FieldState::Coeff>(expList);
            Field<TData, FieldState::Phys> in("boundary phys", physBlocks,
                                              nComp, 1);
            Field<TData, FieldState::Coeff> out("boundary coeff", coeffBlocks,
                                                nComp, 1);

            const auto nphys   = expList->GetTotPoints();
            const auto ncoeffs = expList->GetNcoeffs();
            Array<OneD, TData> localIn(nComp * nphys, 0.0);

            for (unsigned int i = 0; i < nComp; ++i)
            {
                std::copy(inPhys.data() + i * totalPhys + physOffset,
                          inPhys.data() + i * totalPhys + physOffset + nphys,
                          localIn.data() + i * nphys);
            }

            in.template CopyArray<NektarSpaces::HostSpace>(localIn);
            out.template Initialize<NektarSpaces::HostSpace>(0.0);

            auto op = FwdTransBCOp<TData>::Create(expList, components);
            op->Apply(in, out);

            const auto outVec = out.template ToVector<TData>();
            for (unsigned int i = 0; i < nComp; ++i)
            {
                std::copy(outVec.begin() + i * ncoeffs,
                          outVec.begin() + (i + 1) * ncoeffs,
                          result.begin() + i * totalCoeffs + coeffOffset);
            }

            physOffset += nphys;
            coeffOffset += ncoeffs;
        }

        this->fixt_out->template CopyVector<NektarSpaces::HostSpace>(result);
    }

    void ExpectedSolution()
    {
        const auto &components = this->session->GetVariables();
        const auto nComp       = components.size();
        const auto boundaryExp = GetBoundaryConditionExpansions();
        const auto totalPhys   = GetTotalBoundaryPoints(boundaryExp);
        const auto totalCoeffs = GetTotalBoundaryCoeffs(boundaryExp);
        auto inPhys            = this->fixt_in->template ToArray<double>();
        Array<OneD, double> result(totalCoeffs * nComp, 0.0);
        size_t physOffset  = 0;
        size_t coeffOffset = 0;

        for (const auto &expList : boundaryExp)
        {
            const auto nphys   = expList->GetTotPoints();
            const auto ncoeffs = expList->GetNcoeffs();
            Array<OneD, double> tmp;

            for (unsigned int i = 0; i < nComp; ++i)
            {
                if (expList->GetExpType() == MultiRegions::e0D)
                {
                    std::copy(inPhys.data() + i * totalPhys + physOffset,
                              inPhys.data() + i * totalPhys + physOffset +
                                  ncoeffs,
                              result.data() + i * totalCoeffs + coeffOffset);
                }
                else
                {
                    expList->FwdTransBndConstrained(
                        inPhys + i * totalPhys + physOffset,
                        tmp = result + i * totalCoeffs + coeffOffset);
                }
            }

            physOffset += nphys;
            coeffOffset += ncoeffs;
        }

        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(
            result);
    }

private:
    std::shared_ptr<MultiRegions::ContField> CreateBoundaryField(
        const std::string &variable) const
    {
        auto graph = SpatialDomains::MeshGraphIO::Read(this->session);
        auto bcfield =
            MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
                this->session, graph, variable, true, false,
                Collections::eNoCollection);
        bcfield->EvaluateBoundaryConditions(0.0);

        return bcfield;
    }

    std::vector<MultiRegions::ExpListSharedPtr> GetBoundaryConditionExpansions()
        const
    {
        auto bcfield = CreateBoundaryField(this->session->GetVariables()[0]);
        const auto &bndExp = bcfield->GetBndCondExpansions();

        std::vector<MultiRegions::ExpListSharedPtr> result;
        result.reserve(bndExp.size());
        for (size_t i = 0; i < bndExp.size(); ++i)
        {
            result.push_back(bndExp[i]);
        }

        return result;
    }

    std::vector<TData> GetBoundaryPhysValues() const
    {
        const auto &components = this->session->GetVariables();
        const auto nComp       = components.size();
        const auto boundaryExp = GetBoundaryConditionExpansions();
        const auto totalPhys   = GetTotalBoundaryPoints(boundaryExp);
        const auto expType     = this->fixt_explist->GetExpType();
        std::vector<TData> result(totalPhys * nComp, 0.0);

        for (unsigned int i = 0; i < nComp; ++i)
        {
            auto bcfield       = CreateBoundaryField(components[i]);
            const auto &bndExp = bcfield->GetBndCondExpansions();
            size_t physOffset  = 0;

            for (size_t j = 0; j < bndExp.size(); ++j)
            {
                const auto &expList = bndExp[j];
                const auto nphys    = expList->GetTotPoints();

                if (expType == MultiRegions::e1D)
                {
                    const auto &coeffs = expList->GetCoeffs();
                    std::copy(coeffs.begin(), coeffs.begin() + nphys,
                              result.begin() + i * totalPhys + physOffset);
                }
                else
                {
                    const auto &phys = expList->GetPhys();
                    std::copy(phys.begin(), phys.begin() + nphys,
                              result.begin() + i * totalPhys + physOffset);
                }

                physOffset += nphys;
            }
        }

        return result;
    }

    size_t GetTotalBoundaryPoints(
        const std::vector<MultiRegions::ExpListSharedPtr> &boundaryExp) const
    {
        size_t totalPhys = 0;
        for (const auto &expList : boundaryExp)
        {
            totalPhys += expList->GetTotPoints();
        }

        return totalPhys;
    }

    size_t GetTotalBoundaryCoeffs(
        const std::vector<MultiRegions::ExpListSharedPtr> &boundaryExp) const
    {
        size_t totalCoeffs = 0;
        for (const auto &expList : boundaryExp)
        {
            totalCoeffs += expList->GetNcoeffs();
        }

        return totalCoeffs;
    }

    template <FieldState TState>
    std::vector<BlockAttributes<TState>> GetBoundaryBlockAttributes() const
    {
        const auto boundaryExp = GetBoundaryConditionExpansions();
        std::vector<BlockAttributes<TState>> blockAttr;
        size_t nBlocks = 0;
        for (const auto &expList : boundaryExp)
        {
            nBlocks += GetBlockAttributes<TData, TState>(expList).size();
        }

        blockAttr.reserve(nBlocks);
        for (const auto &expList : boundaryExp)
        {
            auto bcBlockAttr = GetBlockAttributes<TData, TState>(expList);
            for (const auto &attr : bcBlockAttr)
            {
                blockAttr.emplace_back(attr);
            }
        }

        return blockAttr;
    }
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public TestFwdTransBC<float>                           \
    {                                                                          \
    public:                                                                    \
        type##float()                                                          \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };
#else
#define TESTFLOAT(type, filename)
#endif
#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
#define TESTDOUBLE(type, filename)                                             \
    class type : public TestFwdTransBC<double>                                 \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };
#else
#define TESTDOUBLE(type, filename)
#endif
#define TEST(type, filename)                                                   \
    TESTFLOAT(type, filename)                                                  \
    TESTDOUBLE(type, filename)
// clang-format on

TEST(Helmholtz1D_Seg, "run/Helmholtz1D_P8.xml")
TEST(Helmholtz1D_Seg_3C, "run/Helmholtz1D_3C_mixedBC.xml")
TEST(Helmholtz2D_Tri, "run/Helmholtz2D_Tri.xml")
TEST(Helmholtz2D_Quad, "run/Helmholtz2D_Quad.xml")
TEST(Helmholtz2D_Tri_Quad, "run/Helmholtz2D_varP.xml")
TEST(Helmholtz2D_Tri_Quad_3C, "run/Helmholtz2D_3C.xml")
TEST(Helmholtz2D_AllBCs, "run/Helmholtz2D_P7_AllBCs.xml")
TEST(Helmholtz3D_Hex, "run/Helmholtz3D_Hex_Heterogeneous.xml")
TEST(Helmholtz3D_Hex_3C, "run/Helmholtz3D_Hex_3C.xml")
TEST(Helmholtz3D_Prism, "run/Helmholtz3D_Prism_VarP.xml")
TEST(Helmholtz3D_Pyr, "run/Helmholtz3D_Pyr_VarP.xml")
TEST(Helmholtz3D_Tet, "run/Helmholtz3D_Tet_VarP.xml")
