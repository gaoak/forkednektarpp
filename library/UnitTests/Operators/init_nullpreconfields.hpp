#include "init_fields.hpp"
#include <MultiRegions/GlobalLinSys.h>
#include <MultiRegions/Preconditioner.h>

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;
using namespace Nektar;

class NullPreconField
    : public InitFields<double, FieldState::Coeff, FieldState::Coeff,
                        MultiRegions::ContField>
{
public:
    NullPreconField()
        : InitFields<double, FieldState::Coeff, FieldState::Coeff,
                     MultiRegions::ContField>()
    {
    }

    void SetTestCase(const std::vector<BlockAttributes> &blocks, double *inptr,
                     bool padding = true)
    {
        for (auto const &block : blocks)
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
                {
                    *(inptr++) = 1.0;
                }
            }
            if (padding)
            {
                for (size_t el = 0; el < block.num_padding_elements; ++el)
                {
                    for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
                    {
                        inptr++;
                    }
                }
            }
        }
    }

    void ExpectedSolution(const std::vector<BlockAttributes> &blocks,
                          double *inptr)
    {
        Array<OneD, NekDouble> incoeffs(fixt_explist->GetNcoeffs());
        Array<OneD, NekDouble> outcoeffs(fixt_explist->GetNcoeffs(), 0.0);

        // Set test case
        SetTestCase(fixt_in->GetBlocks(), incoeffs.get(), false);

        // Calculate expected result from Nektar++
        auto map = fixt_explist->GetLocalToGlobalMap();
        GlobalLinSysKey key(StdRegions::eHelmholtz, map);
        auto globalSys = GetGlobalLinSysFactory().CreateInstance(
            "IterativeFull", key, fixt_explist, map);
        auto precond =
            GetPreconFactory().CreateInstance("Null", globalSys, map);
        precond->DoPreconditioner(incoeffs, outcoeffs, true);

        // Copy expected result from Array to pointer
        double *coeffptr = outcoeffs.get();
        for (auto const &block : blocks)
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
                {
                    (*inptr++) = (*coeffptr++);
                }
            }
            for (size_t el = 0; el < block.num_padding_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
                {
                    inptr++;
                }
            }
        }
    }
};

class Helmholtz1D_Seg : public NullPreconField
{
public:
    Helmholtz1D_Seg()
    {
        meshName = "run/Helmholtz1D_P8.xml";
    }
};

class Helmholtz2D_Tri_Quad : public NullPreconField
{
public:
    Helmholtz2D_Tri_Quad()
    {
        meshName = "run/Helmholtz2D_P7_AllBCs.xml";
    }
};

class Helmholtz3D_Hex : public NullPreconField
{
public:
    Helmholtz3D_Hex()
    {
        meshName = "run/Helmholtz3D_Hex_Heterogeneous.xml";
    }
};

class Helmholtz3D_Prism : public NullPreconField
{
public:
    Helmholtz3D_Prism()
    {
        meshName = "run/Helmholtz3D_Prism_VarP.xml";
    }
};

class Helmholtz3D_Pyr : public NullPreconField
{
public:
    Helmholtz3D_Pyr()
    {
        meshName = "run/Helmholtz3D_Pyr_VarP.xml";
    }
};

class Helmholtz3D_Tet : public NullPreconField
{
public:
    Helmholtz3D_Tet()
    {
        meshName = "run/Helmholtz3D_Tet_VarP.xml";
    }
};
