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
        for (auto const &block : fixt_in->GetBlocks())
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
        SetTestCase(blocks, incoeffs.get(), false);

        // Calculate expected result from Nektar++
        auto map = fixt_explist->GetLocalToGlobalMap();
        GlobalLinSysKey key(StdRegions::eHelmholtz, map);
        auto globalSys = GetGlobalLinSysFactory().CreateInstance(
            "IterativeFull", key, fixt_explist, map);
        auto precond =
            GetPreconFactory().CreateInstance("Null", globalSys, map);
        precond->DoPreconditioner(incoeffs, outcoeffs, true);

        // Copy expected result from Array to fixt_expected
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

class Seg : public NullPreconField
{
public:
    Seg()
    {
        meshName = "run/line.xml";
    }
};

class Quad : public NullPreconField
{
public:
    Quad()
    {
        meshName = "run/square.xml";
    }
};

class Tri : public NullPreconField
{
public:
    Tri()
    {
        meshName = "run/tri.xml";
    }
};

class SquareAllElements : public NullPreconField
{
public:
    SquareAllElements()
    {
        meshName = "run/square_all_elements.xml";
    }
};

class Hex : public NullPreconField
{
public:
    Hex()
    {
        meshName = "run/hex.xml";
    }
};

class Prism : public NullPreconField
{
public:
    Prism()
    {
        meshName = "run/prism.xml";
    }
};

class Pyr : public NullPreconField
{
public:
    Pyr()
    {
        meshName = "run/pyr.xml";
    }
};

class Tet : public NullPreconField
{
public:
    Tet()
    {
        meshName = "run/tet.xml";
    }
};

class CubePrismHex : public NullPreconField
{
public:
    CubePrismHex()
    {
        meshName = "run/cube_prismhex.xml";
    }
};

class CubeAllElements : public NullPreconField
{
public:
    CubeAllElements()
    {
        meshName = "run/cube_all_elements.xml";
    }
};

class Helmholtz2D_P7_AllBCs : public NullPreconField
{
public:
    Helmholtz2D_P7_AllBCs()
    {
        meshName = "run/Helmholtz2D_P7_AllBCs.xml";
    }
};

class Helmholtz3D_Hex_AllBCs_P6 : public NullPreconField
{
public:
    Helmholtz3D_Hex_AllBCs_P6()
    {
        meshName = "run/Helmholtz3D_Hex_AllBCs_P6.xml";
    }
};
