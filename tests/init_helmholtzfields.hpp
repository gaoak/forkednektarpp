#include "init_fields.hpp"

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class HelmholtzField
    : public InitFields<double, FieldState::Coeff, FieldState::Coeff>
{
public:
    HelmholtzField()
        : InitFields<double, FieldState::Coeff, FieldState::Coeff>()
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
                    *(inptr++) = coeff;
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
        Array<OneD, NekDouble> outcoeffs(fixt_explist->GetNcoeffs());
        Array<OneD, NekDouble> tmp;

        // Set test case
        SetTestCase(blocks, incoeffs.get(), false);

        // Calculate expected result from Nektar++
        auto e = 0, offset = 0;
        StdRegions::FactorMap factors;
        factors[StdRegions::eFactorLambda] = 1.0;
        for (auto const &block : blocks)
        {
            auto nmTot = fixt_explist->GetExp(e)->GetNcoeffs();
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                StdRegions::StdMatrixKey mkey(
                    StdRegions::eHelmholtz,
                    fixt_explist->GetExp(e)->DetShapeType(),
                    *(fixt_explist->GetExp(e)), factors);
                fixt_explist->GetExp(e)->GeneralMatrixOp(
                    incoeffs + offset, tmp = outcoeffs + offset, mkey);
                e++;
                offset += nmTot;
            }
        }

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

class Seg : public HelmholtzField
{
public:
    Seg()
    {
        meshName = "run/line.xml";
    }
};

class Quad : public HelmholtzField
{
public:
    Quad()
    {
        meshName = "run/square.xml";
    }
};

class Tri : public HelmholtzField
{
public:
    Tri()
    {
        meshName = "run/tri.xml";
    }
};

class SquareAllElements : public HelmholtzField
{
public:
    SquareAllElements()
    {
        meshName = "run/square_all_elements.xml";
    }
};

class Hex : public HelmholtzField
{
public:
    Hex()
    {
        meshName = "run/hex.xml";
    }
};

class Prism : public HelmholtzField
{
public:
    Prism()
    {
        meshName = "run/prism.xml";
    }
};

class Pyr : public HelmholtzField
{
public:
    Pyr()
    {
        meshName = "run/pyr.xml";
    }
};

class Tet : public HelmholtzField
{
public:
    Tet()
    {
        meshName = "run/tet.xml";
    }
};

class CubePrismHex : public HelmholtzField
{
public:
    CubePrismHex()
    {
        meshName = "run/cube_prismhex.xml";
    }
};

class CubeAllElements : public HelmholtzField
{
public:
    CubeAllElements()
    {
        meshName = "run/cube_all_elements.xml";
    }
};
