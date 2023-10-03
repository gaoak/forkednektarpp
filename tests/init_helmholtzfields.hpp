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
        Array<OneD, NekDouble> bwdtrans(fixt_explist->GetTotPoints());

        Array<OneD, NekDouble> deriv(fixt_explist->GetCoordim(0) *
                                     fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> deriv0 = deriv;
        Array<OneD, NekDouble> deriv1 = deriv0 + fixt_explist->GetTotPoints();
        Array<OneD, NekDouble> deriv2 = deriv1 + fixt_explist->GetTotPoints();
        Array<OneD, Array<OneD, NekDouble>> derivarray(
            fixt_explist->GetCoordim(0));
        if (fixt_explist->GetCoordim(0) > 0)
        {
            derivarray[0] = deriv0;
        }
        if (fixt_explist->GetCoordim(0) > 1)
        {
            derivarray[1] = deriv1;
        }
        if (fixt_explist->GetCoordim(0) > 2)
        {
            derivarray[2] = deriv2;
        }
        Array<OneD, NekDouble> mass(fixt_explist->GetNcoeffs());
        Array<OneD, NekDouble> outcoeffs(fixt_explist->GetNcoeffs());

        // Set test case
        SetTestCase(blocks, incoeffs.get(), false);

        // Calculate expected result from Nektar++
        fixt_explist->BwdTrans(incoeffs, bwdtrans);
        fixt_explist->PhysDeriv(bwdtrans, deriv0, deriv1, deriv2);
        fixt_explist->IProductWRTDerivBase(derivarray, outcoeffs);
        fixt_explist->IProductWRTBase(bwdtrans, mass);

        // Copy expected result from Array to fixt_expected
        double *coeffptr = outcoeffs.get();
        double *massptr  = mass.get();
        for (auto const &block : blocks)
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
                {
                    (*inptr++) = (*coeffptr++) + (*massptr++);
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
        meshName = "line.xml";
    }
};

class Quad : public HelmholtzField
{
public:
    Quad()
    {
        meshName = "square.xml";
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
