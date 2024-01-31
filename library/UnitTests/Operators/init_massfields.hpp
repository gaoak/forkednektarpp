#include "init_fields.hpp"

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class MassField
    : public InitFields<double, FieldState::Coeff, FieldState::Coeff>
{
public:
    MassField() : InitFields<double, FieldState::Coeff, FieldState::Coeff>()
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
        Array<OneD, NekDouble> outcoeffs(fixt_explist->GetNcoeffs());

        // Set test case
        SetTestCase(fixt_in->GetBlocks(), incoeffs.get(), false);

        // Calculate expected result from Nektar++
        fixt_explist->BwdTrans(incoeffs, bwdtrans);
        fixt_explist->IProductWRTBase(bwdtrans, outcoeffs);

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

class Seg : public MassField
{
public:
    Seg()
    {
        meshName = "run/line.xml";
    }
};

class Quad : public MassField
{
public:
    Quad()
    {
        meshName = "run/square.xml";
    }
};

class Tri : public MassField
{
public:
    Tri()
    {
        meshName = "run/tri.xml";
    }
};

class SquareAllElements : public MassField
{
public:
    SquareAllElements()
    {
        meshName = "run/square_all_elements.xml";
    }
};

class Hex : public MassField
{
public:
    Hex()
    {
        meshName = "run/hex.xml";
    }
};

class Prism : public MassField
{
public:
    Prism()
    {
        meshName = "run/prism.xml";
    }
};

class Pyr : public MassField
{
public:
    Pyr()
    {
        meshName = "run/pyr.xml";
    }
};

class Tet : public MassField
{
public:
    Tet()
    {
        meshName = "run/tet.xml";
    }
};

class CubePrismHex : public MassField
{
public:
    CubePrismHex()
    {
        meshName = "run/cube_prismhex.xml";
    }
};

class CubeAllElements : public MassField
{
public:
    CubeAllElements()
    {
        meshName = "run/cube_all_elements.xml";
    }
};
