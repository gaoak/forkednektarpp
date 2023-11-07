#include "init_fields.hpp"

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class BwdTransField
    : public InitFields<double, FieldState::Coeff, FieldState::Phys>
{
public:
    BwdTransField() : InitFields<double, FieldState::Coeff, FieldState::Phys>()
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
        Array<OneD, NekDouble> outphys(fixt_explist->GetTotPoints());

        // Set test case
        SetTestCase(blocks, incoeffs.get(), false);

        // Calculate expected result from Nektar++
        fixt_explist->BwdTrans(incoeffs, outphys);

        // Copy expected result from Array to fixt_expected
        double *ptr = outphys.get();
        for (auto const &block : blocks)
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    (*inptr++) = (*ptr++);
                }
            }
            for (size_t el = 0; el < block.num_padding_elements; ++el)
            {
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    inptr++;
                }
            }
        }
    }
};

class Seg : public BwdTransField
{
public:
    Seg()
    {
        meshName = "run/line.xml";
    }
};

class Quad : public BwdTransField
{
public:
    Quad()
    {
        meshName = "run/square.xml";
    }
};

class Tri : public BwdTransField
{
public:
    Tri()
    {
        meshName = "run/tri.xml";
    }
};

class SquareAllElements : public BwdTransField
{
public:
    SquareAllElements()
    {
        meshName = "run/square_all_elements.xml";
    }
};

class Hex : public BwdTransField
{
public:
    Hex()
    {
        meshName = "run/hex.xml";
    }
};

class Prism : public BwdTransField
{
public:
    Prism()
    {
        meshName = "run/prism.xml";
    }
};

class Pyr : public BwdTransField
{
public:
    Pyr()
    {
        meshName = "run/pyr.xml";
    }
};

class Tet : public BwdTransField
{
public:
    Tet()
    {
        meshName = "run/tet.xml";
    }
};

class CubePrismHex : public BwdTransField
{
public:
    CubePrismHex()
    {
        meshName = "run/cube_prismhex.xml";
    }
};

class CubeAllElements : public BwdTransField
{
public:
    CubeAllElements()
    {
        meshName = "run/cube_all_elements.xml";
    }
};
