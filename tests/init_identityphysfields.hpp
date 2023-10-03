#include "init_fields.hpp"

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class IdentityField
    : public InitFields<double, FieldState::Phys, FieldState::Phys>
{
public:
    IdentityField() : InitFields<double, FieldState::Phys, FieldState::Phys>()
    {
    }

    void SetTestCase(const std::vector<BlockAttributes> &blocks, double *inptr,
                     bool padding = true)
    {
        for (auto const &block : fixt_in->GetBlocks())
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    *(inptr++) = phys;
                }
            }
            if (padding)
            {
                for (size_t el = 0; el < block.num_padding_elements; ++el)
                {
                    for (size_t phys = 0; phys < block.num_pts; ++phys)
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
        // Copy expected result to fixt_expected
        SetTestCase(blocks, fixt_expected->GetStorage().GetCPUPtr());
    }
};

class Seg : public IdentityField
{
public:
    Seg()
    {
        meshName = "line.xml";
    }
};

class Quad : public IdentityField
{
public:
    Quad()
    {
        meshName = "square.xml";
    }
};

class Tri : public IdentityField
{
public:
    Tri()
    {
        meshName = "run/tri.xml";
    }
};

class SquareAllElements : public IdentityField
{
public:
    SquareAllElements()
    {
        meshName = "run/square_all_elements.xml";
    }
};

class Hex : public IdentityField
{
public:
    Hex()
    {
        meshName = "run/hex.xml";
    }
};

class Prism : public IdentityField
{
public:
    Prism()
    {
        meshName = "run/prism.xml";
    }
};

class Pyr : public IdentityField
{
public:
    Pyr()
    {
        meshName = "run/pyr.xml";
    }
};

class Tet : public IdentityField
{
public:
    Tet()
    {
        meshName = "run/tet.xml";
    }
};

class CubePrismHex : public IdentityField
{
public:
    CubePrismHex()
    {
        meshName = "run/cube_prismhex.xml";
    }
};

class CubeAllElements : public IdentityField
{
public:
    CubeAllElements()
    {
        meshName = "run/cube_all_elements.xml";
    }
};
