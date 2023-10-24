#include "init_fields.hpp"
#include <LibUtilities/LinearAlgebra/NekLinSysIter.h>
#include <MultiRegions/ContField.h>
#include <MultiRegions/GlobalLinSysIterativeFull.h>

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;
using namespace Nektar;

class AssmbScatrField
    : public InitFields<double, FieldState::Coeff, FieldState::Coeff, ContField>
{
public:
    AssmbScatrField()
        : InitFields<double, FieldState::Coeff, FieldState::Coeff, ContField>()
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
        Array<OneD, NekDouble> incoeff(fixt_explist->GetNcoeffs());
        Array<OneD, NekDouble> outcoeff(fixt_explist->GetNcoeffs());

        // Set test case
        SetTestCase(blocks, incoeff.get(), false);

        // Calculate expected result from Nektar++
        auto map = fixt_explist->GetLocalToGlobalMap();
        map->Assemble(incoeff, outcoeff);
        //Vmath::Zero(map->GetNumGlobalDirBndCoeffs(), outcoeff, 1);
        map->GlobalToLocal(outcoeff, outcoeff);

        // Copy expected result from Array to fixt_expected
        double *ptr = outcoeff.get();
        for (auto const &block : blocks)
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
                {
                    (*inptr++) = (*ptr++);
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

class Seg : public AssmbScatrField
{
public:
    Seg()
    {
        meshName = "line.xml";
    }
};

class Quad : public AssmbScatrField
{
public:
    Quad()
    {
        meshName = "square.xml";
    }
};

class Tri : public AssmbScatrField
{
public:
    Tri()
    {
        meshName = "run/tri.xml";
    }
};

class SquareAllElements : public AssmbScatrField
{
public:
    SquareAllElements()
    {
        meshName = "run/square_all_elements.xml";
    }
};

class Hex : public AssmbScatrField
{
public:
    Hex()
    {
        meshName = "run/hex.xml";
    }
};

class Prism : public AssmbScatrField
{
public:
    Prism()
    {
        meshName = "run/prism.xml";
    }
};

class Pyr : public AssmbScatrField
{
public:
    Pyr()
    {
        meshName = "run/pyr.xml";
    }
};

class Tet : public AssmbScatrField
{
public:
    Tet()
    {
        meshName = "run/tet.xml";
    }
};

class CubePrismHex : public AssmbScatrField
{
public:
    CubePrismHex()
    {
        meshName = "run/cube_prismhex.xml";
    }
};

class CubeAllElements : public AssmbScatrField
{
public:
    CubeAllElements()
    {
        meshName = "run/cube_all_elements.xml";
    }
};
