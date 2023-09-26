#include "init_fields.hpp"

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class IProductWRTDerivBaseField
    : public InitFields<double, FieldState::Phys, FieldState::Coeff>
{
public:
    IProductWRTDerivBaseField()
        : InitFields<double, FieldState::Phys, FieldState::Coeff>()
    {
    }

    void SetTestCase(const std::vector<BlockAttributes> &blocks, double *inptr,
                     bool padding = true)
    {
        for (size_t k = 0; k < fixt_explist->GetCoordim(0); k++)
        {
            for (auto const &block : fixt_in->GetBlocks())
            {
                for (size_t el = 0; el < block.num_elements; ++el)
                {
                    for (size_t phys = 0; phys < block.num_pts; ++phys)
                    {
                        *(inptr++) = phys + k;
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
    }

    void ExpectedSolution(const std::vector<BlockAttributes> &blocks,
                          double *inptr)
    {
        Array<OneD, NekDouble> inphys(fixt_explist->GetCoordim(0) * fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> outcoeffs(fixt_explist->GetNcoeffs(), 0.0);

        // Set test case
        SetTestCase(blocks, inphys.get(), false);

        // Calculate expected result from Nektar++
        Array<OneD, Array<OneD, NekDouble>> inphysarray(fixt_explist->GetCoordim(0));
        if (fixt_explist->GetCoordim(0) > 0)
        {
            inphysarray[0] = inphys;
        }
        if (fixt_explist->GetCoordim(0) > 1)
        {
            inphysarray[1] = inphysarray[0] + fixt_explist->GetTotPoints();
        }
        if (fixt_explist->GetCoordim(0) > 2)
        {
            inphysarray[2] = inphysarray[1] + fixt_explist->GetTotPoints();
        }
        fixt_explist->IProductWRTDerivBase(inphysarray, outcoeffs);

        // Copy expected result from Array to fixt_expected
        double *ptr = outcoeffs.get();
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

class Seg : public IProductWRTDerivBaseField
{
public:
    Seg()
    {
        meshName = "line.xml";
    }
};

class Quad : public IProductWRTDerivBaseField
{
public:
    Quad()
    {
        meshName = "square.xml";
    }
};

class Tri : public IProductWRTDerivBaseField
{
public:
    Tri()
    {
        meshName = "run/tri.xml";
    }
};

class SquareAllElements : public IProductWRTDerivBaseField
{
public:
    SquareAllElements()
    {
        meshName = "run/square_all_elements.xml";
    }
};

class Hex : public IProductWRTDerivBaseField
{
public:
    Hex()
    {
        meshName = "run/hex.xml";
    }
};

class Prism : public IProductWRTDerivBaseField
{
public:
    Prism()
    {
        meshName = "run/prism.xml";
    }
};

class Pyr : public IProductWRTDerivBaseField
{
public:
    Pyr()
    {
        meshName = "run/pyr.xml";
    }
};

class Tet : public IProductWRTDerivBaseField
{
public:
    Tet()
    {
        meshName = "run/tet.xml";
    }
};

class CubePrismHex : public IProductWRTDerivBaseField
{
public:
    CubePrismHex()
    {
        meshName = "run/cube_prismhex.xml";
    }
};

class CubeAllElements : public IProductWRTDerivBaseField
{
public:
    CubeAllElements()
    {
        meshName = "run/cube_all_elements.xml";
    }
};
