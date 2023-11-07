#include "init_fields.hpp"

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class DirichletField
    : public InitFields<double, FieldState::Coeff, FieldState::Coeff,
                        MultiRegions::ContField>
{
public:
    DirichletField()
        : InitFields<double, FieldState::Coeff, FieldState::Coeff,
                     MultiRegions::ContField>()
    {
    }

    void ExpectedSolution(const std::vector<BlockAttributes> &blocks,
                          double *inptr)
    {
        Array<OneD, NekDouble> outcoeffs(fixt_explist->GetNcoeffs(), 0.0);

        // Calculate expected result from Nektar++
        fixt_explist->ImposeDirichletConditions(outcoeffs);

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

class Helmholtz2D_P7_AllBCs : public DirichletField
{
public:
    Helmholtz2D_P7_AllBCs()
    {
        meshName = "run/Helmholtz2D_P7_AllBCs.xml";
    }
};

class Helmholtz3D_Hex_AllBCs_P6 : public DirichletField
{
public:
    Helmholtz3D_Hex_AllBCs_P6()
    {
        meshName = "run/Helmholtz3D_Hex_AllBCs_P6.xml";
    }
};
