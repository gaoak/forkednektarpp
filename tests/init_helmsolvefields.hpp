#include "init_fields.hpp"

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class HelmSolveField
    : public InitFields<double, FieldState::Phys, FieldState::Coeff,
                        MultiRegions::ContField>
{
public:
    HelmSolveField()
        : InitFields<double, FieldState::Phys, FieldState::Coeff,
                     MultiRegions::ContField>()
    {
    }

    void SetTestCase(const std::vector<BlockAttributes> &blocks, double *inptr,
                     bool padding = true)
    {
        Array<OneD, NekDouble> x(fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> y(fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> z(fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> fce(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y, z);

        if (fixt_explist->GetSession()->DefinesFunction("Forcing"))
        {
            auto func = fixt_explist->GetSession()->GetFunction("Forcing", 0);
            func->Evaluate(x, y, z, fce);
        }

        double *xptr = x.get(), *yptr = y.get(), *zptr = z.get(),
               *fceptr = fce.get();
        for (auto const &block : fixt_in->GetBlocks())
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    if (fixt_explist->GetSession()->DefinesFunction("Forcing"))
                    {
                        *(inptr++) = *(fceptr++);
                    }
                    else
                    {
                        *inptr = 1.0;
                        if (fixt_explist->GetCoordim(0) == 1)
                        {
                            for (size_t n = 1; n < 4; n++)
                            {
                                *inptr += n * std::pow(*xptr, n);
                            }
                            inptr++;
                            xptr++;
                        }
                        else if (fixt_explist->GetCoordim(0) == 2)
                        {
                            for (size_t n = 1; n < 4; n++)
                            {
                                *inptr +=
                                    n * std::pow(*xptr, n) * std::pow(*yptr, n);
                            }
                            inptr++;
                            xptr++;
                            yptr++;
                        }
                        else
                        {
                            for (size_t n = 1; n < 4; n++)
                            {
                                *inptr += n * std::pow(*xptr, n) *
                                          std::pow(*yptr, n) *
                                          std::pow(*zptr, n);
                            }
                            inptr++;
                            xptr++;
                            yptr++;
                            zptr++;
                        }
                    }
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
        Array<OneD, NekDouble> inphys(fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> outcoeffs(fixt_explist->GetNcoeffs(), 0.0);

        // Set test case
        SetTestCase(blocks, inphys.get(), false);

        // Calculate expected result from Nektar++
        StdRegions::ConstFactorMap factors;
        factors[StdRegions::eFactorLambda] =
            fixt_explist->GetSession()->DefinesParameter("Lambda")
                ? fixt_explist->GetSession()->GetParameter("Lambda")
                : 1.0;
        fixt_explist->HelmSolve(inphys, outcoeffs, factors);

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

class Seg : public HelmSolveField
{
public:
    Seg()
    {
        meshName = "run/line.xml";
    }
};

class Quad : public HelmSolveField
{
public:
    Quad()
    {
        meshName = "run/square.xml";
    }
};

class Tri : public HelmSolveField
{
public:
    Tri()
    {
        meshName = "run/tri.xml";
    }
};

class SquareAllElements : public HelmSolveField
{
public:
    SquareAllElements()
    {
        meshName = "run/square_all_elements.xml";
    }
};

class Hex : public HelmSolveField
{
public:
    Hex()
    {
        meshName = "run/hex.xml";
    }
};

class Prism : public HelmSolveField
{
public:
    Prism()
    {
        meshName = "run/prism.xml";
    }
};

class Pyr : public HelmSolveField
{
public:
    Pyr()
    {
        meshName = "run/pyr.xml";
    }
};

class Tet : public HelmSolveField
{
public:
    Tet()
    {
        meshName = "run/tet.xml";
    }
};

class CubePrismHex : public HelmSolveField
{
public:
    CubePrismHex()
    {
        meshName = "run/cube_prismhex.xml";
    }
};

class CubeAllElements : public HelmSolveField
{
public:
    CubeAllElements()
    {
        meshName = "run/cube_all_elements.xml";
    }
};

class Helmholtz2D_P7_AllBCs : public HelmSolveField
{
public:
    Helmholtz2D_P7_AllBCs()
    {
        meshName = "run/Helmholtz2D_P7_AllBCs.xml";
    }
};

class Helmholtz3D_Hex_AllBCs_P6 : public HelmSolveField
{
public:
    Helmholtz3D_Hex_AllBCs_P6()
    {
        meshName = "run/Helmholtz3D_Hex_AllBCs_P6.xml";
    }
};
