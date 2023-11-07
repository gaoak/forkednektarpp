#include "init_fields.hpp"

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class PhysDerivField
    : public InitFields<double, FieldState::Phys, FieldState::Phys>
{
public:
    PhysDerivField() : InitFields<double, FieldState::Phys, FieldState::Phys>()
    {
    }
};

class PhysDerivField1D : public PhysDerivField
{
public:
    PhysDerivField1D()
    {
    }

    void SetTestCase(const std::vector<BlockAttributes> &blocks, double *inptr,
                     bool padding = true)
    {
        size_t el = 0, pts = 0;
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x);
        for (auto const &block : blocks)
        {
            for (size_t e = 0; e < block.num_elements; ++e, ++el)
            {
                size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                for (size_t phys = 0; phys < block.num_pts; ++phys, ++pts)
                {
                    double tmp = 0.0;
                    for (size_t i = 0; i < M / 2; i++)
                    {
                        tmp += std::pow(x[pts], i);
                    }
                    *(inptr++) = tmp;
                }
            }
            if (padding)
            {
                for (size_t el = 0; el < block.num_padding_elements; ++el)
                {
                    for (size_t phys = 0; phys < block.num_pts; ++phys)
                    {
                        *(inptr++) = 0.0;
                    }
                }
            }
        }
    }

    void NektarSolution(const std::vector<BlockAttributes> &blocks,
                        double *inptr)
    {
        Array<OneD, NekDouble> inphys(fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> outphys(fixt_explist->GetCoordim(0) *
                                       fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> outphys0 = outphys;
        Array<OneD, NekDouble> outphys1 = NullNekDouble1DArray;
        Array<OneD, NekDouble> outphys2 = NullNekDouble1DArray;

        // Set test case
        SetTestCase(blocks, inphys.get(), false);

        // Calculate expected result from Nektar++
        fixt_explist->PhysDeriv(inphys, outphys0, outphys1, outphys2);

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
                    *(inptr++) = 0.0;
                }
            }
        }
    }

    void ExpectedSolution(const std::vector<BlockAttributes> &blocks,
                          double *inptr)
    {
        size_t el = 0, pts = 0;
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x);
        for (auto const &block : blocks)
        {
            for (size_t e = 0; e < block.num_elements; ++e, ++el)
            {
                size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                for (size_t phys = 0; phys < block.num_pts; ++phys, ++pts)
                {
                    double tmp = 0.0;
                    for (size_t i = 1; i < M / 2; i++)
                    {
                        tmp += i * std::pow(x[pts], i - 1);
                    }
                    *(inptr++) = tmp;
                }
            }
            for (size_t el = 0; el < block.num_padding_elements; ++el)
            {
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    *(inptr++) = 0.0;
                }
            }
        }
    }
};

class PhysDerivField2D : public PhysDerivField
{
public:
    PhysDerivField2D()
    {
    }

    void SetTestCase(const std::vector<BlockAttributes> &blocks, double *inptr,
                     bool padding = true)
    {
        size_t el = 0, pts = 0;
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        Array<OneD, double> y(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y);
        for (auto const &block : blocks)
        {
            for (size_t e = 0; e < block.num_elements; ++e, ++el)
            {
                size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                size_t N = fixt_explist->GetExp(el)->GetNumPoints(1);
                for (size_t phys = 0; phys < block.num_pts; ++phys, ++pts)
                {
                    double tmp = 0.0;
                    for (size_t i = 0; i < M / 2; i++)
                    {
                        for (size_t j = 0; j < N / 2; j++)
                        {
                            tmp += std::pow(x[pts], i) * std::pow(y[pts], j);
                        }
                    }
                    *(inptr++) = tmp;
                }
            }
            if (padding)
            {
                for (size_t el = 0; el < block.num_padding_elements; ++el)
                {
                    for (size_t phys = 0; phys < block.num_pts; ++phys)
                    {
                        *(inptr++) = 0.0;
                    }
                }
            }
        }
    }

    void NektarSolution(const std::vector<BlockAttributes> &blocks,
                        double *inptr)
    {
        Array<OneD, NekDouble> inphys(fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> outphys(fixt_explist->GetCoordim(0) *
                                       fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> outphys0 = outphys;
        Array<OneD, NekDouble> outphys1 =
            outphys0 + fixt_explist->GetTotPoints();
        Array<OneD, NekDouble> outphys2 = NullNekDouble1DArray;

        // Set test case
        SetTestCase(blocks, inphys.get(), false);

        // Calculate expected result from Nektar++
        fixt_explist->PhysDeriv(inphys, outphys0, outphys1, outphys2);

        // Copy expected result from Array to fixt_expected
        double *ptr = outphys.get();
        for (size_t k = 0; k < fixt_explist->GetCoordim(0); k++)
        {
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
                        *(inptr++) = 0.0;
                    }
                }
            }
        }
    }

    void ExpectedSolution(const std::vector<BlockAttributes> &blocks,
                          double *inptr)
    {
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        Array<OneD, double> y(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y);
        for (size_t n = 0; n < fixt_explist->GetCoordim(0); n++)
        {
            size_t el = 0, pts = 0;
            for (auto const &block : blocks)
            {
                for (size_t e = 0; e < block.num_elements; ++e, ++el)
                {
                    size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                    size_t N = fixt_explist->GetExp(el)->GetNumPoints(1);
                    for (size_t phys = 0; phys < block.num_pts; ++phys, ++pts)
                    {
                        double tmp = 0.0;
                        for (size_t i = 0; i < M / 2; i++)
                        {
                            for (size_t j = 0; j < N / 2; j++)
                            {
                                if (n == 0 && i >= 1)
                                {
                                    tmp += i * std::pow(x[pts], i - 1) *
                                           std::pow(y[pts], j);
                                }
                                if (n == 1 && j >= 1)
                                {
                                    tmp += j * std::pow(x[pts], i) *
                                           std::pow(y[pts], j - 1);
                                }
                            }
                        }
                        *(inptr++) = tmp;
                    }
                }
                for (size_t el = 0; el < block.num_padding_elements; ++el)
                {
                    for (size_t phys = 0; phys < block.num_pts; ++phys)
                    {
                        *(inptr++) = 0.0;
                    }
                }
            }
        }
    }
};

class PhysDerivField3D : public PhysDerivField
{
public:
    PhysDerivField3D()
    {
    }

    void SetTestCase(const std::vector<BlockAttributes> &blocks, double *inptr,
                     bool padding = true)
    {
        size_t el = 0, pts = 0;
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        Array<OneD, double> y(fixt_explist->GetTotPoints());
        Array<OneD, double> z(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y, z);
        for (auto const &block : blocks)
        {
            for (size_t e = 0; e < block.num_elements; ++e, ++el)
            {
                size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                size_t N = fixt_explist->GetExp(el)->GetNumPoints(1);
                size_t K = fixt_explist->GetExp(el)->GetNumPoints(2);
                for (size_t phys = 0; phys < block.num_pts; ++phys, ++pts)
                {
                    double tmp = 0.0;
                    for (size_t i = 0; i < M / 2; i++)
                    {
                        for (size_t j = 0; j < N / 2; j++)
                        {
                            for (size_t k = 0; k < K / 2; k++)
                            {
                                tmp += std::pow(x[pts], i) *
                                       std::pow(y[pts], j) *
                                       std::pow(z[pts], k);
                            }
                        }
                    }
                    *(inptr++) = tmp;
                }
            }
            if (padding)
            {
                for (size_t el = 0; el < block.num_padding_elements; ++el)
                {
                    for (size_t phys = 0; phys < block.num_pts; ++phys)
                    {
                        *(inptr++) = 0.0;
                    }
                }
            }
        }
    }

    void NektarSolution(const std::vector<BlockAttributes> &blocks,
                        double *inptr)
    {
        Array<OneD, NekDouble> inphys(fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> outphys(fixt_explist->GetCoordim(0) *
                                       fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> outphys0 = outphys;
        Array<OneD, NekDouble> outphys1 =
            outphys0 + fixt_explist->GetTotPoints();
        Array<OneD, NekDouble> outphys2 =
            outphys1 + fixt_explist->GetTotPoints();

        // Set test case
        SetTestCase(blocks, inphys.get(), false);

        // Calculate expected result from Nektar++
        fixt_explist->PhysDeriv(inphys, outphys0, outphys1, outphys2);

        // Copy expected result from Array to fixt_expected
        double *ptr = outphys.get();
        for (size_t k = 0; k < fixt_explist->GetCoordim(0); k++)
        {
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
                        *(inptr++) = 0.0;
                    }
                }
            }
        }
    }

    void ExpectedSolution(const std::vector<BlockAttributes> &blocks,
                          double *inptr)
    {
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        Array<OneD, double> y(fixt_explist->GetTotPoints());
        Array<OneD, double> z(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y, z);
        for (size_t n = 0; n < fixt_explist->GetCoordim(0); n++)
        {
            size_t el = 0, pts = 0;
            for (auto const &block : blocks)
            {
                for (size_t e = 0; e < block.num_elements; ++e, ++el)
                {
                    size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                    size_t N = fixt_explist->GetExp(el)->GetNumPoints(1);
                    size_t K = fixt_explist->GetExp(el)->GetNumPoints(2);
                    for (size_t phys = 0; phys < block.num_pts; ++phys, ++pts)
                    {
                        double tmp = 0.0;
                        for (size_t i = 0; i < M / 2; i++)
                        {
                            for (size_t j = 0; j < N / 2; j++)
                            {
                                for (size_t k = 0; k < K / 2; k++)
                                {
                                    if (n == 0 && i >= 1)
                                    {
                                        tmp += i * std::pow(x[pts], i - 1) *
                                               std::pow(y[pts], j) *
                                               std::pow(z[pts], k);
                                    }
                                    if (n == 1 && j >= 1)
                                    {
                                        tmp += j * std::pow(x[pts], i) *
                                               std::pow(y[pts], j - 1) *
                                               std::pow(z[pts], k);
                                    }
                                    if (n == 2 && k >= 1)
                                    {
                                        tmp += k * std::pow(x[pts], i) *
                                               std::pow(y[pts], j) *
                                               std::pow(z[pts], k - 1);
                                    }
                                }
                            }
                        }
                        *(inptr++) = tmp;
                    }
                }
                for (size_t el = 0; el < block.num_padding_elements; ++el)
                {
                    for (size_t phys = 0; phys < block.num_pts; ++phys)
                    {
                        *(inptr++) = 0.0;
                    }
                }
            }
        }
    }
};

class Seg : public PhysDerivField1D
{
public:
    Seg()
    {
        meshName = "run/line.xml";
    }
};

class Quad : public PhysDerivField2D
{
public:
    Quad()
    {
        meshName = "run/square.xml";
    }
};

class Tri : public PhysDerivField2D
{
public:
    Tri()
    {
        meshName = "run/tri.xml";
    }
};

class SquareAllElements : public PhysDerivField2D
{
public:
    SquareAllElements()
    {
        meshName = "run/square_all_elements.xml";
    }
};

class Hex : public PhysDerivField3D
{
public:
    Hex()
    {
        meshName = "run/hex.xml";
    }
};

class Prism : public PhysDerivField3D
{
public:
    Prism()
    {
        meshName = "run/prism.xml";
    }
};

class Pyr : public PhysDerivField3D
{
public:
    Pyr()
    {
        meshName = "run/pyr.xml";
    }
};

class Tet : public PhysDerivField3D
{
public:
    Tet()
    {
        meshName = "run/tet.xml";
    }
};

class CubePrismHex : public PhysDerivField3D
{
public:
    CubePrismHex()
    {
        meshName = "run/cube_prismhex.xml";
    }
};

class CubeAllElements : public PhysDerivField3D
{
public:
    CubeAllElements()
    {
        meshName = "run/cube_all_elements.xml";
    }
};
