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

    void SetTestCase(const std::vector<BlockAttributes> &blocks, double *inptr)
    {
        size_t pts = 0, el = 0;
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x);
        for (auto const &block : blocks)
        {
            for (; el < block.num_elements; ++el)
            {
                size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    double tmp = 0.0;
                    for (size_t i = 0; i < M; i++)
                    {
                        tmp += std::pow(x[pts], i);
                    }
                    pts++;
                    *(inptr++) = tmp;
                }
            }
        }
    }

    void ExpectedSolution(const std::vector<BlockAttributes> &blocks,
                          double *inptr)
    {
        size_t pts = 0, el = 0;
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x);
        for (auto const &block : blocks)
        {
            for (; el < block.num_elements; ++el)
            {
                size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    double tmp = 0.0;
                    for (size_t i = 0; i < M; i++)
                    {
                        tmp += i * std::pow(x[pts], i - 1);
                    }
                    pts++;
                    *(inptr++) = tmp;
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

    void SetTestCase(const std::vector<BlockAttributes> &blocks, double *inptr)
    {
        size_t pts = 0, el = 0;
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        Array<OneD, double> y(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y);
        for (auto const &block : blocks)
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                size_t N = fixt_explist->GetExp(el)->GetNumPoints(1);
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    double tmp = 0.0;
                    for (size_t i = 0; i < M; i++)
                    {
                        for (size_t j = 0; j < N; j++)
                        {
                            tmp += std::pow(x[pts], i) * std::pow(y[pts], j);
                        }
                    }
                    pts++;
                    *(inptr++) = tmp;
                }
            }
        }
    }

    void ExpectedSolution(const std::vector<BlockAttributes> &blocks,
                          double *inptr)
    {
        size_t pts = 0, el = 0;
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        Array<OneD, double> y(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y);
        for (auto const &block : blocks)
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                size_t N = fixt_explist->GetExp(el)->GetNumPoints(1);
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    double tmp = 0.0;
                    for (size_t i = 0; i < M; i++)
                    {
                        for (size_t j = 0; j < N; j++)
                        {
                            tmp += i * std::pow(x[pts], i - 1) *
                                   std::pow(y[pts], j);
                        }
                    }
                    pts++;
                    *(inptr++) = tmp;
                }
            }
        }
        pts = 0;
        for (auto const &block : blocks)
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                size_t N = fixt_explist->GetExp(el)->GetNumPoints(1);
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    double tmp = 0.0;
                    for (size_t i = 0; i < M; i++)
                    {
                        for (size_t j = 0; j < N; j++)
                        {
                            tmp += j * std::pow(x[pts], i) *
                                   std::pow(y[pts], j - 1);
                        }
                    }
                    pts++;
                    *(inptr++) = tmp;
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

    void SetTestCase(const std::vector<BlockAttributes> &blocks, double *inptr)
    {
        size_t pts = 0, el = 0;
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        Array<OneD, double> y(fixt_explist->GetTotPoints());
        Array<OneD, double> z(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y, z);
        for (auto const &block : blocks)
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                size_t N = fixt_explist->GetExp(el)->GetNumPoints(1);
                size_t K = fixt_explist->GetExp(el)->GetNumPoints(2);
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    double tmp = 0.0;
                    for (size_t i = 0; i < M; i++)
                    {
                        for (size_t j = 0; j < N; j++)
                        {
                            for (size_t k = 0; k < K; k++)
                            {
                                tmp += std::pow(x[pts], i) *
                                       std::pow(y[pts], j) *
                                       std::pow(z[pts], k);
                            }
                        }
                    }
                    pts++;
                    *(inptr++) = tmp;
                }
            }
        }
    }

    void ExpectedSolution(const std::vector<BlockAttributes> &blocks,
                          double *inptr)
    {
        size_t pts = 0, el = 0;
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        Array<OneD, double> y(fixt_explist->GetTotPoints());
        Array<OneD, double> z(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y, z);
        for (auto const &block : blocks)
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                size_t N = fixt_explist->GetExp(el)->GetNumPoints(1);
                size_t K = fixt_explist->GetExp(el)->GetNumPoints(2);
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    double tmp = 0.0;
                    for (size_t i = 0; i < M; i++)
                    {
                        for (size_t j = 0; j < N; j++)
                        {
                            for (size_t k = 0; k < K; k++)
                            {
                                tmp += i * std::pow(x[pts], i - 1) *
                                       std::pow(y[pts], j) *
                                       std::pow(z[pts], k);
                            }
                        }
                    }
                    pts++;
                    *(inptr++) = tmp;
                }
            }
        }
        pts = 0;
        for (auto const &block : blocks)
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                size_t N = fixt_explist->GetExp(el)->GetNumPoints(1);
                size_t K = fixt_explist->GetExp(el)->GetNumPoints(2);
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    double tmp = 0.0;
                    for (size_t i = 0; i < M; i++)
                    {
                        for (size_t j = 0; j < N; j++)
                        {
                            for (size_t k = 0; k < K; k++)
                            {
                                tmp += j * std::pow(x[pts], i) *
                                       std::pow(y[pts], j - 1) *
                                       std::pow(z[pts], k);
                            }
                        }
                    }
                    pts++;
                    *(inptr++) = tmp;
                }
            }
        }
        pts = 0;
        for (auto const &block : blocks)
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                size_t N = fixt_explist->GetExp(el)->GetNumPoints(1);
                size_t K = fixt_explist->GetExp(el)->GetNumPoints(2);
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    double tmp = 0.0;
                    for (size_t i = 0; i < M; i++)
                    {
                        for (size_t j = 0; j < N; j++)
                        {
                            for (size_t k = 0; k < K; k++)
                            {
                                tmp += k * std::pow(x[pts], i) *
                                       std::pow(y[pts], j) *
                                       std::pow(z[pts], k - 1);
                            }
                        }
                    }
                    pts++;
                    *(inptr++) = tmp;
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
        meshName = "line.xml";
    }
};

class Quad : public PhysDerivField2D
{
public:
    Quad()
    {
        meshName = "square.xml";
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
