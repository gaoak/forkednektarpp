#include "init_fields.hpp"

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class NeumannField
    : public InitFields<double, FieldState::Coeff, FieldState::Coeff,
                        MultiRegions::ContField>
{
public:
    NeumannField()
        : InitFields<double, FieldState::Coeff, FieldState::Coeff,
                     MultiRegions::ContField>()
    {
    }
};

class Helmholtz1D_Seg : public NeumannField
{
public:
    Helmholtz1D_Seg()
    {
        meshName = "run/Helmholtz1D_P8.xml";
    }
};

class Helmholtz2D_Tri_Quad : public NeumannField
{
public:
    Helmholtz2D_Tri_Quad()
    {
        meshName = "run/Helmholtz2D_P7_AllBCs.xml";
    }
};

class Helmholtz3D_Hex : public NeumannField
{
public:
    Helmholtz3D_Hex()
    {
        meshName = "run/Helmholtz3D_Hex_Heterogeneous.xml";
    }
};

class Helmholtz3D_Prism : public NeumannField
{
public:
    Helmholtz3D_Prism()
    {
        meshName = "run/Helmholtz3D_Prism_VarP.xml";
    }
};

class Helmholtz3D_Pyr : public NeumannField
{
public:
    Helmholtz3D_Pyr()
    {
        meshName = "run/Helmholtz3D_Pyr_VarP.xml";
    }
};

class Helmholtz3D_Tet : public NeumannField
{
public:
    Helmholtz3D_Tet()
    {
        meshName = "run/Helmholtz3D_Tet_VarP.xml";
    }
};
