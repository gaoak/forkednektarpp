///////////////////////////////////////////////////////////////////////////////
//
// File: init_neumannfields.hpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description:
//
///////////////////////////////////////////////////////////////////////////////

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
