///////////////////////////////////////////////////////////////////////////////
//
// File: init_ipwrtbasefields.hpp
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

class IProductWRTBaseField
    : public InitFields<double, FieldState::Phys, FieldState::Coeff>
{
public:
    IProductWRTBaseField()
        : InitFields<double, FieldState::Phys, FieldState::Coeff>()
    {
    }

    void SetTestCase(const std::vector<BlockAttributes> &blocks, double *inptr,
                     bool padding = true)
    {
        for (auto const &block : blocks)
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
        Array<OneD, NekDouble> inphys(fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> outcoeffs(fixt_explist->GetNcoeffs());

        // Set test case
        SetTestCase(fixt_in->GetBlocks(), inphys.get(), false);

        // Calculate expected result from Nektar++
        fixt_explist->IProductWRTBase(inphys, outcoeffs);

        // Copy expected result from Array to pointer
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

class Seg : public IProductWRTBaseField
{
public:
    Seg()
    {
        meshName = "run/line.xml";
    }
};

class Quad : public IProductWRTBaseField
{
public:
    Quad()
    {
        meshName = "run/square.xml";
    }
};

class Tri : public IProductWRTBaseField
{
public:
    Tri()
    {
        meshName = "run/tri.xml";
    }
};

class SquareAllElements : public IProductWRTBaseField
{
public:
    SquareAllElements()
    {
        meshName = "run/square_all_elements.xml";
    }
};

class Hex : public IProductWRTBaseField
{
public:
    Hex()
    {
        meshName = "run/hex.xml";
    }
};

class Prism : public IProductWRTBaseField
{
public:
    Prism()
    {
        meshName = "run/prism.xml";
    }
};

class Pyr : public IProductWRTBaseField
{
public:
    Pyr()
    {
        meshName = "run/pyr.xml";
    }
};

class Tet : public IProductWRTBaseField
{
public:
    Tet()
    {
        meshName = "run/tet.xml";
    }
};

class CubePrismHex : public IProductWRTBaseField
{
public:
    CubePrismHex()
    {
        meshName = "run/cube_prismhex.xml";
    }
};

class CubeAllElements : public IProductWRTBaseField
{
public:
    CubeAllElements()
    {
        meshName = "run/cube_all_elements.xml";
    }
};
