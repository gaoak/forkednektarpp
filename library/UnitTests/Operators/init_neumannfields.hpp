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

#include "Operators/BndCondOps/OperatorNeuBndCond.hpp"

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

    void SetTestCase(
        [[maybe_unused]] const std::vector<BlockAttributes> &blocks,
        [[maybe_unused]] double *inptr)
    {
    }

    void ExpectedSolution(const std::vector<BlockAttributes> &blocks,
                          double *inptr)
    {
        using ExecSpace     = NektarSpaces::Serial;
        using Impl          = Operators::StdMat;
        const auto stateOut = FieldState::Coeff;

        auto blocks_tmp = GetBlockAttributes(stateOut, fixt_explist,
                                             NektarSpaces::Serial::width);
        auto f_tmp =
            Field<double, stateOut>::template create<NektarSpaces::HostSpace>(
                "f_out", blocks_tmp, 1, NektarSpaces::Serial::alignment);
        auto fixt_tmp = new Field<double, stateOut>(std::move(f_tmp));

        NeuBndCond<>::template create<ExecSpace, Impl>(fixt_explist)
            ->apply(*fixt_tmp);

        // Copy expected result from Array to fixt_expected
        const double *ptr =
            fixt_tmp->template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        for (size_t bl = 0; bl < blocks.size(); bl++)
        {
            size_t cnt = 0;
            for (size_t el = 0; el < blocks[bl].num_elements; ++el)
            {
                for (size_t coeff = 0; coeff < blocks[bl].num_pts;
                     ++coeff, ++cnt)
                {
                    inptr[cnt] = ptr[cnt];
                }
            }

            inptr += blocks[bl].block_size;
            ptr += blocks_tmp[bl].block_size;
        }

        delete fixt_tmp;
    }
};

#define TEST(type, filename)                                                   \
    class type : public NeumannField                                           \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };

TEST(Helmholtz1D_Seg, "run/Helmholtz1D_P8.xml")

TEST(Helmholtz2D_Tri_Quad, "run/Helmholtz2D_P7_AllBCs.xml")

TEST(Helmholtz3D_Hex, "run/Helmholtz3D_Hex_Heterogeneous.xml")

TEST(Helmholtz3D_Prism, "run/Helmholtz3D_Prism_VarP.xml")

TEST(Helmholtz3D_Pyr, "run/Helmholtz3D_Pyr_VarP.xml")

TEST(Helmholtz3D_Tet, "run/Helmholtz3D_Tet_VarP.xml")
