#include "init_fields.hpp"

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class AddTraceIntegralField
    : public InitFields<double, FieldState::Phys, FieldState::Coeff,
                        MultiRegions::DisContField>
{
public:
    AddTraceIntegralField()
        : InitFields<double, FieldState::Phys, FieldState::Coeff,
                     MultiRegions::DisContField>()
    {
    }

    /*
     *  Re-Initialise the input blocks based on the Trace-ExpList for this
     * operator Delete previouisly defined fixt_in (also for CUDA) and re-define
     * input based on TraceExpList
     */
    void ReConfigure(size_t nin = 1, size_t nout = 1)
    {
        if (fixt_in)
        {
            delete fixt_in;
        }
        const FieldState stateIn = FieldState::Phys;
        auto blocks_in =
            GetBlockAttributes(stateIn, fixt_explist->GetTrace(), vec_t::width);
        auto f_in =
            Field<double, stateIn>::create(blocks_in, nin, vec_t::alignment);
        fixt_in = new Field<double, stateIn>(std::move(f_in));

#ifdef NEKTAR_USE_CUDA
        if (fixtcuda_in)
        {
            delete fixtcuda_in;
        }
        auto fcuda_in =
            Field<double, stateIn>::template create<MemoryRegionCUDA>(blocks_in,
                                                                      nin);
        fixtcuda_in = new Field<double, stateIn>(std::move(fcuda_in));
#endif
    }

    void SetTestCase(const std::vector<BlockAttributes> &blocks, double *inptr,
                     bool padding = true)
    {
        for (auto const &block : fixt_in->GetBlocks())
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
        auto fixt_explist_trace = fixt_explist->GetTrace();
        Array<OneD, NekDouble> inTracephys(fixt_explist_trace->GetNpoints(),
                                           0.0);
        Array<OneD, NekDouble> outFieldcoeffs(fixt_explist->GetNcoeffs(), 0.0);

        // Set test case
        SetTestCase(blocks, inTracephys.get(), false);

        // Calculate expected result from Nektar++
        fixt_explist->AddTraceIntegral(inTracephys, outFieldcoeffs);

        // Copy expected result from Array to fixt_expected
        double *ptr = outFieldcoeffs.get();
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

class Seg : public AddTraceIntegralField
{
public:
    Seg()
    {
        meshName = "run/line.xml";
    }
};

class Quad : public AddTraceIntegralField
{
public:
    Quad()
    {
        meshName = "run/square.xml";
    }
};

class Tri : public AddTraceIntegralField
{
public:
    Tri()
    {
        meshName = "run/tri.xml";
    }
};

class SquareAllElements : public AddTraceIntegralField
{
public:
    SquareAllElements()
    {
        meshName = "run/square_all_elements.xml";
    }
};

class Hex : public AddTraceIntegralField
{
public:
    Hex()
    {
        meshName = "run/hex.xml";
    }
};

class Prism : public AddTraceIntegralField
{
public:
    Prism()
    {
        meshName = "run/prism.xml";
    }
};

class Pyr : public AddTraceIntegralField
{
public:
    Pyr()
    {
        meshName = "run/pyr.xml";
    }
};

class Tet : public AddTraceIntegralField
{
public:
    Tet()
    {
        meshName = "run/tet.xml";
    }
};

class CubePrismHex : public AddTraceIntegralField
{
public:
    CubePrismHex()
    {
        meshName = "run/cube_prismhex.xml";
    }
};

class CubeAllElements : public AddTraceIntegralField
{
public:
    CubeAllElements()
    {
        meshName = "run/cube_all_elements.xml";
    }
};
