#include "init_fields.hpp"

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class CUDAKernelsField
    : public InitFields<double, FieldState::Phys, FieldState::Phys,
                        MultiRegions::ContField>
{
public:
    CUDAKernelsField()
        : InitFields<double, FieldState::Phys, FieldState::Phys,
                     MultiRegions::ContField>()
    {
    }

    void SetTestCase()
    {
        Array<OneD, NekDouble> x(fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> y(fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> z(fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> fce(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y, z);
        auto func1 = fixt_explist->GetSession()->GetFunction("Forcing", 0);
        func1->Evaluate(x, y, z, fce);
        std::copy(fce.get(), fce.get() + fixt_explist->GetTotPoints(),
                  fixt_in->GetStorage().GetCPUPtr());
        std::copy(
            fce.get(), fce.get() + fixt_explist->GetTotPoints(),
            fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
        auto func2 =
            fixt_explist->GetSession()->GetFunction("ExactSolution", 0);
        func2->Evaluate(x, y, z, fce);
        std::copy(fce.get(), fce.get() + fixt_explist->GetTotPoints(),
                  fixt_out->GetStorage().GetCPUPtr());
        std::copy(
            fce.get(), fce.get() + fixt_explist->GetTotPoints(),
            fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    }
};

class CUDAKernels : public CUDAKernelsField
{
public:
    CUDAKernels()
    {
        meshName = "run/Helmholtz3D_Hex_AllBCs_P6.xml";
    }
};
