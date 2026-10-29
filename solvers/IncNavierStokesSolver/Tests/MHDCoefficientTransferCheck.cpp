///////////////////////////////////////////////////////////////////////////////
// The MIT License
// Copyright (c) 2026 Nektar++ contributors.
// See the LICENSE file in the root directory for details.
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ContField.h>
#include <SolverUtils/Core/Coupling.h>
#include <SpatialDomains/MeshGraphIO.h>
#include <cmath>

using namespace Nektar;

// Integration test: asymmetric polynomials distinguish both element order
// and modal order in both directions. No PDE solution is involved here.
int main(int argc, char *argv[])
{
    auto session = LibUtilities::SessionReader::CreateInstance(argc, argv);
    auto graph   = SpatialDomains::MeshGraphIO::Read(session);
    auto field   = MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
        session, graph, session->GetVariable(0));
    auto coupling =
        SolverUtils::GetCouplingFactory().CreateInstance("Coefficients", field);
    auto options = coupling->GetConfig();
    bool fluid   = options.at("ROLE") == "Fluid";
    bool paired =
        session->GetComm()->GetRank() < std::stoi(options.at("FLUIDRANKS"));
    int nq = field->GetTotPoints();
    Array<OneD, NekDouble> x(nq), y(nq), z(nq);
    field->GetCoords(x, y, z);
    auto u = [](double x, double y) {
        return 1 + x + 2 * y + 3 * x * y + 4 * x * x * y;
    };
    auto v   = [](double x, double y) { return 5 - 2 * x + 7 * y + x * y * y; };
    auto phi = [](double x, double y) {
        return 2 + 3 * x * x + 5 * x * y + 7 * y * y + x * x * x * y;
    };
    std::vector<std::string> velocity{"u", "v", "mhd_time", "mhd_stop"};
    Array<OneD, Array<OneD, NekDouble>> values(4), reply(1);
    for (auto &a : values)
        a = Array<OneD, NekDouble>(nq, 0.0);
    reply[0] = Array<OneD, NekDouble>(nq, 0.0);
    Array<OneD, NekDouble> coeffs(field->GetNcoeffs(), 0.0);
    NekDouble error      = 0.0;
    const NekDouble time = 0.125;
    if (fluid)
    {
        for (int q = 0; q < nq; ++q)
        {
            values[0][q] = u(x[q], y[q]);
            values[1][q] = v(x[q], y[q]);
            values[2][q] = time;
            values[3][q] = -1;
        }
        coupling->Send(0, time, values, velocity);
        coupling->ReceiveCoefficients(0, time, coeffs);
        field->BwdTrans(coeffs, reply[0]);
        for (int q = 0; q < nq; ++q)
            error = std::max(error, std::abs(reply[0][q] - phi(x[q], y[q])));
    }
    else
    {
        coupling->Receive(0, time, values, velocity);
        for (int q = 0; q < nq; ++q)
        {
            error = std::max(
                error, std::abs(values[0][q] - (paired ? u(x[q], y[q]) : 0.0)));
            error = std::max(
                error, std::abs(values[1][q] - (paired ? v(x[q], y[q]) : 0.0)));
            error =
                std::max(error, std::abs(values[2][q] - (paired ? time : 0.0)));
            error =
                std::max(error, std::abs(values[3][q] - (paired ? -1.0 : 0.0)));
            reply[0][q] = phi(x[q], y[q]);
        }
        field->FwdTransLocalElmt(reply[0], coeffs);
        coupling->SendCoefficients(0, time, coeffs);
    }
    session->GetComm()->AllReduce(error, LibUtilities::ReduceMax);
    ASSERTL0(error < 1e-10,
             "Asymmetric polynomial coefficient transfer failed.");
    coupling->Finalize();
    if (session->GetComm()->GetRank() == 0)
        std::cout << "Coefficient polynomial transfer error: " << error
                  << std::endl;
    session->Finalise();
    return 0;
}
