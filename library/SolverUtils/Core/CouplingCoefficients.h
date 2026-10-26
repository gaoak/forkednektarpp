///////////////////////////////////////////////////////////////////////////////
//
// The MIT License
// Copyright (c) 2026 Nektar++ contributors.
// See the LICENSE file in the root directory for details.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SOLVERUTILS_COUPLINGCOEFFICIENTS_H
#define NEKTAR_SOLVERUTILS_COUPLINGCOEFFICIENTS_H

#include <SolverUtils/Core/Coupling.h>
#include <mpi.h>

namespace Nektar::SolverUtils
{

/// Matched MHD fluid partitions exchange element-local coefficients directly.
/// CWIPI supplies application-local communicators; a private intercommunicator
/// carries the paired messages without geometric location or interpolation.
class CouplingCoefficients : public Coupling
{
public:
    friend class MemoryManager<CouplingCoefficients>;
    static std::string className;
    static CouplingSharedPtr create(MultiRegions::ExpListSharedPtr field)
    {
        auto p = MemoryManager<CouplingCoefficients>::AllocateSharedPtr(field);
        p->Init();
        return p;
    }
    bool ExchangesCoefficients() const override
    {
        return true;
    }
    void SendCoefficients(int step, NekDouble time,
                          const Array<OneD, const NekDouble> &coeffs) override;
    void ReceiveCoefficients(int step, NekDouble time,
                             Array<OneD, NekDouble> &coeffs) override;

protected:
    explicit CouplingCoefficients(MultiRegions::ExpListSharedPtr field);
    void v_Init() override;
    void v_Send(int step, NekDouble time,
                const Array<OneD, const Array<OneD, NekDouble>> &fields,
                std::vector<std::string> &names) override;
    void v_Receive(int step, NekDouble time,
                   Array<OneD, Array<OneD, NekDouble>> &fields,
                   std::vector<std::string> &names) override;
    void v_Finalize() override;

private:
    std::vector<NekDouble> BuildLayout();
    void SendPacket(int step, NekDouble time, int stop,
                    const std::vector<NekDouble> &values);
    void ReceivePacket(int step, int nvalues);
    MPI_Comm m_local = MPI_COMM_NULL, m_bridge = MPI_COMM_NULL;
    bool m_fluid = false, m_paired = false;
    int m_rank = 0, m_dimension = 0, m_stop = -1;
    NekDouble m_time = 0.0;
    std::vector<int> m_indices;
    std::vector<NekDouble> m_packet;
    Array<OneD, NekDouble> m_coeffs;
};

} // namespace Nektar::SolverUtils
#endif
