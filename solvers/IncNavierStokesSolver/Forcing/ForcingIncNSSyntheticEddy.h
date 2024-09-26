///////////////////////////////////////////////////////////////////////////////
//
// File:  ForcingIncNSSyntheticEddy.h
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
// Description: Derived base class - Synthetic turbulence forcing
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SOLVERUTILS_FORCINGINCNSSYNTHETICEDDY
#define NEKTAR_SOLVERUTILS_FORCINGINCNSSYNTHETICEDDY

#include <SolverUtils/Forcing/Forcing.h>
#include <string>

namespace Nektar::SolverUtils
{

class ForcingIncNSSyntheticEddy : public SolverUtils::Forcing
{
public:
    friend class MemoryManager<ForcingIncNSSyntheticEddy>;

    /// Creates an instance of this class
    static ForcingSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const std::weak_ptr<EquationSystem> &pEquation,
        const Array<OneD, MultiRegions::ExpListSharedPtr> &pFields,
        const unsigned int &pNumForcingFields, const TiXmlElement *pForce)
    {
        ForcingSharedPtr p =
            MemoryManager<ForcingIncNSSyntheticEddy>::AllocateSharedPtr(
                pSession, pEquation);
        p->InitObject(pFields, pNumForcingFields, pForce);
        return p;
    }

    /// Name of the class
    static std::string className;

protected:
    void v_InitObject(
        const Array<OneD, MultiRegions::ExpListSharedPtr> &pFields,
        const unsigned int &pNumForcingFields,
        const TiXmlElement *pForce) override;

    void v_Apply(const Array<OneD, MultiRegions::ExpListSharedPtr> &pFields,
                 const Array<OneD, Array<OneD, NekDouble>> &inarray,
                 Array<OneD, Array<OneD, NekDouble>> &outarray,
                 const NekDouble &time) override;

    // Set Cholesky decomposition of the Reynolds Stresses in the domain
    void SetCholeskyReyStresses(
        const Array<OneD, MultiRegions::ExpListSharedPtr> &pFields);
    /// Set the Number of the Eddies in the box
    void SetNumberOfEddies();
    /// Set reference lengths
    void ComputeRefLenghts();
    /// Set Xi max
    void ComputeXiMax();
    /// Compute Random 1 or -1 value
    Array<OneD, Array<OneD, int>> GenerateRandomOneOrMinusOne();
    /// Compute Constant C
    NekDouble ComputeConstantC(int row, int col);
    /// Compute Gaussian
    NekDouble ComputeGaussian(NekDouble coord, NekDouble xiMaxVal,
                              NekDouble constC = 1.0);
    /// Check if point is inside the box of eddies
    bool InsideBoxOfEddies(NekDouble coord0, NekDouble coord1,
                           NekDouble coord2);
    /// Compute Stochastic Signal
    Array<OneD, Array<OneD, NekDouble>> ComputeStochasticSignal(
        const Array<OneD, MultiRegions::ExpListSharedPtr> &pFields);
    /// Compute Velocity Fluctuation
    Array<OneD, Array<OneD, NekDouble>> ComputeVelocityFluctuation(
        const Array<OneD, MultiRegions::ExpListSharedPtr> &pFields,
        Array<OneD, Array<OneD, NekDouble>> stochasticSignal);
    /// Compute Characteristic Convective Turbulent Time
    Array<OneD, Array<OneD, NekDouble>> ComputeCharConvTurbTime(
        const Array<OneD, MultiRegions::ExpListSharedPtr> &pFields);
    /// Compute Smoothing Factor
    Array<OneD, Array<OneD, NekDouble>> ComputeSmoothingFactor(
        const Array<OneD, MultiRegions::ExpListSharedPtr> &pFields,
        Array<OneD, Array<OneD, NekDouble>> convTurb);
    /// Set box of eddies mask
    void SetBoxOfEddiesMask(
        const Array<OneD, MultiRegions::ExpListSharedPtr> &pFields);
    /// Compute the initial position of the eddies inside the box
    void ComputeInitialRandomLocationOfEddies();
    /// Update positions of the eddies inside the box
    void UpdateEddiesPositions();
    /// Calculate Forcing
    void CalculateForcing(
        const Array<OneD, MultiRegions::ExpListSharedPtr> &pFields);
    /// Compute the initial location of the eddies for the test case
    void ComputeInitialLocationTestCase();
    /// Remove eddy from forcing term
    void RemoveEddiesFromForcing(
        const Array<OneD, MultiRegions::ExpListSharedPtr> &pFields);
    /// Initialise forcing term for each eddy
    void InitialiseForcingEddy(
        const Array<OneD, MultiRegions::ExpListSharedPtr> &pFields);

    // Members
    // Expressions (functions) of the prescribed Reynolds stresses
    std::map<int, LibUtilities::EquationSharedPtr> m_R;
    /// Cholesky decomposition of the Reynolds Stresses
    /// for all points in the mesh
    Array<OneD, Array<OneD, NekDouble>> m_Cholesky;
    /// Bulk velocity
    NekDouble m_Ub;
    /// Standard deviation
    NekDouble m_sigma;
    /// Inlet length in the y-direction and z-direction
    Array<OneD, NekDouble> m_lyz;
    /// Center of the inlet plane
    Array<OneD, NekDouble> m_rc;
    /// Number of eddies in the box
    int m_N;
    /// Characteristic lenght scales
    Array<OneD, NekDouble> m_l;
    /// Reference lenghts
    Array<OneD, NekDouble> m_lref;
    /// Xi max
    Array<OneD, NekDouble> m_xiMax;
    // XiMaxMin - Value form Giangaspero et al. 2022
    NekDouble m_xiMaxMin = 0.4;
    /// Space dimension
    int m_spacedim;
    /// Ration of specific heats
    NekDouble m_gamma;
    /// Box of eddies mask
    Array<OneD, int> m_mask;
    /// Eddy position
    Array<OneD, Array<OneD, NekDouble>> m_eddyPos;
    /// Check when the forcing should be applied
    bool m_calcForcing{true};
    /// Eddies that add to the domain
    std::vector<unsigned int> m_eddiesIDForcing;
    /// Current time
    NekDouble m_currTime;
    /// Keep applying force during GMRES iteration
    bool m_implicitForcing{false};
    /// Check for test case
    bool m_tCase;
    /// Forcing for each eddy
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> m_ForcingEddy;

private:
    ForcingIncNSSyntheticEddy(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const std::weak_ptr<EquationSystem> &pEquation);
    ~ForcingIncNSSyntheticEddy(void) override{};
};

} // namespace Nektar::SolverUtils

#endif
