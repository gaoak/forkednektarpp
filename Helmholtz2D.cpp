///////////////////////////////////////////////////////////////////////////////
//
// File: Helmholtz2D.cpp
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

#include <cstdio>
#include <cstdlib>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/Communication/Comm.h>
#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <MultiRegions/ContField.h>
#include <SpatialDomains/MeshGraph.h>

#include "Field.hpp"
#include "Operators/OperatorBwdTrans.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"
#include "Operators/OperatorIdentity.hpp"
#include "Operators/OperatorConjGrad.hpp"
#include "Operators/OperatorMass.hpp"
#include "Operators/OperatorFwdTrans.hpp"
#include "Operators/OperatorPhysDeriv.hpp"
#include "Operators/OperatorMatrix.hpp"
#include "Operators/OperatorDiagPrecon.hpp"
#include "Operators/OperatorHelmSolve.hpp"

using namespace std;
using namespace Nektar;
using namespace Nektar::SpatialDomains;
using namespace Nektar::MultiRegions;
using namespace Nektar::Operators;

//#define TIMING
#ifdef TIMING
#include <time.h>
#define Timing(s)                                                              \
    fprintf(stdout, "%s Took %g seconds\n", s,                                 \
            (clock() - st) / (double)CLOCKS_PER_SEC);                          \
    st = clock();
#else
#define Timing(s) /* Nothing */
#endif

int NoCaseStringCompare(const string &s1, const string &s2);

int main(int argc, char *argv[])
{
    LibUtilities::SessionReaderSharedPtr vSession = LibUtilities::SessionReader::CreateInstance(argc, argv);

    MultiRegions::ContFieldSharedPtr Exp, Fce;

    int i, nq, coordim;
    Array<OneD, NekDouble> fce;
    Array<OneD, NekDouble> xc0, xc1, xc2;
    StdRegions::ConstFactorMap factors;
    StdRegions::VarCoeffMap varcoeffs;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: Helmholtz2D meshfile [SysSolnType]   or   \n");
        exit(1);
    }

    try
    {
        LibUtilities::FieldIOSharedPtr fld = LibUtilities::FieldIO::CreateDefault(vSession);

        //----------------------------------------------
        // Read in mesh from input file
        SpatialDomains::MeshGraphSharedPtr graph2D = SpatialDomains::MeshGraph::Read(vSession);
        //----------------------------------------------

        //----------------------------------------------
        // Print summary of solution details
        factors[StdRegions::eFactorLambda] = vSession->GetParameter("Lambda");
        const SpatialDomains::ExpansionInfoMap &expansions = graph2D->GetExpansionInfo();
        LibUtilities::BasisKey bkey0 = expansions.begin()->second->m_basisKeyVector[0];

        if (vSession->GetComm()->GetRank() == 0)
        {
            cout << "Solving 2D Helmholtz: " << endl;
            cout << "         Communication: " << vSession->GetComm()->GetType()
                 << endl;
            cout << "         Solver type  : "
                 << vSession->GetSolverInfo("GlobalSysSoln") << endl;
            cout << "         Lambda       : "
                 << factors[StdRegions::eFactorLambda] << endl;
            cout << "         No. modes    : " << bkey0.GetNumModes() << endl;
            cout << endl;
        }
        //----------------------------------------------

        //----------------------------------------------
        // Define Expansion
        // Targets the u variable (only variable in session)
        Exp = MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(vSession, graph2D, vSession->GetVariable(0));
        
        // get blocks from expansion list
        auto blocks_phys = GetBlockAttributes(FieldState::Phys, Exp);
        auto blocks_coeff = GetBlockAttributes(FieldState::Coeff, Exp);

        //----------------------------------------------

        Timing("Read files and define exp ..");

        //----------------------------------------------
        // Set up coordinates of mesh for Forcing function evaluation
        coordim = Exp->GetCoordim(0);
        nq      = Exp->GetTotPoints();
        
        /*
        // PRINT MAPPING VALUES FROM THE ASSEMBLY
        auto locToGloMap = Exp->GetLocalToGlobalMap();
        std::cout << "Num elements = " << Exp->GetNumElmts() << "\n";
        std::cout << "Number of global coeffs = " << locToGloMap->GetNumGlobalCoeffs() << "\n";
        std::cout << "Number of local  coeffs = " << locToGloMap->GetNumLocalCoeffs() << "\n";
        std::cout << "Number of global bnd coeffs = " << locToGloMap->GetNumGlobalBndCoeffs() << "\n";
        std::cout << "Number of local bnd coeffs = " << locToGloMap->GetNumLocalBndCoeffs() << "\n";
        std::cout << "Number of local dir bnd coeffs = " << locToGloMap->GetNumLocalDirBndCoeffs() << "\n";
        int maxGloIdx = 0;
        for (int i = 0; i < locToGloMap->GetNumLocalCoeffs(); ++i)
        {
            int gloIdx = locToGloMap->GetLocalToGlobalMap(i);
            if (gloIdx > maxGloIdx)
                maxGloIdx = gloIdx;
            std::cout << "Local " << i << " --> Global " << gloIdx << "\n";
        }
        std::cout << "Max global idx = " << locToGloMap->GetNumGlobalCoeffs() - 1 << " --- " << maxGloIdx << "\n";
        */

        xc0 = Array<OneD, NekDouble>(nq, 0.0);
        xc1 = Array<OneD, NekDouble>(nq, 0.0);
        xc2 = Array<OneD, NekDouble>(nq, 0.0);

        // Get Coords calculates the values of all elemental quadrature points
        //std::cout << "Total points = " << nq << "\n";
        //std::cout << "Total points / number of elements = " << nq / Exp->GetNumElmts() << "\n";
        switch (coordim)
        {
            case 2:
                Exp->GetCoords(xc0, xc1);
                break;
            case 3:
                Exp->GetCoords(xc0, xc1, xc2);
                break;
            default:
                ASSERTL0(false, "Coordim not valid");
                break;
        }
        //----------------------------------------------


        //----------------------------------------------
        // Define forcing function for first variable defined in file
        // calculate the values of the forcing function at the quadrature points
        fce = Array<OneD, NekDouble>(nq);
        LibUtilities::EquationSharedPtr ffunc = vSession->GetFunction("Forcing", 0);
        ffunc->Evaluate(xc0, xc1, xc2, fce);

        // copy fce into field
        auto in = Field<double, FieldState::Phys>::create(blocks_phys);
        std::copy(fce.data(), fce.data() + in.GetStorage().size(), in.GetStorage().GetCPUPtr());

        //----------------------------------------------

        //----------------------------------------------
        // Setup expansion containing the  forcing function
        // Copies the expansion list describing the mesh and expansion polynomials for the problem
        // The physical point values are initially set to be equal to the fce array
        // the physical points represent function evaluatations at the quadrature points
        // used for integration / differentiation
        
        // not needed
        //Fce = MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(*Exp);
        //Fce->SetPhys(fce);
        
        //----------------------------------------------
        Timing("Define forcing ..");

        // *********************************************************************************
        // *********************************************************************************

        //----------------------------------------------
        // Helmholtz solution taking physical forcing after setting
        // initial condition to zero
        // Vmath::Zero(Exp->GetNcoeffs(), Exp->UpdateCoeffs(), 1); // initially set coefficients to zero
        auto out = Field<double, FieldState::Coeff>::create(blocks_coeff);

        // note Exp->UpdateCoeffs() returns reference to underlying array
        // Do HelmSolve using a const reference to the physical point values, a reference to the coeffs,
        // the factors map and the variable coeffs map
        //Exp->HelmSolve(Fce->GetPhys(), Exp->UpdateCoeffs(), factors, varcoeffs);
        
        auto helmSolveOp = HelmSolve<double>::create(Exp);
        auto diagPreconOp = DiagPrecon<double>::create(Exp);

        helmSolveOp->setPrecon(diagPreconOp);

        helmSolveOp->setLambda(double(vSession->GetParameter("Lambda")));

        std::cout << "Applying HelmSolve...\n";
        helmSolveOp->apply(in, out);
        std::cout << "Applied HelmSolve\n";

        //----------------------------------------------
        // Backward Transform Solution to get solved values
        // Undertakes backward transform which converts from coefficients to physical points
        // Reads in a constant reference to the coeffs and writes to a refernece of the physical points
        
        //Exp->BwdTrans(Exp->GetCoeffs(), Exp->UpdatePhys());
        
        // BwdTrans<>::create()->apply(out,.)
        auto outPhys = Field<double, FieldState::Phys>::create(blocks_phys);
        BwdTrans<double>::create(Exp)->apply(out, outPhys);
        
        std::cout << "BwdTrans done\n";

        //----------------------------------------------

        // *********************************************************************************
        // *********************************************************************************

        //-----------------------------------------------
        // Write solution to file
        string out_file = vSession->GetSessionName() + ".fld";
        std::vector<LibUtilities::FieldDefinitionsSharedPtr> FieldDef =
            Exp->GetFieldDefinitions();
        std::vector<std::vector<NekDouble>> FieldData(FieldDef.size());

        for (i = 0; i < FieldDef.size(); ++i)
        {
            FieldDef[i]->m_fields.push_back("u");
            Exp->AppendFieldData(FieldDef[i], FieldData[i]);
        }
        fld->Write(out_file, FieldDef, FieldData);

        //-----------------------------------------------

        //----------------------------------------------
        // See if there is an exact solution, if so
        // evaluate and plot errors
        LibUtilities::EquationSharedPtr ex_sol = vSession->GetFunction("ExactSolution", 0);

        if (ex_sol)
        {
            //----------------------------------------------
            // evaluate exact solution
            ex_sol->Evaluate(xc0, xc1, xc2, fce);

            // Segmentation fault here!
            Fce->SetPhys(fce);
            Fce->SetPhysState(true);

            //--------------------------------------------

            //--------------------------------------------
            // Calculate errors
            NekDouble vLinfError = Exp->Linf(Exp->GetPhys(), Fce->GetPhys());
            NekDouble vL2Error   = Exp->L2(Exp->GetPhys(), Fce->GetPhys());
            NekDouble vH1Error   = Exp->H1(Exp->GetPhys(), Fce->GetPhys());

            if (vSession->GetComm()->GetRank() == 0)
            {
                cout << "L infinity error: " << vLinfError << endl;
                cout << "L 2 error:        " << vL2Error << endl;
                cout << "H 1 error:        " << vH1Error << endl;
            }
            //--------------------------------------------
        }

        
        std::cout << "Test1\n";
        //----------------------------------------------
    }
    catch (const std::runtime_error &)
    {
        cout << "Caught an error" << endl;
        return 1;
    }

    vSession->Finalise();   
}

/**
 * Performs a case-insensitive string comparison (from web).
 * @param   s1          First string to compare.
 * @param   s2          Second string to compare.
 * @returns             0 if the strings match.
 */
int NoCaseStringCompare(const string &s1, const string &s2)
{
    string::const_iterator it1 = s1.begin();
    string::const_iterator it2 = s2.begin();

    // stop when either string's end has been reached
    while ((it1 != s1.end()) && (it2 != s2.end()))
    {
        if (::toupper(*it1) != ::toupper(*it2)) // letters differ?
        {
            // return -1 to indicate smaller than, 1 otherwise
            return (::toupper(*it1) < ::toupper(*it2)) ? -1 : 1;
        }

        // proceed to the next character in each string
        ++it1;
        ++it2;
    }

    size_t size1 = s1.size();
    size_t size2 = s2.size(); // cache lengths

    // return -1,0 or 1 according to strings' lengths
    if (size1 == size2)
    {
        return 0;
    }

    return (size1 < size2) ? -1 : 1;
}
