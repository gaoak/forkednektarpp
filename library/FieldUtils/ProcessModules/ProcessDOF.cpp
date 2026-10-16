////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessDOF.cpp
//
//  For more information, please see: http://www.nektar.info/
//
//  The MIT License
//
//  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
//  Department of Aeronautics, Imperial College London (UK), and Scientific
//  Computing and Imaging Institute, University of Utah (USA).
//
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//  Description: Outputs the number of global continuous Galerkin DOF.
//
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/DisContField.h>

#include "ProcessDOF.h"

using namespace std;

namespace Nektar::FieldUtils
{
namespace
{
// Reuse the solver's periodic identification on the base mesh. Restrict CG
// collectives to non-empty partitions of the same homogeneous plane/line.
class DOFField : public MultiRegions::DisContField
{
public:
    DOFField(const MultiRegions::ExpList &exp,
             const LibUtilities::CommSharedPtr &comm,
             const SpatialDomains::BoundaryConditions &bcs,
             const string &variable)
        : DisContField(exp)
    {
        m_comm = comm;
        FindPeriodicTraces(bcs, variable);
    }
};
} // namespace

ModuleKey ProcessDOF::className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eProcessModule, "dof"), ProcessDOF::create,
    "Computes number of global continuous Galerkin DOF.");

ProcessDOF::ProcessDOF(FieldSharedPtr f) : ProcessModule(f)
{
}

ProcessDOF::~ProcessDOF()
{
}

void ProcessDOF::v_Process(po::variables_map &vm)
{
    m_f->SetUpExp(vm);

    auto comm                = m_f->m_comm->GetSpaceComm();
    vector<string> variables = m_f->m_variables;
    if (variables.empty())
    {
        variables = m_f->m_session->GetVariables();
    }
    if (variables.empty())
    {
        variables.push_back("DefaultVar");
    }
    // Empty spatial partitions may not have read the field-file variables.
    // Share the names before reducing arrays whose size depends on them.
    int source = m_f->m_variables.empty() ? comm->GetSize() : comm->GetRank();
    comm->AllReduce(source, LibUtilities::ReduceMin);
    if (source == comm->GetSize())
    {
        source = 0;
    }
    int nfields = variables.size();
    comm->Bcast(nfields, source);
    variables.resize(nfields);
    for (auto &var : variables)
    {
        int length = var.size();
        comm->Bcast(length, source);
        vector<char> name(var.begin(), var.end());
        name.resize(length);
        comm->Bcast(name, source);
        var.assign(name.begin(), name.end());
    }

    // As in ProcessForceDecomposeBnd, boundary region IDs are distinct from
    // local expansion indices and may only be present on some partitions.
    SpatialDomains::BoundaryRegionCollection bregions;
    SpatialDomains::BoundaryConditionCollection bconditions;
    int nBnds = 0;
    if (m_f->m_session->DefinesElement("Nektar/Conditions"))
    {
        SpatialDomains::BoundaryConditions bcs(m_f->m_session, m_f->m_graph);
        bregions    = bcs.GetBoundaryRegions();
        bconditions = bcs.GetBoundaryConditions();
        for (const auto &breg : bregions)
        {
            nBnds = max(nBnds, breg.first + 1);
        }
    }
    comm->AllReduce(nBnds, LibUtilities::ReduceMax);
    Array<OneD, int> bndPresent(nBnds, 0);
    for (const auto &breg : bregions)
    {
        bndPresent[breg.first] = 1;
    }
    if (nBnds != 0)
    {
        comm->AllReduce(bndPresent, LibUtilities::ReduceMax);
    }

    // CG maps cannot be constructed on empty expansion lists. Exclude empty
    // partitions while retaining the row communicator used by the CG solver.
    auto cgComm = m_f->m_session->GetComm()->GetRowComm()->CommCreateIf(
        m_f->m_exp[0]->GetNumElmts() != 0);

    Array<OneD, int> fieldDOF(nfields, 0);
    Array<OneD, int> bndDOF(nfields * nBnds, 0);
    if (cgComm)
    {
        int nstrips =
            m_f->m_variables.empty() ? 1 : m_f->m_exp.size() / nfields;
        for (int i = 0; i < nfields; ++i)
        {
            const auto &var = variables[i];
            // Each variable may use a different expansion order.
            MultiRegions::ExpList baseExp(m_f->m_session, m_f->m_graph, false,
                                          var, Collections::eNoCollection);
            int nCopies = nstrips;
            if (m_f->m_numHomogeneousDir != 0)
            {
                nCopies *= m_f->m_exp[0]->GetExpSize() / baseExp.GetExpSize();
            }

            Array<OneD, MultiRegions::ExpListSharedPtr> bndExp(bregions.size());
            Array<OneD, SpatialDomains::BoundaryConditionShPtr> bndConditions(
                bregions.size());
            SpatialDomains::BoundaryConditions fieldBcs;
            map<int, pair<int, int>> bndRanges;
            int bnd = 0, offset = 0;
            for (const auto &breg : bregions)
            {
                auto region = breg.second;
                fieldBcs.AddBoundaryRegions(breg.first, region);
                auto conditions =
                    make_shared<SpatialDomains::BoundaryConditionMap>();
                auto it = bconditions.find(breg.first);
                if (it != bconditions.end() && it->second->count(var))
                {
                    (*conditions)[var] = it->second->at(var);
                }
                else
                {
                    // With mesh-only input no essential constraints are
                    // imposed. Neumann conditions leave all CG DOF present.
                    (*conditions)[var] =
                        make_shared<SpatialDomains::NeumannBoundaryCondition>(
                            m_f->m_session, "0", "NoUserDefined", "", cgComm);
                }
                fieldBcs.AddBoundaryConditions(breg.first, conditions);
                bndConditions[bnd] = conditions->at(var);
                bndExp[bnd] =
                    MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
                        m_f->m_session, *region, m_f->m_graph, false, var,
                        false, cgComm, Collections::eNoCollection);
                int ncoeffs           = bndExp[bnd]->GetNcoeffs();
                bndRanges[breg.first] = make_pair(offset, ncoeffs);
                offset += ncoeffs;
                ++bnd;
            }

            DOFField field(baseExp, cgComm, fieldBcs, var);
            MultiRegions::PeriodicMap periodicVerts, periodicEdges,
                periodicFaces;
            field.GetPeriodicEntities(periodicVerts, periodicEdges,
                                      periodicFaces);
            auto cgMap =
                MemoryManager<MultiRegions::AssemblyMapCG>::AllocateSharedPtr(
                    m_f->m_session, field.GetNcoeffs(), field, bndExp,
                    bndConditions, false, var, periodicVerts, periodicEdges,
                    periodicFaces);

            // A process-local global index can still be shared with other
            // partitions. Count only the owner selected by the CG map.
            const auto &unique = cgMap->GetGlobalToUniversalMapUnique();
            for (int g = 0; g < cgMap->GetNumGlobalCoeffs(); ++g)
            {
                if (unique[g] > 0)
                {
                    fieldDOF[i] += nCopies;
                }
            }

            const auto &bndToLocal = cgMap->GetBndCondCoeffsToLocalCoeffsMap();
            const auto &localToGlobal = cgMap->GetLocalToGlobalMap();
            for (int b = 0; b < nBnds; ++b)
            {
                if (!bndPresent[b])
                {
                    continue;
                }
                Array<OneD, NekDouble> onBoundary(
                    cgMap->GetNumGlobalBndCoeffs(), 0.0);
                auto range = bndRanges.find(b);
                if (range != bndRanges.end())
                {
                    int start = range->second.first;
                    int end   = start + range->second.second;
                    for (int k = start; k < end; ++k)
                    {
                        int local = bndToLocal[k];
                        // Variable-order CG may suppress excess trace modes.
                        if (!cgMap->GetSignChange() ||
                            cgMap->GetLocalToGlobalSign(local) != 0.0)
                        {
                            onBoundary[localToGlobal[local]] = 1.0;
                        }
                    }
                }
                // The owner may not itself touch this boundary. Propagate
                // membership before applying the unique ownership mask.
                cgMap->UniversalAssembleBnd(onBoundary);
                for (int g = 0; g < cgMap->GetNumGlobalBndCoeffs(); ++g)
                {
                    if (onBoundary[g] > 0.0 && unique[g] > 0)
                    {
                        bndDOF[i * nBnds + b] += nCopies;
                    }
                }
            }
        }
    }
    comm->AllReduce(fieldDOF, LibUtilities::ReduceSum);
    if (nBnds != 0)
    {
        comm->AllReduce(bndDOF, LibUtilities::ReduceSum);
    }

    if (comm->GetRank() == 0)
    {
        long long totalDOF = 0;
        for (int i = 0; i < nfields; ++i)
        {
            cout << "Number of DOF for variable " << variables[i] << ": "
                 << fieldDOF[i] << endl;
            totalDOF += fieldDOF[i];
            for (int b = 0; b < nBnds; ++b)
            {
                if (bndPresent[b])
                {
                    cout << "Number of DOF for variable " << variables[i]
                         << " on boundary " << b << ": "
                         << bndDOF[i * nBnds + b] << endl;
                }
            }
        }
        for (int b = 0; b < nBnds; ++b)
        {
            if (bndPresent[b])
            {
                long long totalBndDOF = 0;
                for (int i = 0; i < nfields; ++i)
                {
                    totalBndDOF += bndDOF[i * nBnds + b];
                }
                cout << "Total number of DOF on boundary " << b << ": "
                     << totalBndDOF << endl;
            }
        }
        cout << "Total number of DOF: " << totalDOF << endl;
    }
}
} // namespace Nektar::FieldUtils
