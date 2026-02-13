///////////////////////////////////////////////////////////////////////////////
//
// File: RiemannSolverOp.hpp
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
// Description: RiemannSolver operator base class.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/Common/Operator.hpp"

#include "Operators/Math/MathKernels.hpp"
#include "Operators/SolverUtilsOps//RiemannSolvers/RiemannSolverKernels.hpp"

namespace Nektar::Operators
{

// RiemannSolver operator base class
template <typename TData> class RiemannSolverOp : public Operator<TData>
{
public:
    static std::shared_ptr<RiemannSolverOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &method = "", const std::string &execStr = "")
    {
        auto session = expansionList->GetSession();

        std::string method0 = method;
        if (method == "" && session->DefinesSolverInfo("UpwindType"))
        {
            method0 = session->GetSolverInfo("UpwindType");
        }

        std::string execStr0 = (execStr == "")
                                   ? Operator<TData>::GetOpExecSpace(session)
                                   : execStr;

        std::string requestedKey = method0 + execStr0;

        OperatorFactory<TData> &factory = GetOperatorFactory<TData>();

        // No suitable operator was found.
        if (!factory.ModuleExists(requestedKey))
        {
            std::stringstream msg;
            msg << "No such operator: " << requestedKey << std::endl;
            factory.PrintAvailableClasses(msg);
            NEKERROR(ErrorUtil::efatal, msg.str());
        }

        return std::static_pointer_cast<RiemannSolverOp<TData>>(
            factory.CreateInstance(requestedKey, expansionList, components));
    }

    void Apply(Field<TData, FieldState::Phys> &Fwd,
               Field<TData, FieldState::Phys> &Bwd,
               Field<TData, FieldState::Phys> &flux)
    {
        this->v_Apply(Fwd, Bwd, flux);
    }

    void operator()(Field<TData, FieldState::Phys> &Fwd,
                    Field<TData, FieldState::Phys> &Bwd,
                    Field<TData, FieldState::Phys> &flux)
    {
        this->v_Apply(Fwd, Bwd, flux);
    }

    void SetTraceAdvVel(Field<TData, FieldState::Phys> &traceAdvVel)
    {
        this->m_traceAdvVel = std::move(traceAdvVel);
    }

    void SetNormals(Field<TData, FieldState::Phys> &normals)
    {
        this->m_normals = std::move(normals);
    }

    void SetTraceNormals(Field<TData, FieldState::Phys> &traceNormals)
    {
        this->m_traceNormals = std::move(traceNormals);
    }

protected:
    Field<TData, FieldState::Phys> m_traceAdvVel;
    Field<TData, FieldState::Phys> m_normals;
    Field<TData, FieldState::Phys> m_traceNormals;

    RiemannSolverOp(const MultiRegions::ExpListSharedPtr &expansionList,
                    const std::vector<std::string> &components)
        : Operator<TData>(expansionList, components)
    {
        unsigned int coordDim  = expansionList->GetCoordim(0);
        size_t nTracePointsTot = expansionList->GetTrace()->GetTotPoints();
        Array<OneD, double> normals(coordDim * nTracePointsTot, 0.0);

        // Setting up the normals
        Array<OneD, Array<OneD, double>> traceNormals(coordDim);

        for (unsigned int i = 0; i < coordDim; ++i)
        {
            traceNormals[i] = Array<OneD, double>(nTracePointsTot, 0.0);
        }

        expansionList->GetTrace()->GetNormals(traceNormals);

        for (unsigned int i = 0; i < coordDim; ++i)
        {
            for (size_t j = 0; j < nTracePointsTot; ++j)
            {
                normals[i * nTracePointsTot + j] = traceNormals[i][j];
            }
        }

        // Create blocks.
        auto blocks_trace = GetBlockAttributes<TData, FieldState::Phys>(
            expansionList->GetTrace());

        // Create fields.
        unsigned int numHomoModes = 1;
        m_traceNormals            = Field<double, FieldState::Phys>(
            "m_traceNormals", blocks_trace, coordDim, numHomoModes);

        // Initialise fields
        m_traceNormals.template CopyArray<NektarSpaces::HostSpace>(normals);
    }

    ~RiemannSolverOp() override = default;

    virtual void v_Apply(Field<TData, FieldState::Phys> &Fwd,
                         Field<TData, FieldState::Phys> &Bwd,
                         Field<TData, FieldState::Phys> &flux) = 0;
};

} // namespace Nektar::Operators
