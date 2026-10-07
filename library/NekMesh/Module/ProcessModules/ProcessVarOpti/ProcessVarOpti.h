////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessVarOpti.h
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
//  Description:
//
////////////////////////////////////////////////////////////////////////////////

#ifndef UTILITIES_NEKMESH_PROCESSVAROPTI
#define UTILITIES_NEKMESH_PROCESSVAROPTI

#include <NekMesh/Module/Module.h>

#include "ElUtil.h"

namespace Nektar::NekMesh
{

struct DerivUtil
{
    NekMatrix<NekDouble> VdmD[3];
    NekMatrix<NekDouble> VdmDStd[3]; // deriv matrix without interp
    NekVector<NekDouble> quadW;

    std::vector<std::vector<NekDouble>> basisDeriv;

    int pts;
    int ptsStd;
};
typedef std::shared_ptr<DerivUtil> DerivUtilSharedPtr;

/**
 * @brief Build the derivative and interpolation operators, and the quadrature
 * weights, for each element type the optimiser supports.
 *
 * @param nummode  Number of modes of the mesh, i.e. its polynomial order + 1.
 * @param overInt  Orders of over-integration to use when evaluating the
 *                 functional, over and above the order of the mesh.
 *
 * A free function rather than a member so that the evaluation of the
 * functional can be set up, and checked, without a mesh or a module.
 */
NEKMESH_EXPORT std::map<LibUtilities::ShapeType, DerivUtilSharedPtr> BuildDerivUtil(
    int nummode, int overInt);

enum optiType
{
    eLinEl,
    eWins,
    eRoca,
    eHypEl
};

struct Residual
{
    NekDouble val;
    int n;
    int nDoF;
    /// Free nodes confined to a CAD curve or surface, which slide along it
    /// rather than moving through space. Zero without CAD, and zero with it
    /// if the boundary is being held fixed.
    int nOnCAD;
    int startInv;
    int nReset[3];
    NekDouble worstJac;
    /// The smallest Jacobian anywhere in the mesh, at the integration points,
    /// as of the start of the current iteration. The Jacobian regularisation
    /// is set from this: it is a property of the mesh rather than of any one
    /// node, so that every local problem within an iteration minimises the
    /// same functional.
    NekDouble minJac;
    NekDouble func;
    int alphaI;
    /// Nodes left alone because the functional there was not a finite number.
    int nSkipped;
};

typedef std::shared_ptr<Residual> ResidualSharedPtr;

class ProcessVarOpti : public ProcessModule
{
public:
    /// Creates an instance of this class
    static std::shared_ptr<Module> create(MeshSharedPtr m)
    {
        return MemoryManager<ProcessVarOpti>::AllocateSharedPtr(m);
    }
    static ModuleKey className;

    ProcessVarOpti(MeshSharedPtr m);
    ~ProcessVarOpti() override;

    void Process() override;

    std::string GetModuleName() override
    {
        return "ProcessVarOpti";
    }

private:
    void Analytics();

    // Keyed by node rather than by node id: the high-order nodes that
    // MakeOrder generates all carry id 0, so identity is the pointer.
    typedef std::unordered_map<SpatialDomains::PointGeom *,
                               std::vector<ElUtilSharedPtr>>
        NodeElMap;

    void GetElementMap(
        int o, std::map<LibUtilities::ShapeType, DerivUtilSharedPtr> derMap);
    std::vector<ElUtilSharedPtr> GetLockedElements(NekDouble thres);
    std::vector<std::vector<SpatialDomains::PointGeom *>> CreateColoursets(
        std::vector<SpatialDomains::PointGeom *> remain);
    std::vector<std::vector<SpatialDomains::PointGeom *>> GetColouredNodes(
        std::vector<ElUtilSharedPtr> elLock);

    void RemoveLinearCurvature();

    LibUtilities::Interpolator GetScalingFieldFromFile(std::string file);
    LibUtilities::Interpolator GetField(
        Array<OneD, Array<OneD, NekDouble>> inPts);

    NodeElMap m_nodeElMap;
    std::vector<ElUtilSharedPtr> m_dataSet;
    AdaptCurveVector m_adaptCurves;
    bool m_radaptCAD;
    int m_nummode = 0;

    ResidualSharedPtr m_res;
    optiType m_opti;
};
} // namespace Nektar::NekMesh

#endif
