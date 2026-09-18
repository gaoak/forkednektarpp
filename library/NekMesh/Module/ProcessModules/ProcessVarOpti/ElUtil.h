////////////////////////////////////////////////////////////////////////////////
//
//  File: ElUtil.h
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

#ifndef UTILITIES_NEKMESH_PROCESSVAROPTI_ELUTIL
#define UTILITIES_NEKMESH_PROCESSVAROPTI_ELUTIL

#include <LibUtilities/BasicUtils/Thread.h>

#include <NekMesh/Module/Module.h>

#include <LibUtilities/BasicUtils/Interpolator.h>
#include <LibUtilities/BasicUtils/PtsField.h>

#include <SpatialDomains/CADSystem/CADCurve.h>
#include <SpatialDomains/Geometry.h>

typedef Nektar::LibUtilities::PtsFieldSharedPtr PtsFieldSharedPtr;

namespace Nektar::NekMesh
{

struct DerivUtil;
struct Residual;

typedef std::shared_ptr<DerivUtil> DerivUtilSharedPtr;
typedef std::shared_ptr<Residual> ResidualSharedPtr;

/**
 * @brief A CAD curve to refine towards, paired with its bounding box in the
 * same {xmin, ymin, zmin, xmax, ymax, zmax} layout as
 * SpatialDomains::Geometry::GetBoundingBox(), so that the two can be tested
 * for overlap directly.
 */
typedef std::vector<
    std::pair<SpatialDomains::CADCurveSharedPtr, std::array<NekDouble, 6>>>
    AdaptCurveVector;

class ElUtilJob;

class ElUtil : public std::enable_shared_from_this<ElUtil>
{
public:
    ElUtil(SpatialDomains::Geometry *e, DerivUtilSharedPtr d, ResidualSharedPtr,
           int n, int o);

    ElUtilJob *GetJob(bool update = false);
    ElUtilJob *GetAdaptJob(AdaptCurveVector &adaptCurves, NekDouble scale,
                           NekDouble rad, SpatialDomains::MeshGraph *graph);

    int GetId()
    {
        return m_el->GetGlobalID();
    }

    // Leaving these varibles as public for sake of efficiency
    std::vector<std::vector<NekDouble *>> nodes;
    std::vector<std::vector<NekDouble>> maps, mapsStd;

    void Evaluate();
    void InitialMinJac();

    SpatialDomains::Geometry *GetEl()
    {
        return m_el;
    }

    int NodeId(SpatialDomains::PointGeom *in)
    {
        return m_idmap[in];
    }

    NekDouble GetScaledJac()
    {
        return m_scaledJac;
    }

    NekDouble &GetMinJac()
    {
        return m_minJac;
    }

    void SetScaling(LibUtilities::Interpolator interp)
    {
        m_interp = interp;
        UpdateMapping();
    }

    void SetScalingFromInput(
        NekDouble scale, NekDouble radius,
        std::vector<SpatialDomains::CADCurveSharedPtr> curves,
        SpatialDomains::MeshGraph *graph)
    {
        m_radapt       = true;
        m_adapt_scale  = scale;
        m_adapt_radius = radius;
        m_adaptcurves  = curves;
        m_graph        = graph;
    }

    bool PreUpdateMapping(AdaptCurveVector &adaptCurves, NekDouble scale,
                          NekDouble rad, SpatialDomains::MeshGraph *graph);

    void UpdateMapping();

private:
    void MappingIdealToRef();

    SpatialDomains::Geometry *m_el;
    int m_dim;
    int m_mode;
    int m_order;
    std::unordered_map<SpatialDomains::PointGeom *, int> m_idmap;

    NekDouble m_scaledJac;
    NekDouble m_minJac;

    PtsFieldSharedPtr m_interpField;
    LibUtilities::Interpolator m_interp;

    DerivUtilSharedPtr m_derivUtil;
    ResidualSharedPtr m_res;

    // Initial maps
    std::vector<std::vector<NekDouble>> m_maps, m_mapsStd;
    // r-adaption
    bool m_radapt;
    std::vector<SpatialDomains::CADCurveSharedPtr> m_adaptcurves;
    /// Needed to ask which CAD curves a vertex sits on; only set when
    /// r-adapting against CAD curves.
    SpatialDomains::MeshGraph *m_graph = nullptr;
    NekDouble m_adapt_scale;
    NekDouble m_adapt_radius;
};
typedef std::shared_ptr<ElUtil> ElUtilSharedPtr;

class ElUtilJob : public Thread::ThreadJob
{
public:
    ElUtilJob(ElUtil *e, AdaptCurveVector &adaptCurves, NekDouble scale,
              NekDouble rad, SpatialDomains::MeshGraph *graph)
        : el(e), m_update(false), m_adaptCurves(adaptCurves),
          m_adaptScale(scale), m_adaptRad(rad)
    {
        m_update = el->PreUpdateMapping(m_adaptCurves, m_adaptScale, m_adaptRad,
                                        graph);
    }

    ElUtilJob(ElUtil *e, bool update) : el(e), m_update(update)
    {
    }

    void Run() override
    {
        el->Evaluate();

        if (m_update)
        {
            el->UpdateMapping();
        }
    }

private:
    ElUtil *el;
    bool m_update;
    AdaptCurveVector m_adaptCurves;
    NekDouble m_adaptScale;
    NekDouble m_adaptRad;
};

} // namespace Nektar::NekMesh

#endif
