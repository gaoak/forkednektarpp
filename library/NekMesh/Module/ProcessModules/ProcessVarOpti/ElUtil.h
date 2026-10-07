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
    NEKMESH_EXPORT ElUtil(SpatialDomains::Geometry *e, DerivUtilSharedPtr d,
                          ResidualSharedPtr, int n, int o);

    ElUtilJob *GetJob(bool update = false);
    ElUtilJob *GetAdaptJob(AdaptCurveVector &adaptCurves, NekDouble scale,
                           NekDouble rad, SpatialDomains::MeshGraph *graph);

    int GetId()
    {
        return m_el->GetGlobalID();
    }

    // Leaving these varibles as public for sake of efficiency
    /// Pointers to the coordinates of each node, as nodes[i * dim + c].
    std::vector<NekDouble *> nodes;
    /// The inverse ideal mapping at each integration point (maps) and at each
    /// nodal point (mapsStd), ten entries per point: the nine matrix entries
    /// and then its determinant. Flat rather than a vector per point, since
    /// the innermost loop of the optimiser reads one point after another.
    std::vector<NekDouble> maps, mapsStd;
    /// The stride between points of the arrays above, which is zero where the
    /// mapping does not vary over the element -- as it does not for a
    /// triangle or a tetrahedron, whose ideal element is affine.
    int mapStride = 10, mapStdStride = 10;

    void Evaluate();
    /// Find the smallest Jacobian of this element at the integration points,
    /// and reduce it into the residual's mesh-wide minimum.
    NEKMESH_EXPORT void CalcMinJac();

    /**
     * @brief Differentiate the element's mapping at the integration points
     * into @p deriv.
     *
     * Held per element rather than worked out per node: an element belongs to
     * the patch of every one of its nodes, so computing this where it is used
     * did the same multiplication once for each of them -- around thirty-five
     * times over for a fourth-order tetrahedron.
     */
    NEKMESH_EXPORT void CalcDeriv();

    /// d x_c / d xi_d at integration point k, as deriv[(d * dim + c) * pts +
    /// k]. Kept current by CalcDeriv() once per iteration and by the rank-one
    /// update below as each node moves.
    std::vector<NekDouble> deriv;

    /**
     * @brief Move node @p id of this element by @p offset, and carry the
     * change through to deriv.
     *
     * The mapping is linear in the node positions, so moving one node changes
     * its derivatives by a rank-one update rather than by anything that needs
     * recomputing.
     */
    NEKMESH_EXPORT void MoveNode(int id, const NekDouble *offset, int dim);

    /**
     * @brief Roughly how much memory this element holds for the optimiser.
     *
     * Dominated by the derivatives and the ideal mapping, both of which carry
     * an entry per integration point, so it grows with the over-integration
     * order -- steeply for a quadrilateral or hexahedron, whose rules are
     * tensor products.
     */
    size_t Footprint() const
    {
        return sizeof(ElUtil) + deriv.capacity() * sizeof(NekDouble) +
               maps.capacity() * sizeof(NekDouble) +
               mapsStd.capacity() * sizeof(NekDouble) +
               m_orig.capacity() * sizeof(NekDouble) +
               m_origStd.capacity() * sizeof(NekDouble) +
               nodes.capacity() * sizeof(NekDouble *);
    }

    DerivUtil *GetDerivUtil()
    {
        return m_derivUtil.get();
    }

    SpatialDomains::Geometry *GetEl()
    {
        return m_el;
    }

    /**
     * @brief Where @p in sits in this element's node list.
     *
     * A linear scan: this is asked once per element when the optimiser is
     * built and not in any loop that matters, and the map it replaces cost
     * more memory than the node list itself.
     */
    int NodeId(SpatialDomains::PointGeom *in)
    {
        const NekDouble *x = &(*in)[0];
        for (int i = 0; i < m_nNodes; ++i)
        {
            if (nodes[i * m_dim] == x)
            {
                return i;
            }
        }

        NEKERROR(ErrorUtil::efatal, "node is not one of this element's");
        return -1;
    }

    NekDouble GetScaledJac()
    {
        return m_scaledJac;
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
    int m_nNodes = 0;
    /// Whether the ideal mapping is the same at every point.
    bool m_constantMap = false;

    NekDouble m_scaledJac;

    PtsFieldSharedPtr m_interpField;
    LibUtilities::Interpolator m_interp;

    DerivUtilSharedPtr m_derivUtil;
    ResidualSharedPtr m_res;

    /// Scratch for building the mappings, emptied once they are flattened.
    std::vector<std::vector<NekDouble>> m_maps, m_mapsStd;
    /// The unscaled mappings, kept only once an r-adaptation scaling has been
    /// applied and the originals are needed to work from.
    std::vector<NekDouble> m_orig, m_origStd;
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
        // CalcMinJac reads what CalcDeriv leaves behind, so the order here
        // matters.
        el->CalcDeriv();
        el->CalcMinJac();

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
