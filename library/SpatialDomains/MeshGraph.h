////////////////////////////////////////////////////////////////////////////////
//
//  File: MeshGraph.h
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

#ifndef NEKTAR_SPATIALDOMAINS_MESHGRAPH_H
#define NEKTAR_SPATIALDOMAINS_MESHGRAPH_H

#include <unordered_map>

#include <boost/container/flat_map.hpp>

#include <LibUtilities/BasicUtils/DomainRange.h>
#include <LibUtilities/BasicUtils/FieldIO.h>
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/Memory/ObjectPool.hpp>

#include <SpatialDomains/HexGeom.h>
#include <SpatialDomains/MeshEntities.hpp>
#include <SpatialDomains/PrismGeom.h>
#include <SpatialDomains/PyrGeom.h>
#include <SpatialDomains/QuadGeom.h>
#include <SpatialDomains/SegGeom.h>
#include <SpatialDomains/TetGeom.h>
#include <SpatialDomains/TriGeom.h>

#include <SpatialDomains/Curve.hpp>
#include <SpatialDomains/SpatialDomainsDeclspec.h>

class TiXmlDocument;

namespace Nektar::SpatialDomains
{

template <typename T>
using GeomMap = boost::container::flat_map<int, unique_ptr_objpool<T>>;

// Point geom type defs
typedef unique_ptr_objpool<PointGeom> PointGeomUniquePtr;

// Geometry typedefs
typedef unique_ptr_objpool<SegGeom> SegGeomUniquePtr;
typedef unique_ptr_objpool<TriGeom> TriGeomUniquePtr;
typedef unique_ptr_objpool<QuadGeom> QuadGeomUniquePtr;
typedef unique_ptr_objpool<TetGeom> TetGeomUniquePtr;
typedef unique_ptr_objpool<PyrGeom> PyrGeomUniquePtr;
typedef unique_ptr_objpool<PrismGeom> PrismGeomUniquePtr;
typedef unique_ptr_objpool<HexGeom> HexGeomUniquePtr;
typedef unique_ptr_objpool<Geometry2D> Geometry2DUniquePtr;
typedef unique_ptr_objpool<Geometry3D> Geometry3DUniquePtr;
typedef unique_ptr_objpool<GeomFactors> GeomFactorsUniquePtr;
typedef unique_ptr_objpool<Curve> CurveUniquePtr;

// Minimal owner of Geom objects not owned by a MeshGraph.
class EntityHolder
{
public:
    std::vector<PointGeomUniquePtr> m_pointVec;
    std::vector<SegGeomUniquePtr> m_segVec;
    std::vector<TriGeomUniquePtr> m_triVec;
    std::vector<QuadGeomUniquePtr> m_quadVec;
    std::vector<TetGeomUniquePtr> m_tetVec;
    std::vector<PyrGeomUniquePtr> m_pyrVec;
    std::vector<PrismGeomUniquePtr> m_prismVec;
    std::vector<HexGeomUniquePtr> m_hexVec;
    std::vector<CurveUniquePtr> m_curveVec;
};

// Composite type def
typedef std::map<int, std::pair<LibUtilities::ShapeType, std::vector<int>>>
    CompositeDescriptor;

enum ExpansionType
{
    eNoExpansionType,
    eModified,
    eModifiedQuadPlus1,
    eModifiedQuadPlus2,
    eModifiedGLLRadau10,
    eOrthogonal,
    eGLL_Lagrange,
    eGLL_Lagrange_SEM,
    eGauss_Lagrange,
    eGauss_Lagrange_SEM,
    eFourier,
    eFourierSingleMode,
    eFourierHalfModeRe,
    eFourierHalfModeIm,
    eChebyshev,
    eFourierChebyshev,
    eChebyshevFourier,
    eFourierModified,
    eExpansionTypeSize
};

// Keep this consistent with the enums in ExpansionType.
// This is used in the BC file to specify the expansion type.
const std::string kExpansionTypeStr[] = {"NOTYPE",
                                         "MODIFIED",
                                         "MODIFIEDQUADPLUS1",
                                         "MODIFIEDQUADPLUS2",
                                         "MODIFIEDGLLRADAU10",
                                         "ORTHOGONAL",
                                         "GLL_LAGRANGE",
                                         "GLL_LAGRANGE_SEM",
                                         "GAUSS_LAGRANGE",
                                         "GAUSS_LAGRANGE_SEM",
                                         "FOURIER",
                                         "FOURIERSINGLEMODE",
                                         "FOURIERHALFMODERE",
                                         "FOURIERHALFMODEIM",
                                         "CHEBYSHEV",
                                         "FOURIER-CHEBYSHEV",
                                         "CHEBYSHEV-FOURIER",
                                         "FOURIER-MODIFIED"};

typedef std::map<int, std::vector<unsigned int>> CompositeOrdering;
typedef std::map<int, std::vector<unsigned int>> BndRegionOrdering;

struct Composite
{
    std::vector<Geometry *> m_geomVec;
};

typedef std::shared_ptr<Composite> CompositeSharedPtr;
typedef std::map<int, CompositeSharedPtr> CompositeMap;

struct ExpansionInfo;

typedef std::shared_ptr<ExpansionInfo> ExpansionInfoShPtr;
typedef std::map<int, ExpansionInfoShPtr> ExpansionInfoMap;

typedef std::shared_ptr<ExpansionInfoMap> ExpansionInfoMapShPtr;
typedef std::map<std::string, ExpansionInfoMapShPtr> ExpansionInfoMapShPtrMap;

struct ExpansionInfo
{
    ExpansionInfo(Geometry *geomPtr,
                  const LibUtilities::BasisKeyVector basiskeyvec)
        : m_geomPtr(geomPtr), m_basisKeyVector(basiskeyvec)
    {
    }

    ExpansionInfo(ExpansionInfoShPtr ExpInfo)
        : m_geomPtr(ExpInfo->m_geomPtr),
          m_basisKeyVector(ExpInfo->m_basisKeyVector)
    {
    }

    Geometry *m_geomPtr;
    LibUtilities::BasisKeyVector m_basisKeyVector;
};

typedef std::map<std::string, std::string> GeomInfoMap;
typedef std::shared_ptr<std::vector<std::pair<Geometry *, int>>>
    GeometryLinkSharedPtr;

/**
 * @brief Structure to describe the elements bordering a facet, and which of
 * their local facets it corresponds to.
 *
 * A facet is the codimension-one entity of an element, i.e. an edge of a
 * two-dimensional element, or a face of a three-dimensional one. The same
 * lookup is wanted in both cases, so it is held once, in terms of facets.
 *
 * Because it is codimension one, a facet separates two elements or bounds the
 * domain, so two entries are held inline and the link lives by value in the
 * map.
 *
 * Two is not a hard limit: periodic alignment, tetrahedron splitting and
 * boundary layer splitting all attach more than two, so anything past the
 * second spills to a vector. That path is rare enough not to matter and the
 * common case never allocates.
 */
struct FacetElementLink
{
    typedef std::pair<Geometry *, int> value_type;
    static constexpr size_t kInline = 2;

    void push_back(const value_type &v)
    {
        if (m_num < kInline)
        {
            m_link[m_num++] = v;
            return;
        }
        m_spill.push_back(v);
        ++m_num;
    }

    size_t size() const
    {
        return m_num;
    }
    bool empty() const
    {
        return m_num == 0;
    }
    const value_type &operator[](size_t i) const
    {
        return i < kInline ? m_link[i] : m_spill[i - kInline];
    }
    const value_type &at(size_t i) const
    {
        ASSERTL1(i < m_num, "Facet-element link index out of range.");
        return (*this)[i];
    }

    /// Iteration has to walk both halves, so it is an index rather than a
    /// pointer.
    class Iterator
    {
    public:
        Iterator(const FacetElementLink *l, size_t i) : m_l(l), m_i(i)
        {
        }
        const value_type &operator*() const
        {
            return (*m_l)[m_i];
        }
        Iterator &operator++()
        {
            ++m_i;
            return *this;
        }
        bool operator!=(const Iterator &o) const
        {
            return m_i != o.m_i;
        }

    private:
        const FacetElementLink *m_l;
        size_t m_i;
    };

    Iterator begin() const
    {
        return Iterator(this, 0);
    }
    Iterator end() const
    {
        return Iterator(this, m_num);
    }

private:
    std::array<value_type, kInline> m_link{};
    /// Empty, and so unallocated, for every facet that has at most two.
    std::vector<value_type> m_spill;
    size_t m_num = 0;
};

// Forward declaration
class RefRegion;

typedef std::map<std::string, std::string> MeshMetaDataMap;

class MeshGraph;
typedef std::shared_ptr<MeshGraph> MeshGraphSharedPtr;

class Movement;
typedef std::shared_ptr<Movement> MovementSharedPtr;

/// Forward declared so that the CAD headers are not pulled into every
/// translation unit that uses a MeshGraph; see CADAssociation.h.
class CADAssociation;
typedef std::shared_ptr<CADAssociation> CADAssociationSharedPtr;
class CADSystem;
typedef std::shared_ptr<CADSystem> CADSystemSharedPtr;

template <typename T> class GeomMapView
{
public:
    class Iterator
    {
        using value_type = std::pair<int, T *>;
        typename GeomMap<T>::const_iterator m_it;
        mutable value_type m_cache;

    public:
        explicit Iterator(typename GeomMap<T>::const_iterator it) : m_it(it)
        {
        }

        const value_type &operator*() const
        {
            m_cache = {m_it->first, m_it->second.get()};
            return m_cache;
        }
        const value_type *operator->() const
        {
            m_cache = {m_it->first, m_it->second.get()};
            return &m_cache;
        }

        Iterator &operator++()
        {
            ++m_it;
            return *this;
        }
        bool operator!=(const Iterator &other) const
        {
            return m_it != other.m_it;
        }
        bool operator==(const Iterator &other) const
        {
            return m_it == other.m_it;
        }
    };

    class ReverseIterator
    {
        using value_type = std::pair<int, T *>;
        typename GeomMap<T>::const_reverse_iterator m_it;
        mutable value_type m_cache;

    public:
        explicit ReverseIterator(typename GeomMap<T>::const_reverse_iterator it)
            : m_it(it)
        {
        }

        const value_type &operator*() const
        {
            m_cache = {m_it->first, m_it->second.get()};
            return m_cache;
        }
        const value_type *operator->() const
        {
            m_cache = {m_it->first, m_it->second.get()};
            return &m_cache;
        }

        ReverseIterator &operator++()
        {
            ++m_it;
            return *this;
        }

        bool operator!=(const ReverseIterator &other) const
        {
            return m_it != other.m_it;
        }

        bool operator==(const ReverseIterator &other) const
        {
            return m_it == other.m_it;
        }
    };

    explicit GeomMapView(const GeomMap<T> &map) : m_map(map)
    {
    }

    Iterator begin() const
    {
        return Iterator(m_map.begin());
    }
    Iterator end() const
    {
        return Iterator(m_map.end());
    }

    ReverseIterator rbegin() const
    {
        return ReverseIterator(m_map.rbegin());
    }
    ReverseIterator rend() const
    {
        return ReverseIterator(m_map.rend());
    }

    std::size_t size() const
    {
        return m_map.size();
    }

    bool empty() const
    {
        return m_map.empty();
    }

    Iterator find(int id) const
    {
        return Iterator(m_map.find(id));
    }

    T *at(int id) const
    {
        auto it = m_map.find(id);
        return it != m_map.end() ? it->second.get() : nullptr;
    }

private:
    const GeomMap<T> &m_map;
};

/// Base class for a spectral/hp element mesh.
class MeshGraph
{
public:
    SPATIAL_DOMAINS_EXPORT MeshGraph();
    SPATIAL_DOMAINS_EXPORT virtual ~MeshGraph();

    void Empty(int dim, int space)
    {
        m_meshDimension  = dim;
        m_spaceDimension = space;
    }

    /*transfers the minial data structure to full meshgraph*/
    SPATIAL_DOMAINS_EXPORT void FillGraph();

    SPATIAL_DOMAINS_EXPORT void FillBoundingBoxTree();

    SPATIAL_DOMAINS_EXPORT std::vector<int> GetElementsContainingPoint(
        PointGeom *p);

    ////////////////////
    SPATIAL_DOMAINS_EXPORT void ReadExpansionInfo(TiXmlElement *expansionTypes);

    /// Read refinement info.
    SPATIAL_DOMAINS_EXPORT void ReadRefinementInfo();

    /* ---- Helper functions ---- */
    /// Dimension of the mesh (can be a 1D curve in 3D space).
    int GetMeshDimension()
    {
        return m_meshDimension;
    }

    /// Dimension of the space (can be a 1D curve in 3D space).
    int GetSpaceDimension()
    {
        return m_spaceDimension;
    }

    void SetMeshDimension(int dim)
    {
        m_meshDimension = dim;
    }

    void SetSpaceDimension(int dim)
    {
        m_spaceDimension = dim;
    }

    /* Range definitions for postprorcessing */
    SPATIAL_DOMAINS_EXPORT void SetDomainRange(
        NekDouble xmin, NekDouble xmax,
        NekDouble ymin = NekConstants::kNekUnsetDouble,
        NekDouble ymax = NekConstants::kNekUnsetDouble,
        NekDouble zmin = NekConstants::kNekUnsetDouble,
        NekDouble zmax = NekConstants::kNekUnsetDouble);

    SPATIAL_DOMAINS_EXPORT void SetDomainRange(
        LibUtilities::DomainRangeShPtr rng);

    /// Check if goemetry is in range definition if activated
    SPATIAL_DOMAINS_EXPORT bool CheckRange(Geometry2D &geom);

    /// Check if goemetry is in range definition if activated
    SPATIAL_DOMAINS_EXPORT bool CheckRange(Geometry3D &geom);

    /// Check if goemetry is in range definition if activated
    SPATIAL_DOMAINS_EXPORT bool CheckRange(MeshEntity &e);

    /* ---- Composites and Domain ---- */
    CompositeSharedPtr GetComposite(int whichComposite)
    {
        if (m_meshComposites.find(whichComposite) == m_meshComposites.end())
        {
            NEKERROR(ErrorUtil::efatal, "Composite not found.");
        }
        return m_meshComposites.find(whichComposite)->second;
    }

    SPATIAL_DOMAINS_EXPORT Geometry *GetCompositeItem(int whichComposite,
                                                      int whichItem);

    SPATIAL_DOMAINS_EXPORT void GetCompositeList(
        const std::string &compositeStr, CompositeMap &compositeVector) const;

    std::map<int, CompositeSharedPtr> &GetComposites()
    {
        return m_meshComposites;
    }

    std::map<int, std::string> &GetCompositesLabels()
    {
        return m_compositesLabels;
    }

    std::map<int, std::map<int, CompositeSharedPtr>> &GetDomain()
    {
        return m_domain;
    }

    std::map<int, CompositeSharedPtr> &GetDomain(int domain)
    {
        ASSERTL1(m_domain.count(domain),
                 "Request for domain which does not exist");
        return m_domain[domain];
    }

    LibUtilities::DomainRangeShPtr &GetDomainRange()
    {
        return m_domainRange;
    }

    SPATIAL_DOMAINS_EXPORT const ExpansionInfoMap &GetExpansionInfo(
        const std::string variable = "DefaultVar");

    SPATIAL_DOMAINS_EXPORT ExpansionInfoShPtr
    GetExpansionInfo(Geometry *geom, const std::string variable = "DefaultVar");

    /// Sets expansions given field definitions
    SPATIAL_DOMAINS_EXPORT void SetExpansionInfo(
        std::vector<LibUtilities::FieldDefinitionsSharedPtr> &fielddef);

    /// Sets expansions given field definition, quadrature points.
    SPATIAL_DOMAINS_EXPORT void SetExpansionInfo(
        std::vector<LibUtilities::FieldDefinitionsSharedPtr> &fielddef,
        std::vector<std::vector<LibUtilities::PointsType>> &pointstype);

    /// Sets expansions to have equispaced points
    SPATIAL_DOMAINS_EXPORT void SetExpansionInfoToEvenlySpacedPoints(
        int npoints = 0);

    /// Reset expansion to have specified polynomial order \a nmodes
    SPATIAL_DOMAINS_EXPORT void SetExpansionInfoToNumModes(int nmodes);

    /// Reset expansion to have specified point order \a
    /// npts
    SPATIAL_DOMAINS_EXPORT void SetExpansionInfoToPointOrder(int npts);
    /// This function sets the expansion #exp in map with
    /// entry #variable

    /// Set refinement info.
    SPATIAL_DOMAINS_EXPORT void SetRefinementInfo(
        ExpansionInfoMapShPtr &expansionMap);

    /// Perform the p-refinement in the selected elements
    SPATIAL_DOMAINS_EXPORT void PRefinementElmts(
        ExpansionInfoMapShPtr &expansionMap, RefRegion *&region,
        Geometry *geomVecIter);

    inline void SetExpansionInfo(const std::string variable,
                                 ExpansionInfoMapShPtr &exp);

    inline void SetSession(LibUtilities::SessionReaderSharedPtr pSession);

    inline LibUtilities::SessionReaderSharedPtr GetSession()
    {
        return m_session;
    }

    /// Sets the basis key for all expansions of the given shape.
    SPATIAL_DOMAINS_EXPORT void SetBasisKey(LibUtilities::ShapeType shape,
                                            LibUtilities::BasisKeyVector &keys,
                                            std::string var = "DefaultVar");

    SPATIAL_DOMAINS_EXPORT void ResetExpansionInfoToBasisKey(
        ExpansionInfoMapShPtr &expansionMap, LibUtilities::ShapeType shape,
        LibUtilities::BasisKeyVector &keys);

    SPATIAL_DOMAINS_EXPORT void ResetExpansionInfoToModified(std::string var);

    inline bool SameExpansionInfo(const std::string var1,
                                  const std::string var2);

    inline bool ExpansionInfoDefined(const std::string var);

    inline bool CheckForGeomInfo(std::string parameter);

    inline const std::string GetGeomInfo(std::string parameter);

    SPATIAL_DOMAINS_EXPORT static LibUtilities::BasisKeyVector
    DefineBasisKeyFromExpansionType(Geometry *in, ExpansionType type,
                                    const int order);

    SPATIAL_DOMAINS_EXPORT LibUtilities::BasisKeyVector
    DefineBasisKeyFromExpansionTypeHomo(Geometry *in, ExpansionType type_x,
                                        ExpansionType type_y,
                                        ExpansionType type_z,
                                        const int nummodes_x,
                                        const int nummodes_y,
                                        const int nummodes_z);

    /* ---- Manipulation of mesh ---- */
    int GetNvertices()
    {
        return m_pointGeoms.size();
    }

    /**
     * @brief Returns vertex @p id from the MeshGraph.
     */
    [[deprecated("since 5.8.0, use GetPointGeom() instead")]] PointGeom *GetVertex(
        int id)
    {
        return this->GetGeom(id, m_pointGeoms);
    }

    /**
     * @brief Returns vertex @p id from the MeshGraph.
     */
    PointGeom *GetPointGeom(int id)
    {
        return this->GetGeom(id, m_pointGeoms);
    }

    /**
     * @brief Returns segment @p id from the MeshGraph.
     */
    SegGeom *GetSegGeom(int id)
    {
        return this->GetGeom(id, m_segGeoms);
    }

    /**
     * @brief Returns triangle @p id from the MeshGraph.
     */
    TriGeom *GetTriGeom(int id)
    {
        return this->GetGeom(id, m_triGeoms);
    }

    /**
     * @brief Returns quadrilateral @p id from the MeshGraph.
     */
    QuadGeom *GetQuadGeom(int id)
    {
        return this->GetGeom(id, m_quadGeoms);
    }

    /**
     * @brief Returns tetrahedron @p id from the MeshGraph.
     */
    TetGeom *GetTetGeom(int id)
    {
        return this->GetGeom(id, m_tetGeoms);
    }

    /**
     * @brief Returns pyramid @p id from the MeshGraph.
     */
    PyrGeom *GetPyrGeom(int id)
    {
        return this->GetGeom(id, m_pyrGeoms);
    }

    /**
     * @brief Returns prism @p id from the MeshGraph.
     */
    PrismGeom *GetPrismGeom(int id)
    {
        return this->GetGeom(id, m_prismGeoms);
    }

    /**
     * @brief Returns hex @p id from the MeshGraph.
     */
    HexGeom *GetHexGeom(int id)
    {
        return this->GetGeom(id, m_hexGeoms);
    }

    /**
     * @brief Convenience function to add a geometry @p geom to the MeshGraph
     * with geometry ID @p id. Retains ownership of the passed unique_ptr.
     *
     * @p id    Geometry ID
     * @p geom  unique_ptr to geometry object.
     */
    template <typename T> void AddGeom(int id, unique_ptr_objpool<T> geom)
    {
        ASSERTL2(geom->GetGlobalID() == id,
                 "Mismatch between geometry ID and global ID");

        if constexpr (std::is_same_v<T, PointGeom>)
        {
            InsertGeomChecked(m_pointGeoms, id, std::move(geom));
        }
        else if constexpr (std::is_same_v<T, SegGeom>)
        {
            InsertGeomChecked(m_segGeoms, id, std::move(geom));
        }
        else if constexpr (std::is_same_v<T, QuadGeom>)
        {
            InsertGeomChecked(m_quadGeoms, id, std::move(geom));
        }
        else if constexpr (std::is_same_v<T, TriGeom>)
        {
            InsertGeomChecked(m_triGeoms, id, std::move(geom));
        }
        else if constexpr (std::is_same_v<T, TetGeom>)
        {
            InsertGeomChecked(m_tetGeoms, id, std::move(geom));
        }
        else if constexpr (std::is_same_v<T, PyrGeom>)
        {
            InsertGeomChecked(m_pyrGeoms, id, std::move(geom));
        }
        else if constexpr (std::is_same_v<T, PrismGeom>)
        {
            InsertGeomChecked(m_prismGeoms, id, std::move(geom));
        }
        else if constexpr (std::is_same_v<T, HexGeom>)
        {
            InsertGeomChecked(m_hexGeoms, id, std::move(geom));
        }
        else if constexpr (std::is_same_v<T, Geometry>)
        {
            switch (geom->GetShapeType())
            {
                case LibUtilities::ePoint:
                    InsertGeomChecked(
                        m_pointGeoms, id,
                        unique_ptr_objpool_cast<PointGeom>(std::move(geom)));
                    break;
                case LibUtilities::eSegment:
                    InsertGeomChecked(
                        m_segGeoms, id,
                        unique_ptr_objpool_cast<SegGeom>(std::move(geom)));
                    break;
                case LibUtilities::eTriangle:
                    InsertGeomChecked(
                        m_triGeoms, id,
                        unique_ptr_objpool_cast<TriGeom>(std::move(geom)));
                    break;
                case LibUtilities::eQuadrilateral:
                    InsertGeomChecked(
                        m_quadGeoms, id,
                        unique_ptr_objpool_cast<QuadGeom>(std::move(geom)));
                    break;
                case LibUtilities::eTetrahedron:
                    InsertGeomChecked(
                        m_tetGeoms, id,
                        unique_ptr_objpool_cast<TetGeom>(std::move(geom)));
                    break;
                case LibUtilities::ePyramid:
                    InsertGeomChecked(
                        m_pyrGeoms, id,
                        unique_ptr_objpool_cast<PyrGeom>(std::move(geom)));
                    break;
                case LibUtilities::ePrism:
                    InsertGeomChecked(
                        m_prismGeoms, id,
                        unique_ptr_objpool_cast<PrismGeom>(std::move(geom)));
                    break;
                case LibUtilities::eHexahedron:
                    InsertGeomChecked(
                        m_hexGeoms, id,
                        unique_ptr_objpool_cast<HexGeom>(std::move(geom)));
                    break;
                default:
                    ASSERTL0(false, "Unknown or unsupported shape type");
                    break;
            }
        }
        else
        {
            ASSERTL0(false, "Unknown geometry type");
        }
    }

    /**
     */
    template <typename MapT, typename U>
    void InsertGeomChecked(MapT &map, int id, unique_ptr_objpool<U> geom)
    {
        const bool inserted =
            map.insert(std::make_pair(id, std::move(geom))).second;
        ASSERTL0(inserted, "MeshGraph already contains a geometry with ID " +
                               std::to_string(id));
    }

    template <typename T>
    void BulkAddGeom(
        std::vector<std::pair<int, unique_ptr_objpool<T>>> &insertVector)
    {
        if constexpr (std::is_same_v<T, PointGeom>)
        {
            m_pointGeoms.insert(std::make_move_iterator(insertVector.begin()),
                                std::make_move_iterator(insertVector.end()));
        }
        else if constexpr (std::is_same_v<T, SegGeom>)
        {
            m_segGeoms.insert(std::make_move_iterator(insertVector.begin()),
                              std::make_move_iterator(insertVector.end()));
        }
        else if constexpr (std::is_same_v<T, QuadGeom>)
        {
            m_quadGeoms.insert(std::make_move_iterator(insertVector.begin()),
                               std::make_move_iterator(insertVector.end()));
        }
        else if constexpr (std::is_same_v<T, TriGeom>)
        {
            m_triGeoms.insert(std::make_move_iterator(insertVector.begin()),
                              std::make_move_iterator(insertVector.end()));
        }
        else if constexpr (std::is_same_v<T, TetGeom>)
        {
            m_tetGeoms.insert(std::make_move_iterator(insertVector.begin()),
                              std::make_move_iterator(insertVector.end()));
        }
        else if constexpr (std::is_same_v<T, PyrGeom>)
        {
            m_pyrGeoms.insert(std::make_move_iterator(insertVector.begin()),
                              std::make_move_iterator(insertVector.end()));
        }
        else if constexpr (std::is_same_v<T, PrismGeom>)
        {
            m_prismGeoms.insert(std::make_move_iterator(insertVector.begin()),
                                std::make_move_iterator(insertVector.end()));
        }
        else if constexpr (std::is_same_v<T, HexGeom>)
        {
            m_hexGeoms.insert(std::make_move_iterator(insertVector.begin()),
                              std::make_move_iterator(insertVector.end()));
        }
        else
        {
            ASSERTL0(false, "Unknown geometry type");
        }
    }

    /**
     * @brief Function to extract a geometry from the MeshGraph
     * with ID @p id, returning a move interator.
     *
     * @p id    Geometry ID
     */
    template <typename T>
    unique_ptr_objpool<T> ExtractGeom(int id, bool erase = false)
    {
        if constexpr (std::is_same_v<T, PointGeom>)
        {
            auto ret = std::move(m_pointGeoms[id]);
            if (erase)
            {
                m_pointGeoms.erase(id);
            }
            return ret;
        }
        else if constexpr (std::is_same_v<T, SegGeom>)
        {
            auto ret = std::move(m_segGeoms[id]);
            if (erase)
            {
                m_segGeoms.erase(id);
            }
            return ret;
        }
        else if constexpr (std::is_same_v<T, QuadGeom>)
        {
            auto ret = std::move(m_quadGeoms[id]);
            if (erase)
            {
                m_quadGeoms.erase(id);
            }
            return ret;
        }
        else if constexpr (std::is_same_v<T, TriGeom>)
        {
            auto ret = std::move(m_triGeoms[id]);
            if (erase)
            {
                m_triGeoms.erase(id);
            }
            return ret;
        }
        else if constexpr (std::is_same_v<T, TetGeom>)
        {
            auto ret = std::move(m_tetGeoms[id]);
            if (erase)
            {
                m_tetGeoms.erase(id);
            }
            return ret;
        }
        else if constexpr (std::is_same_v<T, PyrGeom>)
        {
            auto ret = std::move(m_pyrGeoms[id]);
            if (erase)
            {
                m_pyrGeoms.erase(id);
            }
            return ret;
        }
        else if constexpr (std::is_same_v<T, PrismGeom>)
        {
            auto ret = std::move(m_prismGeoms[id]);
            if (erase)
            {
                m_prismGeoms.erase(id);
            }
            return ret;
        }
        else if constexpr (std::is_same_v<T, HexGeom>)
        {
            auto ret = std::move(m_hexGeoms[id]);
            if (erase)
            {
                m_hexGeoms.erase(id);
            }
            return ret;
        }
    }

    SPATIAL_DOMAINS_EXPORT PointGeom *CreateCurveNode(const int coordim,
                                                      const int vid,
                                                      NekDouble x, NekDouble y,
                                                      NekDouble z)
    {
        auto geom =
            ObjPoolManager<PointGeom>::AllocateUniquePtr(coordim, vid, x, y, z);
        auto ret = geom.get();
        m_nodeSet.push_back(std::move(geom));
        return ret;
    }

    void AddCurveNode(PointGeomUniquePtr n)
    {
        m_nodeSet.push_back(std::move(n));
    }

    void AddCurveNodes(std::vector<PointGeomUniquePtr> &n)
    {
        for (std::size_t i = 0; i < n.size(); ++i)
        {
            m_nodeSet.push_back(std::move(n[i]));
        }
    }

    void AddCurvedEdge(CurveUniquePtr curve)
    {
        m_curvedEdges[curve->m_curveID] = std::move(curve);
    }

    void AddCurvedFace(CurveUniquePtr curve)
    {
        m_curvedFaces[curve->m_curveID] = std::move(curve);
    }

    /// Adopt the interior nodes of a 3D element, filed under the element's own
    /// curve ID. The counterpart of AddCurvedEdge/AddCurvedFace for the one
    /// remaining dimension: unlike an edge or a face, a volume's interior
    /// nodes are not shared with any neighbour, so they belong to the element
    /// alone.
    void AddCurvedVolume(CurveUniquePtr curve)
    {
        m_curvedVolumes[curve->m_curveID] = std::move(curve);
    }

    SPATIAL_DOMAINS_EXPORT PointGeom *CreatePointGeom(const int coordim,
                                                      const int vid,
                                                      NekDouble x, NekDouble y,
                                                      NekDouble z)
    {
        auto geom =
            ObjPoolManager<PointGeom>::AllocateUniquePtr(coordim, vid, x, y, z);
        auto ret = geom.get();
        AddGeom(vid, std::move(geom));
        return ret;
    }

    SPATIAL_DOMAINS_EXPORT SegGeom *CreateSegGeom(
        int id, int coordim, std::array<PointGeom *, SegGeom::kNverts> vertex,
        Curve *curve = nullptr)
    {
        auto geom = ObjPoolManager<SegGeom>::AllocateUniquePtr(id, coordim,
                                                               vertex, curve);
        auto ret  = geom.get();
        AddGeom(id, std::move(geom));
        return ret;
    }

    SPATIAL_DOMAINS_EXPORT QuadGeom *CreateQuadGeom(
        int id, std::array<SegGeom *, QuadGeom::kNedges> edges,
        Curve *curve = nullptr)
    {
        auto geom =
            ObjPoolManager<QuadGeom>::AllocateUniquePtr(id, edges, curve);
        auto ret = geom.get();
        AddGeom(id, std::move(geom));
        return ret;
    }

    SPATIAL_DOMAINS_EXPORT TriGeom *CreateTriGeom(
        int id, std::array<SegGeom *, TriGeom::kNedges> edges,
        Curve *curve = nullptr)
    {
        auto geom =
            ObjPoolManager<TriGeom>::AllocateUniquePtr(id, edges, curve);
        auto ret = geom.get();
        AddGeom(id, std::move(geom));
        return ret;
    }

    SPATIAL_DOMAINS_EXPORT TetGeom *CreateTetGeom(
        int id, std::array<TriGeom *, TetGeom::kNfaces> faces)
    {
        auto geom = ObjPoolManager<TetGeom>::AllocateUniquePtr(id, faces);
        auto ret  = geom.get();
        AddGeom(id, std::move(geom));
        return ret;
    }

    SPATIAL_DOMAINS_EXPORT HexGeom *CreateHexGeom(
        int id, std::array<QuadGeom *, HexGeom::kNfaces> faces)
    {
        auto geom = ObjPoolManager<HexGeom>::AllocateUniquePtr(id, faces);
        auto ret  = geom.get();
        AddGeom(id, std::move(geom));
        return ret;
    }

    SPATIAL_DOMAINS_EXPORT PrismGeom *CreatePrismGeom(
        int id, std::array<Geometry2D *, PrismGeom::kNfaces> faces)
    {
        auto geom = ObjPoolManager<PrismGeom>::AllocateUniquePtr(id, faces);
        auto ret  = geom.get();
        AddGeom(id, std::move(geom));
        return ret;
    }

    SPATIAL_DOMAINS_EXPORT PyrGeom *CreatePyrGeom(
        int id, std::array<Geometry2D *, PyrGeom::kNfaces> faces)
    {
        auto geom = ObjPoolManager<PyrGeom>::AllocateUniquePtr(id, faces);
        auto ret  = geom.get();
        AddGeom(id, std::move(geom));
        return ret;
    }

    SPATIAL_DOMAINS_EXPORT CurveMap &GetCurvedEdges()
    {
        return m_curvedEdges;
    }

    SPATIAL_DOMAINS_EXPORT CurveMap &GetCurvedFaces()
    {
        return m_curvedFaces;
    }

    SPATIAL_DOMAINS_EXPORT CurveMap &GetCurvedVolumes()
    {
        return m_curvedVolumes;
    }

    /**
     * @brief Re-read every geometry's curvature and rebuild what depends on it.
     *
     * Attaching a curve to a geometry only stores a pointer: the \f$\chi\f$
     * map, the coefficients sized from it and the cached regular/deformed
     * classification are all left describing the shape as it was. They are
     * rebuilt here, in ascending dimension, so that each face sees its edges
     * already at their new order and each element sees its faces.
     *
     * Anything that adds or moves curvature after the graph has been read --
     * Mesh::MakeOrder, the CAD projection modules -- has to call this before
     * the mesh is used again, or a geometry whose edges were curved underneath
     * it will still carry the linear map it was set up with.
     */
    SPATIAL_DOMAINS_EXPORT void ResetGeometry();

    template <typename T> GeomMapView<T> &GetGeomMap()
    {
        if constexpr (std::is_same_v<T, PointGeom>)
        {
            return m_pointMapView;
        }
        else if constexpr (std::is_same_v<T, SegGeom>)
        {
            return m_segMapView;
        }
        else if constexpr (std::is_same_v<T, TriGeom>)
        {
            return m_triMapView;
        }
        else if constexpr (std::is_same_v<T, QuadGeom>)
        {
            return m_quadMapView;
        }
        else if constexpr (std::is_same_v<T, TetGeom>)
        {
            return m_tetMapView;
        }
        else if constexpr (std::is_same_v<T, PrismGeom>)
        {
            return m_prismMapView;
        }
        else if constexpr (std::is_same_v<T, PyrGeom>)
        {
            return m_pyrMapView;
        }
        else if constexpr (std::is_same_v<T, HexGeom>)
        {
            return m_hexMapView;
        }

        ASSERTL0(false,
                 "MeshGraph does not support the supplied geometry type.");
    }

    /// The number of geometries of type @tparam T held by this graph.
    template <typename T> std::size_t GetNumGeoms()
    {
        return GetGeomMap<T>().size();
    }

    /// Whether this graph holds any geometry of type @tparam T.
    template <typename T> bool HasGeoms()
    {
        return !GetGeomMap<T>().empty();
    }

    template <typename T> void SetGeomMap(GeomMap<T> newMap)
    {
        if constexpr (std::is_same_v<T, PointGeom>)
        {
            m_pointGeoms = std::move(newMap);
        }
        else if constexpr (std::is_same_v<T, SegGeom>)
        {
            m_segGeoms = std::move(newMap);
        }
        else if constexpr (std::is_same_v<T, TriGeom>)
        {
            m_triGeoms = std::move(newMap);
        }
        else if constexpr (std::is_same_v<T, QuadGeom>)
        {
            m_quadGeoms = std::move(newMap);
        }
        else if constexpr (std::is_same_v<T, TetGeom>)
        {
            m_tetGeoms = std::move(newMap);
        }
        else if constexpr (std::is_same_v<T, PrismGeom>)
        {
            m_prismGeoms = std::move(newMap);
        }
        else if constexpr (std::is_same_v<T, PyrGeom>)
        {
            m_pyrGeoms = std::move(newMap);
        }
        else if constexpr (std::is_same_v<T, HexGeom>)
        {
            m_hexGeoms = std::move(newMap);
        }
        else
        {
            ASSERTL0(false,
                     "MeshGraph does not support the supplied geometry type.");
        }
    }

    SPATIAL_DOMAINS_EXPORT std::unordered_map<Geometry *, FacetElementLink> &
    GetAllFacetToElMap()
    {
        BuildFacetToElMap();
        return m_facetToElMap;
    }

    SPATIAL_DOMAINS_EXPORT std::vector<PointGeomUniquePtr> &GetAllCurveNodes()
    {
        return m_nodeSet;
    }

    SPATIAL_DOMAINS_EXPORT int GetNumElements(int dim = -1);

    Geometry2D *GetGeometry2D(int gID)
    {
        auto it1 = m_triGeoms.find(gID);
        if (it1 != m_triGeoms.end())
        {
            return it1->second.get();
        }

        auto it2 = m_quadGeoms.find(gID);
        if (it2 != m_quadGeoms.end())
        {
            return it2->second.get();
        }

        return nullptr;
    };

    Geometry3D *GetGeometry3D(int gID)
    {
        auto it1 = m_tetGeoms.find(gID);
        if (it1 != m_tetGeoms.end())
        {
            return it1->second.get();
        }

        auto it2 = m_pyrGeoms.find(gID);
        if (it2 != m_pyrGeoms.end())
        {
            return it2->second.get();
        }

        auto it3 = m_prismGeoms.find(gID);
        if (it3 != m_prismGeoms.end())
        {
            return it3->second.get();
        }

        auto it4 = m_hexGeoms.find(gID);
        if (it4 != m_hexGeoms.end())
        {
            return it4->second.get();
        }

        return nullptr;
    };

    /// The elements bordering @p facet. Valid for whichever entity is
    /// codimension one in this graph: an edge in 2D, a face in 3D.
    SPATIAL_DOMAINS_EXPORT const FacetElementLink &GetElementsFromFacet(
        Geometry *facet);

    /// @copydoc MeshGraph::GetElementsFromFacet
    /// Typed front door for a two-dimensional graph.
    SPATIAL_DOMAINS_EXPORT const FacetElementLink &GetElementsFromEdge(
        Geometry1D *edge);

    /// @copydoc MeshGraph::GetElementsFromFacet
    /// Typed front door for a three-dimensional graph.
    SPATIAL_DOMAINS_EXPORT const FacetElementLink &GetElementsFromFace(
        Geometry2D *face);

    void SetPartition(SpatialDomains::MeshGraphSharedPtr graph);

    CompositeOrdering &GetCompositeOrdering()
    {
        return m_compOrder;
    }

    void SetCompositeOrdering(CompositeOrdering p_compOrder)
    {
        m_compOrder = p_compOrder;
    }

    BndRegionOrdering &GetBndRegionOrdering()
    {
        return m_bndRegOrder;
    }

    void SetBndRegionOrdering(BndRegionOrdering p_bndRegOrder)
    {
        m_bndRegOrder = p_bndRegOrder;
    }

    SPATIAL_DOMAINS_EXPORT std::map<int, MeshEntity> CreateMeshEntities();
    SPATIAL_DOMAINS_EXPORT CompositeDescriptor CreateCompositeDescriptor();

    SPATIAL_DOMAINS_EXPORT inline MovementSharedPtr &GetMovement()
    {
        return m_movement;
    }

    /// True if a CAD system has been attached to this graph.
    SPATIAL_DOMAINS_EXPORT bool HasCAD() const;

    /**
     * @brief The mesh entity to CAD associations for this graph.
     *
     * Brought into being by SetCAD, so a graph with no CAD carries nothing
     * beyond a null pointer; asking one for its associations is an error
     * rather than an empty answer, since the caller has nothing to do with the
     * result either way. Guard with HasCAD() where a mesh may arrive without
     * CAD. See CADAssociation.
     */
    SPATIAL_DOMAINS_EXPORT CADAssociationSharedPtr &GetCADAssociation();

    /// Convenience forwarder for the very common GetCADAssociation()->GetCAD().
    SPATIAL_DOMAINS_EXPORT CADSystemSharedPtr &GetCAD();

    SPATIAL_DOMAINS_EXPORT void SetCAD(CADSystemSharedPtr cad);

    void Clear();

    /// Fill #m_facetToElMap from the elements in the graph, if it is not
    /// already filled. Cheap once built.
    SPATIAL_DOMAINS_EXPORT void BuildFacetToElMap();
    SPATIAL_DOMAINS_EXPORT void PopulateFacetToElMap(Geometry *element);

    bool GetMeshPartitioned()
    {
        return m_meshPartitioned;
    }

    void SetMeshPartitioned(bool meshPartitioned)
    {
        m_meshPartitioned = meshPartitioned;
    }

    int GetPartitionNumber()
    {
        return m_partition;
    }

    void SetPartitionNumber(int partition)
    {
        m_partition = partition;
    }

    ExpansionInfoMapShPtrMap &GetExpansionInfoMap()
    {
        return m_expansionMapShPtrMap;
    }

private:
    /**
     * @brief Helper function for geometry lookups
     */
    template <typename T> T *GetGeom(int id, GeomMap<T> &geomMap)
    {
        auto it = geomMap.find(id);
        ASSERTL0(it != geomMap.end(),
                 "Unable to find geometry with ID " + std::to_string(id));
        return it->second.get();
    }

protected:
    ExpansionInfoMapShPtr SetUpExpansionInfoMap();
    std::string GetCompositeString(CompositeSharedPtr comp);

    LibUtilities::SessionReaderSharedPtr m_session;

    CurveMap m_curvedEdges;
    CurveMap m_curvedFaces;
    CurveMap m_curvedVolumes;

    GeomMap<PointGeom> m_pointGeoms;
    GeomMap<SegGeom> m_segGeoms;
    GeomMap<TriGeom> m_triGeoms;
    GeomMap<QuadGeom> m_quadGeoms;
    GeomMap<TetGeom> m_tetGeoms;
    GeomMap<PyrGeom> m_pyrGeoms;
    GeomMap<PrismGeom> m_prismGeoms;
    GeomMap<HexGeom> m_hexGeoms;

    GeomMapView<PointGeom> m_pointMapView;
    GeomMapView<SegGeom> m_segMapView;
    GeomMapView<TriGeom> m_triMapView;
    GeomMapView<QuadGeom> m_quadMapView;
    GeomMapView<TetGeom> m_tetMapView;
    GeomMapView<PyrGeom> m_pyrMapView;
    GeomMapView<PrismGeom> m_prismMapView;
    GeomMapView<HexGeom> m_hexMapView;

    /// Vector of all unique curve nodes, not including vertices
    std::vector<PointGeomUniquePtr> m_nodeSet;

    int m_meshDimension;
    int m_spaceDimension;
    int m_partition;
    bool m_meshPartitioned = false;
    bool m_useExpansionType;

    // Refinement attributes (class members)
    /// Link the refinement id with the composites
    std::map<int, CompositeMap> m_refComposite;
    // std::map<int, LibUtilities::BasisKeyVector> m_refBasis;
    /// Link the refinement id with the surface region data
    std::map<int, RefRegion *> m_refRegion;
    bool m_refFlag = false;

    CompositeMap m_meshComposites;
    std::map<int, std::string> m_compositesLabels;
    std::map<int, CompositeMap> m_domain;
    LibUtilities::DomainRangeShPtr m_domainRange;

    ExpansionInfoMapShPtrMap m_expansionMapShPtrMap;

    std::unordered_map<Geometry *, FacetElementLink> m_facetToElMap;
    /// Whether #m_facetToElMap reflects the elements currently in the graph.
    bool m_facetToElMapBuilt = false;

    TiXmlElement *m_xmlGeom;

    CompositeOrdering m_compOrder;
    BndRegionOrdering m_bndRegOrder;

    struct GeomRTree;
    std::unique_ptr<GeomRTree> m_boundingBoxTree;
    MovementSharedPtr m_movement;

    /// The CAD system and the mesh entity to CAD associations. Null until a
    /// CAD system is attached, since most meshes have none.
    CADAssociationSharedPtr m_cadAssoc;
};

typedef std::shared_ptr<MeshGraph> MeshGraphSharedPtr;
typedef LibUtilities::NekFactory<std::string, MeshGraph> MeshGraphFactory;

SPATIAL_DOMAINS_EXPORT MeshGraphFactory &GetMeshGraphFactory();

/**
 *
 */
void MeshGraph::SetExpansionInfo(const std::string variable,
                                 ExpansionInfoMapShPtr &exp)
{
    if (m_expansionMapShPtrMap.count(variable) != 0)
    {
        NEKERROR(
            ErrorUtil::efatal,
            (std::string("ExpansionInfo field is already set for variable ") +
             variable)
                .c_str());
    }
    else
    {
        m_expansionMapShPtrMap[variable] = exp;
    }
}

/**
 *
 */
void MeshGraph::SetSession(LibUtilities::SessionReaderSharedPtr pSession)
{
    m_session = pSession;
}

/**
 *
 */
inline bool MeshGraph::SameExpansionInfo(const std::string var1,
                                         const std::string var2)
{
    ExpansionInfoMapShPtr expVec1 = m_expansionMapShPtrMap.find(var1)->second;
    ExpansionInfoMapShPtr expVec2 = m_expansionMapShPtrMap.find(var2)->second;

    if (expVec1.get() == expVec2.get())
    {
        return true;
    }

    return false;
}

/**
 *
 */
inline bool MeshGraph::ExpansionInfoDefined(const std::string var)
{
    return m_expansionMapShPtrMap.count(var);
}

} // namespace Nektar::SpatialDomains

#endif
