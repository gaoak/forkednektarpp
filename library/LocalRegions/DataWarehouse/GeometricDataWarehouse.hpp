///////////////////////////////////////////////////////////////////////////////
//
// File: GeometricDataWarehouse.hpp
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

#pragma once

#include <Collections/Collection.h>
#include <LibUtilities/BasicUtils/DataWarehouse/NekDataWarehouse.hpp>

#include <LibUtilities/BasicUtils/ShapeType.hpp>

namespace Nektar::LocalRegions
{

class GeometricDataCreator;

template <typename TData> class WeightsKey : public LibUtilities::BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef TData value_type;

    ~WeightsKey() override = default;

    WeightsKey(const unsigned int block_idx,
               const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "WeightsKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

template <typename TData> class JacobianKey : public LibUtilities::BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef TData value_type;

    ~JacobianKey() override = default;

    JacobianKey(const unsigned int block_idx,
                const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "JacobianKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

template <typename TData> class DerivFactorKey : public LibUtilities::BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef TData value_type;

    ~DerivFactorKey() override = default;

    DerivFactorKey(const unsigned int block_idx,
                   const unsigned int interleave_width, const bool transpose)
        : m_block_idx(block_idx), m_interleave_width(interleave_width),
          m_transpose(transpose)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width, m_transpose,
                     typeid(value_type).name(), "DerivFactorKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
    bool m_transpose;
};

template <typename TData> class CoordKey : public LibUtilities::BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef TData value_type;

    ~CoordKey() override = default;

    CoordKey(const unsigned block_idx, const unsigned int interleave_width,
             const bool transpose)
        : m_block_idx(block_idx), m_interleave_width(interleave_width),
          m_transpose(transpose)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width, m_transpose,
                     typeid(value_type).name(), "CoordKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
    bool m_transpose;
};

template <typename TData> class OrientKey : public LibUtilities::BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef unsigned int value_type;

    ~OrientKey() override = default;

    OrientKey(const unsigned int block_idx, const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "OrientKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

template <typename TData> class TraceToElmtMapKey : public LibUtilities::BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef unsigned int value_type;

    ~TraceToElmtMapKey() override = default;

    TraceToElmtMapKey(const unsigned int block_idx,
                      const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "TraceToElmtMapKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

template <typename TData>
class TraceToElmtSignKey : public LibUtilities::BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef int value_type;

    ~TraceToElmtSignKey() override = default;

    TraceToElmtSignKey(const unsigned int block_idx,
                       const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "TraceToElmtSignKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

template <typename TData> class InteriorMapKey : public LibUtilities::BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef unsigned int value_type;

    ~InteriorMapKey() override = default;

    InteriorMapKey(const unsigned int block_idx,
                   const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "InteriorMapKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

template <typename TData> class JacobianTraceKey : public LibUtilities::BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef TData value_type;

    ~JacobianTraceKey() override = default;

    JacobianTraceKey(const unsigned int block_idx,
                     const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "JacobianTraceKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

template <typename TData>
class JacobianLocTraceKey : public LibUtilities::BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef TData value_type;

    ~JacobianLocTraceKey() override = default;

    JacobianLocTraceKey(const unsigned int block_idx,
                        const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "JacobianLocTraceKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

/**
 * Whether the trace normal derivative factors have to be stored once per
 * trace quadrature point rather than once per trace.
 *
 * For a Seg, Quad or Hex this is just the element's deformed flag: a regular
 * element has a constant normal and constant deriv factors, so one value per
 * trace is enough.
 *
 * A collapsed shape is different. Its factors carry the regular part of the
 * collapsed chain rule, for a triangle
 *
 *     f_0 + (1 + eta_0)/2 f_1,
 *
 * and eta_0 varies along edge 0 even when f_0 and f_1 are constant. So a
 * straight sided triangle, which Nektar calls regular, still needs a factor
 * at every trace point. Storing per point also means the trace is free to
 * carry any quadrature it likes.
 *
 * The singular 2/(1 - eta) part of the chain rule is deliberately not in
 * this data - the operator applies it on the volume - so evaluating these
 * factors at every trace point is safe even at the collapsed apex.
 */
inline bool TraceDerivFactorsArePointwise(LibUtilities::ShapeType shape,
                                          bool isDeformed)
{
    switch (shape)
    {
        case LibUtilities::Seg:
        case LibUtilities::Quad:
        case LibUtilities::Hex:
            return isDeformed;
        default:
            return true;
    }
}

template <typename TData>
class JacNormGeomFactorLocTraceKey : public LibUtilities::BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef TData value_type;

    ~JacNormGeomFactorLocTraceKey() override = default;

    /// @param dir -1 (the default) contracts the derivative factors with
    ///            the trace normal, the classic scalar form; 0..dim-1 keeps
    ///            the factors of that Cartesian direction uncontracted, for
    ///            the vector-input lift.
    JacNormGeomFactorLocTraceKey(const unsigned int block_idx,
                                 const unsigned int interleave_width,
                                 const int dir = -1)
        : m_block_idx(block_idx), m_interleave_width(interleave_width),
          m_dir(dir)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width, m_dir,
                     typeid(value_type).name(), "JacNormGeomFactorLocTraceKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
    int m_dir;
};

class GeometricDataCreator : public LibUtilities::DataCreatorClass
{
public:
    ~GeometricDataCreator() override = default;
    GeometricDataCreator(const Collections::CollectionVector collections)
        : m_collections(collections)
    {
    }

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<TData> Create(
        const WeightsKey<TData> &weightsKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<TData> Create(
        const JacobianKey<TData> &jacobianKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<TData> Create(
        const DerivFactorKey<TData> &derivFactorKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<TData> Create(const CoordKey<TData> &coordKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<unsigned int> Create(
        const OrientKey<TData> &orientKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<unsigned int> Create(
        const TraceToElmtMapKey<TData> &traceToElmtMapKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<int> Create(
        const TraceToElmtSignKey<TData> &traceToElmtSignKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<unsigned int> Create(
        const InteriorMapKey<TData> &interiorMapKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<TData> Create(
        const JacobianTraceKey<TData> &jacobianTraceKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<TData> Create(
        const JacobianLocTraceKey<TData> &jacobianLocTraceKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<TData> Create(
        const JacNormGeomFactorLocTraceKey<TData> &jacnormgeomfacLocTraceKey);

    inline static const std::string m_name = "GeometricDataCreator";

private:
    Collections::CollectionVector m_collections;
};

} // namespace Nektar::LocalRegions
