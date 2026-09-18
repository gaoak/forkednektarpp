///////////////////////////////////////////////////////////////////////////////
//
// File: ObjectPool.hpp
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
// Description: Fast geometry object allocator using boost::fast_pool_allocator.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIBUTILITIES_OBJECTPOOL_HPP
#define NEKTAR_LIBUTILITIES_OBJECTPOOL_HPP

#include <iostream>
#include <memory>
#include <vector>

#include <boost/pool/pool.hpp>
#include <boost/pool/pool_alloc.hpp>

namespace Nektar
{

template <typename DataType>
using PoolAllocator =
    boost::fast_pool_allocator<DataType,
                               boost::default_user_allocator_new_delete,
                               boost::details::pool::null_mutex>;

/**
 * @brief Generic object pool allocator/deallocator.
 *
 * This class provides an allocator based on the boost::fast_pool_allocator for
 * creating object pools.
 */
template <typename DataType> class ObjPoolManager
{
public:
    template <typename... Args> static DataType *Allocate(const Args &...args)
    {
        DataType *ptr = m_alloc.allocate();
        new (ptr) DataType(args...);
        return ptr;
    }

    static void Deallocate(DataType *ptr)
    {
        ptr->~DataType();
        return m_alloc.deallocate(ptr);
    }

    struct UniquePtrDeleter
    {
        void operator()(DataType *ptr) const
        {
            // ignore dealloc
            ObjPoolManager<DataType>::Deallocate(ptr);
        }
    };

    template <typename... Args>
    static std::unique_ptr<DataType, UniquePtrDeleter> AllocateUniquePtr(
        const Args &...args)
    {
        return std::unique_ptr<DataType, UniquePtrDeleter>(Allocate(args...));
    }

    /// The pool this type is allocated from; defined just below.
    static PoolAllocator<DataType> m_alloc;
};

/**
 * @brief Definition of ObjPoolManager::m_alloc.
 *
 * Defined out of class rather than as an inline member: an inline member is a
 * definition inside the class, which MSVC instantiates along with the class
 * and so requires DataType to be complete. Geometry.h forms
 * unique_ptr_objpool<Curve> and unique_ptr_objpool<PointGeom> on forward
 * declarations, so that is not available. Out of class this is a template in
 * its own right, instantiated later and only where the type is complete.
 *
 * The pool itself is unaffected either way: PoolAllocator is empty, its
 * storage being boost::singleton_pool keyed on the object size, so this
 * defines no memory. All it adds per library is a guard variable per type,
 * and those are emitted as STB_GNU_UNIQUE, one to a process.
 */
template <typename DataType>
PoolAllocator<DataType> ObjPoolManager<DataType>::m_alloc;

template <typename T>
using unique_ptr_objpool =
    std::unique_ptr<T, typename ObjPoolManager<T>::UniquePtrDeleter>;

template <class To, class From, class D>
std::unique_ptr<To, D> unique_ptr_objpool_cast(std::unique_ptr<From, D> &&p)
{
    auto *as = dynamic_cast<To *>(p.get());
    ASSERTL2(as, "Bad cast in unique_ptr_objpool_cast");

    // Move the deleter out of p, then release the raw pointer and rewrap.
    D &del    = p.get_deleter();
    From *raw = p.release();
    return std::unique_ptr<To, D>(static_cast<To *>(raw), std::move(del));
}

} // namespace Nektar

#endif
