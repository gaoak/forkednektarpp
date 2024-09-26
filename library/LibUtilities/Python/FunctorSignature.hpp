///////////////////////////////////////////////////////////////////////////////
//
// File: FunctorSignature.hpp
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
// Description: Add registration for functors using lambdas in boost::python
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIBRARY_LIBUTILITIES_PYTHON_FUNCTORSIGNATURE_HPP
#define NEKTAR_LIBRARY_LIBUTILITIES_PYTHON_FUNCTORSIGNATURE_HPP

////////////////////////////////////////
// Functor registers with boost::python
////////////////////////////////////////

#include <boost/mpl/erase.hpp>
#include <boost/mpl/vector.hpp>

namespace boost::python::detail
{
template <class Functor> struct functor_signature;

template <class Functor>
typename std::enable_if<
    std::is_member_function_pointer<decltype(&Functor::operator())>::value,
    typename functor_signature<Functor>::type>::type
get_signature(Functor &, void * = nullptr)
{
    return typename functor_signature<Functor>::type();
}
} // namespace boost::python::detail

#include <boost/python/signature.hpp>

namespace boost::python::detail
{
template <class Functor> struct functor_signature
{
    typedef decltype(get_signature(
        &Functor::operator())) member_function_signature;
    typedef typename mpl::advance<
        typename mpl::begin<member_function_signature>::type,
        mpl::int_<1>>::type instance_argument_iterator;
    typedef typename mpl::erase<member_function_signature,
                                instance_argument_iterator>::type type;
};
} // namespace boost::python::detail

#endif
