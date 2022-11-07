#include <memory>
#include <iostream>

#include "NekFactory.hpp"
#include "Field.hpp"

typedef double NekDouble;

struct OpBwdTrans;
struct OpIProductWRTBase;
struct OpPhysDeriv;

struct MethodLocMat;
struct MethodSumFac;
struct MethodMatFree;
using DefaultMethod = MethodLocMat;

// Base class
template<typename TDerived, typename TData, typename TStateIn, typename TStateOut>
class OperatorBase
{
    public:
        void apply(Field<TData, TStateIn> &in, Field<TData, TStateOut> &out)
        {
            static_cast<TDerived *>(this)->apply_impl(in, out);
        }
};

#if 0
template<typename TData, typename TStateIn, typename TStateOut>
class OperatorBase
{
    public:
        virtual void apply(Field<TData, TStateIn> &in, Field<TData, TStateOut> &out) = 0;
};
#endif


using namespace Nektar::LibUtilities;

/*
template<typename TData>
using OperatorFactory = NekFactory<std::string, OperatorBase<TData, TState>>;

template<typename TData>
OperatorFactory<TData> &GetOperatorFactory();
*/

// Templated Operator
template<typename TData,
         typename TOp,
         typename TMethod = DefaultMethod,
         typename TDevice = DefaultDevice>
class Operator;
