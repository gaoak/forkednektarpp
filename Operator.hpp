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
template<typename TData>
class OperatorBase
{
    public:
        virtual void apply(TData in, TData out) = 0;
};

using namespace Nektar::LibUtilities;

template<typename TData>
using OperatorFactory = NekFactory<std::string, OperatorBase<TData>>;

template<typename TData>
OperatorFactory<TData> &GetOperatorFactory();

// Templated Operator
template<typename TData,
         typename TOp,
         typename TMethod = DefaultMethod,
         typename TDevice = DefaultDevice>
class Operator;


