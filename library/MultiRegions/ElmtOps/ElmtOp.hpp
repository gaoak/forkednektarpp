///////////////////////////////////////////////////////////////////////////////
//
// File: ElmtOp.hpp
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

/**
 * @file ElmtOp.hpp
 * @brief Whole-field base class and factory machinery shared by every
 * element-operator family in the Operators library.
 *
 * @details
 * An element operator acts element by element -- it never couples
 * degrees of freedom belonging to different elements -- and maps a
 * Field in one state (coefficient or physical space) to a Field in
 * the same or the other state: BwdTrans, for instance, takes
 * coefficients to physical values, while Mass maps coefficient space
 * to itself. Every family in the subdirectories of ElmtOps (BwdTrans,
 * IProductWRTBase, PhysDeriv, Mass, Helmholtz, ...) derives its public
 * interface class from ElmtOp, which fixes the input and output field
 * states at compile time and provides the two-stage factory
 * construction described in the ElmtOp class notes.
 *
 * The per-block machinery an element operator is built from -- how the
 * elements are grouped into blocks of like shape and basis, the
 * implementation-strategy tags (StdMat, SumFac, SumFacTOP, Generic)
 * and the device launch helpers -- lives in ElmtBlockOp.hpp; the
 * kernel size-parameter types the sum-factorised implementations hand
 * to their kernels live in ElmtHelper.hpp.
 */

#pragma once

#include <MultiRegions/Common/Operator.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Common base class of the element-operator interfaces: an
 * operator applied independently to every element of an expansion,
 * taking a field in state @p TFieldIn to a field in state
 * @p TFieldOut.
 *
 * @details
 * ### Two-level structure
 * An ElmtOp does no numerical work itself. The elements of the
 * expansion list are grouped into blocks -- contiguous runs of
 * elements sharing shape, basis and quadrature, one block per
 * Collection of the expansion list (see LibUtilities::BlockAttributes
 * in LibUtilities/BasicUtils/Field/Block.hpp and
 * MultiRegions::GetBlockAttributes()) -- and the operator holds one
 * block operator per block, stored in the derived interface's
 * m_blockOp vector. Applying the operator loops over the field's
 * blocks and hands each input/output BlockAccessor pair to the
 * matching block operator (see BwdTransOp::v_Apply for the canonical
 * loop).
 *
 * ### Role of this class
 * Like the other operator base classes in this library, it exists
 * chiefly to fix the Apply() parameter types at compile time: a family
 * such as BwdTransOp derives from
 * `ElmtOp<FieldState::Coeff, FieldState::Phys, TData>`, so passing a
 * field in the wrong state fails to compile. It also provides the
 * family-agnostic Create() and the implementation lookup GetOpImpl()
 * used to select between the block-level implementation strategies
 * (StdMat, SumFac, SumFacTOP, Generic; see ElmtBlockOp.hpp).
 *
 * @tparam TFieldIn  Field state consumed by Apply().
 * @tparam TFieldOut Field state produced by Apply().
 * @tparam TData     Floating-point type of the field data.
 *
 * @see ElmtBlockOp for the per-block base class and the implementation
 * tags; BwdTransOp / BwdTransBlockOp for a complete concrete family.
 */
template <FieldState TFieldIn, FieldState TFieldOut, typename TData>
class ElmtOp : public Operator<TData>
{
public:
    /**
     * @brief Build a complete element operator: the whole-field
     * interface object plus one block operator per element block.
     *
     * Construction happens in two factory stages:
     * -# The interface object is created through the operator factory
     *    by Operator::Create() under the key
     *    `TOperator::name + execStr`.
     * -# For every block of the expansion list, a block operator is
     *    created through the block-operator factory (see
     *    ElmtBlockOp::Create()) under the key
     *    `TBlockOperator::name + execStr + implStr` and appended to
     *    the interface object's m_blockOp vector. Each one is bound to
     *    its block index, the block's first expansion -- the
     *    representative element whose shape, basis and quadrature all
     *    elements of the block share -- and the expansion list's data
     *    warehouse.
     *
     * @note Both factories hand their product back through a
     * static_pointer_cast to the requested class, so the creator
     * registered under a key must construct that class or one derived
     * from it.
     *
     * @tparam TOperator      Family interface class (e.g. BwdTransOp).
     *                        Must declare this ElmtOp instantiation a
     *                        friend and provide the static `name`
     *                        string and the m_blockOp vector the block
     *                        operators are pushed into.
     * @tparam TBlockOperator Family block-operator class providing a
     *                        static Create() (e.g. BwdTransBlockOp).
     *
     * @param   expansionList   Expansion list the operator acts on;
     *                          its Collections define the blocks.
     * @param   components      Names of the field components the
     *                          operator is set up for.
     * @param   execStr         Execution space ("Serial", "AVX" or
     *                          "Device"); if empty,
     *                          Operator::GetOpExecSpace() supplies it
     *                          from the session's "opExecSpace"
     *                          command-line argument or, failing that,
     *                          the build's default, with a warning.
     * @param   implStr         Block implementation ("StdMat",
     *                          "SumFac", "SumFacTOP" or "Generic"); if
     *                          empty, the lookup rules of GetOpImpl()
     *                          apply, keyed by `TOperator::name`.
     *
     * @return The fully assembled operator, ready to Apply(). Creation
     * raises a fatal error (throws ErrorUtil::NekError) if no
     * implementation is registered under a requested key.
     */
    template <template <typename> typename TOperator,
              template <typename> typename TBlockOperator>
    static std::shared_ptr<TOperator<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        auto session = expansionList->GetSession();

        std::string execStr0 = (execStr == "")
                                   ? Operator<TData>::GetOpExecSpace(session)
                                   : execStr;

        std::string implStr0 =
            (implStr == "")
                ? GetOpImpl(TOperator<TData>::name, execStr0, session)
                : implStr;

        auto op = Operator<TData>::template Create<TOperator>(
            expansionList, components, execStr0);

        auto blockAttr =
            MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                expansionList);

        // Loop over the blocks.
        for (unsigned int block_idx = 0; block_idx < blockAttr.size();
             block_idx++)
        {
            const auto exp =
                MultiRegions::GetCollection(expansionList, block_idx)
                    .GetExpVector()[0];
            op->m_blockOp.push_back(TBlockOperator<TData>::Create(
                block_idx, exp, expansionList->GetDataWarehouseSharedPtr(),
                execStr0, implStr0));
        }

        return op;
    }

    /**
     * @brief Apply the operator to @p in, writing the result to
     * @p out.
     *
     * Both fields must be built on the operator's expansion list, so
     * that their block decomposition matches the block operators
     * created alongside this object.
     *
     * @param   in      Input field in state @p TFieldIn.
     * @param   out     Output field in state @p TFieldOut.
     */
    void Apply(LibUtilities::Field<TData, TFieldIn> &in,
               LibUtilities::Field<TData, TFieldOut> &out)
    {
        v_Apply(in, out);
    }

    /// @brief Call operator; equivalent to Apply().
    void operator()(LibUtilities::Field<TData, TFieldIn> &in,
                    LibUtilities::Field<TData, TFieldOut> &out)
    {
        v_Apply(in, out);
    }

    /**
     * @brief Return the implementation for an operator.
     *
     * This looks up the implementation (e.g. "StdMat", "SumFac") for a
     * given operator name and session from maps populated by
     * ParseOptimisations().
     *
     * Lookup rules:
     * - The command-line argument "opImpl" overrides everything.
     * - Otherwise the session's backend map for @p opExecSpace is
     *   consulted: the serial map for "Serial", the AVX map for "AVX" and
     *   the device map for "Device" (SessionReader::GetSerialBackendMap()
     *   and its siblings). The map keys use the UPPERCASE of @p opName
     *   (e.g. "Mass" -> "MASS").
     * - An operator named in neither takes "SumFac" and warns that it has
     *   done so.
     *
     * @param opName  Operator name (e.g. "Mass", "Helmholtz").
     * @param opExecSpace  Execution space whose map is consulted; must be
     * "Serial", "AVX" or "Device" (asserted).
     * @param session  Session reader to recover the relevant maps and
     * command-line arguments.
     *
     * @return std::string containing the implementation name (e.g. "SumFac",
     * "StdMat").
     */
    static std::string GetOpImpl(const std::string &opName,
                                 const std::string &opExecSpace,
                                 LibUtilities::SessionReaderSharedPtr session)
    {
        ASSERTL0(
            opExecSpace == "Serial" || opExecSpace == "AVX" ||
                opExecSpace == "Device",
            "Operator execution space must be 'Serial', 'AVX', or, 'Device'");

        std::string OpImpl = "SumFac";

        // command line overrides everything
        if (session->DefinesCmdLineArgument("opImpl"))
        {
            OpImpl = session->GetCmdLineArgument<std::string>("opImpl");
        }
        else
        {
            bool OpImplSet = false;

            // Keys are stored uppercased by ParseOptimisations().
            const std::string opNameUpper = boost::to_upper_copy(opName);
            if (opExecSpace == "Serial")
            {
                LibUtilities::BackendMap &serialBackendInfo =
                    session->GetSerialBackendMap();
                auto opImplIter = serialBackendInfo.find(opNameUpper);
                if (opImplIter != serialBackendInfo.end())
                {
                    OpImpl    = opImplIter->second;
                    OpImplSet = true;
                }
            }
            else if (opExecSpace == "AVX")
            {
                LibUtilities::BackendMap &avxBackendInfo =
                    session->GetAVXBackendMap();
                auto opImplIter = avxBackendInfo.find(opNameUpper);
                if (opImplIter != avxBackendInfo.end())
                {
                    OpImpl    = opImplIter->second;
                    OpImplSet = true;
                }
            }
            else if (opExecSpace == "Device")
            {
                LibUtilities::BackendMap &deviceBackendInfo =
                    session->GetDeviceBackendMap();
                auto opImplIter = deviceBackendInfo.find(opNameUpper);
                if (opImplIter != deviceBackendInfo.end())
                {
                    OpImpl    = opImplIter->second;
                    OpImplSet = true;
                }
            }

            if (OpImplSet == false)
            {
                NEKERROR(ErrorUtil::ewarning,
                         "No implementation space specified "
                         "for operator. Defaulting to " +
                             OpImpl);
            }
        }

        return OpImpl;
    }

protected:
    /**
     * @brief Construct the interface part of a concrete operator;
     * called by the factory-registered creator functions.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Component names the operator is set up
     *                          for.
     */
    ElmtOp(const MultiRegions::ExpListSharedPtr &expansionList,
           const std::vector<std::string> &components)
        : Operator<TData>(expansionList, components)
    {
    }

    ~ElmtOp() override = default;

    /**
     * @brief Implementation hook for Apply().
     *
     * Overrides must produce in @p out the result of the element
     * operation applied to @p in for every component. The standard
     * pattern (see BwdTransOp::v_Apply) checks that the two fields
     * conform and delegates each block of @p in / @p out to the
     * corresponding entry of the family's m_blockOp vector.
     *
     * @param   in      Input field in state @p TFieldIn.
     * @param   out     Output field in state @p TFieldOut.
     */
    virtual void v_Apply(LibUtilities::Field<TData, TFieldIn> &in,
                         LibUtilities::Field<TData, TFieldOut> &out) = 0;
};

} // namespace Nektar::MultiRegions
