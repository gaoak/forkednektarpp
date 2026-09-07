# Coding practices

Conventions agreed in redesign code review. Kept terse by design; where they
and CONTRIBUTING.md disagree, CONTRIBUTING.md wins.

## Construction and application

1. **Decide at construction, not per call.** The interleave format, sizes,
   warehouse data (requested at the implementation's width) and anything
   `Apply()` needs are fixed and cached when the operator is built. No
   `if (GetNumElements() == 0) return;`, no per-apply asserts restating a
   fixed contract, never a silent early return on a mismatch; a guard against
   genuinely unsupported input may stay.
2. **`Apply()` carries only the fields it acts on.** Runtime parameters reach
   an operator through setters, forwarded to its BlockOps. A setter taking
   `Field&` stores a non-owning pointer and never moves from it; the argument
   must outlive the operator.
3. **Minimal ExpansionList dependence.** The block's representative expansion
   is `GetCollection(list, blk).GetExpVector()[0]`; a global element id is
   not an index into the list. Per-block validation lives in the BlockOp.

## Layout and vectorisation

4. **Never assert a storage layout.** Fields are created at interleave width
   1; an implementation reshapes to its own width (`ReshapeStorage`, per
   component, restored afterwards) or supports both layouts.
5. **Serial/AVX code works in the vector type.** One simd lane per element
   via `simd_type_if`; kernels in `*SerialAVXKernels.hpp` take
   `const simd_type *`; no integer `/` or `%` per element; padding zeroed
   inline in the group loop; horizontal reduce at the end. Lane naming is
   `ilane = e % width; iwarp = e / width`.
6. **Device kernels use `NektarSpaces::Device::warpSize`** so access is
   coalesced.
7. **Kernels are fused free functions, not local lambdas.** `DEFORMED` is a
   kernel template parameter, dispatched once in the wrapper:
   `if (m_isDeformed) Kernel<true> else Kernel<false>`.

## Interfaces

8. **Virtual hooks are protected or private accessors** returning the object;
   public access, when needed, is a non-virtual method on the base class.
   "Not available" is the returned object's own uninstantiated state, tested
   with `explicit operator bool()` (`if (field)`), not a bool with output
   parameters or a named query such as `IsInstantiated()`.
9. **Nothing is public without a reason.** Member functions and variables are
   protected or private unless public access is required; no `Get...()` for
   every `Set...()`.
10. **Scalars are `double`** (or the `TData` parameter), not `NekDouble`.

## Names, order and comments

11. **Names say what the quantity is** (`invTimeScale`, not `rate`).
12. **Sibling implementations match** (Serial/AVX vs Device, 1D/2D/3D): the
    same comments, member declaration order, local names and section
    comments. Member variables are declared before member functions. For
    ElmtOps, follow the pattern, names and types of `BwdTrans`, `Helmholtz`,
    `IProductWRTBase`, `LinAdvDiffReaction` and `PhysDeriv`.
13. **Comments say what the code does.** No provenance ("as legacy does"), no
    references to earlier merge requests; that belongs in commit messages and
    MR descriptions.
14. **Descriptions live with the definition** in the `.cpp`; declarations
    keep a one-line brief. Header-only code documents in the header.
