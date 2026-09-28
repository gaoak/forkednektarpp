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
   an operator through setters, forwarded to its BlockOps. An operator takes
   no ownership of a `Field` or `MemoryRegion` passed to a setter: it stores a
   non-owning pointer and never moves from it, and the caller must make
   provision for the object to outlive the operator. Work derived from one -
   a sample onto another frame, an exchange across ranks - belongs in the next
   `Apply()`, behind a stale flag the setter sets, not in the setter itself; a
   caller that refills the object in place does not trip that flag.
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
   parameters or a named query such as `IsInstantiated()`. A hook whose
   default does nothing is pure virtual, because "nothing" is a decision each
   derived class should state; a hook whose default refuses, raising "not
   available for this implementation", keeps that default, because the
   refusal is shared and copying it into every class that does not provide
   the entry point says no more than the base already did.
9. **Nothing is public without a reason.** Member functions and variables are
   protected or private unless public access is required; no `Get...()` for
   every `Set...()`.
10. **Scalars are `double`** (or the `TData` parameter), not `NekDouble`.

## Names, order and comments

11. **Names say what the quantity is** (`invTimeScale`, not `rate`), and where
    one quantity exists in several frames - local trace, global trace, packed
    for a kernel - the name says which. Names differing only in capitalisation
    are not distinct names, of a variable or of a file: two paths differing
    only in case cannot both exist on macOS or Windows.
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

## Files

15. **Every source file looks like its neighbours.** It opens with the MIT
    licence header, its `File:` and `Description:` fields filled in, and a
    header guards itself with `#pragma once`, not an include guard; it is
    named in PascalCase after the class, program or test it holds
    (`EntityResolverDemo.cpp`, not `test_resolver.cpp`); and it is
    `clang-format` clean. Working notes, scratch programs, design notes and
    hand-written build lines do not go in the tree: a convention worth writing
    down at length belongs in the developer guide, and beside the code it
    belongs in the comment of the thing it explains.
16. **XML session and mesh files follow the project's indentation**, which
    `.gitlab-ci/checkformatXMLfile.sh` enforces and `clang-format` does not
    touch. Run `.gitlab-ci/formatXMLfile.sh <file> <file>` over any session or
    mesh file added or edited.

## Layering and namespaces

17. **A file's namespace follows the library it lives in.** Under
    `library/Operators` that is `Nektar::Operators`, or
    `Nektar::Operators::detail` below the interface; under
    `library/SolverCore`, `Nektar::SolverCore[::detail]`. A solver's own
    operators are not in the library's namespace: plain `Nektar` for the
    operator header, `Nektar::detail` for kernels, implementations and
    generated registrations, with `OPNAMESPACE` set to match. Library
    facilities a solver calls keep their `Operators::` qualification, written
    at each use rather than implied by an enclosing namespace.
18. **`Operators` does not depend on `SolverCore`.** A class under
    `library/Operators` that needs an include from `SolverCore` is in the
    wrong library: move it rather than add the include.
19. **A solver includes `...Op.hpp`, never `...OpImpl.hpp`.** The
    implementation is reached through the factory, and the registrations
    arrive at link time from the generated sources.
20. **An `...Op.hpp` interface is agnostic of the execution space and 
    implementation**: no `*Kernels.hpp` include at that level, no include 
    the file does not use, and no direct use of ExecSpace, MemSpace, and/or
    Implementation template parameters.
21. **An operator's name lives on its interface class**, as
    `static inline const std::string name = "...";` beside the `Create()` a
    solver calls, and the generated factory declarations build their
    registration key from it. Never a literal in the template, and never a
    class existing only to hold a `name` for the boilerplate to find: if an
    operator has no interface class, that is what to add.
22. **Do not keep a second copy of state another object owns.** A helper whose
    every method reaches into one object - for its storage, its numbering, or
    a list it already holds - belongs inside that object.

## Generated operators and registration

23. **An unregistered operator fails only at run time.** `ADD_OPERATOR` and
    `ADD_BLOCK_OPERATOR` skip a path that is not a directory, or one with no
    `<Name>BlockOp<ExecName>.hpp`, without a warning; the first caller gets
    `No such operator`. After adding, moving or renaming an operator, check its
    sources appear under `<build>/.../src/`.

## Device code

24. **A function called from a `NEKTAR_LAMBDA` is marked for the spaces its
    caller is compiled for**: `NEK_HOSTDEVICE_INLINE` in fully generic code,
    `NEK_DEVICE_INLINE` when reached only from a Device backend, plain
    `NEK_FORCE_INLINE` when the lambda is SerialAVX only. Off CUDA all three
    expand to the same thing, so a host build never catches the wrong choice.
    Generic code including a `*SerialAVXKernels.hpp` header is a layering
    mistake, not a missing marker.

## Tests

25. **A test of an operator that couples elements across a partition fills its
    input from the quadrature point coordinates**, not from the storage index.
    A storage index is a property of the partition: the same element takes
    different values at different rank counts, so serial and parallel runs do
    not start from the same field, and the values grow with the mesh, which
    inflates an absolute tolerance. Element-local operators have no such
    constraint.
26. **A test that writes beside its input needs a working directory of its
    own.** A parallel run partitions its session into `run/<session>_xml/`
    next to the mesh, so two tests sharing a directory race over the same
    files under a parallel `ctest`.

## Device memory

27. **Synchronise every block, not the first.** Memory is synchronised a block
    at a time, while a kernel walks a field from its first block's pointer;
    taking that pointer leaves every later block on the wrong side of the
    device boundary. The fields an operator holds need not share a block
    count, so each is walked to its own length. A host build collapses the
    access modes and cannot see the fault; on a device the symptom is wrong
    values, not a failure.
28. **A stream is used consistently or not at all.** Passing a stream at some
    launch sites of an operator and not others implies an overlap that does
    not happen, and a launch that sets the stream leaves it set for whatever
    runs next.

## Kernels and launches

29. **A `parallel_for` belongs in a wrapper, not in a member function.** The
    launch goes in a free function in the `detail` namespace, in a
    `*Kernels.hpp` file, taking what it needs as arguments. A lambda inside a
    member function must be reached through a public enclosing function under
    nvcc, which opens the class's access control for a reason unrelated to
    its interface, and it captures members implicitly. A parameter is not a
    capture, so the wrapper may also use `if constexpr` where the lambda could
    not.

30. **No assert, of any level, inside a function a kernel can call.** An
    assert builds a `std::string` for its message, and a single-pass device
    compiler has no device pass in which that compiles out, so even a level-1
    assert in a host-device function puts the string into the kernel and the
    module fails to load at run time. The device CI jobs are Debug builds, so
    a level-1 assert is compiled in there regardless. Check the condition on
    the host, before the launch.

## Shared operators

31. **State on a shared operator is set where it is used.** An operator held
    through a `shared_ptr` can be reached by another holder, so a flag or a
    scale set once when it is attached can be changed before it is read. Set
    it immediately before the `Apply()` that depends on it.
