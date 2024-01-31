#ifndef NEKTAR__OPERATORS_OPERATORS_DECLSPEC_H
#define NEKTAR__OPERATORS_OPERATORS_DECLSPEC_H

#if defined(_MSC_VER)
#ifdef OPERATORS_EXPORTS
#define OPERATORS_EXPORT _declspec(dllexport)
#else
#define OPERATORS_EXPORT _declspec(dllimport)
#endif
#else
#define OPERATORS_EXPORT
#endif

#endif // NEKTAR__OPERATORS_OPERATORS_DECLSPEC_H
