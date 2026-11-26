// Built with hidden visibility, the same way a consumer DSO such as
// libseams is. These functions are that DSO's view of linkcell::Error.
// LC_PROBE_B compiles the second consumer. Neither DSO defines the type.

#include "linkcell.hpp"

#include <typeinfo>

#if defined(_WIN32)
#define LINKCELL_PROBE_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define LINKCELL_PROBE_API __attribute__((visibility("default")))
#else
#define LINKCELL_PROBE_API
#endif

#if defined(LC_PROBE_B)
extern "C" LINKCELL_PROBE_API const void *lc_hidden_b_error_type() {
  return &typeid(linkcell::Error);
}

extern "C" LINKCELL_PROBE_API void lc_hidden_b_throw_error() {
  throw linkcell::Error("cross-dso-b");
}
#else
extern "C" LINKCELL_PROBE_API const void *lc_hidden_error_type() {
  return &typeid(linkcell::Error);
}

extern "C" LINKCELL_PROBE_API void lc_hidden_throw_error() {
  throw linkcell::Error("cross-dso");
}
#endif
