// libc++ matches a handler by the typeinfo address. That address is one
// value on the Itanium ABI (Linux and macOS) because the destructor is the
// key function. MSVC matches the decorated type name and gives each module
// its own type_info object, so the addresses differ when the catch still
// hits. This program fails unless a throw from each consumer DSO is caught
// as linkcell::Error and the status and message survive. Address equality
// is required only where the ABI guarantees it.

#include "linkcell.hpp"

#include <cstdio>
#include <cstring>
#include <exception>
#include <typeinfo>

extern "C" const void *lc_hidden_error_type();
extern "C" void lc_hidden_throw_error();
extern "C" const void *lc_hidden_b_error_type();
extern "C" void lc_hidden_b_throw_error();

namespace {

#if !defined(_MSC_VER)
int check_type(const char *name, const void *hidden) {
  const void *here = &typeid(linkcell::Error);
  if (here != hidden) {
    std::fprintf(stderr, "%s typeinfo differs across DSOs (%p vs %p)\n", name,
                 here, hidden);
    return 1;
  }
  return 0;
}
#endif

int check_throw(void (*thrower)(), const char *message) {
  try {
    thrower();
  } catch (const linkcell::Error &err) {
    if (err.status() != 1) {
      std::fprintf(stderr, "linkcell::Error status %d\n", err.status());
      return 1;
    }
    if (std::strcmp(err.what(), message) != 0) {
      std::fprintf(stderr, "linkcell::Error message '%s', want '%s'\n",
                   err.what(), message);
      return 1;
    }
    return 0;
  } catch (const std::exception &err) {
    std::fprintf(stderr, "caught std::exception, not linkcell::Error: %s\n",
                 err.what());
    return 1;
  } catch (...) {
    std::fprintf(stderr, "caught an unknown exception\n");
    return 1;
  }
  std::fprintf(stderr, "hidden library returned without throwing\n");
  return 1;
}

} // namespace

int main() {
#if !defined(_MSC_VER)
  if (check_type("a", lc_hidden_error_type()) != 0 ||
      check_type("b", lc_hidden_b_error_type()) != 0) {
    return 1;
  }
#endif
  if (check_throw(lc_hidden_throw_error, "cross-dso") != 0 ||
      check_throw(lc_hidden_b_throw_error, "cross-dso-b") != 0) {
    return 1;
  }
  return 0;
}
