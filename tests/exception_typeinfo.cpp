// libc++ matches a handler by the typeinfo address. libstdc++ matches the
// type name, so a catch across two DSOs can succeed while the addresses
// still differ. This program fails unless both consumer DSOs see one
// typeinfo and a throw from each is caught as linkcell::Error.

#include "linkcell.hpp"

#include <cstdio>
#include <exception>
#include <typeinfo>

extern "C" const void *lc_hidden_error_type();
extern "C" void lc_hidden_throw_error();
extern "C" const void *lc_hidden_b_error_type();
extern "C" void lc_hidden_b_throw_error();

namespace {

int check_type(const char *name, const void *hidden) {
  const void *here = &typeid(linkcell::Error);
  if (here != hidden) {
    std::fprintf(stderr, "%s typeinfo differs across DSOs (%p vs %p)\n", name,
                 here, hidden);
    return 1;
  }
  return 0;
}

int check_throw(void (*thrower)()) {
  try {
    thrower();
  } catch (const linkcell::Error &err) {
    if (err.status() != 1) {
      std::fprintf(stderr, "linkcell::Error status %d\n", err.status());
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
  if (check_type("a", lc_hidden_error_type()) != 0 ||
      check_type("b", lc_hidden_b_error_type()) != 0) {
    return 1;
  }
  if (check_throw(lc_hidden_throw_error) != 0 ||
      check_throw(lc_hidden_b_throw_error) != 0) {
    return 1;
  }
  return 0;
}
