// Fully static link: error.cpp is in this image once. Two translation
// units must see one typeinfo, and a throw is caught as linkcell::Error.
// A second shared library that also linked the static archive would get
// its own copy; that layout is not this test.

#include "linkcell.hpp"

#include <cstdio>
#include <exception>
#include <typeinfo>

extern const void *lc_static_other_type();

int main() {
  const void *here = &typeid(linkcell::Error);
  const void *other = lc_static_other_type();
  if (here != other) {
    std::fprintf(stderr, "static link has two linkcell::Error typeinfos (%p vs %p)\n",
                 here, other);
    return 1;
  }
  try {
    throw linkcell::Error("static");
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
}
