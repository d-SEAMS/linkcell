#include "linkcell.hpp"

#include <typeinfo>

const void *lc_static_other_type() { return &typeid(linkcell::Error); }
