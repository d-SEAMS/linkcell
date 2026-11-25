#!/bin/sh
# The public headers must be pedantic C99 and C++17.
# A zero-length array is not a valid object type in either language.
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cat > "$tmp/pedantic.c" <<'EOF'
#include "linkcell.h"
typedef char lc_nd_index_size_is_one[(sizeof(lc_nd_index) == 1) ? 1 : -1];
int main(void) {
    lc_nd_index_size_is_one ok;
    (void)ok;
    return sizeof(lc_nd_index) == 1 ? 0 : 1;
}
EOF
cat > "$tmp/pedantic.cpp" <<'EOF'
#include "linkcell.hpp"
typedef char lc_nd_index_size_is_one[(sizeof(lc_nd_index) == 1) ? 1 : -1];
int main() {
    lc_nd_index_size_is_one ok;
    (void)ok;
    return sizeof(lc_nd_index) == 1 ? 0 : 1;
}
EOF
# Both languages require sizeof(lc_nd_index) == 1, so the sizes match.
gcc -std=c99 -Wall -Wextra -pedantic -Werror -I "$root/include" -c "$tmp/pedantic.c" -o "$tmp/pedantic.o"
g++ -std=c++17 -Wall -Wextra -pedantic -Werror -I "$root/include" -c "$tmp/pedantic.cpp" -o "$tmp/pedantic_cpp.o"
