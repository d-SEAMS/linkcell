#!/bin/sh
# The public headers must be pedantic C99 and C++17 under GCC and Clang.
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
for cc in gcc clang; do
    "$cc" -std=c99 -Wall -Wextra -pedantic -Werror -I "$root/include" -c "$tmp/pedantic.c" -o "$tmp/pedantic-$cc.o"
done
for cxx in g++ clang++; do
    "$cxx" -std=c++17 -Wall -Wextra -pedantic -Werror -I "$root/include" -c "$tmp/pedantic.cpp" -o "$tmp/pedantic-$cxx.o"
done

# docs/reference/c-api.md shows the handle as the generated header defines it.
handle() {
    awk '/^typedef struct lc_nd_index/ { on = 1 } on { print } on && /} lc_nd_index;/ { exit }' "$1" |
        tr -s ' \t\n' ' '
}
header=$(handle "$root/include/linkcell.h")
docs=$(handle "$root/docs/reference/c-api.md")
if [ -z "$header" ] || [ "$header" != "$docs" ]; then
    echo "docs/reference/c-api.md: lc_nd_index is '$docs'; include/linkcell.h has '$header'" >&2
    exit 1
fi
