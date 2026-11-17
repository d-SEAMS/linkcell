#include "linkcell.hpp"
#include <iostream>

int main() {
  const linkcell::Cell box = linkcell::Cell::ortho(10.0, 10.0, 10.0);
  const double xyz[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  const linkcell::Neighbours nn = linkcell::knearest(xyz, 2, box, 1);
  std::cout << "0 -> " << nn.neighbour(0, 0) << "\n";
  std::cout << "1 -> " << nn.neighbour(1, 0) << "\n";
}
