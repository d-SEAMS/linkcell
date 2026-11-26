#include "linkcell.hpp"

namespace linkcell {

Error::Error(int status, std::string message)
    : std::runtime_error(message.empty() ? "linkcell: knearest failed"
                                         : std::move(message)),
      status_(status) {}

Error::Error(std::string message) : Error(1, std::move(message)) {}

Error::Error(const Error &) = default;
Error::Error(Error &&) noexcept = default;
Error &Error::operator=(const Error &) = default;
Error &Error::operator=(Error &&) noexcept = default;
Error::~Error() = default;

int Error::status() const noexcept { return status_; }

} // namespace linkcell
