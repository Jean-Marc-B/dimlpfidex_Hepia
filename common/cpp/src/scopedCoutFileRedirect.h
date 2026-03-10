#ifndef SCOPEDCOUTFILEREDIRECT_H
#define SCOPEDCOUTFILEREDIRECT_H

#include "errorHandler.h"
#include <fstream>
#include <iostream>
#include <string>

/**
 * @brief Redirects std::cout to a file for the lifetime of this object.
 *
 * The original stream buffer is restored automatically on destruction,
 * including when exceptions are thrown.
 */
class ScopedCoutFileRedirect {
public:
  explicit ScopedCoutFileRedirect(const std::string &filename)
      : stream_(std::cout), originalBuffer_(std::cout.rdbuf()) {
    output_.open(filename);
    if (!output_.is_open()) {
      throw CannotOpenFileError("Error : Couldn't open console output file " + filename + ".");
    }
    stream_.rdbuf(output_.rdbuf());
  }

  ~ScopedCoutFileRedirect() {
    stream_.rdbuf(originalBuffer_);
  }

  ScopedCoutFileRedirect(const ScopedCoutFileRedirect &) = delete;
  ScopedCoutFileRedirect &operator=(const ScopedCoutFileRedirect &) = delete;

private:
  std::ofstream output_;
  std::ostream &stream_;
  std::streambuf *originalBuffer_;
};

#endif // SCOPEDCOUTFILEREDIRECT_H
