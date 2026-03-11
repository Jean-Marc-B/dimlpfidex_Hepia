#ifndef ERRORHANDLER_H
#define ERRORHANDLER_H

#include <stdexcept>
#include <string>
#include <utility>

/**
 * @brief Base class for handling exceptions with custom messages.
 *
 * ErrorHandler extends the standard std::runtime_error class to allow for exceptions
 * that carry custom messages. These messages are passed during construction and
 * retrieved via the what() method.
 */
class ErrorHandler : public std::runtime_error {
public:
  /**
   * @brief Constructs an ErrorHandler with a custom message.
   * @param message The message detailing the cause of the exception.
   */
  explicit ErrorHandler(std::string message) : std::runtime_error(std::move(message)) {}
};

/**
 * @brief Exception for invalid command line arguments.
 */
class CommandArgumentException : public ErrorHandler {
public:
  /**
   * @brief Constructs a CommandArgumentException with a custom message.
   * @param message The message detailing the cause of the exception.
   */
  explicit CommandArgumentException(std::string message) : ErrorHandler("CommandArgumentException: " + std::move(message)) {}
};

/**
 * @brief Exception indicating that a required file could not be found.
 */
class FileNotFoundError : public ErrorHandler {
public:
  /**
   * @brief Constructs a FileNotFoundError with a custom message.
   * @param message The message detailing the cause of the exception.
   */
  explicit FileNotFoundError(std::string message) : ErrorHandler("FileNotFoundError: " + std::move(message)) {}
};

/**
 * @brief Exception for errors when opening a file.
 */
class CannotOpenFileError : public ErrorHandler {
public:
  /**
   * @brief Constructs a CannotOpenFileError with a custom message.
   * @param message The message detailing the cause of the exception.
   */
  explicit CannotOpenFileError(std::string message) : ErrorHandler("CannotOpenFileError: " + std::move(message)) {}
};

/**
 * @brief Exception for errors related to incorrect file format.
 */
class FileFormatError : public ErrorHandler {
public:
  /**
   * @brief Constructs a FileFormatError with a custom message.
   * @param message The message detailing the cause of the exception.
   */
  explicit FileFormatError(std::string message) : ErrorHandler("FileFormatError: " + std::move(message)) {}
};

/**
 * @brief Exception for errors related to incorrect or unexpected content within a file.
 */
class FileContentError : public ErrorHandler {
public:
  /**
   * @brief Constructs a FileContentError with a custom message.
   * @param message The message detailing the cause of the exception.
   */
  explicit FileContentError(std::string message) : ErrorHandler("FileContentError: " + std::move(message)) {}
};

/**
 * @brief Exception for internal errors not covered by more specific exception types.
 */
class InternalError : public ErrorHandler {
public:
  /**
   * @brief Constructs an InternalError with a custom message.
   * @param message The message detailing the cause of the exception.
   */
  explicit InternalError(std::string message) : ErrorHandler("InternalError: " + std::move(message)) {}
};

/**
 * @brief Exception for errors when creating a directory.
 */
class CannotCreateDirectoryError : public ErrorHandler {
public:
  /**
   * @brief Constructs a CannotCreateDirectoryError with a custom message.
   * @param message The message detailing the cause of the exception.
   */
  explicit CannotCreateDirectoryError(std::string message) : ErrorHandler("CannotCreateDirectoryError: " + std::move(message)) {}
};

#endif // ERRORHANDLER_H
