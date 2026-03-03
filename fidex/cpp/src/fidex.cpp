#include "fidexFct.h"

////////////////////////////////////////////////////////////

/// @cond
/**
 * @brief Entry point for fidex executable.
 *
 * This function serves as the entry point for fidex executable. It constructs a string command
 * from the command line arguments and then calls the fidex function with this command.
 *
 * @param nbParam Number of command line arguments.
 * @param param Array of command line arguments.
 * @return Exit status of the program.
 */
int main(int nbParam, char **param)

{
  std::string command;
  for (int p = 1; p < nbParam; p++) {
    if (!command.empty()) {
      command += ' ';
    }
    command += param[p];
  }
  return fidex(command);
}
/// @endcond
