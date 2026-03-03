#include "fidexGloRulesFct.h"

////////////////////////////////////////////////////////////

/// @cond
/**
 * @brief Entry point for fidexGloRules executable.
 *
 * This function serves as the entry point for fidexGloRules executable. It constructs a string command
 * from the command line arguments and then calls the fidexGloRules function with this command.
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
  return fidexGloRules(command);
}
/// @endcond
