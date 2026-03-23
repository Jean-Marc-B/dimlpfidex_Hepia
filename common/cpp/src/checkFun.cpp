#include "checkFun.h"
#include "errorHandler.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <utility>

namespace {
/**
 * @brief Escapes regex metacharacters so attribute names are matched literally.
 */
std::string escapeRegexLiteral(const std::string &input) {
  std::string escaped;
  escaped.reserve(input.size() * 2);

  for (char c : input) {
    switch (c) {
    case '\\':
    case '^':
    case '$':
    case '.':
    case '|':
    case '?':
    case '*':
    case '+':
    case '(':
    case ')':
    case '[':
    case ']':
    case '{':
    case '}':
      escaped.push_back('\\');
      break;
    default:
      break;
    }
    escaped.push_back(c);
  }

  return escaped;
}
} // namespace

/**
 * @brief Checks if a given string represents a valid integer.
 *
 * @param input A string representing the number to be checked.
 * @return Returns true if the string is a valid integer, false otherwise.
 */
bool checkInt(const std::string &input) {
  try {
    std::size_t pos;
    std::stoi(input, &pos);

    if (pos != input.length()) {
      return false; // There are some non-numerical characters
    }
  } catch (const std::invalid_argument &) {
    return false;
  } catch (const std::out_of_range &) {
    return false;
  }
  return true;
}

/**
 * @brief Checks if a given string represents a valid floating-point number.
 *
 * @param str A string representing the number to be checked.
 * @return Returns true if the string is a valid float, false otherwise.
 */
bool checkFloat(const std::string &input) {
  try {
    std::size_t pos;        // To store the position of the last processed character
    std::stod(input, &pos); // Convert string to double

    if (pos != input.length()) {
      return false; // There are unprocessed characters left in the string
    }
  } catch (const std::invalid_argument &) { // Handle invalid arguments
    return false;
  } catch (const std::out_of_range &) { // Handles cases where the value is out of range
    return false;
  }
  return true;
}

////////////////////////////////////////////////////////

/**
 * @brief Checks if a given string is a valid representation of a boolean value.
 *
 * @param input A string input representing a boolean.
 * @return Returns true if the string is 'true', 'false', '0', or '1' (case-insensitive), false otherwise.
 */
bool checkBool(const std::string &inputTemp) {
  std::string input = inputTemp;
  std::transform(input.begin(), input.end(), input.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return (input == "true" || input == "false" || input == "0" || input == "1");
}

////////////////////////////////////////////////////////

/**
 * @brief Checks if a given string is in the format of a list of floating-point numbers.
 *
 * @param input A string representing a list of floats in any format in (a,b), [a,b], a,b
 * with or without spaces, with or without commas.
 * @return Returns true if the string is a valid list of floats, false otherwise.
 */
bool checkList(const std::string &str) {
  static const std::string numericPattern = "[-+]?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?";
  static const std::regex listPattern("^\\s*(?:\\[|\\()?\\s*" + numericPattern + "(?:[ ,]+\\s*" + numericPattern + ")*\\s*(?:\\]|\\))?\\s*$");

  return std::regex_match(str, listPattern);
}

////////////////////////////////////////////////////////

/**
 * @brief Checks if a given string is in the format of a list of doubles, including nan/inf values.
 *
 * @param input A string representing a list of doubles.
 * @return Returns true if the string is a valid list of doubles, false otherwise.
 */
bool checkDoubleList(const std::string &str) {
  static const std::string doublePattern = "[-+]?(?:\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?|[nN][aA][nN]|[iI][nN][fF](?:[iI][nN][iI][tT][yY])?)";
  static const std::regex listPattern("^\\s*(?:\\[|\\()?\\s*" + doublePattern + "(?:[ ,]+\\s*" + doublePattern + ")*\\s*(?:\\]|\\))?\\s*$");

  return std::regex_match(str, listPattern);
}

////////////////////////////////////////////////////////

/**
 * @brief Checks if a given string is empty or contains only whitespace bytes.
 *
 * @param line A string to be checked.
 * @return Returns true if the string is empty or contains only whitespace, false otherwise.
 */
bool checkStringEmpty(const std::string &line) {
  return std::all_of(line.begin(), line.end(), [](unsigned char c) { return std::isspace(c) != 0; });
}

///////////////////////////////////////////////////////

/**
 * @brief Converts a double value to a string and formats it by removing trailing zeros and decimal point if necessary.
 *
 * @param number The double number to be converted.
 * @return A formatted string representing the number.
 */
std::string formattingDoubleToString(double number) {
  std::string str = std::to_string(number);
  str.erase(str.find_last_not_of('0') + 1, std::string::npos);
  str.erase(str.find_last_not_of('.') + 1, std::string::npos);
  return str;
}

//////////////////////////////////////////////////////

/**
 * @brief Splits a given string into a vector of substrings based on a specified delimiter.
 *
 * This function takes a string and a delimiter string, then iteratively finds and extracts
 * substrings separated by the delimiter. Substrings at the start and end of the input string
 * are handled appropriately to avoid including empty strings in the result.
 *
 * @param str The string to be split.
 * @param delimiter The string used as the delimiter to split the input string.
 * @return A vector containing the substrings obtained by splitting the input string.
 */
std::vector<std::string> splitString(const std::string &str, const std::string &delimiter) {
  std::vector<std::string> tokens;
  if (delimiter.empty()) {
    if (!str.empty()) {
      tokens.push_back(str);
    }
    return tokens;
  }

  size_t start = 0;
  size_t end = str.find(delimiter);

  // Loop to find and add new sub-strings
  while (end != std::string::npos) {
    if (start != end) { // Not adding empty strings
      tokens.push_back(str.substr(start, end - start));
    }
    start = end + delimiter.length();
    end = str.find(delimiter, start);
  }

  // Add last string if not empty
  if (start != str.length()) {
    tokens.push_back(str.substr(start));
  }

  return tokens;
}

//////////////////////////////////////////////////////

/**
 * @brief Parses a string representing a list of doubles and returns them as a vector.
 *
 * @param str A string in in any format in (a,b), [a,b], a,b with or without spaces,
 * with or without commas representing a list of floats.
 * @return A vector of doubles parsed from the string.
 */
std::vector<double> getDoubleVectorFromString(std::string str) {
  static const std::regex floatPattern("[-+]?(?:\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?|[nN][aA][nN]|[iI][nN][fF](?:[iI][nN][iI][tT][yY])?)");
  std::smatch match;
  std::vector<double> numbers;
  while (std::regex_search(str, match, floatPattern)) {
    numbers.push_back(std::stod(match[0].str()));
    str = match.suffix().str(); // Continue with the rest of the string
  }
  return numbers;
}

//////////////////////////////////////////////////////

/**
 * @brief Parses a string representing a list of integers and returns them as a vector.
 *
 * @param str A string in any format in (a,b), [a,b], a,b with or without spaces,
 * with or without commas. representing a list of integers.
 * @return A vector of integers parsed from the string.
 */
std::vector<int> getIntVectorFromString(std::string str) {
  static const std::regex numericTokenPattern("[-+]?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?");
  std::smatch match;
  std::vector<int> numbers;
  while (std::regex_search(str, match, numericTokenPattern)) {
    if (checkInt(match[0])) {
      numbers.push_back(std::stoi(match[0].str()));
    } else {
      throw CommandArgumentException("Error : Invalid integer value in int vector: " + match[0].str());
    }
    str = match.suffix().str(); // Continue with the rest of the string
  }
  return numbers;
}

//////////////////////////////////////////////////////

/**
 * @brief Parses a normalization file to extract statistical data.
 *
 * This function reads a normalization file and extracts statistical information such as attribute indices, wether mean or median was used for normalization
 * mean/median and standard deviations values.
 * It handles files with either numeric indices or attribute names. The function also checks for consistency in the usage of mean or median
 * across the file and detects duplicate indices.
 *
 * @param normalizationFile The path to the normalization file to be parsed.
 * @param nbAttributes The number of attributes expected in the file.
 * @param attributes Optional list of attribute names. If provided, the function will parse the file based on attribute names instead of numeric indices.
 * @return A tuple containing four elements in the following order:
 *         1. A vector of attribute indices (int).
 *         2. A boolean flag indicating whether the file uses 'median' (true) or 'mean' (false).
 *         3. A vector of mean or median values (double) extracted from the file.
 *         4. A vector of standard deviations values (double) extracted from the file.
 * @throws FileContentError If there is a mismatch in the number of attributes, or if the file format is incorrect.
 * @throws FileNotFoundError If the normalization file cannot be opened or found.
 */
std::tuple<std::vector<int>, bool, std::vector<double>, std::vector<double>> parseNormalizationStats(const std::string &normalizationFile, int nbAttributes, const std::vector<std::string> &attributes) {
  std::vector<int> indices_list;
  std::vector<double> mus;
  std::vector<double> sigmas;
  bool withMedian = false;
  std::set<int> unique_indices;

  if (!attributes.empty() && attributes.size() != nbAttributes) {
    throw FileContentError("Error during parsing of " + normalizationFile + ": The number of attributes is not equal to the length of attributes list.");
  }

  // Create some general regex patterns
  // Keep the index regex simple and validate bounds in code (more scalable than "(0|1|...|N)").
  const std::string indexPattern = "(\\d+)";

  std::string attrPattern = "";
  if (!attributes.empty()) {
    attrPattern += "(";
    for (int i = 0; i < nbAttributes; i++) {
      attrPattern += escapeRegexLiteral(attributes[i]);
      if (i < nbAttributes - 1) {
        attrPattern += "|";
      }
    }
    attrPattern += ")";
  }

  std::string floatPattern = "((?:[-+]?(?:\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?|[nN][aA][nN]|[iI][nN][fF](?:[iI][nN][iI][tT][yY])?)))(?=$|[^\\dA-Za-z_])"; // We ask that the float is followed either by the end of the line either by a separator character

  // Create regex patterns for a line
  enum class PatternKind {
    Indices,
    Attributes
  };
  struct PatternSpec {
    const std::regex *regex;
    PatternKind kind;
  };

  std::vector<PatternSpec> patterns;
  std::regex patternIndices("^" + indexPattern + " : original (mean|median): " + floatPattern + ", original std: " + floatPattern);
  std::regex patternAttributes("^" + attrPattern + " : original (mean|median): " + floatPattern + ", original std: " + floatPattern);
  if (!attributes.empty()) {
    patterns.push_back({&patternAttributes, PatternKind::Attributes});
  }
  patterns.push_back({&patternIndices, PatternKind::Indices});

  bool patternError = true;

  for (const auto &pattern : patterns) {
    std::vector<int> currentIndicesList;
    std::vector<double> currentMus;
    std::vector<double> currentSigmas;
    std::set<int> currentUniqueIndices;
    bool currentWithMedian = false;
    bool currentWithMedianInitialized = false;

    std::ifstream file(normalizationFile);
    if (!file) {
      throw FileNotFoundError("Error : file " + std::string(normalizationFile) + " not found or couldn't be opened.");
    }

    patternError = false;
    std::string line;
    while (getline(file, line)) {
      std::string mean_median;
      int index;

      if (line.empty())
        continue;

      std::smatch matches;

      if (std::regex_search(line, matches, *pattern.regex)) {
        mean_median = matches[2];
        currentMus.push_back(stod(matches[3]));
        currentSigmas.push_back(stod(matches[4]));
        if (pattern.kind == PatternKind::Indices) {
          index = stoi(matches[1]);
          if (index < 0 || index >= nbAttributes) {
            throw FileContentError("Error in " + normalizationFile + ": Attribute index " + std::to_string(index) + " out of range [0," + std::to_string(nbAttributes - 1) + "].");
          }
        } else {
          std::string attr = matches[1];
          auto it = std::find(attributes.begin(), attributes.end(), attr);
          if (it == attributes.end()) {
            throw FileContentError("Error in " + normalizationFile + ": Attribute not found.");
          }
          index = static_cast<int>(std::distance(attributes.begin(), it));
        }

      } else {
        patternError = true;
        break;
      }

      currentIndicesList.push_back(index);
      currentUniqueIndices.insert(index);

      if (!currentWithMedianInitialized) {
        currentWithMedian = (mean_median == "median");
        currentWithMedianInitialized = true;
      } else if ((currentWithMedian && mean_median != "median") || (!currentWithMedian && mean_median != "mean")) {
        throw FileContentError("Error in " + normalizationFile + ": Inconsistency in using mean or median.");
      }
    }
    if (!patternError) {
      indices_list = std::move(currentIndicesList);
      mus = std::move(currentMus);
      sigmas = std::move(currentSigmas);
      unique_indices = std::move(currentUniqueIndices);
      withMedian = currentWithMedian;
      break;
    }
  }

  if (patternError) {
    if (attributes.empty()) {
      throw FileContentError("Error in " + normalizationFile + ": File not in the correct format, maybe you forgot to add the attribute file.");
    } else {
      throw FileContentError("Error in " + normalizationFile + ": File not in the correct format.");
    }
  }

  if (indices_list.size() != unique_indices.size()) {
    throw FileContentError("Error in " + normalizationFile + ": Duplicate indices found.");
  }

  return std::make_tuple(indices_list, withMedian, mus, sigmas);
}

//////////////////////////////////////////////////////

/**
 * @brief Checks if a string contains an in-between ASCII space ' ' between words.
 *
 * Tabs are intentionally accepted for backward compatibility.
 *
 * @param str The string to check.
 * @return True if an in-between space is found.
 * @return False otherwise.
 */
bool hasSpaceBetweenWords(const std::string &str) {
  for (size_t i = 0; i < str.length(); ++i) {
    if (str[i] == ' ' &&
        i > 0 &&
        i < str.length() - 1 &&
        str[i - 1] != ' ' &&
        str[i + 1] != ' ') {
      return true; // Space found between words
    }
  }
  return false; // No space found between words or only tabs are present
}

//////////////////////////////////////////////////////

/**
 * @brief Checks if a given file or directory is valid.
 *
 *
 * @param path path of the file or directory to be checked.
 * @return Whether the file or directory exists or not.
 */
bool exists(const std::string &path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

//////////////////////////////////////////////////////

/**
 * @brief Gets the OS's default filesystem separator character.
 *
 * @return The separator as a string.
 */
std::string getOSSeparator() {
// define separator depending on OS
#if defined(__unix__) || defined(__APPLE__)
  return "/";
#elif defined(_WIN32)
  return "/";
#endif
}

//////////////////////////////////////////////////////

/**
 * @brief Prints an option and its description aligned at the specified width.
 *
 * If the option length exceeds a specified width, the description is printed
 * on the next line with indentation matching the option part.
 * Otherwise, the option and description are printed on the same line.
 *
 * @param option The command line option to be printed (e.g., "--train_data_file <str>").
 * @param description The description for the command line option.
 * @param width The fixed width for the start of the description part.
 */
void printOptionDescription(const std::string &option, const std::string &description, int width) {
  if (option.length() >= width) {
    std::cout << option << std::endl;
    std::cout << std::setw(width) << " " << description << std::endl;
  } else {
    std::cout << std::left << std::setw(width) << option << description << std::endl;
    std::cout << std::right; // Reinitialize alignment
  }
}

//////////////////////////////////////////////////////

/**
 * @brief Counts the number of networks in the weights file.
 *
 * @param fileWts Reference to the file stream opened for reading the weight file.
 * @return The number of networks found in the file.
 */
int countNetworksInFile(const std::string &weightsFile) {

  std::filebuf buf;

  if (buf.open(weightsFile, std::ios_base::in) == nullptr) {
    throw CannotOpenFileError("Cannot open weights file " + weightsFile);
  }

  std::istream fileWts(&buf);

  std::string line;
  int count = 0;
  while (std::getline(fileWts, line)) {
    if (line.find("Network") != std::string::npos) {
      ++count;
    }
  }
  return count > 0 ? count : 1; // If no "Network" keyword is found, assume there's one network
}

//////////////////////////////////////////////////////

/**
 * @brief Parses a line from a file and converts it into a vector of double values.
 *
 * It can handle data separated by a space, a comma(CSV), a semicolon(;) or a tab.
 *
 *
 * @param str The string to parse.
 * @param fileName The name of the file from which the string was read. Used for error reporting.
 * @return A vector of double values parsed from the string.
 * @throws FileContentError if a token cannot be converted to a double.
 */
std::vector<double> parseFileLine(std::string str, const std::string &fileName) {

  std::vector<double> valuesData;

  static const std::regex re("([ \\t]+)|[,;]");
  std::sregex_token_iterator first{str.begin(), str.end(), re, -1}; //  -1 makes the regex split, it keeps only what was not matched
  std::sregex_token_iterator last;
  std::vector<std::string> stringTokens{first, last};

  for (const std::string &strToken : stringTokens) {
    try {
      if (!checkStringEmpty(strToken)) {
        valuesData.push_back(std::stod(strToken));
      }
    } catch (const std::invalid_argument &) {
      throw FileContentError("Error : Non number found in file " + fileName + " : " + strToken + ".");
    } catch (const std::out_of_range &) {
      throw FileContentError("Error: Number out of range in file " + fileName + " : " + strToken + ".");
    }
  }

  return valuesData;
}

//////////////////////////////////////////////////////
