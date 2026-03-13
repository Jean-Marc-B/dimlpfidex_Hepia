#include "rule.h"
#include "checkFun.h"
#include "dataSet.h"
#include "errorHandler.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using Json = nlohmann::json;

namespace {
/**
 * @brief Returns true when a path ends with ".json".
 */
bool hasJsonExtension(const std::string &path) {
  const auto dotPos = path.find_last_of('.');
  return dotPos != std::string::npos && path.substr(dotPos + 1) == "json";
}

/**
 * @brief Compares two vectors of doubles with an epsilon tolerance.
 */
bool areDoubleVectorsEqual(const std::vector<double> &left, const std::vector<double> &right, double epsilon) {
  if (left.size() != right.size()) {
    return false;
  }
  for (size_t i = 0; i < left.size(); ++i) {
    if (std::fabs(left[i] - right[i]) > epsilon) {
      return false;
    }
  }
  return true;
}
} // namespace

/**
 * @brief Constructs a Rule object.
 *
 * @param antecedents Vector of antecedents to insert inside a rule.
 * @param coveredSamples Vector of integers containing the covered samples IDs.
 * @param outClass Integer indicating which class is targetted by the rule.
 * @param fidelity Double indicating the fidelity of the rule.
 * @param accuracy Double indicating the accuracy of the rule.
 * @param confidence Double indicating the confidence of the rule.
 */
Rule::Rule(const std::vector<Antecedent> &antecedents, const std::vector<int> &coveredSamples, const std::vector<int> &coveringSizesWithNewAntecedent, int outClass, double fidelity, const std::vector<double> &increasedFidelity, double accuracy, const std::vector<double> &accuracyChanges, double confidence) {
  setAntecedents(antecedents);
  setCoveredSamples(coveredSamples);
  setCoveringSizesWithNewAntecedent(coveringSizesWithNewAntecedent);
  setCoveringSize(static_cast<int>(coveredSamples.size()));
  setOutputClass(outClass);
  setFidelity(fidelity);
  setIncreasedFidelity(increasedFidelity);
  setAccuracy(accuracy);
  setAccuracyChanges(accuracyChanges);
  setConfidence(confidence);
}

/**
 * @brief Builds a string presenting every element and value contained by a given rule.
 *
 * @param attributes Optional vector of strings containing all attributes names, useful to print attribute names instead of integers.
 * @param classes Optional vector of strings containing all class names, useful to print class names instead of an integer.
 * @return A string representation of the rule.
 */
std::string Rule::toString(const std::vector<std::string> &attributes, const std::vector<std::string> &classes) const {
  std::stringstream result;
  int _outputClass = getOutputClass();
  auto nbCoveredSamples = getCoveringSize();
  double _fidelity = getFidelity();
  double _accuracy = getAccuracy();
  double _confidence = getConfidence();
  const auto &_coveringSizesWithNewAntecedent = getCoveringSizesWithNewAntecedent();
  const auto &_increasedFidelity = getIncreasedFidelity();
  const auto &_accuracyChanges = getAccuracyChanges();

  for (const Antecedent &a : getAntecedents()) {
    const int attribute = a.getAttribute();
    if (!attributes.empty()) {
      if (attribute < 0 || static_cast<size_t>(attribute) >= attributes.size()) {
        throw InternalError("Error while formatting rule: antecedent attribute index " + std::to_string(attribute) + " is out of bounds for attributes size " + std::to_string(attributes.size()) + ".");
      }
      result << attributes[attribute];
    } else {
      result << "X" + std::to_string(attribute);
    }

    if (a.getInequality()) {
      result << ">=";
    } else {
      result << "<";
    }

    result << formattingDoubleToString(a.getValue()) << " ";
  }

  if (!classes.empty()) {
    if (_outputClass < 0 || static_cast<size_t>(_outputClass) >= classes.size()) {
      throw InternalError("Error while formatting rule: output class index " + std::to_string(_outputClass) + " is out of bounds for classes size " + std::to_string(classes.size()) + ".");
    }
    result << "-> " << classes[_outputClass] << std::endl;
  } else {
    result << "-> class " << getOutputClass() << std::endl;
  }

  result << "   Train Covering size : " << std::to_string(nbCoveredSamples)
         << std::endl
         << "   Train Fidelity : " << formattingDoubleToString(_fidelity)
         << std::endl
         << "   Train Accuracy : " << formattingDoubleToString(_accuracy)
         << std::endl
         << "   Train Confidence : " << formattingDoubleToString(_confidence)
         << std::endl
         << "   Train Covering size evolution with antecedents : ";
  for (int c : _coveringSizesWithNewAntecedent) {
    result << c << " ";
  }
  result << std::endl
         << "   Train Fidelity increase with antecedents : ";
  for (double f : _increasedFidelity) {
    result << formattingDoubleToString(f) << " ";
  }
  result << std::endl
         << "   Train Accuracy variation with antecedents : ";
  for (double a : _accuracyChanges) {
    result << formattingDoubleToString(a) << " ";
  }

  return result.str();
}

/**
 * @brief Parses a JSON file and returns a vector of the parsed rules.
 *
 * JSON rule format must be like this example:
 * {
 *   "rules": [
 *       {
 *           "antecedents": [
 *               {
 *                   "attribute": 0,
 *                   "inequality": true,
 *                   "value": 0.7231
 *               },
 *               {
 *                   "attribute": 3,
 *                   "inequality": false,
 *                   "value": 0.0031
 *               },
 *               ...
 *           ],
 *           "coveredSamples": [1,2,3, ...],
 *           "outputClass": 0,
 *           "fidelity": 1,
 *           "accuracy": 0.63,
 *           "confidence": 0.8
 *       },
 *       {...}
 *     ]
 *   }
 *
 * @param filename Path of the JSON file to be parsed.
 * @param decisionThreshold Reference to a float where the decision threshold will be stored.
 * @param positiveClassIndex Reference to an int where the positive class index will be stored.
 * @return A vector of parsed rules.
 */
std::vector<Rule> Rule::fromJsonFile(const std::string &filename, float &decisionThreshold, int &positiveClassIndex) {
  std::vector<Rule> result;
  std::ifstream ifs(filename);

  if (!ifs.is_open() || ifs.fail()) {
    throw FileNotFoundError("JSON file to parse named '" + filename + "' was not found or is corrupted, cannot proceed.");
  }

  // this throws an exception if input is not valid JSON
  Json jsonData = Json::parse(ifs);

  if (!jsonData.contains("rules") || !jsonData.contains("positive index class") || !jsonData.contains("threshold")) {
    throw FileContentError("Parsing error in JSON file '" + filename + "': missing required key(s) among 'rules', 'positive index class', and 'threshold'.");
  }

  positiveClassIndex = jsonData["positive index class"];
  decisionThreshold = jsonData["threshold"];
  result = jsonData["rules"];

  return result;
}

/**
 * @brief Writes a vector of rules into a JSON file.
 *
 * @param filename Name of the file to be written.
 * @param rules Vector of rules to be written.
 * @param threshold Decision threshold.
 * @param positiveIndex Positive class index.
 */
void Rule::toJsonFile(const std::string &filename, const std::vector<Rule> &rules, float threshold, int positiveIndex) {
  std::ofstream ofs(filename);

  if (!ofs.is_open() || ofs.fail()) {
    throw FileNotFoundError("JSON file to be written named '" + filename + "' couldn't be opened, cannot proceed.");
  }

  Json jsonData;

  jsonData["threshold"] = threshold;
  jsonData["positive index class"] = positiveIndex;
  jsonData["rules"] = rules;

  ofs << std::setw(4) << jsonData << std::endl;
}

/**
 * @brief Writes vectors of rules with train and test stats into a JSON file.
 *
 * @param filename Name of the file to be written.
 * @param trainRules Vector of rules with train stats rules to be written.
 * @param testRules Vector of rules with test stats to be written.
 * @param threshold Decision threshold.
 * @param positiveIndex Positive class index.
 */
void Rule::toJsonStatsFile(const std::string &filename, const std::vector<Rule> &trainRules, const std::vector<Rule> &testRules, float threshold, int positiveIndex) {
  if (trainRules.size() != testRules.size()) {
    throw InternalError("Error while writing stats JSON: trainRules and testRules must have the same size.");
  }

  std::ofstream ofs(filename);

  if (!ofs.is_open() || ofs.fail()) {
    throw FileNotFoundError("JSON file to be written named '" + filename + "' couldn't be opened, cannot proceed.");
  }

  Json jsonData;

  jsonData["threshold"] = threshold;
  jsonData["positive index class"] = positiveIndex;

  for (size_t i = 0; i < trainRules.size(); ++i) {
    jsonData["rules"].push_back({{"train", trainRules[i]}, {"test", testRules[i]}});
  }

  ofs << std::setw(4) << jsonData << std::endl;
}

/**
 * @brief Writes rules for a given sample into a JSON file. (Ugly workaround for fidexGlo program)
 *
 * @param filename Name of the file to be written.
 * @param sampleId id of the given sample.
 * @param rules Vector of train rules to be written.
 */
void Rule::toJsonGloFile(const std::string &filename, const std::vector<std::vector<Rule>> &rulesPerSamples) {
  std::ofstream ofs(filename);

  if (!ofs.is_open() || ofs.fail()) {
    throw FileNotFoundError("JSON file to be written named '" + filename + "' couldn't be opened, cannot proceed.");
  }

  Json jsonData;

  for (size_t i = 0; i < rulesPerSamples.size(); ++i) {
    jsonData["samples"].push_back({{"sampleId", i}, {"rules", rulesPerSamples[i]}});
  }

  ofs << std::setw(4) << jsonData << std::endl;
}

/**
 * @brief Compares a rule with another to determine whether they're identical.
 *
 * @param other Other rule for comparison.
 * @return Returns true if they're identical, false otherwise.
 */
bool Rule::isEqual(const Rule &other) const {
  double epsilon = 10e-6;

  if (getAntecedents() != other.getAntecedents())
    return false;

  if (getCoveredSamples() != other.getCoveredSamples())
    return false;

  if (getCoveringSizesWithNewAntecedent() != other.getCoveringSizesWithNewAntecedent())
    return false;

  if (getOutputClass() != other.getOutputClass())
    return false;

  if (fabs(getFidelity() - other.getFidelity()) > epsilon)
    return false;

  if (!areDoubleVectorsEqual(getIncreasedFidelity(), other.getIncreasedFidelity(), epsilon))
    return false;

  if (fabs(getAccuracy() - other.getAccuracy()) > epsilon)
    return false;

  if (!areDoubleVectorsEqual(getAccuracyChanges(), other.getAccuracyChanges(), epsilon))
    return false;

  if (fabs(getConfidence() - other.getConfidence()) > epsilon)
    return false;

  return true;
}

/**
 * @brief Generates a regular expression pattern for matching positive integers smaller than the given maximum number.
 *
 * @param maxNumber The maximum number (exclusive and positive) for which the pattern will match smaller integers.
 * @return The generated regular expression pattern as a string.
 */
std::string generateRegexSmallerPositive(int maxNumber) {
  std::ostringstream regexStream;
  std::string maxStr = std::to_string(maxNumber);
  auto nbDigits = static_cast<int>(maxStr.length());

  // Accept numbers with fewer digits
  if (nbDigits > 1) {
    regexStream << "\\d{1," << (nbDigits - 1) << "}|";
  }

  // Accept numbers with the same number of digits but starting with lower digits
  for (int i = 0; i < nbDigits; ++i) {
    if (maxStr[i] != '0') {
      regexStream << maxStr.substr(0, i) << "[0-" << (maxStr[i] - '1') << "]\\d{" << (nbDigits - i - 1) << "}|";
    }
  }

  // Remove the last '|'
  std::string regex = regexStream.str();
  if (!regex.empty() && regex.back() == '|') {
    regex.pop_back();
  }

  return regex;
}

/**
 * @brief Generates a regular expression pattern for matching an antecedent of a rule using the IDs of the attributes.
 *
 * @param nbAttributes The number of attributes that can appear in the rule.
 * @return The compiled regular expression object that can be used to match an antecedent with attribute ids.
 */
std::string getAntStrPatternWithAttrIds(int nbAttributes) {
  std::string pattern = generateRegexSmallerPositive(nbAttributes);
  std::string idPattern("X(" + pattern + ")([<>]=?)(-?(\\d+\\.?\\d*|\\.\\d+)([eE][+-]?\\d+)?)");
  return idPattern;
}

/**
 * @brief Generates a regular expression pattern for matching an antecedent using the names of the attributes.
 *
 * @return The compiled regular expression object that can be used to match an antecedent with attribute names.
 */
std::string getAntStrPatternWithAttrNames() {
  std::string attrPattern = "[^\\s]+";
  std::string attributesPattern("(" + attrPattern + ")([<>]=?)(-?(\\d+\\.?\\d*|\\.\\d+)([eE][+-]?\\d+)?)");
  return attributesPattern;
}

/**
 * @brief Generates a regular expression pattern for matching a class of a rule using the IDs of the classes.
 *
 * @param nbClasses The number of classes that can appear in the rule.
 * @return The compiled regular expression object that can be used to match a rule class id.
 */
std::string getStrPatternWithClassIds(int nbClasses) {
  std::string pattern = generateRegexSmallerPositive(nbClasses);
  std::string idPattern("-> class (" + pattern + "\\b\\s*)");
  return idPattern;
}

/**
 * @brief Generates a regular expression pattern for matching a class of a rule using the names of the classes.
 *
 * @return The compiled regular expression object that can be used to match a rule class name.
 */
std::string getStrPatternWithClassNames() {
  std::string clPattern = "[^\\s]+";
  std::string classesPattern("-> (" + clPattern + ")");
  return classesPattern;
}

/**
 * @brief Checks if a rules file contains rules with attribute names or attribute IDs and with class names or class IDs. A rule needs to start with "Rule".
 *
 * @param rulesFile The rules file to check.
 * @param dataset The dataset containing information about the attributes and classes.
 * @param withClasses Whether to check for class patterns as well.
 * @return Both booleans tell if there are attribute ids or names and class ids or names.
 * @throws FileNotFoundError If the rules file cannot be opened.
 * @throws FileContentError If the rules in the file are not properly formatted.
 */
std::vector<bool> getRulesPatternsFromRuleFile(const std::string &rulesFile, const DataSetFid &dataset, bool withClasses) {
  bool hasAttrIds = true;
  bool hasAttrNames = true;
  bool hasClassIds = true;
  bool hasClassNames = true;

  bool foundARule = false;

  std::ifstream fileDta(rulesFile);
  if (!fileDta) {
    throw FileNotFoundError("Error : file " + rulesFile + " not found");
  }

  std::string line;

  std::regex antecedentsPatternIds(": (" + getAntStrPatternWithAttrIds(dataset.getNbAttributes()) + " )*" + "\\s?->");
  std::regex antecedentsPatternNames;
  if (dataset.getHasAttributeNames()) {
    antecedentsPatternNames = ": (" + getAntStrPatternWithAttrNames() + " )*" + "\\s?->";
  }
  auto patternWithClassIds = std::regex(getStrPatternWithClassIds(dataset.getNbClasses()));
  std::regex patternWithClassNames;
  if (dataset.getHasClassNames()) {
    patternWithClassNames = std::regex(getStrPatternWithClassNames());
  }

  while (getline(fileDta, line)) {
    if (line.find("Rule") == 0) { // If line begins with "Rule"
      foundARule = true;

      bool matched = false;
      if (regex_search(line, antecedentsPatternIds)) {
        matched = true;
        hasAttrIds &= true; // Stays true only if it was already true, can't become true if it's false
      } else {
        hasAttrIds = false;
      }
      if (dataset.getHasAttributeNames() && regex_search(line, antecedentsPatternNames)) {
        matched = true;
        hasAttrNames &= true;
      } else {
        hasAttrNames = false;
      }
      if (!matched) {
        throw FileContentError("Error : in file " + rulesFile + ", the rule " + line + " is not in a good format. Maybe an attribute or class id is wrong or you forgot to add the attribute file.");
      }
      if (withClasses) {
        matched = false;
        if (regex_search(line, patternWithClassIds)) {
          matched = true;
          hasClassIds &= true;
        } else {
          hasClassIds = false;
        }
        if (dataset.getHasClassNames() && regex_search(line, patternWithClassNames)) {
          matched = true;
          hasClassNames &= true;
        } else {
          hasClassNames = false;
        }
        if (!matched) {
          throw FileContentError("Error : in file " + rulesFile + ", the rule " + line + " is not in a good format. Maybe a class id is wrong or you forgot to add the attribute file.");
        }
      }

      // If no pattern matches each rule
      if (!hasAttrIds && !hasAttrNames) {
        throw FileContentError("Error : in file " + rulesFile + ", the rules are not always using the same convention for attributes, or a given attribute name or attribute id is wrong.");
      }
      if (!hasClassIds && !hasClassNames && withClasses) {
        throw FileContentError("Error : in file " + rulesFile + ", the rules are not always using the same convention for classes, or a given class name or class id is wrong.");
      }
    }
  }

  // If there is no rule in the file
  if (!foundARule) {
    throw FileContentError("Error : there is no rule in the file " + rulesFile + ". Note: a rule needs to start with 'Rule' keyword");
  }

  return std::vector<bool>{hasAttrIds, hasClassIds};
}

/**
 * @brief Converts a well-formatted rule string to a Rule object.
 *
 * @param rule The generated rule.
 * @param str The string rule to convert.
 * @param attributePattern The regex pattern to match attribute names or ids.
 * @param classPattern The regex pattern to match class names or ids.
 * @param withAttributeNames Whether the rule contains attribute names.
 * @param withClassNames Whether the rule contains class names.
 * @param dataset The dataset containing information about the attributes and classes.
 * @return Returns true if the rule is created, false otherwise.

 */
bool stringToRule(Rule &rule, const std::string &str, const std::regex &attributePattern, const std::regex &classPattern, bool withAttributeNames, bool withClassNames, const DataSetFid &dataset) {

  std::vector<Antecedent> antecedents;
  bool isRule = false;

  std::istringstream iss(str);
  std::string token;

  while (iss >> token) {
    std::smatch match;
    if (regex_match(token, match, attributePattern)) {
      isRule = true;
      Antecedent antecedent;
      if (!withAttributeNames) {
        antecedent.setAttribute(stoi(match[1]));
      } else {
        auto it = find(dataset.getAttributeNames().begin(), dataset.getAttributeNames().end(), match[1]);
        if (it != dataset.getAttributeNames().end()) {
          antecedent.setAttribute(static_cast<int>(distance(dataset.getAttributeNames().begin(), it)));
        } else {
          throw FileContentError("Error : in rulefile, the rule " + str + " contains unknown named attribute " + std::string(match[1]) + ". If the attributes in the rules are not named, do not specify an attribute file.");
        }
      }
      if (match[2] == ">=") {
        antecedent.setInequality(true);
      } else {
        antecedent.setInequality(false);
      }
      antecedent.setValue(stod(match[3]));
      antecedents.push_back(antecedent);
    } else if (token == "->") {
      std::string classString = token;
      iss >> token;
      classString += " " + token;
      if (!withClassNames) {
        iss >> token;
        classString += " " + token;
      }
      if (regex_match(classString, match, classPattern)) {
        isRule = true;
        if (!withClassNames) {
          rule.setOutputClass(stoi(match[1]));
        } else {
          auto it = find(dataset.getClassNames().begin(), dataset.getClassNames().end(), match[1]);
          if (it != dataset.getClassNames().end()) {
            rule.setOutputClass(static_cast<int>(distance(dataset.getClassNames().begin(), it)));
          } else {
            throw FileContentError("Error : in rulefile, the rule " + str + " contains unknown named class " + std::string(match[1]) + ". If the classes in the rules are not named, do not specify an attribute file.");
          }
        }
      }
    }
  }

  if (isRule) {
    rule.setAntecedents(antecedents);
    return true;
  }

  return false;
}

/**
 * @brief Get the rules from a rules file.
 *
 * @param rules Vector to store the rules obtained from the file.
 * @param rulesFile Path to the rules file.
 * @param dataset The dataset containing information about the attributes and classes.
 * @param decisionThreshold Reference to a float where the decision threshold will be stored.
 * @param positiveClassIndex Reference to an int where the positive class index will be stored.
 * @throws FileNotFoundError If the rules file cannot be opened.
 * @throws FileContentError If the rules in the file are not properly formatted.
 */
void getRules(std::vector<Rule> &rules, const std::string &rulesFile, const DataSetFid &dataset, float &decisionThreshold, int &positiveClassIndex) {
  // if file is JSON read it properly
  if (hasJsonExtension(rulesFile)) {
    rules = Rule::fromJsonFile(rulesFile, decisionThreshold, positiveClassIndex);
  } else {
    // Open rules file
    std::ifstream rulesData;
    rulesData.open(rulesFile, std::ios::in); // Read data file
    if (rulesData.fail()) {
      throw FileNotFoundError("Error : file " + rulesFile + " not found.");
    }

    // Check if the file has attribute names or ids
    std::vector<bool> checkPatterns = getRulesPatternsFromRuleFile(rulesFile, dataset);
    bool attributeIdsInFile = checkPatterns[0];
    bool classeIdsInFile = checkPatterns[1];

    std::regex attributePattern;
    std::regex classPattern;
    if (attributeIdsInFile) {
      attributePattern = getAntStrPatternWithAttrIds(dataset.getNbAttributes());
    } else {
      attributePattern = getAntStrPatternWithAttrNames();
    }

    if (classeIdsInFile) {
      classPattern = getStrPatternWithClassIds(dataset.getNbClasses());
    } else {
      classPattern = getStrPatternWithClassNames();
    }

    std::string line;
    while (getline(rulesData, line)) {
      Rule rule;
      bool isRule = stringToRule(rule, line, attributePattern, classPattern, !attributeIdsInFile, !classeIdsInFile, dataset);

      if (isRule) {
        const std::string ruleHeader = line;
        auto readTokensOrThrow = [&](const std::string &fieldName, size_t minTokens, const std::string &invalidLinePrefix) -> std::vector<std::string> {
          if (!std::getline(rulesData, line)) {
            throw FileFormatError("Error : In file " + rulesFile + ", missing line for " + fieldName + " after rule: " + ruleHeader + ".");
          }
          auto tokens = splitString(line, " ");
          if (tokens.size() < minTokens) {
            throw FileFormatError("Error : In file " + rulesFile + ", " + invalidLinePrefix + line + ".");
          }
          return tokens;
        };

        auto elements = readTokensOrThrow("covering size", 5, "invalid covering size line: ");
        rule.setCoveringSize(std::stoi(elements[4]));
        elements = readTokensOrThrow("fidelity", 4, "invalid fidelity line: ");
        rule.setFidelity(std::stod(elements[3]));
        elements = readTokensOrThrow("accuracy", 4, "invalid accuracy line: ");
        rule.setAccuracy(std::stod(elements[3]));
        elements = readTokensOrThrow("confidence", 4, "invalid confidence line: ");
        rule.setConfidence(std::stod(elements[3]));

        elements = readTokensOrThrow("covering size evolution", 8, "missing covering size for new antecedents in the line ");
        std::vector<int> covSizeAnt;
        for (size_t i = 7; i < elements.size(); i++) {
          covSizeAnt.push_back(std::stoi(elements[i]));
        }
        rule.setCoveringSizesWithNewAntecedent(covSizeAnt);
        elements = readTokensOrThrow("fidelity increase", 7, "missing increased fidelity for new antecedents in the line ");
        std::vector<double> fidIncr;
        for (size_t i = 6; i < elements.size(); i++) {
          fidIncr.push_back(std::stod(elements[i]));
        }
        rule.setIncreasedFidelity(fidIncr);
        elements = readTokensOrThrow("accuracy change", 7, "missing accuracy change for new antecedents in the line ");
        std::vector<double> accChange;
        for (size_t i = 6; i < elements.size(); i++) {
          accChange.push_back(std::stod(elements[i]));
        }
        rule.setAccuracyChanges(accChange);
        rules.push_back(rule);
      }
    }
  }
}

/**
 * @brief Writes a list of rules into a given file. Returns a tuple of two doubles representing the mean covering size and the mean number of antecedents.
 *
 * @param filename Name of the file to be written/overwritten.
 * @param rules List of Rule objects to be written in "filename".
 * @param attributeNames List of attribute names, used to write Rule's antecedents with attribute names instead of a "X" variable.
 * @param classNames List of class names, used to write Rule's class with class names instead of numerical representation.
 * @param threshold Decision threshold.
 * @param positiveIndex Index defining the positive class in the dataset.
 * @return A tuple containing the mean covering size and the mean number of antecedents.
 */
std::tuple<double, double> writeRulesFile(const std::string &filename, const std::vector<Rule> &rules, const std::vector<std::string> &attributeNames,
                                          const std::vector<std::string> &classNames, float threshold, int positiveIndex) {
  if (rules.empty()) {
    std::cout << "Warning: cannot write to file \"" << filename << "\", generated rules list is empty." << std::endl;
    return std::make_tuple(0, 0);
  }

  double meanCovSize = 0;
  double meanNbAntecedents = 0;
  const auto nbRules = static_cast<int>(rules.size());

  if (hasJsonExtension(filename)) {
    for (const Rule &r : rules) { // each rule
      meanCovSize += static_cast<double>(r.getCoveringSize());
      meanNbAntecedents += static_cast<double>(r.getAntecedents().size());
    }
    meanCovSize /= nbRules;
    meanNbAntecedents /= nbRules;

    Rule::toJsonFile(filename, rules, threshold, positiveIndex);

  } else {
    int counter = 0;
    std::stringstream stream;
    std::ofstream file(filename);

    for (const Rule &r : rules) { // each rule
      meanCovSize += static_cast<double>(r.getCoveringSize());
      meanNbAntecedents += static_cast<double>(r.getAntecedents().size());
      counter++;
      stream << "Rule " << counter << ": " << r.toString(attributeNames, classNames);
      stream << std::endl;
    }

    meanCovSize /= nbRules;
    meanNbAntecedents /= nbRules;

    if (file.is_open()) {
      file << "Number of rules : " << nbRules
           << ", mean sample covering number per rule : " << formattingDoubleToString(meanCovSize)
           << ", mean number of antecedents per rule : " << formattingDoubleToString(meanNbAntecedents)
           << std::endl;
      if (threshold != -1) {
        file << "Using a decision threshold of " << threshold << " for class " << positiveIndex
             << std::endl;
      } else {
        file << "No decision threshold is used.";
      }
      file << std::endl
           << std::endl
           << stream.str();

      file.close();

    } else {
      throw CannotOpenFileError("Error : Couldn't open rules extraction file \"" + filename + "\".");
    }
  }
  return std::make_tuple(meanCovSize, meanNbAntecedents);
}

/**
 * @brief Get the indices of the rules activated by a test sample.
 *
 * @param activatedRules Vector to store indices of activated rules.
 * @param rules Vector of rules to check.
 * @param testValues Values of the test sample for which we search activated rules.
 */
void getActivatedRules(std::vector<int> &activatedRules, const std::vector<Rule> &rules, const std::vector<double> &testValues) {
  activatedRules.clear();

  int attr;
  bool ineq;
  double val;
  for (size_t r = 0; r < rules.size(); ++r) { // For each rule
    bool notActivated = false;
    for (const auto &antecedent : rules[r].getAntecedents()) { // For each antecedent
      attr = antecedent.getAttribute();
      if (attr < 0 || static_cast<size_t>(attr) >= testValues.size()) {
        throw InternalError("Error while checking activated rules: antecedent attribute index " + std::to_string(attr) + " is out of bounds for test sample size " + std::to_string(testValues.size()) + ".");
      }
      ineq = antecedent.getInequality();
      val = antecedent.getValue();
      if (ineq == 0 && testValues[attr] >= val) { // If the inequality is not verified
        notActivated = true;
        break;
      }
      if (ineq == 1 && testValues[attr] < val) {
        notActivated = true;
        break;
      }
    }
    if (!notActivated) {
      activatedRules.push_back(static_cast<int>(r));
    }
  }
}

/**
 * @brief Extract the decision threshold and the index of the positive class from a rules file.
 *
 * @param filePath The path to the file containing the rules.
 * @param decisionThreshold Reference to a float variable where the decision threshold will be stored.
 *                          Set to -1.0 if not found in the file.
 * @param positiveClassIndex Reference to an int variable where the index of the positive class will be stored.
 *                           Set to -1 if not found in the file.
 * @throws FileNotFoundError If the file specified by filePath cannot be opened.
 */
void getThresholdFromRulesFile(const std::string &filePath, float &decisionThreshold, int &positiveClassIndex) {
  decisionThreshold = -1.0;
  positiveClassIndex = -1;

  std::ifstream file(filePath);
  std::string line;

  if (hasJsonExtension(filePath)) {
    Rule::fromJsonFile(filePath, decisionThreshold, positiveClassIndex);
  } else {

    if (!file) {
      throw FileNotFoundError("Error : file " + filePath + " not found");
    }

    while (std::getline(file, line)) {
      std::string tokenThresh;
      std::string tokenClass;

      if (line.find("Using a decision threshold of") != std::string::npos) {
        std::istringstream iss(line);
        iss >> tokenThresh >> tokenThresh >> tokenThresh >> tokenThresh >> tokenThresh >> tokenThresh >> tokenClass >> tokenClass >> tokenClass; // Get decision threshold
        decisionThreshold = std::stof(tokenThresh);
        positiveClassIndex = std::stoi(tokenClass);
      }
    }

    file.close();
  }
}
