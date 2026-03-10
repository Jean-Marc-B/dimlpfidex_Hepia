#include "fidexGloStatsFct.h"

#include "../../../common/cpp/src/checkFun.h"
#include "../../../common/cpp/src/dataSet.h"
#include "../../../common/cpp/src/errorHandler.h"
#include "../../../common/cpp/src/parameters.h"
#include "../../../common/cpp/src/rule.h"
#include "../../../common/cpp/src/scopedCoutFileRedirect.h"

#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace {
// ============================================================================
// Local helpers used only by fidexGloStats.cpp
// ============================================================================

struct RuleTestStats {
  std::vector<int> coveredSamples;
  std::vector<int> coverSizes;
  std::vector<double> fidelities; // Percentage of correct covered samples predictions with respect to the rule prediction
  std::vector<double> accuracies; // Percentage of correct covered samples predictions with respect to the samples true class
  std::vector<double> fidelityIncreases;
  std::vector<double> accuracyChanges;
  double finalConfidence = 0.0; // Mean output prediction score of covered samples
};

bool hasJsonExtension(const std::string &path) {
  const size_t dotPos = path.find_last_of('.');
  return dotPos != std::string::npos && path.substr(dotPos + 1) == "json";
}

// Validate the rule output class against the configured number of classes.
void validateRulePredictionIndex(int rulePred, int nbClasses, size_t ruleIndex) {
  if (rulePred < 0 || rulePred >= nbClasses) {
    throw FileContentError("Error : rule " + std::to_string(ruleIndex + 1) + " has output class " + std::to_string(rulePred) +
                           " which is out of valid range [0, " + std::to_string(nbClasses - 1) + "].");
  }
}

// Compute test-time rule statistics for each antecedent step and final rule.
RuleTestStats computeRuleTestStats(const Rule &fullRule,
                                   const std::vector<std::vector<double>> &testData,
                                   const std::vector<int> &testPreds,
                                   const std::vector<int> &testTrueClasses,
                                   const std::vector<std::vector<double>> &testPredictionScores,
                                   size_t ruleIndex) {
  RuleTestStats stats;

  const std::vector<Antecedent> &fullAntecedents = fullRule.getAntecedents();
  const int rulePred = fullRule.getOutputClass();
  const size_t nbSteps = fullAntecedents.empty() ? 1 : fullAntecedents.size();

  stats.coverSizes.reserve(nbSteps);
  stats.fidelities.reserve(nbSteps);
  stats.accuracies.reserve(nbSteps);

  for (size_t step = 1; step <= nbSteps; ++step) {
    Rule partialRule;
    partialRule.setOutputClass(rulePred);

    if (!fullAntecedents.empty()) {
      std::vector<Antecedent> partialAntecedents(fullAntecedents.begin(), fullAntecedents.begin() + step);
      partialRule.setAntecedents(partialAntecedents);
    }

    std::vector<int> stepCoveredSamples;
    getCovering(stepCoveredSamples, partialRule, testData); // contains every samples if there is no antecedents in the rule

    const int coverSize = static_cast<int>(stepCoveredSamples.size());
    double fidelity = 0.0;
    double accuracy = 0.0;
    double confidenceSum = 0.0;

    for (int sampleId : stepCoveredSamples) {
      const int samplePred = testPreds[sampleId];
      const int trueClass = testTrueClasses[sampleId];
      const std::vector<double> &scores = testPredictionScores[sampleId];

      if (rulePred < 0 || static_cast<size_t>(rulePred) >= scores.size()) {
        throw FileContentError("Error : rule " + std::to_string(ruleIndex + 1) + " has output class " + std::to_string(rulePred) +
                               " but prediction scores for sample " + std::to_string(sampleId) + " have size " + std::to_string(scores.size()) + ".");
      }

      const double outputScore = scores[rulePred];

      if (samplePred == rulePred) {
        fidelity += 1.0;
      }
      if (rulePred == trueClass) {
        accuracy += 1.0;
      }

      if (step == nbSteps) {
        confidenceSum += outputScore;
      }
    }

    if (coverSize > 0) {
      fidelity /= static_cast<double>(coverSize);
      accuracy /= static_cast<double>(coverSize);
      if (step == nbSteps) {
        stats.finalConfidence = confidenceSum / static_cast<double>(coverSize);
      }
    }

    stats.fidelities.push_back(fidelity);
    stats.accuracies.push_back(accuracy);
    stats.coverSizes.push_back(coverSize);

    if (step == nbSteps) {
      stats.coveredSamples = std::move(stepCoveredSamples);
    }
  }

  stats.fidelityIncreases.reserve(stats.fidelities.size());
  stats.accuracyChanges.reserve(stats.accuracies.size());
  if (!stats.fidelities.empty()) {
    stats.fidelityIncreases.push_back(stats.fidelities[0]);
    stats.accuracyChanges.push_back(stats.accuracies[0]);

    for (size_t i = 1; i < stats.fidelities.size(); ++i) {
      stats.fidelityIncreases.push_back(stats.fidelities[i] - stats.fidelities[i - 1]);
      stats.accuracyChanges.push_back(stats.accuracies[i] - stats.accuracies[i - 1]);
    }
  }

  return stats;
}

} // namespace

/**
 * @brief Displays the parameters for fidexGloStats.
 */
void showStatsParams() {
  std::cout << std::endl
            << "---------------------------------------------------------------------" << std::endl
            << std::endl;
  std::cout << "Warning! The files are located with respect to the root folder dimlpfidex." << std::endl;
  std::cout << "The arguments can be specified in the command or in a json configuration file with --json_config_file your_config_file.json." << std::endl
            << std::endl;

  std::cout << "----------------------------" << std::endl
            << std::endl;
  std::cout << "Required parameters:" << std::endl
            << std::endl;

  printOptionDescription("--test_data_file <str>", "Path to the file containing the test portion of the dataset");
  printOptionDescription("--test_class_file <str>", "Path to the file containing the test true classes of the dataset, not mandatory if classes are specified in test data file");
  printOptionDescription("--test_pred_file <str>", "Path to the file containing predictions on the test portion of the dataset");
  printOptionDescription("--global_rules_file <str>", "Path to the file containing the global rules obtained with fidexGloRules algorithm.");
  printOptionDescription("--nb_attributes <int [1,inf[>", "Number of attributes in the dataset");
  printOptionDescription("--nb_classes <int [2,inf[>", "Number of classes in the dataset");

  std::cout << std::endl
            << "----------------------------" << std::endl
            << std::endl;
  std::cout << "Optional parameters: " << std::endl
            << std::endl;

  printOptionDescription("-h --help", "Show this help message and exit");
  printOptionDescription("--json_config_file <str>", "Path to the JSON file that configures all parameters. If used, this must be the sole argument and must specify the file's relative path");
  printOptionDescription("--root_folder <str>", "Path to the folder, based on main default folder dimlpfidex, containing all used files and where generated files will be saved. If a file name is specified with another option, its path will be relative to this root folder");
  printOptionDescription("--global_rules_outfile <str>", "Path to the file where the output global rules will be stored with stats on test set, if you want to compute those statistics. If a .json extension is given, rules are saved in JSON format.");
  printOptionDescription("--attributes_file <str>", "Path to the file containing the labels of attributes and classes> Mandatory if rules file contains attribute names, if not, do not add it");
  printOptionDescription("--stats_file <str>", "Path to the file where statistics of the global ruleset will be stored");
  printOptionDescription("--console_file <str>", "Path to the file where the terminal output will be redirected. If not specified, all output will be shown on your terminal");
  printOptionDescription("--positive_class_index <int [0,nb_classes-1]>", "Index of the positive class to compute true/false positive/negative rates, index starts at 0. If it is specified in the rules file, it has to be the same value.");

  std::cout << std::endl
            << "----------------------------" << std::endl
            << std::endl;
  std::cout << "Execution example :" << std::endl
            << std::endl;
  std::cout << "fidex.fidexGloStats(\"--test_data_file test_data.txt --test_pred_file predTest.out --test_class_file test_class.txt --nb_attributes 16 --nb_classes 2 --stats_file stats.txt --global_rules_file globalRules.rls --root_folder dimlp/datafiles\")" << std::endl
            << std::endl;
  std::cout << "---------------------------------------------------------------------" << std::endl
            << std::endl;
}

/**
 * @brief Determines which samples are covered by a given rule.
 *
 * @param sampleIds Vector to store the IDs of the samples covered by the rule.
 * @param rule The rule used to determine coverage.
 * @param testValues Matrix of test sample values.
 */
void getCovering(std::vector<int> &sampleIds, const Rule &rule, const std::vector<std::vector<double>> &testValues) {
  // Get covering index samples
  sampleIds.clear();
  int attr;
  bool ineq;
  double val;
  for (size_t id = 0; id < testValues.size(); ++id) {
    bool notCovered = false;
    for (const auto &antecedent : rule.getAntecedents()) { // For each antecedent
      attr = antecedent.getAttribute();
      ineq = antecedent.getInequality();
      val = antecedent.getValue();
      if ((ineq == 0 && testValues[id][attr] >= val) || // If the inequality is not verified
          (ineq == 1 && testValues[id][attr] < val)) {
        notCovered = true;
        break; // One failed antecedent is enough to reject this sample.
      }
    }
    if (!notCovered) {
      sampleIds.push_back(static_cast<int>(id));
    }
  }
}

/**
 * @brief Computes the number of true positives, false positives, true negatives, and false negatives based on the model's or rules's decision and the true class.
 *
 * @param decision The predicted class by the model or by the rules.
 * @param positiveClassIndex The index of the positive class.
 * @param testTrueClass The true class of the test sample.
 * @param nbTruePositive Counter for true positives.
 * @param nbFalsePositive Counter for false positives.
 * @param nbTrueNegative Counter for true negatives.
 * @param nbFalseNegative Counter for false negatives.
 */
void computeTFPN(int decision, int positiveClassIndex, int testTrueClass, int &nbTruePositive, int &nbFalsePositive, int &nbTrueNegative, int &nbFalseNegative) {
  if (decision == positiveClassIndex) { // Positive prediction
    if (decision == testTrueClass) {
      nbTruePositive += 1;
    } else {
      nbFalsePositive += 1;
    }
  } else { // Negative prediction
    if (testTrueClass == positiveClassIndex) {
      nbFalseNegative += 1;
    } else {
      nbTrueNegative += 1;
    }
  }
}

/**
 * @brief Sets default hyperparameters and checks the logic and validity of the parameters of fidexGloStats.
 *
 * @param p Reference to the Parameters object containing all hyperparameters.
 */
void checkStatsParametersLogicValues(Parameters &p) {
  // setting default values
  p.setDefaultInt(POSITIVE_CLASS_INDEX, -1);

  // this sections check if values comply with program logic

  // asserting mandatory parameters
  p.assertStringExists(TEST_DATA_FILE);
  p.assertStringExists(TEST_PRED_FILE);
  p.assertStringExists(GLOBAL_RULES_FILE);
  p.assertIntExists(NB_ATTRIBUTES);
  p.assertIntExists(NB_CLASSES);

  // verifying logic between parameters, values range and so on...
  p.checkAttributeAndClassCounts();
  const int positiveClassIndex = p.getInt(POSITIVE_CLASS_INDEX);
  const int nbClasses = p.getInt(NB_CLASSES);

  if (positiveClassIndex < -1) {
    throw CommandArgumentException("Error : Positive class index must be positive (>=0)");
  }

  if (positiveClassIndex >= nbClasses) {
    throw CommandArgumentException("Error : The index of positive class cannot be greater or equal to the number of classes (" + std::to_string(nbClasses) + ").");
  }
}

/**
 * @brief Computes the statistics of the global ruleset obtained from fidexGloRules on a test dataset.
 *
 * The statistics computed for the ruleset are:
 * - The global rule fidelity rate.
 * - The global rule accuracy.
 * - The explainability rate (when we can find one or more rules, either correct ones or activated ones which all agree on the same class).
 * - The default rule rate (when we can't find any rule activated for a sample).
 * - The mean number of correct (fidel) activated rules per sample.
 * - The mean number of wrong (not fidel) activated rules per sample.
 * - The model test accuracy.
 * - The model test accuracy when rules and model agree.
 * - The model test accuracy when activated rules and model agree.
 *
 * If there is a positive class, additional statistics are computed with both the model decision and the rules decision (ruleset or Fidex rule if ruleset cannot decide):
 * - The number of true positive test samples.
 * - The number of false positive test samples.
 * - The number of true negative test samples.
 * - The number of false negative test samples.
 * - The false positive rate.
 * - The false negative rate.
 * - The precision.
 * - The recall.
 *
 * Notes:
 * - Each file is located with respect to the root folder dimlpfidex or to the content of the 'root_folder' parameter if specified.
 * - It's mandatory to specify the number of attributes and classes in the data, as well as the test dataset.
 * - True test class labels must be provided, either within the data file or separately through a class file.
 * - Test predictions are also mandatory.
 * - The path of the file containing the global ruleset must be provided.
 * - The path of the global rules output file must be provided to compute statistics of the rules on the test set.
 * - If the positive class index is specified, the true/false positive/negative rates are computed.
 * - Parameters can be defined directly via the command line or through a JSON configuration file.
 * - Providing no command-line arguments or using <tt>-h/-\-help</tt> displays usage instructions, detailing both required and optional parameters for user guidance.
 *
 * Outputs:
 * - stats_file : If specified, contains the statistics of the global ruleset seen above.
 * - global_rules_outfile : If specified, edits the global ruleset file by adding the statistics of global rules on test set.
 * - console_file : If specified, contains the console output.
 *
 * File formats:
 * - **Data files**: These files should contain one sample per line, with numbers separated either by spaces, tabs, semicolons, or commas. Supported formats:
 *   1. Only attributes (floats).
 *   2. Attributes (floats) followed by an integer class ID.
 *   3. Attributes (floats) followed by one-hot encoded class.
 * - **Class files**: These files should contain one class sample per line, with integers separated either by spaces, tabs, semicolons, or commas. Supported formats:
 *   1. Integer class ID.
 *   2. One-hot encoded class.
 * - **Prediction files**: These files should contain one line per data sample, each line consisting of a series of numerical values separated
 *   by a space, a comma (CSV), a semicolon (;), or a tab representing the prediction scores for each class.
 * - **Global rule file**: This file is generated by fidexGloRules. The first line contains general statistics in the form:<br>
 *   'Number of rules : 1171, mean sample covering number per rule : 236.923997, mean number of antecedents per rule : 13.020495'<br>
 *   The second line indicates if a decision threshold has been used. If no, it says: 'No decision threshold is used.'
 *   and if yes, it says something like 'Using a decision threshold of 0.3 for class 0'.
 *   Then there is an empty line and each rule is numbered starting from 1 and separated from each other by an empty line. A rule is in the form:<br>
 *   %Rule 1: X2531>=175.95 X2200>=181.05 X1828>=175.95 X2590>=178.5 X1257>=183.6 X2277>=170.85 X1816>=173.4 X3040>=183.6 -> class 0<br>
 *   Train Covering size : 127<br>
 *   Train Fidelity : 1<br>
 *   Train Accuracy : 1<br>
 *   Train Confidence : 0.999919<br>
 * - **Attributes file**: Each line corresponds to one attribute, each attribute must be specified. Classes can be specified
 *   after the attributes but are not mandatory. Each attribute or class must be in one word without spaces (you can use _ to replace a space).
 *   The order is important as the first attribute/class name will represent the first attribute/class in the dataset.
 *
 * Example of how to call the function:
 * @par
 * <tt>from dimlpfidex import fidex</tt>
 * @par
 * <tt>fidex.fidexGloStats('-\-test_data_file test_data.txt -\-test_pred_file predTest.out -\-test_class_file test_class.txt -\-nb_attributes 16 -\-nb_classes 2 -\-stats_file stats.txt -\-global_rules_file globalRules.rls -\-root_folder dimlp/datafiles')</tt>
 *
 * @param command A single string containing either the path to a JSON configuration file with all specified arguments, or all arguments for the function formatted like command-line input. This includes file paths and options for output.
 * @return Returns 0 for successful execution, -1 for errors encountered during the process.
 */
int fidexGloStats(const std::string &command) {
  try {

    // =========================================================================
    // 1) Initialization and command parsing
    // =========================================================================

    float temps;
    clock_t t1;
    clock_t t2;

    t1 = clock();

    // Parsing the command
    std::vector<std::string> commandList = {"fidexGloStats"};
    std::string s;
    std::stringstream ss(command);

    while (ss >> s) {
      commandList.push_back(s);
    }
    std::size_t nbParam = commandList.size();
    if (nbParam < 2 || commandList[1] == "-h" || commandList[1] == "--help") {
      showStatsParams();
      return 0;
    }

    // =========================================================================
    // 2) Parameter loading and validation
    // =========================================================================

    // Import parameters
    std::unique_ptr<Parameters> params;
    std::vector<ParameterCode> validParams = {TEST_DATA_FILE, TEST_PRED_FILE, TEST_CLASS_FILE,
                                              GLOBAL_RULES_FILE, GLOBAL_RULES_OUTFILE, NB_ATTRIBUTES, NB_CLASSES, ROOT_FOLDER, ATTRIBUTES_FILE,
                                              STATS_FILE, CONSOLE_FILE, POSITIVE_CLASS_INDEX};
    if (commandList[1].compare("--json_config_file") == 0) {
      if (commandList.size() < 3) {
        throw CommandArgumentException("JSON config file name/path is missing");
      } else if (commandList.size() > 3) {
        throw CommandArgumentException("Option " + commandList[1] + " has to be the only option in the command if specified.");
      }
      try {
        params = std::unique_ptr<Parameters>(new Parameters(commandList[2], validParams));
      } catch (const std::out_of_range &) {
        throw CommandArgumentException("Some value inside your JSON config file '" + commandList[2] + "' is out of range.\n(Probably due to a too large or too tiny numeric value).");
      } catch (const std::exception &e) {
        std::string msg(e.what());
        throw CommandArgumentException("Unknown JSON config file error: " + msg);
      }
    } else {
      // Read parameters from CLI
      params = std::unique_ptr<Parameters>(new Parameters(commandList, validParams));
    }

    // getting all program arguments from CLI
    checkStatsParametersLogicValues(*params);

    // =========================================================================
    // 3) Optional console redirection and runtime parameter resolution
    // =========================================================================

    std::unique_ptr<ScopedCoutFileRedirect> coutRedirect;
    if (params->isStringSet(CONSOLE_FILE)) {
      coutRedirect.reset(new ScopedCoutFileRedirect(params->getString(CONSOLE_FILE)));
    }

    // Show chosen parameters
    std::cout << *params;

    // Get parameters values

    int nbAttributes = params->getInt(NB_ATTRIBUTES);
    int nbClasses = params->getInt(NB_CLASSES);
    std::string testDataFile = params->getString(TEST_DATA_FILE);
    std::string testDataFilePred = params->getString(TEST_PRED_FILE);
    std::string rulesFile = params->getString(GLOBAL_RULES_FILE);

    // Get decision threshold and positive class index
    float decisionThreshold;
    int positiveClassIndex;
    getThresholdFromRulesFile(rulesFile, decisionThreshold, positiveClassIndex);

    int positiveClassIndexParam = params->getInt(POSITIVE_CLASS_INDEX);
    if (positiveClassIndexParam != -1) {
      if (positiveClassIndex != -1 && positiveClassIndex != positiveClassIndexParam) {
        throw CommandArgumentException("The positive class index must be the same as the one specified in the rules file (" + std::to_string(positiveClassIndex) + ").");
      }
      positiveClassIndex = positiveClassIndexParam;
    }
    const bool hasPositiveClass = positiveClassIndex != -1;

    // =========================================================================
    // 4) Input data loading (test set, predictions, optional attributes)
    // =========================================================================

    std::cout << "Importing files..." << std::endl
              << std::endl;

    // Get test data

    std::unique_ptr<DataSetFid> testDatas;
    if (!params->isStringSet(TEST_CLASS_FILE)) {
      testDatas.reset(new DataSetFid("testDatas from FidexGloStats", testDataFile, testDataFilePred, nbAttributes, nbClasses, decisionThreshold, positiveClassIndex));
      if (!testDatas->getHasClasses()) {
        throw CommandArgumentException("The test true classes file has to be given with option --test_class_file or classes have to be given in the test data file.");
      }
    } else {
      testDatas.reset(new DataSetFid("testDatas from FidexGloStats", testDataFile, testDataFilePred, nbAttributes, nbClasses, decisionThreshold, positiveClassIndex, params->getString(TEST_CLASS_FILE)));
    }

    std::vector<std::vector<double>> &testData = testDatas->getDatas();
    std::vector<int> &testPreds = testDatas->getPredictions();
    std::vector<int> &testTrueClasses = testDatas->getClasses();

    std::vector<std::vector<double>> &testPredictionScores = testDatas->getPredictionScores();
    int nbTestData = testDatas->getNbSamples();

    // Get attributes
    std::vector<std::string> attributeNames;
    std::vector<std::string> classNames;
    bool hasClassNames = false;
    if (params->isStringSet(ATTRIBUTES_FILE)) {
      testDatas->setAttributes(params->getString(ATTRIBUTES_FILE), nbAttributes, nbClasses);
      attributeNames = testDatas->getAttributeNames();
      hasClassNames = testDatas->getHasClassNames();
      if (hasClassNames) {
        classNames = testDatas->getClassNames();
      }
    }

    // =========================================================================
    // 5) Rule loading and first statistics line retrieval
    // =========================================================================

    // Get rules
    std::vector<std::string> lines;                             // Lines for the output stats
    lines.emplace_back("Global statistics of the rule set : "); // Lines for the output stats

    // Get statistic line at the top of the rulesfile
    std::string statsLine;
    std::fstream rulesData;
    std::vector<Rule> rules;

    if (hasJsonExtension(rulesFile)) {
      rules = Rule::fromJsonFile(rulesFile, decisionThreshold, positiveClassIndex);

      double meanCovering = 0;
      double meanNbAntecedentsPerRule = 0;
      auto nbRules = static_cast<int>(rules.size());

      for (const Rule &r : rules) {
        meanCovering += static_cast<double>(r.getCoveredSamples().size());
        meanNbAntecedentsPerRule += static_cast<double>(r.getAntecedents().size());
      }
      if (nbRules > 0) {
        meanCovering /= nbRules;
        meanNbAntecedentsPerRule /= nbRules;
      }

      statsLine += "Number of rules : " + std::to_string(nbRules);
      statsLine += ", mean sample covering number per rule : ";
      statsLine += std::to_string(meanCovering) + ", mean number of antecedents per rule : ";
      statsLine += std::to_string(meanNbAntecedentsPerRule) + "\n";

    } else {
      rulesData.open(rulesFile, std::ios::in); // Read data file
      if (rulesData.fail()) {
        throw FileNotFoundError("Error : file " + rulesFile + " not found.");
      }
      getline(rulesData, statsLine);
      statsLine += "\n";
      getRules(rules, rulesFile, *testDatas, decisionThreshold, positiveClassIndex);
    }
    lines.emplace_back(statsLine);

    std::cout << "Data imported." << std::endl
              << std::endl;

    // =========================================================================
    // 6) Global statistics computation on test set
    // =========================================================================

    // Compute global statistics on test set

    std::cout << "Compute statistics..." << std::endl
              << std::endl;
    double fidelity = 0;       // Global rule fidelity rate (wrong only if there is activated rules but no correct one(with respect to prediction))
    double accuracy = 0;       // Global rule accuracy (true if there is at least one fidel activated rule and the model is right, or if there is no activated rules and the model is right or if all the activated rules decide the same class and this class is the true one)
    double explainabilityRate; // True if there is an activated rule
    double explainabilityTotal = 0;
    double defaultRuleRate = 0;             // True if there is no activated rule
    double meanNbCorrectActivatedRules = 0; // Mean number of correct activated rules per sample
    double meanNbWrongActivatedRules = 0;   // Mean number of wrong activated rules per sample
    int nbActivatedRulesAndModelAgree = 0;
    double accuracyWhenActivatedRulesAndModelAgree = 0; // Model accuracy when activated rules and model agree (sample has correct activated rules or no activated rules, then percentage of good predictions on them by the model)
    int nbFidelRules = 0;                               // When the model and rule agree on the sample or when no rule is activated
    double accuracyWhenRulesAndModelAgree = 0;          // Model accuracy when activated rules and model agree (sample has correct activated rules, then percentage of good predictions on them by the model)
    double modelAccuracy = 0;
    int testPred;
    int testTrueClass;

    int nbTruePositive = 0;  // Correct positive prediction
    int nbTrueNegative = 0;  // Correct negative prediction
    int nbFalsePositive = 0; // Wrong positive prediction
    int nbFalseNegative = 0; // Wrong negative prediction
    int nbPositive = 0;
    int nbNegative = 0;

    int nbTruePositiveRules = 0;  // Correct positive rule prediction
    int nbTrueNegativeRules = 0;  // Correct negative rule prediction
    int nbFalsePositiveRules = 0; // Wrong positive rule prediction
    int nbFalseNegativeRules = 0; // Wrong negative rule prediction

    for (int t = 0; t < nbTestData; t++) { // For each test value
      const std::vector<double> &testValues = testData[t];
      testPred = testPreds[t];
      testTrueClass = testTrueClasses[t];
      if (hasPositiveClass) {
        if (testTrueClass == positiveClassIndex) {
          nbPositive += 1;
        } else {
          nbNegative += 1;
        }
      }

      if (testPred == testTrueClass) {
        modelAccuracy++;
      }
      if (hasPositiveClass) {
        computeTFPN(testPred, positiveClassIndex, testTrueClass, nbTruePositive, nbFalsePositive, nbTrueNegative, nbFalseNegative);
      }

      // Find rules activated by this sample
      bool noCorrectRuleWithAllSameClass = false; // If there is no correct rule activated but all rules have same class
      std::vector<int> activatedRules;
      getActivatedRules(activatedRules, rules, testValues);

      // Check which rules are correct (same prediction as the model's on the sample for activated rules)
      std::vector<int> correctRules;
      if (activatedRules.empty()) { // If there is no activated rule -> we would launch Fidex and so it will be fidel
        defaultRuleRate++;
        fidelity++; // It is true to the model because we choose his prediction
        nbFidelRules++;
        if (testPred == testTrueClass) { // If the model is right, it's true for the accuracy
          accuracy++;
          accuracyWhenRulesAndModelAgree++;
        }
      } else { // There is some activated rules
        for (int v : activatedRules) {
          if (rules[v].getOutputClass() == testPred) { // Check if the class of the rule is the predicted one
            correctRules.push_back(v);
          }
        }
        if (correctRules.empty()) { // If there is no correct rule
          meanNbWrongActivatedRules += static_cast<double>(activatedRules.size());

          int ancientClass = rules[activatedRules[0]].getOutputClass();
          bool allSameClass = true; // Check if all the rules choose the same class
          for (int v : activatedRules) {
            if (rules[v].getOutputClass() != ancientClass) {
              allSameClass = false;
              break;
            }
          }
          if (allSameClass) {
            explainabilityTotal++; // If all decisions are the same, we have an explanation
            int decision = rules[activatedRules[0]].getOutputClass();
            if (decision == testTrueClass) { // If those decisions are the true class, this is accurate
              accuracy++;
            }
            // The rules' decision is different from the model's (And we don't call Fidex on this sample)
            noCorrectRuleWithAllSameClass = true;
            if (hasPositiveClass) {
              computeTFPN(decision, positiveClassIndex, testTrueClass, nbTruePositiveRules, nbFalsePositiveRules, nbTrueNegativeRules, nbFalseNegativeRules);
            }
          }

        } else { // There is an explanation which is caracterised by the correct rules

          fidelity++; // It is true to the model because we found a correct activated rule
          explainabilityTotal++;
          nbActivatedRulesAndModelAgree++;
          nbFidelRules++;
          if (testPred == testTrueClass) {
            accuracy++;                                // If the model is right, it's true for the accuracy because we found a fidel rule
            accuracyWhenActivatedRulesAndModelAgree++; // It is true for the accuracy because prediction is correct
            accuracyWhenRulesAndModelAgree++;          // It is true for the accuracy because prediction is correct
          }
          meanNbCorrectActivatedRules += static_cast<double>(correctRules.size());
          meanNbWrongActivatedRules += static_cast<double>(activatedRules.size() - correctRules.size());
        }
      }

      if (!noCorrectRuleWithAllSameClass && hasPositiveClass) { // The rules' decision is the same as the model's, which is the case if we can find a correct rule or if we need to compute Fidex because no decision can be made by the ruleset
        computeTFPN(testPred, positiveClassIndex, testTrueClass, nbTruePositiveRules, nbFalsePositiveRules, nbTrueNegativeRules, nbFalseNegativeRules);
      }
    }

    const bool hasTestSamples = nbTestData > 0;
    const bool hasFidelRules = nbFidelRules > 0;
    const bool hasActivatedRulesAgreement = nbActivatedRulesAndModelAgree > 0;

    if (hasTestSamples) {
      fidelity /= nbTestData;
      accuracy /= nbTestData;
      explainabilityRate = explainabilityTotal / nbTestData;
      defaultRuleRate /= nbTestData;
      meanNbCorrectActivatedRules /= nbTestData;
      meanNbWrongActivatedRules /= nbTestData;
      modelAccuracy /= nbTestData;
    } else {
      explainabilityRate = 0.0;
    }

    if (hasFidelRules) {
      accuracyWhenRulesAndModelAgree /= nbFidelRules;
    }
    if (hasActivatedRulesAgreement) {
      accuracyWhenActivatedRulesAndModelAgree /= nbActivatedRulesAndModelAgree;
    }

    auto metricOrNA = [](double value, bool isDefined) {
      return isDefined ? std::to_string(value) : std::string("N/A");
    };

    // =========================================================================
    // 7) Build global report lines
    // =========================================================================

    lines.push_back("Statistics with a test set of " + std::to_string(nbTestData) + " samples :\n");
    if (decisionThreshold < 0.0) {
      lines.emplace_back("No decision threshold is used.");
      if (positiveClassIndex < 0) {
        lines.emplace_back("No positive index class is used.");
      } else {
        lines.push_back("Positive index class used : " + std::to_string(positiveClassIndex));
      }
    } else {
      lines.push_back("Using a decision threshold of " + std::to_string(decisionThreshold) + " for class " + std::to_string(positiveClassIndex));
    }

    lines.push_back("The global rule fidelity rate is : " + metricOrNA(fidelity, hasTestSamples));
    lines.push_back("The global rule accuracy is : " + metricOrNA(accuracy, hasTestSamples));
    lines.push_back("The explainability rate (when we can find one or more rules, either correct ones or activated ones which all agree on the same class) is : " + metricOrNA(explainabilityRate, hasTestSamples));
    lines.push_back("The default rule rate (when we can't find any rule activated for a sample) is : " + metricOrNA(defaultRuleRate, hasTestSamples));
    lines.push_back("The mean number of correct(fidel) activated rules per sample is : " + metricOrNA(meanNbCorrectActivatedRules, hasTestSamples));
    lines.push_back("The mean number of wrong(not fidel) activated rules per sample is : " + metricOrNA(meanNbWrongActivatedRules, hasTestSamples));
    lines.push_back("The model test accuracy is : " + metricOrNA(modelAccuracy, hasTestSamples));
    lines.push_back("The model test accuracy when rules and model agree is : " + metricOrNA(accuracyWhenRulesAndModelAgree, hasFidelRules));
    lines.push_back("The model test accuracy when activated rules and model agree is : " + metricOrNA(accuracyWhenActivatedRulesAndModelAgree, hasActivatedRulesAgreement));

    auto appendBinaryClassMetrics = [&](int truePositive, int falsePositive, int trueNegative, int falseNegative) {
      lines.push_back("The number of true positive test samples is : " + std::to_string(truePositive));
      lines.push_back("The number of false positive test samples is : " + std::to_string(falsePositive));
      lines.push_back("The number of true negative test samples is : " + std::to_string(trueNegative));
      lines.push_back("The number of false negative test samples is : " + std::to_string(falseNegative));
      lines.push_back("The false positive rate is : " + ((nbNegative != 0) ? std::to_string(float(falsePositive) / static_cast<float>(nbNegative)) : "N/A"));
      lines.push_back("The false negative rate is : " + ((nbPositive != 0) ? std::to_string(float(falseNegative) / static_cast<float>(nbPositive)) : "N/A"));
      lines.push_back("The precision is : " + ((truePositive + falsePositive != 0) ? std::to_string(float(truePositive) / static_cast<float>(truePositive + falsePositive)) : "N/A"));
      lines.push_back("The recall is : " + ((truePositive + falseNegative != 0) ? std::to_string(float(truePositive) / static_cast<float>(truePositive + falseNegative)) : "N/A"));
    };

    if (hasPositiveClass) {
      if (hasClassNames) {
        lines.push_back("\nWith positive class " + classNames[positiveClassIndex] + " :");
      } else {
        lines.push_back("\nWith positive class " + std::to_string(positiveClassIndex) + " :");
      }
      lines.emplace_back("\nComputation with model decision :");
      appendBinaryClassMetrics(nbTruePositive, nbFalsePositive, nbTrueNegative, nbFalseNegative);

      lines.emplace_back("\nComputation with rules decision (or Fidex):");
      appendBinaryClassMetrics(nbTruePositiveRules, nbFalsePositiveRules, nbTrueNegativeRules, nbFalseNegativeRules);
    }

    // =========================================================================
    // 8) Print and optionally save global report
    // =========================================================================

    for (const std::string &l : lines) {
      std::cout << l << std::endl;
    }

    // Output statistics
    if (params->isStringSet(STATS_FILE)) {
      std::ofstream outputFile(params->getString(STATS_FILE));
      if (outputFile.is_open()) {
        for (const std::string &line : lines) {
          outputFile << line << std::endl;
        }
        outputFile.close();
      } else {
        throw CannotOpenFileError("Error : Couldn't open explanation extraction file " + params->getString(STATS_FILE) + ".");
      }
    }

    // =========================================================================
    // 9) Optional per-rule test statistics export
    // =========================================================================

    // Compute rules statistics on test set
    if (params->isStringSet(GLOBAL_RULES_OUTFILE)) {
      std::string ruleFile = params->getString(GLOBAL_RULES_OUTFILE);

      if (hasJsonExtension(ruleFile)) {
        // 9.a JSON export path
        std::vector<Rule> testRules;

        for (size_t r = 0; r < rules.size(); ++r) {
          const Rule &fullRule = rules[r];
          const int rulePred = fullRule.getOutputClass();
          validateRulePredictionIndex(rulePred, nbClasses, r);

          const RuleTestStats ruleStats = computeRuleTestStats(fullRule, testData, testPreds, testTrueClasses, testPredictionScores, r);

          // Only keep detailed per-antecedent test stats when final test covering is non-zero.
          if (!ruleStats.coverSizes.empty() && ruleStats.coverSizes.back() > 0) {
            testRules.push_back(Rule(fullRule.getAntecedents(), ruleStats.coveredSamples, ruleStats.coverSizes, rulePred, ruleStats.fidelities.back(),
                                     ruleStats.fidelityIncreases, ruleStats.accuracies.back(), ruleStats.accuracyChanges, ruleStats.finalConfidence));
          } else {
            testRules.push_back(Rule(fullRule.getAntecedents(), {}, {}, rulePred, ruleStats.fidelities.back(), {}, ruleStats.accuracies.back(), {}, ruleStats.finalConfidence));
          }
        }

        Rule::toJsonStatsFile(ruleFile, rules, testRules, decisionThreshold, positiveClassIndex);

      } else {
        // 9.b Text rules export path
        std::ofstream outputFile(ruleFile);
        if (outputFile.is_open()) {
          outputFile << statsLine;

          if (decisionThreshold < 0.0) {
            outputFile << "No decision threshold is used.\n";
          } else {
            outputFile << "Using a decision threshold of " << decisionThreshold << " for class " << positiveClassIndex << "\n";
          }

          for (size_t r = 0; r < rules.size(); ++r) {
            const Rule &fullRule = rules[r];
            const int rulePred = fullRule.getOutputClass();
            validateRulePredictionIndex(rulePred, nbClasses, r);

            const RuleTestStats ruleStats = computeRuleTestStats(fullRule, testData, testPreds, testTrueClasses, testPredictionScores, r);

            const std::vector<std::string> trainStats = splitString(fullRule.toString(attributeNames, classNames), "\n");
            if (trainStats.size() < 8) {
              throw FileFormatError("Error : unexpected rule format while writing test stats for rule " + std::to_string(r + 1) + ".");
            }
            outputFile << "\n"
                       << "Rule " << std::to_string(r + 1) << ": " << trainStats[0] << std::endl;
            outputFile << trainStats[1] << " --- Test Covering size : " << ruleStats.coverSizes.back() << std::endl;
            if (ruleStats.coverSizes.back() == 0) {
              outputFile << trainStats[2] << std::endl;
              outputFile << trainStats[3] << std::endl;
              outputFile << trainStats[4] << std::endl;
              outputFile << trainStats[5] << std::endl;
              outputFile << trainStats[6] << std::endl;
              outputFile << trainStats[7] << std::endl;
              outputFile << std::endl;
            } else {
              outputFile << trainStats[2] << " --- Test Fidelity : " << formattingDoubleToString(ruleStats.fidelities.back()) << std::endl;
              outputFile << trainStats[3] << " --- Test Accuracy : " << formattingDoubleToString(ruleStats.accuracies.back()) << std::endl;
              outputFile << trainStats[4] << " --- Test Confidence : " << formattingDoubleToString(ruleStats.finalConfidence) << std::endl;
              outputFile << trainStats[5] << std::endl;
              outputFile << "   Test Covering size evolution with antecedents : ";
              for (int c : ruleStats.coverSizes) {
                outputFile << c << " ";
              }
              outputFile << std::endl;
              outputFile << trainStats[6] << std::endl;
              outputFile << "   Test Fidelity increase with antecedents : ";
              for (double f : ruleStats.fidelityIncreases) {
                outputFile << formattingDoubleToString(f) << " ";
              }
              outputFile << std::endl;
              outputFile << trainStats[7] << std::endl;
              outputFile << "   Test Accuracy variation with antecedents : ";
              for (double a : ruleStats.accuracyChanges) {
                outputFile << formattingDoubleToString(a) << " ";
              }
              outputFile << std::endl;
              outputFile << std::endl;
            }
          }
          outputFile.close();
        } else {
          throw CannotOpenFileError("Error : Couldn't open global rules file with statistics on test set " + params->getString(GLOBAL_RULES_OUTFILE) + ".");
        }
      }
    }

    // =========================================================================
    // 10) End-of-run timing
    // =========================================================================

    t2 = clock();
    temps = (float)(t2 - t1) / CLOCKS_PER_SEC;
    std::cout << "\nFull execution time = " << temps << " sec" << std::endl;

  } catch (const ErrorHandler &e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
