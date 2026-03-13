#include "dataSet.h"
#include "checkFun.h"
#include "errorHandler.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>

/**
 * @brief Local helper for weights file parsing.
 */
namespace {
/**
 * @brief Parse one non-empty weight line and ensure it only contains numeric values.
 *
 * @param line Raw line read from weight file.
 * @param datasetName Dataset identifier used in error messages.
 * @param weightFile Weight filename used in error messages.
 * @return Parsed numeric values for the line.
 */
std::vector<double> parseWeightValuesLine(const std::string &line, const std::string &datasetName, const std::string &weightFile) {
  std::stringstream myLine(line);
  std::vector<double> values;
  double value;

  while (myLine >> value) {
    values.push_back(value);
  }

  if (values.empty()) {
    throw FileContentError("Error in dataset " + datasetName + " : in file " + weightFile + ", one non-empty weight line does not contain any valid numeric value.");
  }

  myLine >> std::ws;
  if (!myLine.eof()) {
    throw FileContentError("Error in dataset " + datasetName + " : in file " + weightFile + ", one weight line contains non-numeric tokens.");
  }

  return values;
}
} // namespace

/**
 * @brief Construct a new DataSetFid object using separate data, prediction, and optional class files.
 *
 *        - The data file should contain the attributes of each sample, and optionally, the class information can also be included in this file.
 *          If class information is included in the data file, there is no need to provide a separate class file.
 *        - The prediction file should contain the prediction scores for each class per sample.
 *        - The class file (optional) should be used only if class information is not included in the data file. It should contain either
 *          a single class ID or a series of integers in a one-hot encoding scheme per sample.
 *
 *        Expected file formats:
 *        - Data file: Each line should contain a series of numerical values representing the attributes of a sample,
 *          optionally followed by the class information (either as a single integer for class ID or a one-hot encoded vector).
 *        - Prediction file: Each line should contain a series of numerical values.
 *        - Class file (optional): Only needed if class information is not included in the data file. Each line should contain either a single class ID
 *          or a series of integers in a one-hot encoding scheme.
 *
 *        Each number in a line are separated by a space, a comma(CSV), a semicolon(;) or a tab.
 *        The number of attributes and classes in the dataset are used to validate the format and content of the data and class files.
 *
 * @param name The name of the dataSet.
 * @param dataFile The data file name.
 * @param predFile The prediction file name.
 * @param nbAttributes The number of attributes.
 * @param nbClasses The number of classes.
 * @param decisionThresh A double indicating the decision threshold, useful when choosing the decision (-1 for no threshold).
 * @param positiveClassId An integer corresponding to the index of the positive class for which we have the decision threshold (-1 if no threshold).
 * @param trueClassFile The class file name.
 */
DataSetFid::DataSetFid(const std::string &name, const std::string &dataFile, const std::string &predFile, int nbAttributes, int nbClasses, double decisionThresh, int positiveClassId, const std::string &trueClassFile) : datasetName(name) {

  setNbClassAndAttr(nbClasses, nbAttributes);

  if (decisionThresh != -1 && !std::isnan(decisionThresh)) {
    decisionThreshold = decisionThresh;
  }
  if (positiveClassId != -1) {
    positiveClassIndex = positiveClassId;
  }

  checkThreshold();

  // Get data
  setDataFromFile(dataFile, nbAttributes, nbClasses);
  // Get data class
  if (!trueClassFile.empty()) {
    setClassFromFile(trueClassFile, nbClasses);
  }
  // Get predictions
  setPredFromFile(predFile, nbClasses);

  checkDatas();
}

/**
 * @brief Construct a new DataSetFid object using a single data file containing data, predictions, and optionally classes.
 *
 *        The format of each sample in the file is as follows:
 *        - First Line: Contains data attributes. It may be followed by class information (either as an ID or in one-hot format).
 *        - Second Line: Contains prediction values.
 *        - Third Line (optional): Contains class information, only if it was not included in the first line and if present.
 *        There is a blank line between each sample in the file.
 *        Each data in a line is separated by a space, a comma(CSV), a semicolon(;) or a tab.
 *
 *        The presence and format of class data (ID or one-hot) are inferred based on the structure of the lines in the file.
 *
 * @param name A string containing the name of the dataSet.
 * @param dataFile The data file name containing data, predictions and maybe classes(not mandatory).
 * @param nbAttributes The number of attributes.
 * @param nbClasses The number of classes.
 * @param decisionThresh A double indicating the decision threshold, useful when choosing the decision (-1 for no threshold).
 * @param positiveClassId An integer corresponding to the index of the positive class for which we have the decision threshold (-1 if no threshold).
 */
DataSetFid::DataSetFid(const std::string &name, const std::string &dataFile, int nbAttributes, int nbClasses, double decisionThresh, int positiveClassId) : datasetName(name), hasDatas(true), hasPreds(true) {

  setNbClassAndAttr(nbClasses, nbAttributes);

  if (decisionThresh != -1 && !std::isnan(decisionThresh)) {
    decisionThreshold = decisionThresh;
  }
  if (positiveClassId != -1) {
    positiveClassIndex = positiveClassId;
  }

  checkThreshold();

  std::ifstream fileDta;
  fileDta.open(dataFile, std::ios::in); // Read data file
  if (fileDta.fail()) {
    throw FileNotFoundError("Error in dataset " + datasetName + " : file " + dataFile + " not found.");
  }
  std::string line;
  bool firstLine = true;
  while (std::getline(fileDta, line)) {
    if (!checkStringEmpty(line)) {
      setDataLine(line, dataFile);
    } else if (firstLine) {
      throw FileFormatError("Error in dataset " + datasetName + " : in file " + dataFile + ", first line is empty.");
    } else {
      while (std::getline(fileDta, line)) {
        if (!checkStringEmpty(line)) {
          throw FileFormatError("Error in dataset " + datasetName + " : file " + dataFile + " is not on good format, there are more than one empty line between 2 samples.");
        }
      }
      break; // There are just empty lines at the end of the file
    }
    if (!std::getline(fileDta, line)) {
      throw FileContentError("Error in dataset " + datasetName + " : file " + dataFile + " has not enough prediction data.");
    }
    if (!checkStringEmpty(line)) {
      setPredLine(line, dataFile);
    } else {
      while (std::getline(fileDta, line)) {
        if (!checkStringEmpty(line)) {
          throw FileFormatError("Error in dataset " + datasetName + " : file " + dataFile + " is not on good format, there is empty lines inbetween data.");
        }
      }
      throw FileContentError("Error in dataset " + datasetName + " : file " + dataFile + " has not enough prediction data.");
    }
    bool endOfFile = !std::getline(fileDta, line);
    if (!endOfFile && !checkStringEmpty(line)) {
      if (classFormat == ClassFormat::OneHotCombined || classFormat == ClassFormat::IdCombined) {
        throw FileContentError("Error in dataset " + datasetName + " : file " + dataFile + " classes are given two times in the file, with attributes and are after predictions.");
      }
      setClassLine(line, dataFile);

      if (std::getline(fileDta, line)) {
        if (!checkStringEmpty(line)) {
          throw FileFormatError("Error in dataset " + datasetName + " : in file " + dataFile + ", you need to have empty lines between samples. You have chosen to give data, predictions and classes in one file. If you want to separate them, specify a prediction file.");
        }
      }
    }
    firstLine = false;
  }
  if (firstLine) {
    throw FileFormatError("Error in dataset " + datasetName + " : in file " + dataFile + ", first line is empty.");
  }

  if (!trueClasses.empty()) {
    hasClasses = true;
    nbClassData = static_cast<int>(trueClasses.size());
  }

  fileDta.close(); // close data file

  nbSamples = static_cast<int>(datas.size());
  nbPredData = static_cast<int>(predictions.size());
  checkDatas();
}

/**
 * @brief Construct a new DataSetFid object using a weight file.
 *
 * This constructor is capable of handling both single and multiple network weight files.
 *
 * @param name The name of the dataSet.
 * @param weightFile The name of the weight file.
 */
DataSetFid::DataSetFid(const std::string &name, const std::string &weightFile) : datasetName(name) {
  setWeights(weightFile);
}

/**
 * @brief Allows to set weights from file
 *
 * @param weightFile path of file
 */
void DataSetFid::setWeights(const std::string &weightFile) {
  if (hasWeights) {
    throw InternalError("Error in dataset " + datasetName + " : weights have been given two times.");
  }

  // Commit weights only if full parsing succeeds.
  const size_t weightsSizeBefore = weights.size();
  const int nbNetsBefore = nbNets;
  const bool hasWeightsBefore = hasWeights;

  try {
    std::ifstream fileWts;
    std::string line;
    bool multipleNetworks = false;

    fileWts.open(weightFile, std::ios::in); // Read weight file
    if (fileWts.fail()) {
      throw FileNotFoundError("Error in dataset " + datasetName + " : file " + weightFile + " not found.");
    }

    // Check if weight file contains multiple networks
    while (getline(fileWts, line)) {
      if (line.find("Network") != std::string::npos) {
        multipleNetworks = true;
        break;
      }
    }

    // Reinitialise cursor at the beginning
    fileWts.clear();
    fileWts.seekg(0, std::ios::beg);

    if (!multipleNetworks) {
      parseSingleNetwork(fileWts, weightFile);
    } else {
      parseMultipleNetworks(fileWts, weightFile);
    }

    // Validate each parsed network structure once at load time.
    for (size_t netIndex = weightsSizeBefore; netIndex < weights.size(); ++netIndex) {
      if (weights[netIndex].size() < 2) {
        throw FileContentError("Error in dataset " + datasetName + " : malformed weight file, missing first-layer bias/weight vectors for network " + std::to_string(netIndex) + ".");
      }
    }

    nbNets = static_cast<int>(weights.size());
    hasWeights = true;
    fileWts.close(); // close file
  } catch (...) {
    weights.resize(weightsSizeBefore);
    nbNets = nbNetsBefore;
    hasWeights = hasWeightsBefore;
    throw;
  }
}

/**
 * @brief Parses a weight file containing a single network's weights and stores them in the weights vector.
 *
 * @param fileWts Reference to the file stream opened for reading the weight file.
 * @param weightFile Weight filename used to report precise parsing errors.
 */
void DataSetFid::parseSingleNetwork(std::istream &fileWts, const std::string &weightFile) {
  std::string line;
  std::vector<std::vector<double>> networkWeights;

  while (getline(fileWts, line)) {
    if (!checkStringEmpty(line)) {
      networkWeights.push_back(parseWeightValuesLine(line, datasetName, weightFile));
    }
  }

  if (networkWeights.empty()) {
    throw FileContentError("Error in dataset " + datasetName + " : in file " + weightFile + ", no valid weight rows were found.");
  }

  weights.push_back(networkWeights); // Add the network to weights
}

/**
 * @brief Parses a weight file containing multiple networks' weights and stores them in the weights vector.
 *
 * @param fileWts Reference to the file stream opened for reading the weight file.
 * @param weightFile Weight filename used to report precise parsing errors.
 */
void DataSetFid::parseMultipleNetworks(std::istream &fileWts, const std::string &weightFile) {
  std::string line;
  std::vector<std::vector<double>> networkWeights;
  int networkCount = 0;
  bool hasSeenNetworkHeader = false;

  while (getline(fileWts, line)) {
    if (line.find("Network") != std::string::npos) {
      if (hasSeenNetworkHeader) {
        if (networkWeights.empty()) {
          throw FileContentError("Error in dataset " + datasetName + " : in file " + weightFile + ", network " + std::to_string(networkCount) + " contains no weight rows.");
        }
        weights.push_back(networkWeights); // Add network to weights
        networkWeights.clear();
      }
      hasSeenNetworkHeader = true;
      networkCount++;
    } else if (!checkStringEmpty(line)) {
      if (!hasSeenNetworkHeader) {
        throw FileContentError("Error in dataset " + datasetName + " : in file " + weightFile + ", weight data were found before the first 'Network' header.");
      }
      networkWeights.push_back(parseWeightValuesLine(line, datasetName, weightFile));
    }
  }

  if (!hasSeenNetworkHeader) {
    throw FileContentError("Error in dataset " + datasetName + " : in file " + weightFile + ", no 'Network' header was found.");
  }
  if (networkWeights.empty()) {
    throw FileContentError("Error in dataset " + datasetName + " : in file " + weightFile + ", network " + std::to_string(networkCount) + " contains no weight rows.");
  }

  weights.push_back(networkWeights); // Add last network to weights
}

/**
 * @brief Get data from dataFile and save it in datas and trueClasses if it contains class information.
 *
 *        The file should contain one sample per line. Each number in line is separated by a space, a comma(CSV), a semicolon(;) or a tab. Each sample can be in one of the following formats:
 *        1. Attributes only: Each line contains each float attribute.
 *        2. Attributes with Class ID: Each line contains all the float attributes followed by an integer class ID.
 *        3. Attributes with One-Hot Class Encoding: Each line contains all the float attributes followed by a one-hot encoding of the class.
 *           The number of elements in this encoding should match the total number of classes, with exactly one '1' and the rest '0's.
 *
 * @param dataFile A string representing the name of the data file. This file should contain data in one of the supported formats.
 * @param nbAttributes The number of attributes.
 * @param nbClasses The number of classes.
 */
void DataSetFid::setDataFromFile(const std::string &dataFile, int nbAttributes, int nbClasses) {
  if (hasDatas == true) {
    throw InternalError("Error in dataset " + datasetName + " : data have been given two times.");
  }
  // Commit flags/state only if parsing succeeds.
  const size_t datasSizeBefore = datas.size();
  const size_t classesSizeBefore = trueClasses.size();
  const int nbSamplesBefore = nbSamples;
  const int nbClassDataBefore = nbClassData;
  const int nbClassesBefore = _nbClasses;
  const int nbAttributesBefore = _nbAttributes;
  const bool hasClassesBefore = hasClasses;
  const ClassFormat classFormatBefore = classFormat;

  try {
    setNbClassAndAttr(nbClasses, nbAttributes);

    std::string line;
    std::ifstream fileDta;

    fileDta.open(dataFile, std::ios::in); // Read data file
    if (fileDta.fail()) {
      throw FileNotFoundError("Error in dataset " + datasetName + " : file " + dataFile + " not found.");
    }

    while (std::getline(fileDta, line)) {
      if (!checkStringEmpty(line)) {
        setDataLine(line, dataFile);
      }
    }

    fileDta.close(); // close data file

    if (trueClasses.size() > classesSizeBefore) {
      if (hasClassesBefore == true) {
        throw InternalError("Error in dataset " + datasetName + " : classes have been given two times.");
      }
      hasClasses = true;
      nbClassData = static_cast<int>(trueClasses.size());
    }

    nbSamples = static_cast<int>(datas.size());
    hasDatas = true;
    checkDatas();
  } catch (...) {
    datas.resize(datasSizeBefore);
    trueClasses.resize(classesSizeBefore);
    nbSamples = nbSamplesBefore;
    nbClassData = nbClassDataBefore;
    _nbClasses = nbClassesBefore;
    _nbAttributes = nbAttributesBefore;
    hasClasses = hasClassesBefore;
    classFormat = classFormatBefore;
    throw;
  }
}

/**
 * @brief Add predictions to the dataset using a prediction file.
 *
 *        The prediction file should contain one line per data sample, each line consisting of a series of numerical values separated
 *        by a space, a comma(CSV), a semicolon(;) or a tab representing the prediction scores for each class.
 *        The number of values per line should match the specified number of classes.
 *        If a decision threshold is provided, the function uses it to determine the predicted class based on the threshold.
 *
 * @param predFile The prediction file name.
 * @param nbClasses The number of classes.
 * @param decisionThresh An optional double indicating the decision threshold, useful when choosing the decision (-1 for no threshold).
 * @param positiveClassId An optional integer corresponding to the index of the positive class for which we have the decision threshold (-1 if no threshold).
 */
void DataSetFid::setPredFromFile(const std::string &predFile, int nbClasses, double decisionThresh, int positiveClassId) {
  if (hasPreds == true) {
    throw InternalError("Error in dataset " + datasetName + " : predictions have been given two times.");
  }

  // Keep object reusable after any parsing error.
  const size_t predsSizeBefore = predictions.size();
  const size_t predScoresSizeBefore = predictionScores.size();
  const int nbPredDataBefore = nbPredData;
  const int nbClassesBefore = _nbClasses;
  const double decisionThresholdBefore = decisionThreshold;
  const int positiveClassIndexBefore = positiveClassIndex;

  try {
    setNbClass(nbClasses);

    if (decisionThresh != -1 && !std::isnan(decisionThresh)) {
      if (decisionThreshold == -1) {
        decisionThreshold = decisionThresh;
      } else {
        throw InternalError("Error in dataset " + datasetName + " : decision threshold has been given two times.");
      }
    }
    if (positiveClassId != -1) {
      if (positiveClassIndex == -1) {
        positiveClassIndex = positiveClassId;
      } else {
        throw InternalError("Error in dataset " + datasetName + " : index of positive class has been given two times.");
      }
    }

    checkThreshold();

    std::string line;
    std::ifstream filePrd;

    filePrd.open(predFile, std::ios::in); // read predictions data file
    if (filePrd.fail()) {
      throw FileNotFoundError("Error in dataset " + datasetName + " : file " + predFile + " not found.");
    }

    while (std::getline(filePrd, line)) {
      if (!checkStringEmpty(line)) {
        setPredLine(line, predFile);
      }
    }

    filePrd.close(); // close file
    nbPredData = static_cast<int>(predictions.size());
    hasPreds = true;
    checkDatas();
  } catch (...) {
    predictions.resize(predsSizeBefore);
    predictionScores.resize(predScoresSizeBefore);
    nbPredData = nbPredDataBefore;
    _nbClasses = nbClassesBefore;
    decisionThreshold = decisionThresholdBefore;
    positiveClassIndex = positiveClassIndexBefore;
    throw;
  }
}

/**
 * @brief Add classes from a specified file into the dataset.
 *
 *        The class file can contain lines in different formats:
 *        1. Class ID format: Each line contains a single integer representing the class ID.
 *        2. One-hot format: Each line contains a sequence of integers in a one-hot encoding scheme,
 *           where exactly one value is 1 (indicating the class ID) and all others are 0.
 *        Each number in a line is separated by a space, a comma(CSV), a semicolon(;) or a tab.
 *
 *        The function determines the format of each line based on the nbClasses parameter and the structure of the data in the line.
 *
 * @param classFile A string representing the name of the class file. This file should contain class data.
 *                  in one of the supported formats (either class ID or one-hot encoded).
 * @param nbClasses An int specifying the number of classes.
 */
void DataSetFid::setClassFromFile(const std::string &classFile, int nbClasses) {
  if (hasClasses == true) {
    throw InternalError("Error in dataset " + datasetName + " : classes have been given two times.");
  }

  // Rollback class parsing on failure.
  const size_t classesSizeBefore = trueClasses.size();
  const int nbClassDataBefore = nbClassData;
  const int nbClassesBefore = _nbClasses;
  const ClassFormat classFormatBefore = classFormat;

  try {
    setNbClass(nbClasses);

    std::string line;
    std::ifstream fileCl;

    fileCl.open(classFile, std::ios::in); // read true class file
    if (fileCl.fail()) {
      throw FileNotFoundError("Error in dataset " + datasetName + " : file " + classFile + " not found.");
    }

    while (std::getline(fileCl, line)) {
      if (!checkStringEmpty(line)) {
        setClassLine(line, classFile);
      }
    }

    fileCl.close(); // close file
    nbClassData = static_cast<int>(trueClasses.size());
    hasClasses = true;
    checkDatas();
  } catch (...) {
    trueClasses.resize(classesSizeBefore);
    nbClassData = nbClassDataBefore;
    _nbClasses = nbClassesBefore;
    classFormat = classFormatBefore;
    throw;
  }
}

/**
 * @brief Read a data line from a data file and save it in datas and trueClasses if it contains class information.
 *
 *        This function supports different data formats. It can process lines containing:
 *        - Only attributes: A sequence of numerical values.
 *        - Attributes followed by class IDs: A sequence of numerical values followed by an integer representing the class ID.
 *        - Attributes followed by one-hot encoded classes: A sequence of numerical values followed by a one-hot encoding representing the class.
 *          The number of elements in this encoding should match the total number of classes, with exactly one '1' and the rest '0's.
 *        Each data in a line is separated by a space, a comma(CSV), a semicolon(;) or a tab.
 *        The format of the class representation (ID or one-hot encoding) is determined based on the structure of the data in the line.
 *
 * @param line A string containing one line of the data file. This line can include attribute values and, optionally, class information.
 * @param dataFile A string representing the name of the data file. This file may contain lines of data in any of the supported formats.
 */
void DataSetFid::setDataLine(const std::string &line, const std::string &dataFile) {

  std::vector<double> valuesData = parseFileLine(line, dataFile);

  auto lineSize = static_cast<int>(valuesData.size());

  // Identify class format (one-hot, id, one-hot_combined, id_combined)
  if (classFormat == ClassFormat::Unknown) {
    if (lineSize == _nbAttributes + _nbClasses) {
      classFormat = ClassFormat::OneHotCombined;
    } else if (lineSize == _nbAttributes + 1) {
      classFormat = ClassFormat::IdCombined;
    } else if (lineSize != _nbAttributes) {
      throw FileContentError("Error in dataset " + datasetName + " : in file " + dataFile + ", incorrect number of parameters per line, expected the given number of attributes (" + std::to_string(_nbAttributes) + ") or this number plus 1 (classes in id format) or plus the number of classes (one-hot format).");
    }
  }

  if (classFormat == ClassFormat::OneHot || classFormat == ClassFormat::Id || classFormat == ClassFormat::Unknown) { // No classes with the attributes
    if (lineSize != _nbAttributes) {
      throw FileContentError("Error in dataset " + datasetName + " : in file " + dataFile + ", the number of attributes for a sample don't match the given number of attributes (" + std::to_string(_nbAttributes) + "). No class should be given with the data as there was none beforehand.");
    } else {
      datas.push_back(valuesData);
    }
  } else if (classFormat == ClassFormat::OneHotCombined) { // With classes in one-hot format
    if (lineSize != _nbAttributes + _nbClasses) {
      throw FileContentError("Error in dataset " + datasetName + " : in file " + dataFile + ", the number of attributes and classes for a sample don't match the given number of attributes and classes (" + std::to_string(_nbAttributes + _nbClasses) + "). Classes should be given with data on one-hot format as they were given like this beforehand.");
    } else {

      int oneHotIndex;
      if (isOneHot(valuesData, _nbAttributes, oneHotIndex)) {
        datas.emplace_back(valuesData.begin(), valuesData.begin() + _nbAttributes);
        trueClasses.emplace_back(oneHotIndex);
      } else {
        throw FileContentError("Error in dataset " + datasetName + " : in file " + dataFile + ", invalid one-hot format for a given sample.");
      }
    }
  } else if (classFormat == ClassFormat::IdCombined) { // With classes in id format
    if (lineSize != _nbAttributes + 1) {
      throw FileContentError("Error in dataset " + datasetName + " : in file " + dataFile + ", the number of attributes for a sample don't match the given number of attributes (" + std::to_string(_nbAttributes) + "). The id of the class should be given at the end of the line as they were given like this beforehand.");
    } else {
      double classId = valuesData.back();
      if (classId < 0 || classId >= _nbClasses || classId != std::floor(classId)) {
        throw FileContentError("Error in dataset " + datasetName + " : in file " + dataFile + ", invalid class ID for a given sample.");
      }
      datas.emplace_back(valuesData.begin(), valuesData.begin() + _nbAttributes);
      trueClasses.emplace_back(static_cast<int>(classId));
    }
  } else {
    throw InternalError("Error in dataset " + datasetName + " : wrong class format.");
  }
}

/**
 * @brief Read a prediction line from the prediction file and save it in predictions.
 *
 *        Each line in the prediction file should contain a sequence of numerical values representing the prediction scores for each class.
 *        The number of values in each line must match the number of classes specified for the dataset.
 *        If a decision threshold is set, the function also determines the predicted class based on this threshold.
 *        Each line should contain numbers separated by a space, a comma(CSV), a semicolon(;) or a tab.
 *
 * @param line A string containing one line of the prediction file.
 * @param predFile The prediction file name.
 */
void DataSetFid::setPredLine(const std::string &line, const std::string &predFile) {

  std::vector<double> valuesPred = parseFileLine(line, predFile);

  if (static_cast<int>(valuesPred.size()) != _nbClasses) {
    throw FileContentError("Error in dataset " + datasetName + " : in file " + predFile + ", the number of predictions is not equal to the number of classes (" + std::to_string(_nbClasses) + ") for each sample.");
  }

  predictionScores.push_back(valuesPred);

  if (decisionThreshold >= 0 && valuesPred[positiveClassIndex] >= decisionThreshold) {
    predictions.push_back(positiveClassIndex);
  } else {
    predictions.push_back(static_cast<int>(std::max_element(valuesPred.begin(), valuesPred.end()) - valuesPred.begin()));
  }
}

/**
 * @brief Read a class line from the class file and save it in trueClasses.
 *
 *        This function is designed to handle two formats of class representation in the file:
 *        1. Class ID format: The line contains a single integer representing the class ID.
 *        2. One-hot format: The line contains a sequence of integers in a one-hot encoding scheme,
 *           where exactly one value is 1 (indicating the class ID) and all others are 0.
 *        Each data in a line is separated by a space, a comma(CSV), a semicolon(;) or a tab.
 *
 *        The function automatically determines the format based on the structure of the data in the line.
 *
 * @param line A string containing one line from the class file. This line can either have a single integer (class ID)
 *             or multiple integers (one-hot encoded vector) representing the class information.
 * @param classFile A string representing the name of the class file. This file should contain lines in one of the
 *                  supported formats (either class ID or one-hot encoded).
 */
void DataSetFid::setClassLine(const std::string &line, const std::string &classFile) {
  std::vector<double> valuesData = parseFileLine(line, classFile);

  auto lineSize = static_cast<int>(valuesData.size());

  if (classFormat == ClassFormat::Unknown) {
    if (lineSize == _nbClasses) {
      classFormat = ClassFormat::OneHot;
    } else if (lineSize == 1) {
      classFormat = ClassFormat::Id;
    } else {
      throw FileContentError("Error in dataset " + datasetName + " : in file " + classFile + ", incorrect number of classes for a given sample.");
    }
  }

  if (classFormat == ClassFormat::OneHot) {
    if (lineSize != _nbClasses) {
      throw FileContentError("Error in dataset " + datasetName + " : in file " + classFile + ", the number of classes for a sample doesn't match the given number of classes (" + std::to_string(_nbClasses) + "). Classes should be given with data on one-hot format as they were given like this beforehand.");
    } else {
      int oneHotIndex;
      if (isOneHot(valuesData, 0, oneHotIndex)) {
        trueClasses.emplace_back(oneHotIndex);
      } else {
        throw FileContentError("Error in dataset " + datasetName + " : in file " + classFile + ", invalid one-hot format for a given sample.");
      }
    }
  } else if (classFormat == ClassFormat::Id) {
    if (lineSize != 1) {
      throw FileContentError("Error in dataset " + datasetName + " : in file " + classFile + ", only the id of the class should be given on the line as classes were given like this beforehand.");
    } else {
      double classId = valuesData[0];
      if (classId < 0 || classId >= _nbClasses || classId != std::floor(classId)) {
        throw FileContentError("Error in dataset " + datasetName + " : in file " + classFile + ", invalid class ID for a given sample.");
      }
      trueClasses.emplace_back(static_cast<int>(classId));
    }
  } else if (classFormat == ClassFormat::OneHotCombined || classFormat == ClassFormat::IdCombined) {
    throw InternalError("Error in dataset " + datasetName + " : classes were already added combined with data.");
  } else {
    throw InternalError("Error in dataset " + datasetName + " : wrong class format.");
  }
}

/**
 * @brief Set the number of classes for the dataset.
 *
 * This function sets the number of classes for the dataset.
 * If the number of classes is already set (not -1), it checks
 * if the new value matches the existing one. If not, an error is thrown.
 * If the number is not set yet, it validates that the new value is strictly positive.
 *
 * @param nbClasses The number of classes to set.
 */
void DataSetFid::setNbClass(int nbClasses) {
  if (_nbClasses != -1) {
    if (_nbClasses != nbClasses) {
      throw InternalError("Error in dataset " + datasetName + " : the given number of classes (" + std::to_string(nbClasses) + ") doesn't match the number given before (" + std::to_string(_nbClasses) + ").");
    }
  } else {
    if (nbClasses > 0) {
      _nbClasses = nbClasses;
    } else {
      throw InternalError("Error in dataset " + datasetName + " : the number of classes has to be a strictly positive integer.");
    }
  }
}

/**
 * @brief Set the number of attributes and classes for the dataset.
 *
 * This function sets the number of attributes and classes for the dataset.
 * If the number of attributes or classes is already set (not -1), it checks
 * if the new value matches the existing one. If not, an error is thrown.
 * If the number is not set yet, it validates that the new value is strictly positive.
 *
 * @param nbAttributes The number of attributes to set.
 * @param nbClasses The number of classes to set.
 */
void DataSetFid::setNbClassAndAttr(int nbClasses, int nbAttributes) {
  if (_nbAttributes != -1) {
    if (_nbAttributes != nbAttributes) {
      throw InternalError("Error in dataset " + datasetName + " : the given number of attributes (" + std::to_string(nbAttributes) + ") doesn't match the number given before (" + std::to_string(_nbAttributes) + ").");
    }
  } else {
    if (nbAttributes > 0) {
      _nbAttributes = nbAttributes;
    } else {
      throw InternalError("Error in dataset " + datasetName + " : the number of attributes has to be a strictly positive integer.");
    }
  }

  setNbClass(nbClasses);
}

/**
 * @brief Checks if a portion of a std::vector<double> follows a one-hot encoding scheme.
 *
 *        In one-hot encoding, all elements must be either 0.0 or 1.0, with exactly one element being 1.0.
 *
 * @param values The vector of double values to check for one-hot encoding.
 * @param start The starting index in 'values' from where to check for one-hot encoding.
 * @param oneHotIndex Reference to an integer where the index (relative to 'start') of the 1.0 element in one-hot encoding will be stored.
 *                    This is set only if a valid one-hot encoding is found.
 * @return True if the portion of the vector starting from 'start' follows a one-hot encoding.
 * @return False if the portion of the vector does not follow a one-hot encoding,
 *               either because there's more than one 1.0, or because an element other than 0.0 or 1.0 is found.
 */
bool DataSetFid::isOneHot(const std::vector<double> &values, int start, int &oneHotIndex) const {
  bool foundOne = false; // Flag to indicate if 1.0 has been found in the specified range
  oneHotIndex = -1;      // Initialize the one-hot index to -1 (not found)
  if (start < 0 || static_cast<size_t>(start) > values.size()) {
    return false;
  }
  const size_t startIndex = static_cast<size_t>(start);

  for (size_t i = startIndex; i < values.size(); ++i) {
    if (values[i] == 1.0) { // Check if the current element is 1.0
      if (foundOne) {
        // More than one '1.0' found, not a one-hot format
        return false;
      }
      foundOne = true;                                // Mark that we've found a 1.0
      oneHotIndex = static_cast<int>(i - startIndex); // Calculate the relative index of the 1.0 element
    } else if (values[i] != 0.0) {
      // An element other than 0.0 or 1.0 found, not a one-hot format
      return false;
    }
  }

  return foundOne; // Return true if a valid one-hot encoding was found
}

/**
 * @brief Return the samples' data.
 *
 * @return The samples' data.
 */
const std::vector<std::vector<double>> &DataSetFid::getDatas() const {
  if (hasDatas) {
    return datas;
  } else {
    throw CommandArgumentException("Error in dataset " + datasetName + " : data file not specified for this dataset.");
  }
}

/**
 * @brief Return the classes of the samples.
 *
 * @return The classes of the samples.
 */
const std::vector<int> &DataSetFid::getClasses() const {
  if (hasClasses) {
    return trueClasses;
  } else {
    throw CommandArgumentException("Error in dataset " + datasetName + " : class file not specified for this dataset.");
  }
}

/**
 * @brief Return whether the dataset contains classes.
 *
 * @return Whether the dataset contains classes.
 */
bool DataSetFid::getHasClasses() const {
  return hasClasses;
}

/**
 * @brief Return the predictions of the samples.
 *
 * @return The predictions of the samples.
 */
const std::vector<int> &DataSetFid::getPredictions() const {
  if (hasPreds) {
    return predictions;
  } else {
    throw CommandArgumentException("Error in dataset " + datasetName + " : prediction file not specified for this dataset.");
  }
}

/**
 * @brief Return the prediction output values of the samples.
 *
 * @return The prediction output values of the samples.
 */
const std::vector<std::vector<double>> &DataSetFid::getPredictionScores() const {
  if (hasPreds) {
    return predictionScores;
  } else {
    throw CommandArgumentException("Error in dataset " + datasetName + " : prediction file not specified for this dataset.");
  }
}

/**
 * @brief Return the number of classes in the dataset.
 *
 * @return The number of classes in the dataset.
 */
int DataSetFid::getNbClasses() const {
  if (_nbClasses != -1) {
    return _nbClasses;
  } else {
    throw CommandArgumentException("Error in dataset " + datasetName + " : number of classes not specified for this dataset.");
  }
}

/**
 * @brief Return the number of attributes in the dataset.
 *
 * @return The number of attributes in the dataset.
 */
int DataSetFid::getNbAttributes() const {
  if (_nbAttributes != -1) {
    return _nbAttributes;
  } else {
    throw CommandArgumentException("Error in dataset " + datasetName + " : number of attributes not specified for this dataset.");
  }
}

/**
 * @brief Return the number of samples in the dataset.
 *
 * @return The number of samples in the dataset.
 */
int DataSetFid::getNbSamples() const {
  if (hasDatas) {
    return nbSamples;
  } else {
    throw CommandArgumentException("Error in dataset " + datasetName + " : data file not specified for this dataset.");
  }
}

/**
 * @brief Return the weights.
 *
 * @return The weights.
 */
const std::vector<std::vector<std::vector<double>>> &DataSetFid::getWeights() const {
  if (hasWeights) {
    return weights;
  } else {
    throw CommandArgumentException("Error in dataset " + datasetName + " : weight file not specified for this dataset.");
  }
}

/**
 * @brief Return the biases of the first layer.
 *
 * @return The biases of the first layer.
 */
const std::vector<double> &DataSetFid::getInputBias(int netId) const {
  if (!hasWeights) {
    throw CommandArgumentException("Error in dataset " + datasetName + " : weight file not specified for this dataset.");
  }

  if (netId < 0 || netId >= static_cast<int>(weights.size())) {
    throw CommandArgumentException("Error in dataset " + datasetName + " : invalid network index (" + std::to_string(netId) + ").");
  }

  return weights[netId][0];
}

/**
 * @brief Return the weights of the first layer.
 *
 * @return The weights of the first layer.
 */
const std::vector<double> &DataSetFid::getInputWeights(int netId) const {
  if (!hasWeights) {
    throw CommandArgumentException("Error in dataset " + datasetName + " : weight file not specified for this dataset.");
  }

  if (netId < 0 || netId >= static_cast<int>(weights.size())) {
    throw CommandArgumentException("Error in dataset " + datasetName + " : invalid network index (" + std::to_string(netId) + ").");
  }

  return weights[netId][1];
}

/**
 * @brief Return the number of training networks.
 *
 * @return The number of training networks.
 */
int DataSetFid::getNbNets() const {
  if (hasWeights) {
    return nbNets;
  } else {
    throw CommandArgumentException("Error in dataset " + datasetName + " : weight file not specified for this dataset.");
  }
}

/**
 * @brief Check for errors in the dataset.
 *
 */
void DataSetFid::checkDatas() const {
  if (hasDatas && nbSamples < 1) {
    throw FileContentError("Error in dataset " + datasetName + " : There are no data samples.");
  }
  if (hasPreds && nbPredData < 1) {
    throw FileContentError("Error in dataset " + datasetName + " : There are no predictions.");
  }
  if (hasClasses && nbClassData < 1) {
    throw FileContentError("Error in dataset " + datasetName + " : There are no classes.");
  }

  if (hasDatas && hasPreds && nbSamples != nbPredData) {
    throw FileContentError("Error in dataset " + datasetName + " : The number of prediction data does not match the number of samples.");
  }
  if (hasClasses) {
    if (hasDatas && nbSamples != nbClassData) {
      throw FileContentError("Error in dataset " + datasetName + " : The number of class data does not match the number of samples.");
    }
    if (hasPreds && nbPredData != nbClassData) {
      throw FileContentError("Error in dataset " + datasetName + " : The number of class data does not match the number of prediction data.");
    }
  }
}

/**
 * @brief Check for errors when adding a threshold.
 *
 */
void DataSetFid::checkThreshold() const {

  if (decisionThreshold != -1 && positiveClassIndex == -1) { // XOR
    throw InternalError("Error in dataset " + datasetName + " : index of positive class has to be given when decisionThreshold is given.");
  }

  if (decisionThreshold != -1 && (decisionThreshold < 0 || decisionThreshold > 1)) {
    throw CommandArgumentException("Error in dataset " + datasetName + " : the decision threshold has to be a float included in [0,1].");
  }

  if (positiveClassIndex != -1 && positiveClassIndex < 0) {
    throw CommandArgumentException("Error in dataset " + datasetName + " : the index of positive class has to be a positive integer.");
  }

  if (positiveClassIndex >= _nbClasses) {
    throw CommandArgumentException("Error in dataset " + datasetName + " : The index of positive class cannot be greater or equal to the number of classes (" + std::to_string(_nbClasses) + ").");
  }
}

/**
 * @brief Add attributes and eventually classes from attribute file in the dataset.
 *
 * @param attributesFile The attribute file name.
 * @param nbAttributes The number of attributes.
 * @param nbClasses The number of classes (optional).
 */
void DataSetFid::setAttributes(const std::string &attributesFile, int nbAttributes, int nbClasses) {
  if (hasAttributes == true) {
    throw InternalError("Error in dataset " + datasetName + " : attributes have been given two times.");
  }

  // Rollback changes of names/metadata on error.
  const size_t attributeNamesSizeBefore = attributeNames.size();
  const size_t classNamesSizeBefore = classNames.size();
  const int nbAttributesBefore = _nbAttributes;
  const int nbClassesBefore = _nbClasses;
  const bool hasClassNamesBefore = hasClassNames;

  try {
    if (nbClasses != -1) {
      setNbClassAndAttr(nbClasses, nbAttributes);
    } else {
      if (_nbAttributes != -1) {
        if (_nbAttributes != nbAttributes) {
          throw InternalError("Error in dataset " + datasetName + " : the given number of attributes (" + std::to_string(nbAttributes) + ") doesn't match the number given before (" + std::to_string(_nbAttributes) + ").");
        }
      } else {
        if (nbAttributes > 0) {
          _nbAttributes = nbAttributes;
        } else {
          throw InternalError("Error in dataset " + datasetName + " : the number of attributes has to be a strictly positive integer.");
        }
      }
    }

    // Get attributes
    std::ifstream fileAttr;
    std::string line;

    fileAttr.open(attributesFile, std::ios::in); // Read attribute file
    if (fileAttr.fail()) {
      throw FileNotFoundError("Error in dataset " + datasetName + " : file " + attributesFile + " not found.");
    }
    while (std::getline(fileAttr, line)) {
      // Remove invisible characters at the end if exist
      for (int i = static_cast<int>(line.length()) - 1; i >= 0; i--) {
        if (!std::isspace(static_cast<unsigned char>(line[i]))) {
          line.erase(i + 1);
          break;
        }
      }
      if (!checkStringEmpty(line)) {
        if (hasSpaceBetweenWords(line)) {
          throw FileContentError("Error in dataset " + datasetName + ", in file " + attributesFile + " : attribute " + line + " has spaces inbetween. Maybe replace it with an underscore.");
        }
        attributeNames.push_back(line);
      }
    }

    hasClassNames = false;

    if (attributeNames.size() < static_cast<size_t>(_nbAttributes)) { // Not enough attributes
      throw FileContentError("Error in dataset " + datasetName + " : in file " + attributesFile + ", there is not enough attribute names.");
    } else if (attributeNames.size() == static_cast<size_t>(_nbAttributes)) { // Exact correlation, so there are no class names.
      hasClassNames = false;
    } else {
      if (_nbClasses != -1) {
        if (attributeNames.size() != static_cast<size_t>(_nbAttributes + _nbClasses)) { // No correlation with the number of attributes and classes
          throw FileContentError("Error in dataset " + datasetName + " : in file " + attributesFile + ", there is not the correct amount of attribute and class names.");
        } else { // There are attributes and classes
          hasClassNames = true;
          auto firstEl = attributeNames.end() - _nbClasses;
          auto lastEl = attributeNames.end();
          classNames.insert(classNames.end(), firstEl, lastEl);
          attributeNames.erase(firstEl, lastEl);
        }
      } else { // No class specified and too many attributes
        throw FileContentError("Error in dataset " + datasetName + " : in file " + attributesFile + ", there are too many attribute names (no class has been specified, if you have classNames, add the number of classes).");
      }
    }

    fileAttr.close(); // close file
    hasAttributes = true;
  } catch (...) {
    attributeNames.resize(attributeNamesSizeBefore);
    classNames.resize(classNamesSizeBefore);
    _nbAttributes = nbAttributesBefore;
    _nbClasses = nbClassesBefore;
    hasClassNames = hasClassNamesBefore;
    throw;
  }
}

/**
 * @brief Return attribute names.
 *
 * @return Attribute names.
 */
const std::vector<std::string> &DataSetFid::getAttributeNames() const {
  if (hasAttributes) {
    return attributeNames;
  } else {
    throw CommandArgumentException("Error in dataset " + datasetName + " : attribute file not specified for this dataset.");
  }
}

/**
 * @brief Return class names.
 *
 * @return Class names.
 */
const std::vector<std::string> &DataSetFid::getClassNames() const {
  if (hasClassNames) {
    return classNames;
  } else {
    throw CommandArgumentException("Error in dataset " + datasetName + " : classNames not present in attribute file or attribute file not specified.");
  }
}

/**
 * @brief Return whether the dataset contains attribute names.
 *
 * @return Whether the dataset contains attribute names.
 */
bool DataSetFid::getHasAttributeNames() const {
  return hasAttributes;
}

/**
 * @brief Return whether the dataset contains class names.
 *
 * @return Whether the dataset contains class names.
 */
bool DataSetFid::getHasClassNames() const {
  return hasClassNames;
}
