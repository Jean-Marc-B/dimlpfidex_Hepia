#ifndef DATASETFID_H
#define DATASETFID_H

#include <iosfwd>
#include <string>
#include <vector>

/**
 * @brief The DataSetFid class handles the loading, parsing, and storage of datasets including data, predictions, and classes.
 */
class DataSetFid {

private:
  std::string datasetName;                               ///< Name of the dataset.
  std::vector<std::vector<double>> datas;                ///< Data samples.
  std::vector<int> trueClasses;                          ///< True class labels for the data samples.
  std::vector<int> predictions;                          ///< Predictions for the data samples.
  std::vector<std::vector<double>> predictionScores;     ///< Raw prediction scores for the data samples.
  std::vector<std::vector<std::vector<double>>> weights; ///< Weights for the dataset.
  bool hasDatas = false;                                 ///< Flag indicating if data is loaded.
  bool hasPreds = false;                                 ///< Flag indicating if predictions are loaded.
  bool hasClasses = false;                               ///< Flag indicating if classes are loaded.
  bool hasWeights = false;                               ///< Flag indicating if weights are loaded.

  enum class ClassFormat {
    Unknown,
    OneHot,
    Id,
    OneHotCombined,
    IdCombined
  };
  ClassFormat classFormat = ClassFormat::Unknown; ///< Format of the class representation.

  double decisionThreshold = -1; ///< Decision threshold for classifying samples.
  int positiveClassIndex = -1;   ///< Index of the positive class used for threshold-based classification.

  int _nbClasses = -1;    ///< Number of classes in the dataset.
  int _nbAttributes = -1; ///< Number of attributes in the dataset.
  int nbSamples = -1;     ///< Number of samples in the dataset.
  int nbClassData = -1;   ///< Number of class data entries.
  int nbPredData = -1;    ///< Number of prediction data entries.
  int nbNets = 1;         ///< Number of networks used in training found in the weight file.

  std::vector<std::string> attributeNames; ///< Names of the attributes.
  std::vector<std::string> classNames;     ///< Names of the classes.
  bool hasAttributes = false;              ///< Flag indicating if attribute names are loaded.
  bool hasClassNames = false;              ///< Flag indicating if class names are loaded.

  /**
   * @brief Read a data line from a data file and save it in datas and trueClasses if it contains class information.
   */
  void setDataLine(const std::string &line, const std::string &dataFile);

  /**
   * @brief Read a prediction line from the prediction file and save it in predictions.
   */
  void setPredLine(const std::string &line, const std::string &dataFile);

  /**
   * @brief Read a class line from the class file and save it in trueClasses.
   */
  void setClassLine(const std::string &line, const std::string &dataFile);

  /**
   * @brief Parses a weight file containing a single network's weights and stores them in the weights vector.
   *
   * @param fileWts Reference to opened weight input stream.
   * @param weightFile Weight filename used for detailed error messages.
   */
  void parseSingleNetwork(std::istream &fileWts, const std::string &weightFile);

  /**
   * @brief Parses a weight file containing multiple networks' weights and stores them in the weights vector.
   *
   * @param fileWts Reference to opened weight input stream.
   * @param weightFile Weight filename used for detailed error messages.
   */
  void parseMultipleNetworks(std::istream &fileWts, const std::string &weightFile);

  /**
   * @brief Set the number of classes for the dataset.
   */
  void setNbClass(int nbClasses);

  /**
   * @brief Set the number of attributes and classes for the dataset.
   */
  void setNbClassAndAttr(int nbClasses, int nbAttributes);

  /**
   * @brief Checks if a portion of a std::vector<double> follows a one-hot encoding scheme.
   */
  bool isOneHot(const std::vector<double> &values, int start, int &oneHotIndex) const;

  /**
   * @brief Check for errors in the dataset.
   */
  void checkDatas() const;

  /**
   * @brief Check for errors when adding a threshold.
   */
  void checkThreshold() const;

  //////////////////////////////////////////////////////

public:
  /**
   * @brief Construct a new DataSetFid object with a name.
   */
  explicit DataSetFid(const std::string &name) : datasetName(name){};

  /**
   * @brief Construct a new DataSetFid object using separate data, prediction, and optional class files.
   */
  DataSetFid(const std::string &name, const std::string &dataFile, const std::string &predFile, int nbAttributes, int nbClasses, double decisionThresh = -1, int positiveClassId = -1, const std::string &trueClassFile = "");

  /**
   * @brief Construct a new DataSetFid object using a single data file containing data, predictions, and optionally classes.
   */
  DataSetFid(const std::string &name, const std::string &dataFile, int nbAttributes, int nbClasses, double decisionThresh = -1, int positiveClassId = -1); // dataFile with data, predictions and maybe classes

  /**
   * @brief Construct a new DataSetFid object using a weight file.
   */
  explicit DataSetFid(const std::string &name, const std::string &weightFile);

  /**
   * @brief Get data from dataFile and save it in datas and trueClasses if it contains class information.
   */
  void setDataFromFile(const std::string &dataFile, int nbAttributes, int nbClasses);

  /**
   * @brief Set the Weights object by reading and parseing a file
   */
  void setWeights(const std::string &weightFile);

  /**
   * @brief Add predictions to the dataset using a prediction file.
   */
  void setPredFromFile(const std::string &predFile, int nbClasses, double decisionThreshold = -1, int positiveClassId = -1);

  /**
   * @brief Add classes from a specified file into the dataset.
   */
  void setClassFromFile(const std::string &classFile, int nbClasses);

  /**
   * @brief Return the samples' data.
   */
  const std::vector<std::vector<double>> &getDatas() const;

  /**
   * @brief Return the classes of the samples.
   */
  const std::vector<int> &getClasses() const;

  /**
   * @brief Return whether the dataset contains classes.
   */
  bool getHasClasses() const;

  /**
   * @brief Return the predictions of the samples.
   */
  const std::vector<int> &getPredictions() const;

  /**
   * @brief Return the prediction output values of the samples.
   */
  const std::vector<std::vector<double>> &getPredictionScores() const;

  /**
   * @brief Return the number of classes in the dataset.
   */
  int getNbClasses() const;

  /**
   * @brief Return the number of attributes in the dataset.
   */
  int getNbAttributes() const;

  /**
   * @brief Return the number of samples in the dataset.
   */
  int getNbSamples() const;

  /**
   * @brief Add attributes and eventually classes from attribute file in the dataset.
   */
  void setAttributes(const std::string &attributesFile, int nbAttributes, int nbClasses = -1);

  /**
   * @brief Return attribute names.
   */
  const std::vector<std::string> &getAttributeNames() const;

  /**
   * @brief Return class names.
   */
  const std::vector<std::string> &getClassNames() const;

  /**
   * @brief Return whether the dataset contains attribute names.
   */
  bool getHasAttributeNames() const;

  /**
   * @brief Return whether the dataset contains class names.
   */
  bool getHasClassNames() const;

  /**
   * @brief Return the weights.
   */
  const std::vector<std::vector<std::vector<double>>> &getWeights() const;

  /**
   * @brief Return the biases of the first layer.
   */
  const std::vector<double> &getInputBias(int netId) const;

  /**
   * @brief Return the weights of the first layer.
   */
  const std::vector<double> &getInputWeights(int netId) const;

  /**
   * @brief Return the number of training networks.
   */
  int getNbNets() const;
};

#endif // DATASETFID_H
