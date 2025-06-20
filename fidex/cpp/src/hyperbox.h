#ifndef HYPERBOX_H
#define HYPERBOX_H

#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

/**
 * @brief Represents a hyperbox composed of discriminative hyperplanes in the feature space used to construct an explaining rule for a sample of interest, discriminating between different classes of samples.
 *
 * This class encapsulates the attributes and methods needed to handle hyperboxes
 * which include discriminative hyperplanes and the samples they cover.
 */
class Hyperbox {
  std::vector<std::pair<int, int>> discriminativeHyperplans; ///< Vector of discriminative hyperplanes represented by a dimension (attribute index) and a hyperplane value in this dimension.
  std::vector<int> coveredSamples;                           ///< Vector of sample IDs covered by the hyperbox.
  std::vector<int> coveringSizesWithNewAntecedent;           ///< Vector of the number of samples covered by the rule for each new antecedent in the rule (with 1, 2, 3, ... antecedents).
  std::vector<double> increasedFidelity;                     ///< Vector of the increased fidelity for each new antecedent.
  std::vector<double> accuracyChanges;                       ///< Vector of the accuracy changes for each new antecedent.
  double fidelity = -1;                                      ///< Fidelity of the samples covered by the hyperbox with respect to the model's prediction.

public:
  /**
   * @brief Default constructor for Hyperbox.
   */
  Hyperbox();

  /**
   * @brief Constructs a Hyperbox object with specified discriminative hyperplanes.
   */
  explicit Hyperbox(const std::vector<std::pair<int, int>> &m_discriminativeHyperplans);

  /**
   * @brief Sets the covered samples of the hyperbox.
   */
  void setCoveredSamples(const std::vector<int> &m_coveredSamples);

  /**
   * @brief Gets the discriminative hyperplanes of the hyperbox.
   */
  std::vector<std::pair<int, int>> getDiscriminativeHyperplans() const;

  /**
   * @brief Resets the discriminative hyperplanes of the hyperbox.
   */
  void resetDiscriminativeHyperplans();

  /**
   * @brief Computes the new covered samples with a new discriminative hyperplane.
   */
  void computeCoveredSamples(const std::vector<int> &ancienCoveredSamples, int attribut, std::vector<std::vector<double>> &trainData, bool mainSampleGreater, double hypValue);

  /**
   * @brief Computes the fidelity of the samples covered by the hyperbox with respect to the model's prediction.
   */
  void computeFidelity(const int mainsamplePred, std::vector<int> &trainPreds);

  /**
   * @brief Gets the fidelity of the samples covered by the hyperbox.
   */
  double getFidelity() const;

  /**
   * @brief Sets the fidelity of the samples covered by the hyperbox.
   */
  void setFidelity(double x);

  /**
   * @brief Gets the increased fidelity for each new antecedent.
   */
  std::vector<double> getIncreasedFidelity() const;

  /**
   * @brief Adds in the list the increased fidelity with the new antecedent.
   */
  void addIncreasedFidelity(double incrFidelity);

  /**
   * @brief Reset the increased fidelity for each new antecedent.
   */
  void resetIncreasedFidelity();

  /**
   * @brief Gets the accuracy changes for each new antecedent.
   */
  std::vector<double> getAccuracyChanges() const;

  /**
   * @brief Adds in the list the accuracy changes with the new antecedent.
   */
  void addAccuracyChanges(double incrFidelity);

  /**
   * @brief Reset the accuracy changes for each new antecedent.
   */
  void resetAccuracyChanges();

  /**
   * @brief Gets the covered samples of the hyperbox.
   */
  std::vector<int> getCoveredSamples() const;

  /**
   * @brief Gets the covering sizes for each new antecedant(discriminative hypaerplan) in the hyperbox.
   */
  std::vector<int> getCoveringSizesWithNewAntecedent() const;

  /**
   * @brief Adds the new covering size of the rule with the new antecedant in the list.
   */
  void addCoveringSizesWithNewAntecedent(int covSize);

  /**
   * @brief Reset the covering sizes for each new antecedant(discriminative hypaerplan) in the hyperbox.
   */
  void resetCoveringSizesWithNewAntecedent();

  /**
   * @brief Adds a new discriminative hyperplane to the hyperbox.
   */
  void discriminateHyperplan(int dimVal, int hypVal);
};

#endif // HYPERBOX_H
