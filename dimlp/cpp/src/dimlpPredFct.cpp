#include "dimlpPredFct.h"
#include "../../../common/cpp/src/checkFun.h"
#include "../../../common/cpp/src/parameters.h"
#include "../../../common/cpp/src/scopedCoutFileRedirect.h"
#include "dimlp.h"
#include "dimlpCommonFun.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

////////////////////////////////////////////////////////////

namespace {
struct OwnedDataSetDeleter {
  void operator()(DataSet *dataset) const {
    if (dataset != nullptr) {
      dataset->Del();
      delete dataset;
    }
  }
};

using OwnedDataSetPtr = std::unique_ptr<DataSet, OwnedDataSetDeleter>;

template <typename... Args>
OwnedDataSetPtr makeOwnedDataSet(Args &&...args) {
  return OwnedDataSetPtr(new DataSet(std::forward<Args>(args)...));
}
} // namespace

/**
 * @brief Displays the parameters for dimlpPred.
 */
void showDimlpPredParams()

{
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
  printOptionDescription("--weights_file <str>", "Path to the file containing the weights of the model trained with dimlpTrn");
  printOptionDescription("--hidden_layers_file <str>", "Path to the file containing hidden layers sizes");
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
  printOptionDescription("--test_pred_outfile <str>", "Path to the file where the test predictions will be stored (default: dimlpTest.out)");
  printOptionDescription("--console_file <str>", "Path to the file where the terminal output will be redirected. If not specified, all output will be shown on your terminal");
  printOptionDescription("--nb_quant_levels <int [3,inf[>", "Number of stairs in the staircase activation function (default: 50)");

  std::cout << std::endl
            << "----------------------------" << std::endl
            << std::endl;
  std::cout << "Execution example :" << std::endl
            << std::endl;
  std::cout << "dimlp.dimlpPred(\"--test_data_file test_data.txt --weights_file weights.wts --nb_attributes 16 --hidden_layers_file hidden_layers.out --nb_classes 2 --test_pred_outfile predTest.out --root_folder dimlp/datafiles\")" << std::endl
            << std::endl;
  std::cout << "---------------------------------------------------------------------" << std::endl
            << std::endl;
}

////////////////////////////////////////////////////////////

/**
 * @brief Sets default hyperparameters and checks the logic and validity of the parameters of dimlpPred.
 *
 * @param p Reference to the Parameters object containing all hyperparameters.
 */
void checkDimlpPredParametersLogicValues(Parameters &p) {
  // setting default values
  p.setDefaultNbQuantLevels();
  p.setDefaultString(TEST_PRED_OUTFILE, "dimlpTest.out", true);

  // this sections check if values comply with program logic

  // asserting mandatory parameters
  p.assertIntExists(NB_ATTRIBUTES);
  p.assertIntExists(NB_CLASSES);
  p.assertStringExists(TEST_DATA_FILE);
  p.assertStringExists(WEIGHTS_FILE);
  p.assertStringExists(HIDDEN_LAYERS_FILE);

  // verifying logic between parameters, values range and so on...
  p.checkParametersCommon();
}

/**
 * @brief Executes the Dimlp prediction process on test set with specified parameters for a model trained with dimlpTrn.
 *
 * The function performs the following steps:
 * 1. Parses the command string to extract parameters.
 * 2. Sets up the neural network and other necessary objects.
 * 3. Loads the weights from the provided weights file.
 * 4. Performs predictions on the provided test dataset.
 * 5. Saves the network's predictions for the test dataset.
 * 6. Saves the total execution time.
 *
 * Notes:
 * - Each file is located with respect to the root folder dimlpfidex or to the content of the 'root_folder' parameter if specified.
 * - It's mandatory to specify the number of attributes and classes in the data, as well as the test dataset and weights file.
 * - The hidden layers configuration file must also be provided to specify the network architecture.
 * - Parameters can be defined directly via the command line or through a JSON configuration file.
 * - Providing no command-line arguments or using <tt>-h/-\-help</tt> displays usage instructions, detailing both required and optional parameters for user guidance.
 *
 * Outputs:
 * - test_pred_outfile: File containing the model's test predictions.
 * - console_file: If specified, contains the console output.
 *
 * File formats:
 * - **Data files**: These files should contain one sample per line, with numbers separated either by spaces, tabs, semicolons, or commas. Supported formats:
 *   1. Only attributes (floats).
 *   2. Attributes (floats) followed by an integer class ID.
 *   3. Attributes (floats) followed by one-hot encoded class.
 * - **Weights file**: This file should be obtained by training with DimlpTrn and not with DimlpBT(!).
 *   The first row represents bias values of the Dimlp layer and the second row are values of the weight matrix between the previous layer and the Dimlp layer.
 *   Each value is separated by a space. As an example, if the layers are of size 4, the biases are: b1 b2 b3 b4 and the weights are w1 w2 w3 w4.
 * - **Hidden layers file**: This file contains the number of nodes in each hidden layer. In each line there is the layer id and the number of its nodes separated by a space. Ex :<br>
 *   1 16<br>
 *   2 5
 *
 * Example of how to call the function:
 * @par
 * <tt>from dimlpfidex import dimlp</tt>
 * @par
 * <tt>dimlp.dimlpPred('-\-test_data_file test_data.txt -\-weights_file weights.wts -\-nb_attributes 16 -\-hidden_layers_file hidden_layers.out -\-nb_classes 2 -\-test_pred_outfile predTest.out -\-root_folder dimlp/datafiles')</tt>
 *
 * @param command A single string containing either the path to a JSON configuration file with all specified arguments, or all arguments for the function formatted like command-line input. This includes file paths and options for output.
 * @return Returns 0 for successful execution, -1 for errors encountered during the process.
 */
int dimlpPred(const std::string &command) {
  try {

    double temps;
    const auto t1 = std::chrono::steady_clock::now();

    // Parsing the command
    std::vector<std::string> commandList = {"dimlpPred"};
    std::string s;
    std::stringstream ss(command);

    while (ss >> s) {
      commandList.push_back(s);
    }

    size_t nbParam = commandList.size();
    if (nbParam < 2 || commandList[1] == "-h" || commandList[1] == "--help") {
      showDimlpPredParams();
      return 0;
    }

    // Import parameters
    std::unique_ptr<Parameters> params;
    static const std::vector<ParameterCode> validParams = {TEST_DATA_FILE, WEIGHTS_FILE, NB_ATTRIBUTES, NB_CLASSES, ROOT_FOLDER,
                                                           TEST_PRED_OUTFILE, CONSOLE_FILE, HIDDEN_LAYERS_FILE, NB_QUANT_LEVELS};
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
    checkDimlpPredParametersLogicValues(*params);

    std::unique_ptr<ScopedCoutFileRedirect> coutRedirect;
    if (params->isStringSet(CONSOLE_FILE)) {
      coutRedirect = std::unique_ptr<ScopedCoutFileRedirect>(new ScopedCoutFileRedirect(params->getString(CONSOLE_FILE)));
    }

    // Show chosen parameters
    std::cout << *params;

    // ----------------------------------------------------------------------

    // Get parameters values

    int nbIn = params->getInt(NB_ATTRIBUTES);
    int nbOut = params->getInt(NB_CLASSES);
    std::string testFile = params->getString(TEST_DATA_FILE);
    std::string weightFile = params->getString(WEIGHTS_FILE);
    std::string predFile = params->getString(TEST_PRED_OUTFILE);
    int quant = params->getInt(NB_QUANT_LEVELS);

    const int nbNetworks = countNetworksInFile(weightFile);
    if (nbNetworks != 1) {
      throw FileContentError("Error : " + weightFile + " must contain exactly one network for dimlpPred.");
    }

    OwnedDataSetPtr Test = makeOwnedDataSet();
    int nbLayers;
    int nbWeightLayers;
    std::vector<int> vecNbNeurons;
    StringInt arch;
    StringInt archInd;
    params->readHiddenLayersFile(arch, archInd);

    Test = makeOwnedDataSet(testFile, nbIn, nbOut);

    // ----------------------------------------------------------------------

    if (arch.GetNbEl() == 0) {
      nbLayers = 3;
      nbWeightLayers = nbLayers - 1;

      vecNbNeurons.assign(nbLayers, 0);
      vecNbNeurons[0] = nbIn;
      vecNbNeurons[1] = nbIn;
      vecNbNeurons[2] = nbOut;
    }

    else {
      archInd.GoToBeg();

      if (archInd.GetVal() == 1) {
        arch.GoToBeg();

        if (arch.GetVal() % nbIn != 0) {
          throw InternalError("The number of neurons in the first hidden layer must be a multiple of the number of input neurons.");
        }

        nbLayers = arch.GetNbEl() + 2;
        nbWeightLayers = nbLayers - 1;

        vecNbNeurons.assign(nbLayers, 0);
        vecNbNeurons[0] = nbIn;
        vecNbNeurons[nbLayers - 1] = nbOut;

        arch.GoToBeg();
        for (int p = 1; p <= arch.GetNbEl(); p++, arch.GoToNext()) {
          vecNbNeurons[p] = arch.GetVal();

          if (vecNbNeurons[p] == 0) {
            throw InternalError("The number of neurons must be greater than 0.");
          }
        }
      }

      else {
        nbLayers = arch.GetNbEl() + 3;
        nbWeightLayers = nbLayers - 1;

        vecNbNeurons.assign(nbLayers, 0);
        vecNbNeurons[0] = nbIn;
        vecNbNeurons[1] = nbIn;
        vecNbNeurons[nbLayers - 1] = nbOut;
        arch.GoToBeg();
        for (int p = 1; p <= arch.GetNbEl(); p++, arch.GoToNext()) {
          vecNbNeurons[p + 1] = arch.GetVal();

          if (vecNbNeurons[p + 1] == 0) {
            throw InternalError("The number of neurons must be greater than 0.");
          }
        }
      }
    }

    // ----------------------------------------------------------------------

    auto net = std::make_shared<Dimlp>(weightFile, nbLayers, vecNbNeurons, quant);

    SaveOutputs(*Test, net, nbOut, nbWeightLayers, predFile);

    std::cout << "\n-------------------------------------------------\n"
              << std::endl;

    const auto t2 = std::chrono::steady_clock::now();
    temps = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "\nFull execution time = " << temps << " sec" << std::endl;

    BpNN::resetInitRandomGen();

  } catch (const ErrorHandler &e) {
    std::cerr << e.what() << std::endl;
    return -1;
  } catch (const std::exception &e) {
    std::cerr << "Unhandled standard exception in dimlpPred: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cerr << "Unhandled unknown exception in dimlpPred." << std::endl;
    return -1;
  }
  return 0;
}
