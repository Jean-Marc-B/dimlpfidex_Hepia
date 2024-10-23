# FROM QUENTIN LEBLANC
# https://githepia.hesge.ch/quentin.leblanc/pre-act/-/blob/main/src/utils/data_helper.py

import re
import os
from collections import defaultdict
from typing import Optional, Any
from typing import Union

import pandas as pd
import numpy as np


def read_file(file: Union[str, Any], sep: str = ',') -> pd.DataFrame:
    """
    Read the file precised and put the Subject ID as index.
    :param sep: separator for the CSV or TSV file
    :param file: path to file to open and read or bytes of the file. Must be a CSV file.
    :return: a DataFrame with the data in the file
    """
    data = pd.read_csv(file, sep=sep)
    data = data.set_index('Subject ID', drop=True)
    # Remove index columns from CSV
    return data.loc[:, ~data.columns.str.contains("^Unnamed")]


def get_labels(data: pd.DataFrame, *columns_names: str) -> np.array:
    """
    Extract the labels from the data in the DataReader and return them as a numpy array
    Labels have values 0 for NEGATIVE class, 1 for POSITIVE class and 2 for NaN
    :param data: DataFrame from which to extract labels
    :param columns_names: column names for the labels
    :return: a numpy array of shape (n_data, 1)
    """
    res = data[columns_names[0]].copy(deep=True)
    for c in columns_names[1:]:
        res = pd.concat([res, data[c].copy(deep=True)], axis=1)

    # # aggregate columns to get one column that says NEGATIVE = 0 // POSITIVE = 1 // NaN = 2
    # # if NaN are in all columns of labels that means the sum is 2*nb_columns -> replace value by 2
    # # Else if at least a 1 is in one of the columns, that counts as a positive -> value will be 1
    # # Else 0
    # def __get_label(x: pd.Series):
    #     acc: int = 2
    #     for c in x.values:
    #         if (acc == c and acc == 2) or c == 2:
    #             pass
    #         elif c == 0 and acc != 1:
    #             acc = 0
    #         else:
    #             acc = 1
    #     return acc
    def __get_label(x: pd.Series) -> int:
        # counting in trits
        # each 12M/24M/36 represents a 3 trits number
        # refers to (this table)[../../Ressources/trits.png] to understand the numbers below
        # 0 0 0 = 0 / 0 1 0 = 3 / 1 0 0 = 9
        negs = [0, 2, 6]
        nans = [8, 18, 20, 24, 26]
        case = int("".join(map(str, x.values.astype(np.uint8).tolist())), 3)
        if case in negs:
            return 0
        elif case in nans:
            return 2
        return 1

    res = res.replace(np.nan, 2.0)
    if len(res.shape) != 1:
        res = res.apply(lambda x: __get_label(x), axis=1)

    res.rename('label', inplace=True)
    return res


def remove_attributes_with_keyword(data: pd.DataFrame, *keywords: str) -> pd.DataFrame:
    """
    Remove all columns in data with keywords passed int their names
    :param data: DataFrame
    :param keywords: one or more keywords
    :return: A new DataFrame with the Keywords removed
    """
    res = data.copy(deep=True)
    for k in keywords:
        res = res.loc[:, ~res.columns.str.contains(k)]
    return res


def one_hot_encoding(data: pd.DataFrame, dummy_na: bool) -> pd.DataFrame:
    """
    One hot encode categorical attributes contained in data
    :param data: DataFrame with all the data to encode
    :param dummy_na: Add a column to indicate NaNs, if False NaNs are ignored.
    :return: A new DataFrame with encoded values
    """
    # one hot encoding data
    res = data.copy(deep=True)

    str_columns = res.select_dtypes(include=object)
    for c in str_columns:
        str_attribute = res[c]
        res = res.drop(c, axis=1)
        str_attribute = pd.get_dummies(str_attribute, prefix=c, dtype=np.float64, dummy_na=dummy_na)
        res = pd.concat([res, str_attribute], axis=1)

    return res


def one_hot_encoding_ext(data: pd.DataFrame, dummy_na: bool, *columns: str) -> pd.DataFrame:
    """
    One hot encode categorical attributes contained in data
    :param data: DataFrame with all the data to encode
    :param dummy_na: Add a column to indicate NaNs, if False NaNs are ignored.
    :return: A new DataFrame with encoded values
    """

    def __int_to_str(x: Any) -> str:
        if x == 1:
            return "YES"
        elif x == 0:
            return "NO"
        else:
            return str(x)

    # one hot encoding data
    res = data.copy(deep=True)

    for c in columns:
        if c in res.columns:
            attribute = res[c]
            res = res.drop(c, axis=1)
            attribute = attribute.map(lambda x: __int_to_str(x))
            attribute = pd.get_dummies(attribute, prefix=c, dtype=np.float64)
            res = pd.concat([res, attribute], axis=1)

    return res


def infer_missing_values(data: pd.DataFrame, method: str = 'median', value: Optional[float] = None) -> pd.DataFrame:
    """
    Replace NaN vales in the DataFrame with values from chosen method.
    :param data: explicit
    :param method: either 'mean', 'median', 'value'. If value, only numerical values are compatible. Default is median.
    :param value: Optional if value is chosen.
    :return: pd.DataFrame
    """
    res = data.copy(deep=True)
    if method == "median":
        for c in data:
            res.loc[:, c] = res.loc[:, c].replace(np.nan, data[c].median())
        return res

    if method == "mean":
        for c in data:
            res.loc[:, c] = res.loc[:, c].replace(np.nan, data[c].mean())
        return res

    if method == "value":
        if value is None:
            raise Exception("A value must be passed to infer missing value")
        if type(value) != float and type(value) != int:
            raise Exception(f"Infered value must be of type int or float, passed value was of type {type(value)}")
        for c in data:
            res.loc[:, c] = res.loc[:, c].replace(np.nan, value)
        return res

    raise Exception("Method must be one of 'median', 'mean', value'")


def __obtain_clinical_data(file_path: str, training: bool=True) -> tuple[pd.DataFrame, np.array]:
    """
    Get the data and apply some tranformations to be usable for experiences
    Read the data
    Extract label 12/24/36m Arm Lymphedema
    Remove useless columns
    One hot encode categorical data
    :param file_path: path to the file to read
    :return: a pd.DataFrame with the data and a np.array with the labels.
    """
    data = read_file(file_path)
    label_names = ['12m follow-up Arm Lymphedema', '24m follow-up Arm Lymphedema', '36m follow-up Arm Lymphedema']
    label_lymph = get_labels(data, *label_names)

    keywords = ["follow-up", "DB", "Post-RT", "Date", 'Baseline Telangiectasia', 'Baseline Edema',
                'Baseline SkinInduration', 'Baseline Erythema', 'Baseline SkinHyperpigment', 'Mean Heart Dose']

    attributes = remove_attributes_with_keyword(data, *keywords)
    # Surgery is a special case since some attributes have surgery in their names we have to delete only this column
    attributes.drop('Surgery', inplace=True, axis=1)
    attributes = one_hot_encoding(attributes, dummy_na=True)
    categorical_cols = ['Diabeties', 'IMRT', '3D', 'Boost', 'neoadjuvant chemotherapy', 'adjuvant chemotherapy',
                        'Baseline Arm Lymphedema']
    attributes = one_hot_encoding_ext(attributes, True, *categorical_cols)
    attributes = infer_missing_values(attributes)
    label_lymph.rename("label", inplace=True)

    # Management of T1-4
    def __manage_T_retroactively(attributes: pd.DataFrame) -> pd.DataFrame:
        # Transforming to imply that T4a, T4b, T4d <=> T4
        # Transforming to imply that T4 -> T3 -> T2 -> T1
        t4 = attributes.filter(regex=r"^Clinical T-Stage_T4[a-d]?$")
        ts = attributes.filter(regex=r"^Clinical T-Stage_T[1-3]$")
        t4 = np.add.reduce(t4, axis=1)
        t4.rename("Clinical T-Stage_T4", inplace=True)
        ts = pd.concat([ts, t4], axis=1)
        attributes = attributes.drop(ts.columns, axis=1)

        ts = ts.sort_index(axis=1, ascending=False)
        new_ts = []
        for c in ts.columns[:-1]:
            new_ts.append(ts[c])
            ts = ts.drop(c, axis=1)
            ts = ts.add(new_ts[-1], axis=0)

        ts = pd.concat([*new_ts, ts], axis=1).apply(lambda x: x.apply(lambda y: 1 if y >= 1 else 0), axis=1)

        attributes = pd.concat([attributes, ts], axis=1)

        # Tis are T0s
        tis = attributes["Clinical T-Stage_Tis"]
        t0s = attributes["Clinical T-Stage_T0"].add(tis, axis=0).apply(lambda x: 1 if x >=1 else 0).rename("Clinical T-Stage_T0")
        attributes = attributes.drop(["Clinical T-Stage_T0"], axis=1)
        attributes = pd.concat([attributes, t0s], axis=1)

        # T+ (T1->T4/T4a,b,c,d) can't be T0
        t_plus = attributes.filter(regex=r"^Clinical T-Stage_T[1-4][a-d]?$")
        t_plus = np.add.reduce(t_plus, axis=1).rename("Clinical T-Stage_T+")
        t0s = attributes["Clinical T-Stage_T0"]
        ts = pd.concat([t0s, t_plus], axis=1)
        t0s = ts.apply(lambda x: 1 if x["Clinical T-Stage_T+"] == 0 and x["Clinical T-Stage_T0"] == 1 else 0, axis=1).rename("Clinical T-Stage_T0")
        attributes = attributes.drop(["Clinical T-Stage_T0"], axis=1)
        attributes = pd.concat([attributes, t0s], axis=1)

        return attributes.sort_index(axis=1, ascending=True)

    attributes = __manage_T_retroactively(attributes)

    def __manage_N_retroactively(attributes: pd.DataFrame) -> pd.DataFrame:
        ns = attributes.filter(regex=r"^Clinical N-Stage_N[0-3]$")
        attributes = attributes.drop(ns.columns, axis=1)

        ns = ns.sort_index(axis=1, ascending=False)
        new_ns = []
        for c in ns.columns[:-1]:
            new_ns.append(ns[c])
            ns = ns.drop(c, axis=1)
            ns = ns.add(new_ns[-1], axis=0)

        ns = pd.concat([*new_ns, ns], axis=1)
        ns = ns.apply(lambda x: x.apply(lambda y: 1 if y >= 1 else 0), axis=1)

        attributes = pd.concat([attributes, ns], axis=1)
        return attributes.sort_index(axis=1, ascending=True)

    attributes = __manage_N_retroactively(attributes)

    if not training:
        # clinical tests case
        # we have to rename some columns, add some more to correspond to training data
        # CF. ressources and file "precautionsDeCodage.pdf" for further information
        pass

    return attributes, label_lymph


def __obtain_file_with_ids(file_with_ids: str) -> pd.DataFrame:
    """
    Link genomic data to clinical data and labels
    :param clinical_file: path to clinical data file
    :param genomic_file: path to genomic data file
    :return: A tuple Containing clinical data in pd.DataFrame, genomic Data in pd.DataFrame and labels in np.array
    """
    genom_data = read_file(file_with_ids, sep='\t')
    genom_data = infer_missing_values(genom_data, method='value', value=-1.0)
    return genom_data


def obtain_data(clinical_file: str, file_with_ids: Optional[str] = None) -> Any:
    """
    Return data contained in files passed in argument.
    Must be either clinical data or clinical data and genomic data.
    :param clinical_file: path to clinical data file
    :param file_with_ids: path to data file containing patient ids and data related to them
    :return: Clinical data and Labels or Clinical data, Genomic data and Labels
    """
    if not file_with_ids:
        return __obtain_clinical_data(clinical_file)

    clinical_data, labels = __obtain_clinical_data(clinical_file)
    clinical_columns = clinical_data.columns
    ids_data = __obtain_file_with_ids(file_with_ids)
    ids_column = ids_data.columns
    labels = pd.DataFrame(labels, index=clinical_data.index, columns=['label'])
    # might be a bug in other experience files
    res = pd.concat([ids_data, clinical_data, labels], axis=1, join="outer")

    # remove indexes that does not exist in labels
    labels = res['label'].dropna()

    res_clinical = res[clinical_columns].loc[res[clinical_columns].index.isin(labels.index)]
    res_ids = res[ids_column].loc[res[ids_column].index.isin(labels.index)]

    return res_clinical, res_ids, labels


def separate_data_by_db(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Separate Data by their database of origin using the subject ID.
    DataBase are in this order : Requite, Canto and HypoG
    :param data: pd.DataFrame
    :return: a DataFrame per database (Requite, Canto, HypoG)
    """
    requite = data.filter(like="RQ", axis=0)
    canto = data.filter(like="Canto", axis=0)
    hypoG = data.filter(like="FR", axis=0)
    return requite, canto, hypoG


def intersect(df1: pd.DataFrame, df2: pd.DataFrame, labels: pd.DataFrame) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Intersect 2 data frames without filling lines with nan if not present in one of them.
    :param df1: clinical data
    :param df2: genomic data
    :param labels: labels
    :return: clinical data, genomic data and labels purged of nan values
    """
    df1 = df1.dropna(axis=0, how='all')
    df2 = df2.dropna(axis=0, how='all')
    clinical_col = df1.columns

    res = pd.concat([df1, df2, labels], axis=1, join="inner")
    labels = res['label']
    res = res.drop("label", axis=1)

    return res[clinical_col], res.drop(clinical_col, axis=1), labels


def aggregate_patient_id(filename: str) -> str:
    """
    Transform a filename into a valid Subject ID.
    Warning: this function uses a regex based on what was observed in the filenames provided.
    This could not work with further files if they are not in the right format.
    :param filename:
    :return:
    """
    # (go to https://regex101.com/ for an explanation of the regex)
    search = re.search(r"canto-[0-9]{2}-[0-9]{5}|RQ[0-9]{5}-[0-9]|FR-[0-9]{2}-[0-9]{4}", filename)

    if search:
        name = search.group(0)
        # Canto and HypoG ID are not in the same format we expect something like :
        # FR[0-9]{2}-[0-9]{4} for HypoG
        # Canto[0-9]{2}-[0-9]{5} for Canto
        if "canto" in name or "FR" in name:
            name = name.replace('-', '', 1)
        if "canto" in name:
            name = name.capitalize()
        return name
    else:
        return ''


def create_2d_dataset(root_folder: str, columns: list[str], labels: pd.Series) -> dict[str, dict[str, np.ndarray]]:
    """
    Create a dataset containing 2D tabular data indexed by Subject ID
    :param root_folder: root folder for 2D dosiomics or radiomics data
    :param columns: name of all the columns for the data. A column named 'Unnamed: 0' should absolutely be in this list
    This represent the name of the rows in the dosiomic or radiomic file.
    :param labels: labels indexed by subject ids
    :return: a dict associating subject ids to their data and labels.
    """
    # Keeping subject ids in case someone would want to filter by database....
    patient_files = [f for f in os.listdir(root_folder) if
                     os.path.isfile(os.path.join(root_folder, f)) and 'lock' not in f]
    empty_df = pd.DataFrame([], columns=columns)
    aggregate = defaultdict(dict)

    for file in patient_files:
        print(f"reading {file}...", end=" ")
        subject_id = aggregate_patient_id(file)

        if subject_id in labels.index:
            patient_data = pd.read_excel(os.path.join(root_folder, file))

            # If there are missing columns
            patient_data = pd.concat([patient_data, empty_df])
            # Ensure row order is the same for all patient data
            patient_data = patient_data.sort_values(by=['Unnamed: 0'])
            rows = patient_data.iloc[:, 0]
            # 1st column with names is now useless
            patient_data = patient_data.iloc[:, 1:]
            # Ensure column order is the same for all patient data
            patient_data = patient_data.reindex(sorted(patient_data.columns), axis=1)
            patient_data = infer_missing_values(patient_data, "value", 0.0)
            aggregate[subject_id] = {'data': patient_data.to_numpy(), 'label': labels[subject_id]}

    return aggregate


def obtain_standalone_data(filename: Union[str, Any], sep: str = ";") -> (pd.DataFrame, pd.Series):
    """
    Read CSV standalone file. A Standalone file contains flatten data with one Subject ID per line.
    There must also be a label column.
    :param filename: path to the file to open
    :param sep: separator for CSV file
    :return: Data indexed by Subject ID and labels separately
    """
    data = pd.read_csv(filename, sep=sep, index_col=[0])
    labels = data['label']
    data = data.drop(['label'], axis=1)
    data.dropna(axis=0, how='all', inplace=True)
    data = infer_missing_values(data)
    return data, labels
