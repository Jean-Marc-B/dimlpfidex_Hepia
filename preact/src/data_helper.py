import re
import os
import copy
from collections import defaultdict
from typing import Optional, Any
from typing import Union

import pandas as pd
import numpy as np
from numpy import dtype

# Global Variables to ensure consistency
__columns_train2trial = {
    "SUBJID": "SUBJECT ID",
    "SMOKE": "SMOKER",
    "DIABE": "DIABETES",
    "MENO": "MENOPAUSAL",
    "SIDE": "SIDE OF PRIMARY",
    "SURGTYP": "TYPE SURGERY",
    "AXI": "PLANNED AXILLARY DISSECTION",
    "NODREM": "NODES REMOVED",
    "NODINV": "NODES INVOLVED",
    "SENTNOD": "SENTINEL NODE BIOPSY",
    "HISTO": "HISTOLOGICAL TYPE",
    "PATHOSIZ": "PATHOLOGICAL TUMOUR SIZE",
    "CT": "CLINICAL T-STAGE",
    "CN": "CLINICAL N-STAGE",
    "KISTAT": "KI-67 STATUS",
    "ERSTAT": "ER STATUS",
    "PRSTAT": "PR STATUS",
    "HERSTAT": "HER-2 STATUS",
    "NBFRAC": "NUMBER OF FRACTIONS",
    "TBRSIDE": "TREATED BREAST RADIO",
    "BEDBOOST": "BOOST",
    "NEOADJCH": "NEOADJUVANT CHEMOTHERAPY",
    "ADJCH": "ADJUVANT CHEMOTHERAPY",
    "ARMLYM": "BASELINE ARM LYMPHEDEMA"
}

__original_columns = {
    'HEIGHT': (dtype('float64'), dtype('int64')),
    'WEIGHT': (dtype('float64'), dtype('int64')),
    'AGE': (dtype('float64'), dtype('int64')),
    'SMOKER': dtype('O'),
    'MENOPAUSAL': dtype('O'),
    'TYPE SURGERY': dtype('O'),
    'NODES INVOLVED': (dtype('float64'), dtype('int64')),
    'SIDE OF PRIMARY': dtype('O'),
    'HISTOLOGICAL TYPE': dtype('O'),
    'PATHOLOGICAL TUMOUR SIZE': (dtype('float64'), dtype('int64')),
    'CLINICAL T-STAGE': dtype('O'),
    'CLINICAL N-STAGE': dtype('O'),
    'KI-67 STATUS': (dtype('float64'), dtype('int64')),
    'ER STATUS': dtype('O'),
    'HER-2 STATUS': dtype('O'),
    'PR STATUS': dtype('O'),
    'NUMBER OF FRACTIONS': (dtype('float64'), dtype('int64')),
    'IMRT': (dtype('float64'), dtype('int64')),
    'IMRT_YES': (dtype('float64'), dtype('int64')),
    'IMRT_NO': (dtype('float64'), dtype('int64')),
    'IMRT_UNKNOWN': (dtype('float64'), dtype('int64')),
    '3D-CRT': dtype('float64'),
    '3D-CRT_YES': (dtype('float64'), dtype('int64')),
    '3D-CRT_NO': (dtype('float64'), dtype('int64')),
    '3D-CRT_UNKNOWN': (dtype('float64'), dtype('int64')),
    'TREATED BREAST RADIO': dtype('O'),
    'BOOST': (dtype('float64'), dtype('int64'), dtype('O')),
    'NEOADJUVANT CHEMOTHERAPY': (dtype('float64'), dtype('int64'), dtype('O')),
    'ADJUVANT CHEMOTHERAPY': (dtype('float64'), dtype('int64'), dtype('O')),
    'NODES REMOVED': (dtype('float64'), dtype('int64')),
    'BASELINE ARM LYMPHEDEMA': (dtype('float64'), dtype('int64'), dtype('O')),
    'DIABETES': (dtype('float64'), dtype('int64'), dtype('O')),
    'PLANNED AXILLARY DISSECTION': (dtype('float64'), dtype('int64'), dtype('O')),
    'SENTINEL NODE BIOPSY': (dtype('float64'), dtype('int64'), dtype('O'))
}

__discrete_attributes = ['KI-67 STATUS', 'NODES INVOLVED', 'NODES REMOVED', 'NUMBER OF FRACTIONS', 'PATHOLOGICAL TUMOUR SIZE']

__training_columns = {'MENOPAUSAL_POST_MENOPAUSAL', 'IMRT_UNKNOWN', 'NEOADJUVANT_CHEMOTHERAPY_UNKNOWN', 'AGE', 'NODES_REMOVED', 'TREATED_BREAST_RADIO_LEFT', 'HER_2_STATUS_UNKNOWN', 'SIDE_OF_PRIMARY_UNKNOWN', 'PR_STATUS_UNKNOWN', 'CLINICAL_N_STAGE_NX', 'IMRT_YES', 'BOOST_UNKNOWN', 'DIABETES_UNKNOWN', '3D_CRT_UNKNOWN', 'CLINICAL_T_STAGE_TX', 'TYPE_SURGERY_MASTECTOMY', '3D_CRT_NO', 'SMOKER_FORMER', 'ER_STATUS_UNKNOWN', 'CLINICAL_T_STAGE_TIS', 'SMOKER_UNKNOWN', 'CLINICAL_T_STAGE_UNKNOWN', 'NODES_INVOLVED', 'HISTOLOGICAL_TYPE_UNKNOWN', 'SIDE_OF_PRIMARY_RIGHT', 'BOOST_YES', 'KI_67_STATUS', 'IMRT_NO', 'HISTOLOGICAL_TYPE_OTHER', 'BOOST_NO', 'SENTINEL_NODE_BIOPSY', 'ADJUVANT_CHEMOTHERAPY_YES', 'NEOADJUVANT_CHEMOTHERAPY_NO', 'CLINICAL_T_STAGE_T4D', 'SMOKER_CURRENT', '3D_CRT_YES', 'HER_2_STATUS_NEGATIVE', 'SIDE_OF_PRIMARY_LEFT', 'NUMBER_OF_FRACTIONS', 'PATHOLOGICAL_TUMOUR_SIZE', 'NEOADJUVANT_CHEMOTHERAPY_YES', 'HER_2_STATUS_POSITIVE', 'ADJUVANT_CHEMOTHERAPY_UNKNOWN', 'CLINICAL_T_STAGE_T0', 'CLINICAL_T_STAGE_T3', 'PR_STATUS_NEGATIVE', 'CLINICAL_T_STAGE_T4B', 'ADJUVANT_CHEMOTHERAPY_NO', 'WEIGHT', 'CLINICAL_N_STAGE_N1', 'PR_STATUS_POSITIVE', 'DIABETES_NO', 'MENOPAUSAL_PRE_MENOPAUSAL', 'HISTOLOGICAL_TYPE_LOBULAR', 'CLINICAL_N_STAGE_UNKNOWN', 'CLINICAL_N_STAGE_N2', 'TYPE_SURGERY_UNKNOWN', 'PLANNED_AXILLARY_DISSECTION', 'BASELINE_ARM_LYMPHEDEMA_YES', 'CLINICAL_T_STAGE_T4', 'TREATED_BREAST_RADIO_UNKNOWN', 'HISTOLOGICAL_TYPE_DUCTAL/INVASIVE_CARCINOMA_NST', 'SMOKER_NEVER', 'HEIGHT', 'ER_STATUS_NEGATIVE', 'CLINICAL_T_STAGE_T2', 'TYPE_SURGERY_BREAST_CONSERVING_SURGERY', 'MENOPAUSAL_NOT_MENOPAUSAL', 'MENOPAUSAL_UNKNOWN', 'BASELINE_ARM_LYMPHEDEMA_NO', 'CLINICAL_N_STAGE_N3', 'DIABETES_YES', 'BASELINE_ARM_LYMPHEDEMA_UNKNOWN', 'CLINICAL_T_STAGE_T4A', 'CLINICAL_N_STAGE_N0', 'CLINICAL_T_STAGE_T1', 'ER_STATUS_POSITIVE', 'TREATED_BREAST_RADIO_RIGHT'}

__possible_values = {'SMOKER': ("CURRENT", "FORMER", "NEVER", "UNKNOWN"),
                     'MENOPAUSAL': ("NOT-MENOPAUSAL", "POST-MENOPAUSAL", "UNKNOWN"),
                     'DIABETES': ("YES", "NO"),
                     'SIDE OF PRIMARY': ("RIGHT", "LEFT"),
                     'TYPE SURGERY': ("MASTECTOMY", "BREAST CONSERVING SURGERY"),
                     'PLANNED AXILLARY DISSECTION': ("YES", "NO", "UNKNOWN"),
                     'SENTINEL NODE BIOPSY': ("YES", "NO", "UNKNOWN"),
                     'HISTOLOGICAL TYPE': ("DUCTAL/INVASIVE CARCINOMA NST", "LOBULAR", "OTHER"),
                     'CLINICAL T-STAGE': ( "T0", "TIS", "T1", "T2", "T3", "T4", "T4A", "T4B", "T4C", "T4D", "TX"),
                     'CLINICAL N-STAGE': ("N0", "N1", "N2", "N3", "NX"),
                     'ER STATUS': ("POSITIVE", "NEGATIVE", "UNKNOWN"),
                     'PR STATUS': ("POSITIVE", "NEGATIVE", "UNKNOWN"),
                     'HER-2 STATUS': ("POSITIVE", "NEGATIVE", "UNKNOWN"),
                     'TREATED BREAST RADIO': ("RIGHT", "LEFT"),
                     'BOOST': ("YES", "NO", "UNKNOWN"),
                     'NEOADJUVANT CHEMOTHERAPY': ("YES", "NO", "UNKNOWN"),
                     'ADJUVANT CHEMOTHERAPY': ("YES", "NO", "UNKNOWN"),
                     'BASELINE ARM LYMPHEDEMA': ("YES", "NO", "UNKNOWN")
                     }

__training_means = {'WEIGHT': 69.25586658255796, 'TYPE_SURGERY_UNKNOWN': 0.0018864958339883666, 'TYPE_SURGERY_MASTECTOMY': 0.18519100770319133, 'TYPE_SURGERY_BREAST_CONSERVING_SURGERY': 0.8129224964628203, 'TREATED_BREAST_RADIO_UNKNOWN': 0.48420059739034743, 'TREATED_BREAST_RADIO_RIGHT': 0.2501179059896243, 'TREATED_BREAST_RADIO_LEFT': 0.2656814966200283, 'SMOKER_UNKNOWN': 0.009746895142273228, 'SMOKER_NEVER': 0.6109102342398994, 'SMOKER_FORMER': 0.221191636535136, 'SMOKER_CURRENT': 0.1581512340826914, 'SIDE_OF_PRIMARY_UNKNOWN': 0.0, 'SIDE_OF_PRIMARY_RIGHT': 0.4856154692658387, 'SIDE_OF_PRIMARY_LEFT': 0.5143845307341613, 'SENTINEL_NODE_BIOPSY': 0.7074359377456375, 'PR_STATUS_UNKNOWN': 0.07027196981606666, 'PR_STATUS_POSITIVE': 0.663574909605408, 'PR_STATUS_NEGATIVE': 0.2661531205785254, 'PLANNED_AXILLARY_DISSECTION': 0.40418173243200756, 'PATHOLOGICAL_TUMOUR_SIZE': 17.107667833280303, 'NUMBER_OF_FRACTIONS': 20.30259691381257, 'NODES_REMOVED': 6.396994483545749, 'NODES_INVOLVED': 1.1941463414634146, 'NEOADJUVANT_CHEMOTHERAPY_YES': 0.13409841219933974, 'NEOADJUVANT_CHEMOTHERAPY_UNKNOWN': 0.00015720798616569723, 'NEOADJUVANT_CHEMOTHERAPY_NO': 0.8657443798144946, 'MENOPAUSAL_UNKNOWN': 0.015091966671906933, 'MENOPAUSAL_PRE_MENOPAUSAL': 0.2681968243986795, 'MENOPAUSAL_POST_MENOPAUSAL': 0.6406225436252162, 'MENOPAUSAL_NOT_MENOPAUSAL': 0.07608866530419746, 'KI_67_STATUS': 23.250853970964986, 'IMRT_YES': 0.2600220091180632, 'IMRT_UNKNOWN': 0.4845150133626788, 'IMRT_NO': 0.255462977519258, 'HISTOLOGICAL_TYPE_UNKNOWN': 0.0029869517371482472, 'HISTOLOGICAL_TYPE_OTHER': 0.13881465178431066, 'HISTOLOGICAL_TYPE_LOBULAR': 0.11664832573494734, 'HISTOLOGICAL_TYPE_DUCTAL/INVASIVE_CARCINOMA_NST': 0.7415500707435938, 'HER_2_STATUS_UNKNOWN': 0.04323219619556674, 'HER_2_STATUS_POSITIVE': 0.1359849080333281, 'HER_2_STATUS_NEGATIVE': 0.8207828957711052, 'HEIGHT': 162.67628053585503, 'ER_STATUS_UNKNOWN': 0.021380286118534823, 'ER_STATUS_POSITIVE': 0.8344599905675208, 'ER_STATUS_NEGATIVE': 0.14415972331394436, 'DIABETES_YES': 0.05895299481213646, 'DIABETES_UNKNOWN': 0.4577896557145103, 'DIABETES_NO': 0.48325734947335325, 'CLINICAL_T_STAGE_UNKNOWN': 0.10249960698003459, 'CLINICAL_T_STAGE_TX': 0.010061311114604623, 'CLINICAL_T_STAGE_TIS': 0.04480427605722371, 'CLINICAL_T_STAGE_T4D': 0.00031441597233139445, 'CLINICAL_T_STAGE_T4B': 0.0012576638893255778, 'CLINICAL_T_STAGE_T4A': 0.00031441597233139445, 'CLINICAL_T_STAGE_T4': 0.0031441597233139444, 'CLINICAL_T_STAGE_T3': 0.05046376355918881, 'CLINICAL_T_STAGE_T2': 0.2906775664203742, 'CLINICAL_T_STAGE_T1': 0.7278729759471781, 'CLINICAL_T_STAGE_T0': 0.15956610595818269, 'CLINICAL_N_STAGE_UNKNOWN': 0.0507781795315202, 'CLINICAL_N_STAGE_NX': 0.03285646910863072, 'CLINICAL_N_STAGE_N3': 0.007074359377456375, 'CLINICAL_N_STAGE_N2': 0.029712309385316774, 'CLINICAL_N_STAGE_N1': 0.2145889011161767, 'CLINICAL_N_STAGE_N0': 0.9163653513598491, 'BOOST_YES': 0.310485772677252, 'BOOST_UNKNOWN': 0.4865587171828329, 'BOOST_NO': 0.2029555101399151, 'BASELINE_ARM_LYMPHEDEMA_YES': 0.014777550699575539, 'BASELINE_ARM_LYMPHEDEMA_UNKNOWN': 0.4853010532935073, 'BASELINE_ARM_LYMPHEDEMA_NO': 0.49992139600691715, 'AGE': 57.49154951824356, 'ADJUVANT_CHEMOTHERAPY_YES': 0.3975789970130483, 'ADJUVANT_CHEMOTHERAPY_UNKNOWN': 0.00015720798616569723, 'ADJUVANT_CHEMOTHERAPY_NO': 0.6022637950007861, '3D_CRT_YES': 0.34963056123251063, '3D_CRT_UNKNOWN': 0.48781638107215847, '3D_CRT_NO': 0.16255305769533093}


def read_file(file: Union[str, Any], sep: str = ',') -> pd.DataFrame:
    """
    Read the file precised and put the Subject ID as index.
    :param file: path to file to open and read or bytes of the file. Must be a CSV file.
    :param sep: separator for the CSV or TSV file. Default to ','.
    :return: a DataFrame with the data in the file
    """
    data = None
    ext = os.path.splitext(file)[1].lower()
    encodings = ["utf-8", "latin1"]

    if ext in [".xls", ".xlsx", ".xlsm", ".xlsb", ".odf", ".ods", ".odt"]:

        for enc in encodings:
            try:
                data = pd.read_excel(file, encoding=enc)
                break
            except Exception:
                print(f"Data helper: Bad excel file encoding: '{enc}', trying another one.")

    elif ext == ".csv":

        for enc in encodings:
            try:
                data = pd.read_csv(file, sep=sep, encoding="latin1")
                # Remove index columns from CSV
                data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
                break
            except Exception:
                print(f"Data helper: Bad CSV file encoding: '{enc}', trying another one.")

    else:
        raise NotImplementedError(f"Support for {ext} extension in {file} file is not implemented.")

    # All capital letters columns
    data.columns = [re.sub(r"^AI", "", c.upper()) for c in data.columns]
    # Rename columns to have same names than in our experiences / trials.
    # If the name is not in the dict provided, it will not change
    data.rename(columns=__columns_train2trial, inplace=True)
    data = data.set_index("SUBJECT ID", drop=True)
    return data


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
        # negs = [0, 2, 6, 8]
        # nans = [8, 18, 20, 24, 26]
        negs = [0, 2]
        nans = [6, 8]
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


def infer_missing_values(data: pd.DataFrame, method: str = 'mean', values: Optional[dict[str, float]] = None) -> pd.DataFrame:
    """
    Replace NaN vales in the DataFrame with values from chosen method.
    :param data: explicit
    :param method: either 'mean', 'median', 'value'. If value, only numerical values are compatible. Default is median.
    :param values: Optional to replace the value in the precised column by the value
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
        if values is None:
            raise ValueError("A value must be passed to infer missing value")
        if type(values) != dict:
            raise ValueError(f"Values must be dict-like, was {type(values)}")
        # replace nan values in columns precized in the dict by their value
        for c in values:
            if c not in res.columns:
                raise ValueError(f"Column {c} not found in data")
            res.fillna(values, inplace=True)
        return res

    raise Exception("Method must be one of 'median', 'mean', value'")


def __manage_T_retroactively(attributes: pd.DataFrame) -> pd.DataFrame:
    # Transforming to imply that T4a, T4b, T4d <=> T4
    # Transforming to imply that T4 -> T3 -> T2 -> T1
    t4 = attributes.filter(regex=r"^CLINICAL_T_STAGE_T4[A-D]?$")
    ts = attributes.filter(regex=r"^CLINICAL_T_STAGE_T[1-3]$")
    t4 = np.add.reduce(t4, axis=1)
    t4.rename("CLINICAL_T_STAGE_T4", inplace=True)
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
    tis = attributes["CLINICAL_T_STAGE_TIS"]
    t0s = attributes["CLINICAL_T_STAGE_T0"].add(tis, axis=0).apply(lambda x: 1 if x >=1 else 0).rename("CLINICAL_T_STAGE_T0")
    attributes = attributes.drop(["CLINICAL_T_STAGE_T0"], axis=1)
    attributes = pd.concat([attributes, t0s], axis=1)

    # T+ (T1->T4/T4a,b,c,d) can't be T0
    t_plus = attributes.filter(regex=r"^CLINICAL_T_STAGE_T[1-4][a-d]?$")
    t_plus = np.add.reduce(t_plus, axis=1).rename("CLINICAL_T_STAGE_T+")
    t0s = attributes["CLINICAL_T_STAGE_T0"]
    ts = pd.concat([t0s, t_plus], axis=1)
    t0s = ts.apply(lambda x: 1 if x["CLINICAL_T_STAGE_T+"] == 0 and x["CLINICAL_T_STAGE_T0"] == 1 else 0, axis=1).rename("CLINICAL_T_STAGE_T0")
    attributes = attributes.drop(["CLINICAL_T_STAGE_T0"], axis=1)
    attributes = pd.concat([attributes, t0s], axis=1)
    return attributes.sort_index(axis=1, ascending=True)


def __manage_N_retroactively(attributes: pd.DataFrame) -> pd.DataFrame:
    # cf. ressources/PrecautionsDeCodage.pdf
    # N3 -> N2 -> N1 -> N0
    ns = attributes.filter(regex=r"^CLINICAL_N_STAGE_N[0-3]$")
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


def __check_dtypes(data: pd.DataFrame) -> None:
    def unknown2nan(c, x):
        try:
            x = float(x)
            return float(x)
        except ValueError:
            if x == "UNKNOWN":
                return np.nan
            else:
                raise ValueError(f"Value '{x}' not valid for column {c}. Should be either of type [np.int64, np.float64] or have UNKNOWN value")

    dtypes = data.dtypes
    for c in dtypes.index:
        if type(__original_columns[c]) is tuple:
            if dtypes[c] not in __original_columns[c]:
                # Manage UNKNOWN in discrete columns
                if c in __discrete_attributes and dtypes[c] == dtype('O'):
                    data[c] = data[c].apply(lambda x: unknown2nan(c, x))
                else:
                    raise ValueError(f"Column {c} should be of dtype '{__original_columns[c]}. Got {dtypes[c]} instead.")
                # print(f"Column {c} should be of dtype '{__original_columns[c]}. Got {dtypes[c]} instead.")
        elif dtypes[c] != __original_columns[c]:
            raise ValueError(f"Column {c} should be of dtype '{__original_columns[c]}. Got {dtypes[c]} instead.")
            # print(f"Column {c} should be of dtype '{__original_columns[c]}. Got {dtypes[c]} instead.")


def __postprocess_attributes(attributes: pd.DataFrame) -> pd.DataFrame:
    # replace NAN in column's name by UNKNOWN, spaces by _ and - by _
    attributes.rename(
        columns=lambda col: col.strip().upper().replace("NAN", "UNKNOWN").replace("-", "_").replace(" ", "_"),
        inplace=True)

    # Clinical Ts cases
    # we have to rename some columns, add some more to correspond to training data
    # CF. ressources and file "precautionsDeCodage.pdf" for further information
    if 'CLINICAL_T_STAGE_T4C' in attributes.columns:
        t4c = attributes['CLINICAL_T_STAGE_T4C']
        t4at4b = attributes.filter(regex=r"^CLINICAL_T_STAGE_T4[A-B]$")
        t4at4b = t4at4b.add(t4c, axis=0)
        attributes = attributes.drop(columns=['CLINICAL_T_STAGE_T4A', 'CLINICAL_T_STAGE_T4B', 'CLINICAL_T_STAGE_T4C'])
        attributes = pd.concat([attributes, t4at4b], axis=1)

    # Management of T1-4
    attributes = __manage_T_retroactively(attributes)

    # Management of N0-3
    attributes = __manage_N_retroactively(attributes)

    attributes = attributes.sort_index(axis=1, ascending=False)

    return attributes


def __obtain_clinical_data(file_path: str) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Get the data and apply some tranformations to be usable for experiences
    Read the data
    Extract label 12/24/36m Arm Lymphedema
    Remove useless columns
    One hot encode categorical data
    :param file_path: path to the file to read
    :return: a pd.DataFrame with the data and a pd.Series with the labels if in training case, otherwise only clinical Data.
    """
    data = read_file(file_path)

    # Managing labels and train drop columns
    # label_names = ['12m follow-up Arm Lymphedema', '24m follow-up Arm Lymphedema', '36m follow-up Arm Lymphedema']
    label_names = ['12M FOLLOW-UP ARM LYMPHEDEMA', '24M FOLLOW-UP ARM LYMPHEDEMA']
    label_lymph = get_labels(data, *label_names)
    label_lymph.rename("label", inplace=True)

    keywords = ['FOLLOW-UP', 'DB', 'POST-RT', 'DATE', 'BASELINE TELANGIECTASIA', 'BASELINE EDEMA',
                'BASELINE SKININDURATION', 'BASELINE ERYTHEMA', 'BASELINE SKINHYPERPIGMENT', 'MEAN HEART DOSE']

    attributes = remove_attributes_with_keyword(data, *keywords)
    # Surgery is a special case since some attributes have surgery in their names we have to delete only this column
    attributes.drop('SURGERY', inplace=True, axis=1)

    # match trial names
    attributes.rename(columns={"3D": "3D-CRT", "NODES EXAMINED": "NODES REMOVED"}, inplace=True)

    # check columns dtypes for consistancy before transformation and one hot encoding
    __check_dtypes(attributes)

    attributes = one_hot_encoding(attributes, dummy_na=True)
    # Typo in Diabetes column for training data
    attributes.rename(columns={"DIABETIES": "DIABETES", "HISTOLOGICAL TYPE_DUCTAL": "HISTOLOGICAL TYPE_DUCTAL/INVASIVE_CARCINOMA_NST"}, inplace=True)
    categorical_cols = ['DIABETES', 'IMRT', '3D-CRT', 'BOOST', 'NEOADJUVANT CHEMOTHERAPY', 'ADJUVANT CHEMOTHERAPY',
                        'BASELINE ARM LYMPHEDEMA']
    attributes = one_hot_encoding_ext(attributes, True, *categorical_cols)

    # Match trial columns
    attributes.rename(columns={"SMOKER_YES": "SMOKER_CURRENT", "SMOKER_NO": "SMOKER_NEVER"}, inplace=True)
    attributes = infer_missing_values(attributes)

    attributes = __postprocess_attributes(attributes)

    return attributes, label_lymph


def __obtain_trial_data(file_path: str) -> pd.DataFrame:
    """
    Get the data and apply some tranformations to be usable for experiences
    Read the data from trial file and fit to correspond to the train columns
    Remove useless columns
    One hot encode categorical data
    :param file_path: path to the file to read
    :return: a pd.DataFrame with the data indexed by SubjectID
    """
    data = read_file(file_path)
    # Managing RTTEC -> 3D/IMRT case and trial drop columns
    trial_keywords = ['STUDYID', 'SITEIDN', 'SITENAME', 'VISIT']
    attributes = remove_attributes_with_keyword(data, *trial_keywords)
    # SURG is a special case
    attributes.drop(columns="SURG", inplace=True)

    def __convert_rttec(value: str, df: pd.DataFrame) -> None:
        """
        Utilitary function to transform column RTTEC in clinical trials to 2 columns 3D and IMRT to match training data
        :param value: Value of the RTTEC column
        :param df: temporary DataFrame in the form Subject_ID -> (3D, IMRT)
        :return : None
        """
        if value == "3D-CRT":
            df["3D-CRT_YES"] = 1
            df["IMRT_NO"] = 1
        elif value == "IMRT":
            df["3D-CRT_NO"] = 1
            df["IMRT_YES"] = 1
        else:
            df["3D-CRT_UNKNOWN"] = 1
            df["IMRT_UNKNOWN"] = 1

    rttec = pd.DataFrame(np.zeros((data.shape[0], 6)), columns=["3D-CRT_YES", "IMRT_YES", "3D-CRT_NO", "IMRT_NO", "3D-CRT_UNKNOWN", "IMRT_UNKNOWN"], index=data.index, )
    attributes["RTTEC"].apply(lambda x: __convert_rttec(x, rttec))
    attributes.drop("RTTEC", axis=1, inplace=True)
    attributes = pd.concat([attributes, rttec], axis=1)

    # CLINICAL N Stages and T Stages are encoded with just their numbers must add prefix N and T
    attributes["CLINICAL N-STAGE"] = "N" + attributes["CLINICAL N-STAGE"].apply(lambda x: str(x))
    attributes["CLINICAL T-STAGE"] = "T" + attributes["CLINICAL T-STAGE"].apply(lambda x: str(x) if x != 'Tis' else 'is')
    attributes["CLINICAL T-STAGE"] = attributes["CLINICAL T-STAGE"].apply(lambda x: str(x).upper())

    def __manage_categorical(x: str):
        if x.upper() == "YES":
            return 1
        elif x.upper() == "NO":
            return 0
        return np.nan

    # Remove smoker postfix in Smoker column
    attributes['SMOKER'] = attributes['SMOKER'].apply(lambda x: x.upper().replace(" SMOKER", ""))

    # Seen during adaptation, NOT Menopausal has been transformed to NON Menopausal, changing it
    attributes['MENOPAUSAL'] = attributes['MENOPAUSAL'].apply(lambda x: x.upper().replace("NON", "NOT"))

    __check_dtypes(attributes)

    # Check that values are consistant
    for c in __possible_values:
        for v in set(attributes[c]):
            if str(v).upper() not in __possible_values[c]:
                raise ValueError(f"Value {v} is not applicable for attribute {c}. Must be one of {__possible_values[c]}")

    # planned axillary columns is categorical in trial, must transform to 0/1/nan to match training
    attributes["PLANNED AXILLARY DISSECTION"] = attributes["PLANNED AXILLARY DISSECTION"].apply(
        lambda x: __manage_categorical(x))

    # same for sentinel node biopsy
    attributes["SENTINEL NODE BIOPSY"] = attributes["SENTINEL NODE BIOPSY"].apply(lambda x: __manage_categorical(x))

    attributes = one_hot_encoding(attributes, dummy_na=False) # There should not be any empty value for categorical data

    # check missing columns and adding them filled with 0s since they are categorical and imply that one of the value is already present
    attributes.rename(
        columns=lambda col: col.strip().upper().replace("NAN", "UNKNOWN").replace("-", "_").replace(" ", "_"),
        inplace=True)
    trial_columns = set(attributes.columns)
    diff = list(__training_columns.difference(trial_columns))
    missing_columns = pd.DataFrame(np.zeros((attributes.shape[0], len(diff))), columns=diff, index=attributes.index)
    attributes = pd.concat([attributes, missing_columns], axis=1)

    attributes = infer_missing_values(attributes, method='value', values=__training_means)

    attributes = __postprocess_attributes(attributes)

    return attributes


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


def obtain_data(clinical_file: str, file_with_ids: Optional[str] = None, training: bool = True) -> Any:
    """
    Return data contained in files passed in argument.
    Must be either clinical data or clinical data and genomic data.
    :param clinical_file: path to clinical data file
    :param file_with_ids: path to data file containing patient ids and data related to them
    :return: Clinical data and Labels or Clinical data, Genomic data and Labels
    """
    if not training:
        return __obtain_trial_data(clinical_file)

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


def obtain_standalone_data(filename: Union[str, Any], sep: str = ";") -> tuple[pd.DataFrame, pd.Series]:
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

def filter_clinical(data: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Filter Clinical data with the rules given from doctors.
    Clinical T-Stage_nan != 1 AND Clinical N-Stage_nan != 1
    AND
    not (Clinical N-Stage_N0 == 1 AND Clinical N-Stage_N1 == 0 AND Clinical T-Stage_T0 == 1 AND Clinical T-Stage_Tis == 0)
    AND
    Planned axillary dissection == 1 OR Sentinel Node Biopsy == 1
    :param data: the data to filter
    :param labels: labels indexed by subject ids
    :return: filtered data and corresponding labels
    """
    labels = labels[labels != 2]  # enlève les données où les labels sont indéterminés

    data = data[((data["CLINICAL_T_STAGE_UNKNOWN"] != 1) & (data["CLINICAL_N_STAGE_UNKNOWN"] != 1)) &
                ~((data["CLINICAL_N_STAGE_N0"] == 1) & (data["CLINICAL_N_STAGE_N1"] == 0) & (
                            data["CLINICAL_T_STAGE_T0"] == 1) & (data["CLINICAL_T_STAGE_TIS"] == 0)) &
                ((data["PLANNED_AXILLARY_DISSECTION"] == 1) | (data[
                                                                   "SENTINEL_NODE_BIOPSY"] == 1))]  # filtre les données selon les critères envoyés par Guido

    data = data.loc[data.index.isin(labels.index)]  # enlève les lignes correspondantes aux labels indéterminés
    labels = labels.loc[labels.index.isin(data.index)]  # harmonise en ne gardant que les indexs contenus dans Data

    return data, labels
