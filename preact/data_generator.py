from matplotlib import pyplot as plt
from sys import argv
import pandas as pd
import random as r
from math import ceil
from os import makedirs
from src.data_helper import obtain_data

__values_translation_dict = {
    "AISMOKE":          (["NO", "YES", "FORMER"], ["NEVER", "CURRENT", "FORMER"]),
    "AIMENO":           (["PRE-MENOPAUSAL", "POST-MENOPAUSAL", "NOT MENOPAUSAL"], ["UNKNOWN", "POST-MENOPAUSAL", "NOT-MENOPAUSAL"]),
    "AIDIABE":          ([0.0, 1.0], ["NO", "YES"]), 
    "AISIDE":           (["Left", "Right"],["LEFT", "RIGHT"]),
    "AISURG":           ([0,1],["NO", "YES"]),
    "AISURGTYP":        (["Breast conserving surgery", "Mastectomy"],["BREAST CONSERVING SURGERY", "MASTECTOMY"]),
    "AIAXI":            ([0,1], ["NO", "YES"]),
    "AISENTNOD":        ([0,1], ["NO", "YES"]),
    "AIHISTO":          (["DUCTAL", "LOBULAR", "OTHER"],["DUCTAL/INVASIVE CARCINOMA NST", "LOBULAR", "OTHER"]),
    "AICT":             (["T0", "Tis", "T1", "T2", "T3", "T4", "T4a", "T4b","T4c", "T4d", "TX"],["0", "Tis", "1", "2", "3", "4", "4a", "4b", "4c", "4d", "X"]),
    "AICN":             (["N0", "N1", "N2", "NX", "N3"],["0", "1", "2", "3", "X"]),
    "AIERSTAT":         (["Positive", "Negative"], ["POSITIVE", "NEGATIVE"]),
    "AIPRSTAT":         (["Positive", "Negative"], ["POSITIVE", "NEGATIVE"]),
    "AITBRSIDE":        (["Left", "Right"], ["LEFT", "RIGHT"]),
    "AIBEDBOOST":       ([0.0, 1.0], ["NO", "YES"]),
    "AINEOADJCH":       ([0.0, 1.0], ["NO", "YES"]),
    "AIADJCH":          ([0.0, 1.0], ["NO", "YES"]),
    "AIARMLYM":         ([0.0, 1.0], ["NO", "YES"]),
}


def convert_dataset_to_trial_format(selected_data: pd.DataFrame) -> pd.DataFrame:
    def __convert(x, src_values: list, dest_values: list):
        if x in src_values:
            i = src_values.index(x)
            return dest_values[i]
        
    for key, values in __values_translation_dict.items():
        src_values = values[0]
        dest_values = values[1]
        selected_data[key] = selected_data[key].apply(lambda x: __convert(x, src_values, dest_values))

    return selected_data


def today_str() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d-%H-%M")


def init_subplots(nrows: int, ncols: int, nplots: int) -> list:
    subplots = [plt.subplots(nrows, ncols, figsize=(15, 10)) for _ in range(nplots)]

    for i, subplot in enumerate(subplots):
        subplot[0].suptitle(f"Data gen. comparision page {i+1}/{nplots}")
        subplot[0].tight_layout()

    return subplots


def save_data_gen_cmp_charts(original_data: pd.DataFrame, generated_data: pd.DataFrame) -> None:
    today = today_str()
    save_path = f"plots/{today}"
    makedirs(save_path)

    n = 0
    i_sp = 0
    nplots = 10
    n_per_plot = ceil(len(generated_data.columns) / nplots)
    ncols = 4
    nrows = ceil(n_per_plot / ncols)

    subplots = init_subplots(nrows, ncols, nplots)

    current_fig = subplots[i_sp][0]
    current_subplot_axes = subplots[i_sp][1]

    for name in generated_data:
        # ignore useless columns
        if name in ["Subject ID", "DB"]:
            continue

        if n == n_per_plot:
            current_fig.savefig(f"{save_path}/subplot_{i_sp+1}of{nplots}.png")
            i_sp += 1
            n = 0
            current_subplot_axes = subplots[i_sp][1]
            current_fig = subplots[i_sp][0]

        ir = n // ncols
        ic = n % ncols

        original_col = original_data[name].dropna()
        generated_col = generated_data[name].dropna()

        current_subplot_axes[ir, ic].set_title(name, fontsize=10)
        current_subplot_axes[ir, ic].hist(original_col, alpha=0.5)
        current_subplot_axes[ir, ic].hist(generated_col, alpha=0.5)
        n += 1

    current_fig.savefig(f"{save_path}/subplot_{nplots}of{nplots}.png")


def rand_number(column: pd.Series, n: int, precision: int = 0) -> list[float]:
    column = column.dropna()
    mean = column.mean()
    std = column.std()
    res = [round(r.gauss(mean, std), precision) for _ in range(n)]

    # abs used to remove -0.0
    return [abs(n) if n >= 0.0 else 0.0 for n in res]


def rand_choice(column: pd.Series, n: int) -> list[bool]:
    cleaned_col = column.dropna()
    size = cleaned_col.count()
    values = cleaned_col.unique()
    weights = (column.value_counts(sort=False) / size).to_list()

    return r.choices(values, weights=weights, k=n)


def write_fake_data_test_file(source_data: pd.DataFrame, nb_records: int = 10):
    original_colnames = ["Age", "height", "weight", "Smoker", "Menopausal", "Diabetes", "Side of Primary", "Surgery", "Type Surgery", "Planned axillary dissection", "Nodes examined", "Nodes involved", "Sentinel node biopsy", "Histological Type", "Pathological tumour size", "Clinical T-Stage", "Clinical N-Stage", "Ki-67 status", "ER Status", "PR Status", "HER-2 Status", "Number of fractions", "IMRT", "3D", "Treated breast radio","Boost",  "neoadjuvant chemotherapy", "adjuvant chemotherapy", "Baseline Arm Lymphedema"]
    target_colnames = ["AIAGE","AIHEIGHT","AIWEIGHT","AISMOKE","AIMENO","AIDIABE","AISIDE","AISURG","AISURGTYP","AIAXI","AINODREM","AINODINV","AISENTNOD","AIHISTO","AIPATHOSIZ","AICT","AICN","AIKISTAT","AIERSTAT","AIPRSTAT","AIHERSTAT","AINBFRAC","AITBRSIDE","AIBEDBOOST","AINEOADJCH","AIADJCH","AIARMLYM"]
    ordered_colnames = ["STUDYID", "SITEIDN", "SITENAME", "SUBJID","VISIT", "AIAGE","AIHEIGHT","AIWEIGHT","AISMOKE","AIMENO","AIDIABE","AISIDE","AISURG","AISURGTYP","AIAXI","AINODREM","AINODINV","AISENTNOD","AIHISTO","AIPATHOSIZ","AICT","AICN","AIKISTAT","AIERSTAT","AIPRSTAT","AIHERSTAT","AINBFRAC","AIRTTEC","AITBRSIDE","AIBEDBOOST","AINEOADJCH","AIADJCH","AIARMLYM"]
    integer_columns = ["AIAGE", "AIHEIGHT", "AIWEIGHT", "AINODREM", "AINODINV", "AIPATHOSIZ", "AIKISTAT", "AINBFRAC"]

    filename = f"ecrf_test_{today_str()}.csv"

    def __rttec(row: pd.Series) -> str:
        if row["IMRT"] == 1.0 or row["IMRT"] == 1:
            return "IMRT"
        if row["3D"] == 1.0 or row["3D"] == 1:
            return "3D-CRT"
        
        return r.choices(["IMRT", "3D-CRT"], k=1)[0]

    selected_data = source_data[original_colnames].iloc[0:nb_records,:].copy()
    selected_data["AIRTTEC"] = selected_data.apply(__rttec, axis=1)
    selected_data = selected_data.drop(["IMRT", "3D"], axis=1)
    original_colnames.remove("IMRT")
    original_colnames.remove("3D")
    
    renaming_dict = dict(zip(original_colnames, target_colnames))
    selected_data = selected_data.rename(renaming_dict, axis=1)
    selected_data = convert_dataset_to_trial_format(selected_data)
    selected_data[integer_columns] = selected_data[integer_columns].astype("int32")

    zero_pad = len(str(n))-1 
    selected_data["STUDYID"]    =   ["PRE-ACT-00-TEST"] * nb_records
    selected_data["SITEIDN"]    =   ["CH-25"] * nb_records
    selected_data["SITENAME"]   =   ["HEPIA-TEST"] * nb_records
    selected_data["SUBJID"]     =   [f"CH-25-{str(i).zfill(zero_pad)}" for i in range(nb_records)]
    selected_data["VISIT"]      =   ["BASELINE"] * nb_records

    selected_data = selected_data.reindex(ordered_colnames, axis=1)

    selected_data.to_csv(filename, index=False)


if __name__ == "__main__":
    n = 6000
    dataset_path = "dataset/clinical_complete_rev1.csv"

    if len(argv) == 2:
        nb_rows_to_gen = argv[1]

    data = pd.read_csv(dataset_path, index_col=False)
    data = data.iloc[1:,]
    zero_pad = len(str(n))-1 

    generated_data = pd.DataFrame({
    "Subject ID":                               [f"testgen-{str(i).zfill(zero_pad)}" for i in range(n)],
    "height":                                   rand_number(data["height"], n, 0),
    "weight":                                   rand_number(data["weight"], n, 0),
    "Age":                                      rand_number(data["Age"], n, 0),
    "Smoker":                                   rand_choice(data["Smoker"], n),
    "Menopausal":                               rand_choice(data["Menopausal"], n),
    "Diabetes":                                 rand_choice(data["Diabetes"], n),
    "Surgery":                                  rand_choice(data["Surgery"], n),
    "Type Surgery":                             rand_choice(data["Type Surgery"], n),
    "Nodes involved":                           rand_choice(data["Nodes involved"], n),
    "Side of Primary":                          rand_choice(data["Side of Primary"], n),
    "Histological Type":                        rand_choice(data["Histological Type"], n),
    "Pathological tumour size":                 rand_number(data["Pathological tumour size"], n, 0),
    "Clinical T-Stage":                         rand_choice(data["Clinical T-Stage"], n),
    "Clinical N-Stage" :                        rand_choice(data["Clinical N-Stage"], n),
    "Ki-67 status":                             rand_number(data["Ki-67 status"], n, 0),
    "ER Status":                                rand_choice(data["ER Status"], n),
    "HER-2 Status":                             rand_choice(data["HER-2 Status"], n),
    "PR Status":                                rand_choice(data["PR Status"], n),
    "Number of fractions":                      rand_choice(data["Number of fractions"], n),
    "Mean Heart Dose":                          rand_number(data["Mean Heart Dose"], n, 2),
    "IMRT":                                     rand_choice(data["IMRT"], n),
    "3D":                                       rand_choice(data["3D"], n),
    "Treated breast radio":                     rand_choice(data["Treated breast radio"], n),
    "Boost":                                    rand_choice(data["Boost"], n),
    "neoadjuvant chemotherapy":                 rand_choice(data["neoadjuvant chemotherapy"], n),
    "adjuvant chemotherapy":                    rand_choice(data["adjuvant chemotherapy"], n),
    "Nodes examined":                           rand_number(data["Nodes examined"], n, 0),
    "Baseline Arm Lymphedema":                  rand_choice(data["Baseline Arm Lymphedema"], n),
    "Post-RT Arm Lymphedema":                   rand_choice(data["Post-RT Arm Lymphedema"], n),
    "12m follow-up Arm Lymphedema":             rand_choice(data["12m follow-up Arm Lymphedema"], n),
    "24m follow-up Arm Lymphedema":             rand_choice(data["24m follow-up Arm Lymphedema"], n),
    "36m follow-up Arm Lymphedema":             rand_choice(data["36m follow-up Arm Lymphedema"], n),
    "60m follow-up Arm Lymphedema":             rand_choice(data["60m follow-up Arm Lymphedema"], n),
    "48m follow-up Arm Lymphedema":             rand_choice(data["48m follow-up Arm Lymphedema"], n),
    "72m follow-up Arm Lymphedema":             rand_choice(data["72m follow-up Arm Lymphedema"], n),
    "Baseline Telangiectasia":                  rand_choice(data["Baseline Telangiectasia"], n),
    "Post-RT Telangiectasia":                   rand_choice(data["Post-RT Telangiectasia"], n),
    "12m follow-up Telangiectasia":             rand_choice(data["12m follow-up Telangiectasia"], n),
    "24m follow-up Telangiectasia":             rand_choice(data["24m follow-up Telangiectasia"], n),
    "36m follow-up Telangiectasia":             rand_choice(data["36m follow-up Telangiectasia"], n),
    "60m follow-up Telangiectasia":             rand_choice(data["60m follow-up Telangiectasia"], n),
    "48m follow-up Telangiectasia":             rand_choice(data["48m follow-up Telangiectasia"], n),
    "72m follow-up Telangiectasia":             rand_choice(data["72m follow-up Telangiectasia"], n),
    "Baseline Edema":                           rand_choice(data["Baseline Edema"], n),
    "Post-RT Edema":                            rand_choice(data["Post-RT Edema"], n),
    "12m follow-up Edema":                      rand_choice(data["12m follow-up Edema"], n),
    "24m follow-up Edema":                      rand_choice(data["24m follow-up Edema"], n),
    "36m follow-up Edema":                      rand_choice(data["36m follow-up Edema"], n),
    "60m follow-up Edema":                      rand_choice(data["60m follow-up Edema"], n),
    "48m follow-up Edema":                      rand_choice(data["48m follow-up Edema"], n),
    "72m follow-up Edema":                      rand_choice(data["72m follow-up Edema"], n),
    "Baseline SkinInduration":                  rand_choice(data["Baseline SkinInduration"], n),
    "Post-RT SkinInduration":                   rand_choice(data["Post-RT SkinInduration"], n),
    "12m follow-up SkinInduration":             rand_choice(data["12m follow-up SkinInduration"], n),
    "24m follow-up SkinInduration":             rand_choice(data["24m follow-up SkinInduration"], n),
    "36m follow-up SkinInduration":             rand_choice(data["36m follow-up SkinInduration"], n),
    "60m follow-up SkinInduration":             rand_choice(data["60m follow-up SkinInduration"], n),
    "48m follow-up SkinInduration":             rand_choice(data["48m follow-up SkinInduration"], n),
    "72m follow-up SkinInduration":             rand_choice(data["72m follow-up SkinInduration"], n),
    "Baseline Erythema":                        rand_choice(data["Baseline Erythema"], n),
    "Post-RT Erythema":                         rand_choice(data["Post-RT Erythema"], n),
    "12m follow-up Erythema":                   rand_choice(data["12m follow-up Erythema"], n),
    "24m follow-up Erythema":                   rand_choice(data["24m follow-up Erythema"], n),
    "36m follow-up Erythema":                   rand_choice(data["36m follow-up Erythema"], n),
    "60m follow-up Erythema":                   rand_choice(data["60m follow-up Erythema"], n),
    "48m follow-up Erythema":                   rand_choice(data["48m follow-up Erythema"], n),
    "72m follow-up Erythema":                   rand_choice(data["72m follow-up Erythema"], n),
    "Baseline SkinHyperpigment":                rand_choice(data["Baseline SkinHyperpigment"], n),
    "Post-RT SkinHyperpigment":                 rand_choice(data["Post-RT SkinHyperpigment"], n),
    "12m follow-up SkinHyperpigment":           rand_choice(data["12m follow-up SkinHyperpigment"], n),
    "24m follow-up SkinHyperpigment":           rand_choice(data["24m follow-up SkinHyperpigment"], n),
    "36m follow-up SkinHyperpigment":           rand_choice(data["36m follow-up SkinHyperpigment"], n),
    "60m follow-up SkinHyperpigment":           rand_choice(data["60m follow-up SkinHyperpigment"], n),
    "48m follow-up SkinHyperpigment":           rand_choice(data["48m follow-up SkinHyperpigment"], n),
    "72m follow-up SkinHyperpigment":           rand_choice(data["72m follow-up SkinHyperpigment"], n),
    "Sentinel node biopsy":                     rand_choice(data["Sentinel node biopsy"], n),
    "Planned axillary dissection":              rand_choice(data["Planned axillary dissection"], n),
    "DB":                                       [0] * n,
    })


    # save_data_gen_cmp_charts(data, generated_data)
    
    today = today_str()
    generated_filename =f"gendata_{today}.csv"
    generated_data.to_csv(generated_filename)
    data = obtain_data(generated_filename)

    write_fake_data_test_file(generated_data, 2000)
