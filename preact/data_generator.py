from sys import argv
import pandas as pd
import random as r


def rand_float(column: pd.Series, delta: float, n:int) -> list[float]:
    mean = column.mean()
    low_bound = mean - delta
    up_bound = mean + delta
    return [r.uniform(low_bound, up_bound) for _ in range(n)]


def rand_int(column: pd.Series, delta: int, n: int) -> list[int]:
    mean = round(column.mean(), 0)
    low_bound = mean - delta
    up_bound = mean + delta
    return [r.randint(low_bound, up_bound) for _ in range(n)]


def rand_choice(column: pd.Series, n: int) -> list:
    return [r.choice(column.dropna().unique()) for _ in range(n)]


if __name__ == "__main__":
    n = 10
    dataset_path = "dataset/clinical_complete_rev1.csv"

    if len(argv) == 2:
        nb_rows_to_gen = argv[1]

    data = pd.read_csv(dataset_path, index_col=False)

    print(data.info())

    generated_data = pd.DataFrame(
        {
    "height":                                   rand_int(data["height"], 20.0, n),
    "weight":                                   rand_int(data["weight"], 20.0, n),
    "Age":                                      rand_int(data["Age"], 40.0, n),
    "Smoker":                                   rand_choice(data["Smoker"], n),
    "Menopausal":                               rand_choice(data["Menopausal"], n),
    "Diabetes":                                 rand_choice(data["Diabetes"], n),
    "Surgery":                                  rand_choice(data["Surgery"], n),
    "Type Surgery":                             rand_choice(data["Type Surgery"], n),
    "Nodes involved":                           rand_float(data["Nodes involved"], 1.0, n),
    "Side of Primary":                          rand_choice(data["Side of Primary"], n),
    "Histological Type":                        rand_choice(data["Histological Type"], n),
    "Pathological tumour size":                 rand_int(data["Pathological tumour size"], 10.0, n),
    "Clinical T-Stage":                         rand_choice(data["Clinical T-Stage"], n),
    "Clinical N-Stage" :                        rand_choice(data["Clinical N-Stage"], n),
    "Ki-67 status":                             rand_int(data["Ki-67 status"], 20.0, n),
    "ER Status":                                rand_choice(data["ER Status"], n),
    "HER-2 Status":                             rand_choice(data["HER-2 Status"], n),
    "PR Status":                                rand_choice(data["PR Status"], n),
    "Number of fractions":                      rand_choice(data["Number of fractions"], n),
    "Mean Heart Dose":                          rand_float(data["Mean Heart Dose"], 2.5, n),
    "IMRT":                                     rand_choice(data["IMRT"], n),
    "3D":                                       rand_choice(data["3D"], n),
    "Treated breast radio":                     rand_choice(data["Treated breast radio"], n),
    "Boost":                                    rand_choice(data["Boost"], n),
    "neoadjuvant chemotherapy":                 rand_choice(data["neoadjuvant chemotherapy"], n),
    "adjuvant chemotherapy":                    rand_choice(data["adjuvant chemotherapy"], n),
    "Nodes examined":                           rand_int(data["Nodes examined"], 20.0, n),
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
        }
    )
    generated_data = generated_data.assign(BMI=lambda x: round(x["weight"] / (x["height"] / 100.0) ** 2, 3))
    print(generated_data.info())
    print(generated_data)
