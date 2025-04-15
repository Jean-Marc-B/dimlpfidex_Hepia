from sys import argv
import pandas as pd
from src.data_helper import obtain_data


if __name__ == "__main__":
    nb_rows_to_gen = 1000
    dataset_path = "dataset/clinical_complete_rev1.csv"

    if len(argv) == 2:
        nb_rows_to_gen = argv[1]

    data, _ = obtain_data(dataset_path)
    data = data.assign(BMI=lambda x: round(x.WEIGHT / (x.HEIGHT / 100.0) ** 2, 3))
    print(data.head())

    generated_data = pd.DataFrame(
        {column.unique() for column in data if column.dtype in []}
    )
