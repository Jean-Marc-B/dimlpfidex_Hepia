from pathlib import Path

INPUT_DIRNAME = "input"
MODEL_DIRNAME = "model"
OUTPUT_DIRNAME = "output"
PATIENTS_DATA_DIRNAME = "patients_data"

def check_directories_existance(project_abspath: str) -> None:
    abspath = Path(project_abspath)
    abspath.joinpath(INPUT_DIRNAME).mkdir(exist_ok=True)
    abspath.joinpath(MODEL_DIRNAME).mkdir(exist_ok=True)
    abspath.joinpath(OUTPUT_DIRNAME).mkdir(exist_ok=True)
    abspath.joinpath(PATIENTS_DATA_DIRNAME).mkdir(exist_ok=True)