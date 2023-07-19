import joblib
from pathlib import Path


def save_file(model, folder_name, file_name):
    FILES_PATH = Path() / folder_name
    FILES_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_name)


def load_file(folder_name, file_name):
    FILES_PATH = Path() / folder_name / file_name
    return joblib.load(FILES_PATH)
