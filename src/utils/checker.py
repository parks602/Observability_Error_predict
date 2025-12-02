import os


def model_date_checker(model_path: str, model_name: str, train_value: int):
    if not os.path.exists(f"{model_path}/{model_name}") & train_value == 2:
        print(f"Model file does not exist: {model_path}/{model_name}")
        return True
    elif os.path.getsize(model_path + model_name) == 0:
        print(f"Model file is empty: {model_path}/{model_name}")
        return False
    else:
        print(f"Model file exists: {model_path}/{model_name}")
        return False
