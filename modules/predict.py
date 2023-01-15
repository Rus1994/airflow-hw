# <YOUR_IMPORTS>
import dill
import os
import pandas as pd
from datetime import datetime


path = os.environ.get('PROJECT_PATH', '.')

def get_path_last_model():
    path_model = f'{path}/data/models'
    models = os.listdir(path_model)
    if len(models) == 0:
        return ''

    for name in reversed(models):
        if name.endswith('.pkl'):
            return path_model + '/' + name
    else:
        return ''


def get_last_pipe():
    file_name = get_path_last_model()
    if file_name == '':
        return None
    with open(file_name, 'rb') as file:
        return dill.load(file)


def get_test_data_frame():
    path_test = f'{path}/data/test'
    files_name = [f for f in os.listdir(path_test) if f.endswith('.json')]
    frames = []
    for f in files_name:
        name = path_test + '/' + f
        frames.append(pd.read_json(name, typ='series'))
    return pd.DataFrame(frames, index=range(len(frames)))


def predict():
    best_pipe = get_last_pipe()
    if best_pipe is None:
        return

    df = get_test_data_frame()
    prediction = best_pipe.predict(df)
    res_frame = pd.DataFrame({'cars_id': df['id'], 'prediction': prediction})
    file_name = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    res_frame.to_csv(file_name, index=False)


if __name__ == '__main__':
    predict()
