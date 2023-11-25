import pickle
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def save_pkl(data, path):
    with open(path, "wb") as saved_data:
        pickle.dump(data, saved_data)
    saved_data.close()

def load_pkl(path):
    to_return = None
    with open(path, "rb") as loaded_data:
        to_return = pickle.load(loaded_data)
    loaded_data.close()
    return to_return


def train_val_test_split(test_fold):
    test_df = load_pkl(f"./cnn_folds_dataframes/fold{test_fold}_df.pkl")
    train_df = pd.DataFrame(columns=test_df.columns)
    val_df = pd.DataFrame(columns=test_df.columns)
    print(test_df.columns)
    print(train_df.columns)
    print(val_df.columns)
    for i in range(1, 10+1):
        if i == test_fold: continue
        fold_df = load_pkl(f"./cnn_folds_dataframes/fold{i}_df.pkl")
        train_fold, val_fold = train_test_split(fold_df, test_size=0.1, random_state=42, stratify=['classID'])
        train_df = pd.concat([train_df, train_fold], axis=0, join='outer', ignore_index=True)
        val_df = pd.concat([val_df, val_fold], axis=0, join='outer', ignore_index=True)
        del fold_df
        del train_fold
        del val_fold
    X_train, y_train = train_df.drop(columns=['slice_file_name','classID']), train_df['classID']
    X_val, y_val = val_df.drop(columns=['slice_file_name','classID']), val_df['classID']
    X_test, y_test = test_df.drop(columns=['slice_file_name','classID']), test_df['classID']
    del test_df
    del train_df
    del val_df
    return X_train, y_train, X_val, y_val, X_test, y_test
    

def cross_validation_10_fold(compiled_model:keras.models.Model, model_type="CNN", **fit_params):
    metrics = {'accuracy':[], 'confusion_matrix':[]}
    # run cross validation
    for i in range(1, 10+1):
        # get the train, validation and testing data
        X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(i)
        if model_type == "CNN":
            # separate the information for the different inputs of the model
            X_train = {
                'chromagram_input' : X_train['chromogram'].to_numpy(),
                'mel_spectogram_input': X_train['mel_spectogram'].to_numpy(),
                'fourier_tempogram': X_train['fourier_tempogram'].to_numpy(),
                'features_1d': np.stack([X_train['spectral_centroid'].to_numpy(), X_train['spectral_bandwidth'].to_numpy(), X_train['spectral_flatness'].to_numpy(), X_train['spectral_rolloff'].to_numpy()], axis=-1)
            }
            X_val = {
                'chromagram_input' : X_val['chromogram'].to_numpy(),
                'mel_spectogram_input': X_val['mel_spectogram'].to_numpy(),
                'fourier_tempogram': X_val['fourier_tempogram'].to_numpy(),
                'features_1d': np.stack([X_val['spectral_centroid'].to_numpy(), X_val['spectral_bandwidth'].to_numpy(), X_val['spectral_flatness'].to_numpy(), X_val['spectral_rolloff'].to_numpy()], axis=-1)
            }
            X_test = {
                'chromagram_input' : X_test['chromogram'].to_numpy(),
                'mel_spectogram_input': X_test['mel_spectogram'].to_numpy(),
                'fourier_tempogram': X_test['fourier_tempogram'].to_numpy(),
                'features_1d': np.stack([X_test['spectral_centroid'].to_numpy(), X_test['spectral_bandwidth'].to_numpy(), X_test['spectral_flatness'].to_numpy(), X_test['spectral_rolloff'].to_numpy()], axis=-1)
            }
            # train the model on the current CV iteration data
            compiled_model.fit(
                x=X_train,
                y=y_train,
                #validation_data=(X_val, y_val),
                **fit_params
            )
        
        # test the model and obtain metrics
        y_pred = compiled_model.predict(X_test)
        y_test = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['confusion_matrix'].append(confusion_matrix(y_test, y_pred))

        # memory management
        del X_train
        del y_train
        del X_val
        del y_val
        del X_test
        del y_test
    
    return metrics
