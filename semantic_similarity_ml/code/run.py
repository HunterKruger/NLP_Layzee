import joblib
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from feature_engineering import feature_engineering
from modeling import Modeling
from evaluation import BinClsEvaluation

def main():

    print('Loading data...')
    if config.ALREADY_SPLIT:
        df_all = pd.read_csv(config.TRAIN_FILE)
    else:
        df_all = pd.read_csv(config.ALL_FILE)
        df_train = df_all[:config.TRAIN_SAMPLES]
        df_test = df_all[config.TRAIN_SAMPLES: config.TRAIN_SAMPLES+config.TEST_SAMPLES]
    print('Loading finished!')

    print('Feature engineering...')
    t1 = time.time()
    df_train = feature_engineering(df_train, config.SENTENCE1_FIELD, config.SENTENCE2_FIELD) 
    df_test = feature_engineering(df_test, config.SENTENCE1_FIELD, config.SENTENCE2_FIELD) 
    t2 = time.time()
    print('Feature engineering done in ' + str(round(t2-t1, 2)) + 's.')

    print('Modeling...')
    y_train = df_train[config.LABEL_FIELD]
    y_test = df_test[config.LABEL_FIELD]

    X_train = df_train.drop([config.SENTENCE1_FIELD, config.SENTENCE2_FIELD, config.LABEL_FIELD], axis=1)
    X_test = df_test.drop([config.SENTENCE1_FIELD, config.SENTENCE2_FIELD, config.LABEL_FIELD], axis=1)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)   
    joblib.dump(ss, config.STANDARD_SCALER_PATH)

    model = Modeling(X_train, X_test, y_train, y_test, 'bin')
    y_score, best_model, _, _ = model.modeling(
        config.MODEL, 'roc_auc', cv=config.CV, hp='auto', strategy='random', 
        max_iter = config.MAX_ITER, parallelism_cv=config.PARALLELISM_CV, calibration=None)
    joblib.dump(best_model, config.MODEL_PATH) 

    print('Evaluation...')
    eva = BinClsEvaluation(y_score, y_test)
    eva.detailed_metrics()


if __name__ == "__main__":
    # execute only if run as a script
    main()


