import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

from math import sqrt


def Backend(dataframe, modelType, Xcol, predictorColumn):
    

    class MachineLearningCode():
        def __init__(self, dataframe, modelType, Xcol, predictorColumn):
            self.Xcol = Xcol
            self.predictorColumn = predictorColumn
            df = dataframe
            df2 = df[Xcol].copy()
            df2[predictorColumn] = df[predictorColumn].copy()
            #df2=df[Xcol]
            #df2[predictorColumn]=df[predictorColumn]
            self.encode_categorical_features(df2)
            del df
            self.estimators = []
            self.modelmetrics = [[], []]  # List to store model names and scores
            self.KFold(df2, modelType)
            
            
            
        def encode_categorical_features(self, df):
            label_encoder = LabelEncoder()
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = label_encoder.fit_transform(df[col])
                
        def KFold(self, df, modelType):
            for col in self.Xcol:
                scaler = StandardScaler()
                try:
                    df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
                    
                except Exception as e:
                    print('fail', col, e)
                    pass

            X, Y = df.drop(self.predictorColumn, axis=1), df[self.predictorColumn]

            if modelType == 'regression':
                self.estimators = [['Linear regression', self.Regression_1_Linear_Regression()],
                              ['Random Forest regressor', self.Regression_2_Random_Forest_Regression(X, Y)],
                              ['KNN Regression', self.Regression_3_KNN(X, Y)]]
                scoring = 'neg_mean_squared_error'
            elif modelType == 'classification':
                self.estimators = [['KNN Classifier', self.Classification_1_KNN(X, Y)],
                              ['Random Forest Classifier', self.Classification_2_Random_Forest(X, Y)],
                              ['Logistic Regression', self.Classification_3_Logistic_Regression()]]
                scoring = 'f1'
            self.modelmetrics = [[], []]
            for model in self.estimators:
                try:
                    print(model[0])
                    kf = cross_validate(X=X, y=Y, estimator=model[1], scoring=scoring, cv=5)
                    self.modelmetrics[0].append(model[0])
                    self.modelmetrics[1].append(kf['test_score'].mean())
                except Exception as e:
                    print('Error', e)
            if modelType == 'regression':
                bestmodelindex = np.argmax(self.modelmetrics[1])
            elif modelType == 'classification':
                bestmodelindex = np.argmin(self.modelmetrics[1])
            print('Best model:', bestmodelindex + 1, self.modelmetrics[0][bestmodelindex])
            self.Visualise_Results(self.modelmetrics[0], self.modelmetrics[1], scoring)

        def Visualise_Results(self, x, y, scoring):
            plt.figure(figsize=(13, 7))
            plt.bar(x=x, height=y, color="blue")
            titlelabel = 'A bar chart to show the difference in model performance'
            plt.xticks(rotation=15, fontsize=16)
            plt.yticks(fontsize=14)
            plt.title(titlelabel, fontsize=20, fontweight="bold")
            plt.xlabel('Model', fontsize=17)
            plt.ylabel('Mean ' + scoring, fontsize=17)
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            plt.show()

        def Classification_1_KNN(self, x, y):
            X_train, X_test, Y_train, Y_test = train_test_split(x, y, stratify=y, random_state=0)
            error_rate = []
            for i in range(1, 7):
                knn = KNeighborsClassifier(n_neighbors=i)
                knn.fit(X_train, Y_train)
                pred_i = knn.predict(X_test)
                error_rate.append(np.mean(pred_i != Y_test))
            print(error_rate)
            plt.figure(figsize=(10, 6))
            plt.plot(error_rate)
            plt.title('Error Rate vs. K Value')
            plt.xlabel('K')
            plt.ylabel('Error Rate')
            plt.show()

            K = self.best_k_value(error_rate)
            knn = KNeighborsClassifier(n_neighbors=K)
            return knn

        def Classification_2_Random_Forest(self, train_features, train_labels):
            param_grid = {
                'bootstrap': [False],
                'max_depth': [3],
                'max_features': [3],
                'n_estimators': [10, 30, 60],
                'random_state': [0]
            }

            rf = RandomForestClassifier()
            return self.MyGridSearch(rf, param_grid, train_features, train_labels)

        def Classification_3_Logistic_Regression(self):
            logreg = LogisticRegression(random_state=0)
            return logreg

        def Regression_1_Linear_Regression(self):
            logreg = LinearRegression()
            return logreg

        def Regression_2_Random_Forest_Regression(self, train_features, Y):
            param_grid = {
                'bootstrap': [False],
                'max_depth': [3],
                'n_estimators': [10, 30, 60],
                'random_state': [0]
            }
            rf = RandomForestRegressor()
            return self.MyGridSearch(rf, param_grid, train_features, Y)

        def Regression_3_KNN(self, X, Y):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
            rmse_val = []
            for K in range(1, 7):
                model = KNeighborsRegressor(n_neighbors=K)
                model.fit(X_train, Y_train)
                y_pred = model.predict(X_test)
                rmse_val.append(sqrt(mean_squared_error(Y_test, y_pred)))
            K = self.best_k_value(rmse_val)
            return KNeighborsRegressor(n_neighbors=K)

        def MyGridSearch(self, estimator, param_grid, train_features, train_labels):
            grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2,
                                       error_score='raise')
            grid_search.fit(train_features, train_labels)
            bestparams = grid_search.best_params_
            print(bestparams)
            return grid_search.best_estimator_

        def best_k_value(self, error_array):
            while len(error_array) > 2:
                if error_array[-2] <= error_array[-1] * 1.1:
                    error_array = error_array[:-1]
                else:
                    break
            print('final error_array', error_array[-1])
            K = len(error_array)
            return K

    ml = MachineLearningCode(dataframe, modelType, Xcol, predictorColumn)
    
    if len(ml.modelmetrics) >= 2:
        ModelAName = ml.modelmetrics[0][0]
        ModelA = ml.modelmetrics[0][0]
        ModelAScore = ml.modelmetrics[1][0]

        ModelBName = ml.modelmetrics[0][1]
        ModelB = ml.modelmetrics[0][1]
        ModelBScore = ml.modelmetrics[1][1]

        ModelCName = ml.modelmetrics[0][2]
        ModelC = ml.modelmetrics[0][2]
        ModelCScore = ml.modelmetrics[1][2]

        if max(ModelAScore, ModelBScore, ModelCScore)==ModelAScore:
            Model1Name = ModelAName
            Model1 = ModelA
            Model1Score = ModelAScore
            if min(ModelBScore, ModelCScore) == ModelBScore:
                Model3Name = ModelBName
                Model3 = ModelB
                Model3Score = ModelBScore
                Model2Name = ModelCName
                Model2 = ModelC
                Model2Score = ModelCScore
            else:
                Model3Name = ModelCName
                Model3 = ModelC
                Model3Score = ModelCScore
                Model2Name = ModelBName
                Model2 = ModelB
                Model2Score = ModelBScore
        elif max(ModelAScore, ModelBScore, ModelCScore)==ModelBScore:
            Model1Name = ModelBName
            Model1 = ModelB
            Model1Score = ModelBScore
            if min(ModelAScore, ModelCScore) == ModelAScore:
                Model3Name = ModelAName
                Model3 = ModelA
                Model3Score = ModelAScore
                Model2Name = ModelCName
                Model2 = ModelC
                Model2Score = ModelCScore
            else:
                Model3Name = ModelCName
                Model3 = ModelC
                Model3Score = ModelCScore
                Model2Name = ModelAName
                Model2 = ModelA
                Model2Score = ModelAScore
        else:
            Model1Name = ModelCName
            Model1 = ModelC
            Model1Score = ModelCScore
            if min(ModelAScore, ModelBScore) == ModelAScore:
                Model3Name = ModelAName
                Model3 = ModelA
                Model3Score = ModelAScore
                Model2Name = ModelBName
                Model2 = ModelB
                Model2Score = ModelBScore
            else:
                Model3Name = ModelBName
                Model3 = ModelB
                Model3Score = ModelBScore
                Model2Name = ModelAName
                Model2 = ModelA
                Model2Score = ModelAScore



        return Model1Name, Model1, Model1Score, Model2Name, Model2, Model2Score, Model3Name, Model3, Model3Score
    else:
        print('Not enough models found')
        return None, None, None, None, None, None, None, None, None


    return (Model1Name, Model1, Model1Score, Model2Name, Model2, Model2Score, Model3Name, Model3, Model3Score)
