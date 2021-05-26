import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

if __name__ == '__main__':
    data = pd.read_csv(r'D:\IIStudy\scientist_project\data\update_data.csv', delimiter=',')

    # Размер дата фрейма
    print('Размер Dataframe: ', data.shape)
    data.drop('Unnamed: 0', inplace=True, axis=1)
    print(data.head(5))

    # Получаем обобщенную информацию о DataFrame
    print('\n\nИнформация о Dataframe df.info():')
    print(data.info())

    numeric_data_with_target = ["ProductName", 'CreditSum', 'Employment', 'sex', 'age', 'EducationStatus',
                                'kolichestvo_rabotnikov_v_organizacii', 'OrgStanding_N', 'kolichestvo_detej_mladshe_18',
                                'Residence', 'HaveSalaryCard', 'IsBankWorker',
                                'harakteristika_tekutschego_trudoustrojstva', 'ConfirmedMonthlyIncome (Target)',
                                'LivingRegionName',
                                'otrasl_rabotodatelya',

                                ]

    numeric_data = ["ProductName", 'CreditSum', 'Employment', 'sex', 'age', 'EducationStatus',
                    'kolichestvo_rabotnikov_v_organizacii', 'OrgStanding_N', 'kolichestvo_detej_mladshe_18',
                    'Residence', 'HaveSalaryCard', 'IsBankWorker', 'harakteristika_tekutschego_trudoustrojstva',
                    'otrasl_rabotodatelya',
                    'LivingRegionName', 'month', 'day', 'year']

    numeric_data_test = ['ProductName', 'Employment', 'CreditSum', 'sex', 'age', 'EducationStatus',
                         'kolichestvo_rabotnikov_v_organizacii', 'OrgStanding_N', 'kolichestvo_detej_mladshe_18',
                         'Residence', 'IsBankWorker', 'harakteristika_tekutschego_trudoustrojstva', 'LivingRegionName',
                         'otrasl_rabotodatelya']

    print(data[numeric_data_test].info())

    sn.heatmap(data.corr())

    min_max_scaler = MinMaxScaler()
    data_norm = data[numeric_data_with_target].copy()
    data_norm = min_max_scaler.fit_transform(data_norm)

    data_norm = pd.DataFrame(data_norm, columns=data[numeric_data_with_target].columns)

    target_name = "ConfirmedMonthlyIncome (Target)"
    target = data[target_name]
    data_norm.drop(target_name, axis=1, inplace=True)

    data.drop(target_name, axis=1, inplace=True)

    # clf = RandomForestRegressor()
    # clf.fit(data_norm, target)
    #
    # plt.figure(figsize=(12, 12))
    # plt.bar(data_norm.columns, clf.feature_importances_)
    # plt.xticks(rotation=45)
    # plt.show()

    data_train, data_test, target_values_train, target_values_test = train_test_split(data_norm,
                                                                                      target,
                                                                                      test_size=0.40, random_state=42)
    model_regressor = LinearRegression()
    model_regressor.fit(data_train, target_values_train)

    # model_logistic = LogisticRegression()
    # model_logistic.fit(data_train, target_values_train)

    model_tree = DecisionTreeRegressor(random_state=42,max_depth=5, min_samples_split=2, min_samples_leaf=7)
    model_tree.fit(data_train, target_values_train)

    real_forest = RandomForestRegressor(random_state=42, max_depth=7, n_estimators=60, min_samples_split=2,
                                        min_samples_leaf=5)
    real_forest.fit(data_train, target_values_train)

    parametrs = {
        'max_depth': range(1, 9, 2),
        'min_samples_leaf': range(1, 8),
        'min_samples_split': range(2, 6, 2)}

    random_forest = RandomForestRegressor()
    #
    # random_parametrs = {'n_estimators': range(10, 51, 10),
    #                     'max_depth': range(1, 13, 2),
    #                     'min_samples_leaf': range(1, 8),
    #                     'min_samples_split': range(2, 10, 2)}

    # model_grid_random_forest = GridSearchCV(random_forest, parametrs, cv=5)
    # model_grid_random_forest.fit(data_train, target_values_train)
    # print(model_grid_random_forest.best_params_)

    # model_grid = DecisionTreeRegressor()
    # grid = GridSearchCV(model_grid, parametrs, cv=5)
    # grid.fit(data_train, target_values_train)
    # print(grid.best_params_)

    print('Linear regression')

    print("MAE: {0:7.2f}, RMSE: {1:7.2f}, R2: {2:7.2f} for linear model(test)".format(
        mean_absolute_error(target_values_test, model_regressor.predict(data_test)),
        mean_squared_error(target_values_test, model_regressor.predict(data_test)) ** 0.5,
        r2_score(target_values_test, model_regressor.predict(data_test))))

    print("MAE: {0:7.2f}, RMSE: {1:7.2f}, R2: {2:7.2f} for linear model(train)".format(
        mean_absolute_error(target_values_train, model_regressor.predict(data_train)),
        mean_squared_error(target_values_train, model_regressor.predict(data_train)) ** 0.5,
        r2_score(target_values_train, model_regressor.predict(data_train))))

    print()
    print('DecisionTree')

    print("MAE: {0:7.2f}, RMSE: {1:7.2f}, R2: {2:7.2f} for tree model(test)".format(
        mean_absolute_error(target_values_test, model_tree.predict(data_test)),
        mean_squared_error(target_values_test, model_tree.predict(data_test)) ** 0.5,
        r2_score(target_values_test, model_tree.predict(data_test))))

    print("MAE: {0:7.2f}, RMSE: {1:7.2f}, R2: {2:7.2f} for tree model(train)".format(
        mean_absolute_error(target_values_train, model_tree.predict(data_train)),
        mean_squared_error(target_values_train, model_tree.predict(data_train)) ** 0.5,
        r2_score(target_values_train, model_tree.predict(data_train))))

    print()
    print('Random Tree')

    print("MAE: {0:7.2f}, RMSE: {1:7.2f}, R2: {2:7.2f} for randomTree model(test)".format(
        mean_absolute_error(target_values_test, real_forest.predict(data_test)),
        mean_squared_error(target_values_test, real_forest.predict(data_test)) ** 0.5,
        r2_score(target_values_test, real_forest.predict(data_test))))

    print("MAE: {0:7.2f}, RMSE: {1:7.2f}, R2: {2:7.2f} for randomTree model(train)".format(
        mean_absolute_error(target_values_train, real_forest.predict(data_train)),
        mean_squared_error(target_values_train, real_forest.predict(data_train)) ** 0.5,
        r2_score(target_values_train, real_forest.predict(data_train))))

    plt.figure(figsize=(6, 6))
    plt.scatter(target_values_test, model_regressor.predict(data_test))
    plt.plot([0, max(target_values_test)], [0, max(model_regressor.predict(data_test))])
    plt.xlabel('Настоящее значение', fontsize=15)
    plt.ylabel('Предсказанное значение', fontsize=15)
    plt.title('Test data(Linear regression)', fontsize=15)
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(target_values_train, model_regressor.predict(data_train))
    plt.plot([0, max(target_values_train)], [0, max(model_regressor.predict(data_train))])
    plt.xlabel('Настоящее значение', fontsize=15)
    plt.ylabel('Предсказанное значение', fontsize=15)
    plt.title('Train data(Linear regression)', fontsize=15)

    plt.figure(figsize=(6, 6))
    plt.scatter(target_values_train, model_tree.predict(data_train))
    plt.plot([0, max(target_values_train)], [0, max(model_tree.predict(data_train))])
    plt.xlabel('Настоящее значение', fontsize=15)
    plt.ylabel('Предсказанное значение', fontsize=15)
    plt.title('Train data(Decision tree)', fontsize=15)

    plt.figure(figsize=(6, 6))
    plt.scatter(target_values_train, real_forest.predict(data_train))
    plt.plot([0, max(target_values_train)], [0, max(real_forest.predict(data_train))])
    plt.xlabel('Настоящее значение', fontsize=15)
    plt.ylabel('Предсказанное значение', fontsize=15)
    plt.title('Train data(Random forest)', fontsize=15)

    kfold = 5  # количество подвыборок для валидации
    itog_val = {}  # список для записи результатов кросс валидации разных алгоритмов

    scores = cross_val_score(model_regressor, data_train, target_values_train, cv=kfold)
    itog_val['LinearRegression'] = scores.mean()
    itog_val['DecisionTree'] = cross_val_score(model_tree, data_train, target_values_train, cv=kfold).mean()
    itog_val['RandomForest'] = cross_val_score(real_forest, data_train, target_values_train, cv=kfold).mean()

    DataFrame.from_dict(data=itog_val, orient='index').plot(kind='bar', legend=False)

    plt.show()
