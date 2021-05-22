from datetime import datetime

import pandas as pd
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    data = pd.read_csv(r'D:\IIStudy\scientist_project\data\data.csv', delimiter=',')

    # Размер дата фрейма
    print('Размер Dataframe: ', data.shape)
    print(data.head(5))

    # Получаем обобщенную информацию о DataFrame
    print('\n\nИнформация о Dataframe df.info():')
    print(data.info())

    # На оставшихся данных проведем нормализацию
    object_features = ["NaturalPersonID", "RequestDate", "ProductName", "TypeOfWork", "Employment",
                       "SignIP", "sex", "EducationStatus", "otrasl_rabotodatelya",
                       "kolichestvo_rabotnikov_v_organizacii", "Employment.1", "LivingRegionName",
                       "Residence", "IncomeDocumentKind", "HaveSalaryCard", "IsBankWorker", "TypeOfWork.1", "SignIP.1",
                       "harakteristika_tekutschego_trudoustrojstva"]
    float64_features = ["CreditSum", "OrgStanding_N"]
    # преобразуем nan
    data["CreditSum"] = data["CreditSum"].fillna(0).astype(str)
    data["OrgStanding_N"] = data["OrgStanding_N"].fillna(0).astype(str)
    data["Employment"] = data["Employment"].fillna('нет данных').astype(str)
    data["LivingRegionName"] = data["Employment"].fillna('нет данных').astype(str)
    data["otrasl_rabotodatelya"] = data["otrasl_rabotodatelya"].fillna('нет данных').astype(str)
    data["kolichestvo_rabotnikov_v_organizacii"] = data["kolichestvo_rabotnikov_v_organizacii"].fillna(
        'нет данных').astype(str)
    data["harakteristika_tekutschego_trudoustrojstva"] = data["harakteristika_tekutschego_trudoustrojstva"].fillna(
        'нет данных').astype(str)
    data["otrasl_rabotodatelya"] = data["otrasl_rabotodatelya"].fillna('нет данных').astype(str)

    # меняем тип данных на float
    data["CreditSum"] = data["CreditSum"].str.replace(',', '.').astype(float)
    data["OrgStanding_N"] = data["OrgStanding_N"].str.replace(',', '.').astype(float)
    data["ConfirmedMonthlyIncome (Target)"] = data["ConfirmedMonthlyIncome (Target)"].str.replace(',', '.').astype(
        float)
    data["Residence"] = data["Residence"].fillna('нет данных').astype(str)
    # убираем дубликаты столбцов
    data.drop("Employment.1", axis=1, inplace=True)
    data.drop("TypeOfWork.1", axis=1, inplace=True)
    data.drop("SignIP.1", axis=1, inplace=True)

    label = LabelEncoder()
    dicts = {}

    label.fit(data.sex)
    dicts['sex'] = list(label.classes_)
    data.sex = label.transform(data.sex)

    label.fit(data.otrasl_rabotodatelya)
    dicts['otrasl_rabotodatelya'] = list(label.classes_)
    data.otrasl_rabotodatelya = label.transform(data.otrasl_rabotodatelya)

    label.fit(data.LivingRegionName)
    dicts['LivingRegionName'] = list(label.classes_)
    data.LivingRegionName = label.transform(data.LivingRegionName)

    data['month'] = data['RequestDate'].apply(lambda x: x.split('/')[0]).astype(int)
    data['day'] = data['RequestDate'].apply(lambda x: x.split('/')[1]).astype(int)
    data['year'] = data['RequestDate'].apply(lambda x: x.split('/')[2]).astype(int)

    # print(data["ConfirmedMonthlyIncome (Target)"])
    # получаем уникальные значения
    uniq = data.EducationStatus.unique()
    # создаем словарик приоритета образования
    education_priority = {
        'Незаконченное среднее образование': 1,
        'Среднее образование': 2,
        'Среднее специальное образование': 3,
        'Незаконченное высшее образование': 4,
        'Высшее образование': 5,
        'Несколько высших образований': 6,
        'Академическая степень (кандидат наук, доктор наук и т.д.)': 7
    }
    data.EducationStatus = data.EducationStatus.replace(to_replace=education_priority)

    residence_dict = {
        'город': 2,
        'село': 1,
        'нет данных': 0
    }

    # sex_dict = {
    #     'Женский': 1,
    #     'Мужской': 2
    # }
    product_name = {
        'Кредит на потребительские нужды': 1,
        'Кредитная карта': 2
    }

    bank_worker = {
        'нет': 0,
        'да': 1,
    }

    number_worker = {
        'нет данных': 0,
        'до 20': 1,
        '21-100': 2,
        '101-500': 3,
        'более 500': 4
    }

    type_employment = {
        'нет данных': 0,
        'Сотрудник \\ Рабочий \\ Ассистент': 1,
        'Главный Специалист\\Руководитель среднего звена': 2,
        'Руководитель высшего звена': 3,
        'Эксперт\\Старший или Ведущий Специалист': 4,
        'Индивидуальный предприниматель': 5
    }

    character_work = {
        'Постоянная, полная занятость': 2,
        'Частичная или временная занятость': 1,
        'нет данных': 0
    }
    salary_card = {
        'нет': 0,
        'да': 1
    }

    data.Residence = data.Residence.replace(to_replace=residence_dict)
    # data.sex = data.sex.replace(to_replace=sex_dict)
    data.ProductName = data.ProductName.replace(to_replace=product_name)
    data.IsBankWorker = data.IsBankWorker.replace(to_replace=bank_worker)
    data.kolichestvo_rabotnikov_v_organizacii = data.kolichestvo_rabotnikov_v_organizacii.replace(
        to_replace=number_worker)
    data.Employment = data.Employment.replace(to_replace=type_employment)
    data.harakteristika_tekutschego_trudoustrojstva = data.harakteristika_tekutschego_trudoustrojstva.replace(
        to_replace=character_work)
    data.HaveSalaryCard = data.HaveSalaryCard.replace(to_replace=salary_card)

    data.drop(['RequestDate'], axis=1, inplace=True)
    print(data.info())
    print(data.shape)
    print(data.head(5))

    data.to_csv(r'D:\IIStudy\scientist_project\data\update_data.csv', encoding='utf-8')

    # continouns_features = set(data.columns) - set(object_features)
    # # continouns_features = set(float64_features)
    # min_max_scaler = MinMaxScaler()
    # data_norm = data.copy()
    # print(data_norm.info())
    # data_norm[list(continouns_features)] = min_max_scaler.fit_transform(data[list(continouns_features)])
    #
    # corrCoef = data_norm.corr()
    # data_norm.corr().style.format("{:.2}").background_gradient(cmap='coolwarm', axis=1)
