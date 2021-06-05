import pandas as pd
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    data = pd.read_csv(r'D:\IIStudy\scientist_project\data\data.csv', delimiter=',')

    # Размер дата фрейма
    print('Размер Dataframe: ', data.shape)
    print(data.head(5))
    print(data.SignIP.unique())

    # убираем дубликаты столбцов
    data.drop("Employment.1", axis=1, inplace=True)
    data.drop("TypeOfWork.1", axis=1, inplace=True)
    data.drop("SignIP.1", axis=1, inplace=True)

    # Получаем обобщенную информацию о DataFrame
    print('\n\nИнформация о Dataframe df.info():')
    print(data.info())

    # преобразуем nan
    data['TypeOfWork'] = data.TypeOfWork.fillna('нет данных').astype(str)
    data["Residence"] = data["Residence"].fillna('нет данных').astype(str)
    data['SignIP'] = data.SignIP.fillna('нет данных').astype(str)
    data = data.dropna()

    # меняем тип данных на float
    data["CreditSum"] = data["CreditSum"].str.replace(',', '.').astype(float)
    data["OrgStanding_N"] = data["OrgStanding_N"].str.replace(',', '.').astype(float)
    data["ConfirmedMonthlyIncome (Target)"] = data["ConfirmedMonthlyIncome (Target)"].str.replace(',', '.').astype(
        float)

    label = LabelEncoder()
    dicts = {}

    label.fit(data.otrasl_rabotodatelya)
    dicts['otrasl_rabotodatelya'] = list(label.classes_)
    data.otrasl_rabotodatelya = label.transform(data.otrasl_rabotodatelya)

    # one_encode = pd.get_dummies(data.otrasl_rabotodatelya, prefix='otrasl_rabotodatelya').astype(int)
    # one_encode = pd.DataFrame(one_encode)
    # data = pd.concat([data, one_encode], axis=1).drop(['otrasl_rabotodatelya'], axis=1)

    label.fit(data.IncomeDocumentKind)
    dicts['IncomeDocumentKind'] = list(label.classes_)
    data.IncomeDocumentKind = label.transform(data.IncomeDocumentKind)

    # one_encode = pd.get_dummies(data['IncomeDocumentKind'],prefix='IncomeDocumentKind').astype(int)
    # one_encode = pd.DataFrame(one_encode)
    # data = pd.concat([data, one_encode], axis=1).drop(['IncomeDocumentKind'], axis=1)

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

    sign_ip = {
        'нет данных': 0,
        'ИП': 1
    }

    type_of_work = {
        'нет данных': 0,
        'Собственное дело': 3,
        'по найму': 1,
        'Индивидуальный предприниматель': 4,
        'Агент на комиссионом договоре': 2
    }

    sex_dict = {
        'Женский': 1,
        'Мужской': 2
    }
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

    data.SignIP = data.SignIP.replace(to_replace=sign_ip)
    data['TypeOfWork'] = data.TypeOfWork.replace(to_replace=type_of_work)
    data.Residence = data.Residence.replace(to_replace=residence_dict)
    data.sex = data.sex.replace(to_replace=sex_dict)
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
