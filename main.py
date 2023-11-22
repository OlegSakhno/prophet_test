def hypo():
    import pandas as pd
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    import itertools

    # Загрузка данных
    df = pd.read_csv('/home/osakhno/dev/prophet/forecasting - IQ_DAU.csv')
    df = df[["ds","y"]]
    df["y"] = df["y"].astype(int)
    df['ds'] = df['ds'].astype('datetime64[ns]')

    param_grid = {  
        'changepoint_prior_scale': [0.001, 0.1, 1],
        'seasonality_prior_scale': [0.01, 1.0, 10.0],
        'holidays_prior_scale': [0.01, 1.0, 10.0],
        'seasonality_mode':['additive','multiplicative'],
    }

    # Создание всех комбинаций параметров
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

    mape_scores = []  # Список для сохранения оценок MAPE

    # Перебор всех комбинаций и оценка модели
    for params in all_params:
        model = Prophet(**params).fit(df)  # df - исторические данные
        cv_results = cross_validation(model, horizon='30 days', period='180 days', initial='365 days')

        df_p = performance_metrics(cv_results, rolling_window = 1)
        mape_scores.append(df_p['mape'].values[0])

    # Выбор лучшей модели
    tuning_results = pd.DataFrame(all_params)
    tuning_results['mape'] = mape_scores
    tuning_results.sort_values(by = "mape", ascending = True).to_csv("hypo.csv")
    
    hypo = tuning_results.sort_values(by = "mape", ascending = True).iloc[0]

    return hypo

def forecast():

    from prophet import Prophet
    import pandas as pd
    from sklearn.metrics import mean_absolute_percentage_error
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    # Загрузка данных
    df = pd.read_csv('/home/osakhno/dev/prophet/forecasting - IQ_DAU.csv')
    df = df[["ds","y"]]
    df["y"] = df["y"].astype(int)
    df['ds'] = df['ds'].astype('datetime64[ns]')

    df["budget"] = 1

    params = {  
            'changepoint_prior_scale' : 0.03,
            'seasonality_prior_scale' : 3.0,
            'holidays_prior_scale' : 5.0,
            'seasonality_mode' : 'multiplicative'
    }

    model = Prophet(**params) # Создание модели
    
    model.add_regressor("budget")

    model.fit(df)

    # Создание датафрейма для предсказаний
    future = model.make_future_dataframe(periods=365)
    future["budget"] = 1
    future.loc[future[future['ds']>"2024-04-01"].index, "budget"] = 1.4


    forecast = model.predict(future) # Получение предсказаний

    # расчёт ошибки 
    y_true = df['y']  # 'y' - столбец с фактическими значениями
    y_pred = forecast['yhat'][:len(y_true)]  # 'yhat' - прогнозируемые значения
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    print(mape)
    # ..расчёт ошибки 

    result = forecast.join(
        df[['ds','y']].set_index('ds'), 
        on = 'ds', 
        rsuffix = ''
    )

    ax = result[
        (result["ds"]>="2023-09-01")&
        (result["ds"]<"2025-01-01")
        ].plot(x = "ds", y = ["yhat","y"], figsize=(16,9))

    #ax.set_ylim([10000, 60000])  # Замените min_value и max_value на желаемые значения

    # Установка формата даты на оси X
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Формат YYYY-MM-DD
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Например, деление оси по дням недели

    # Опционально: Поворот меток даты
    plt.setp(ax.get_xticklabels(), rotation=45)

if __name__ == "__main__":
    forecast()
