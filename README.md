Cargamos el fichero Leads.csv en el que vamos a tratar un problema de clasificación binaria. La variable objetivo será la columna Converted, que indicará con un 1 si el cliente se ha convertido en cliente real, o con un 0 en caso contrario.

Primero, remplazamos el nombre de las columnas y los valores a minuscula y remplaza los espacios por guines bajos:

    replacer = lambda str: str.lower().str.replace(' ', '_')
    df.columns = replacer(df.columns.str)
    for col in list(df.dtypes[df.dtypes == 'object'].index):
        df[col] = replacer(df[col].str)
    df.head().T

Sumamos los valores nulos de cada columna y borramos todas aquellas columnas que superen los 3000 nulos:

    df.isnull().sum()

    for i in df.columns:
    if df[i].isna().sum()>3000:
        df.drop(i, axis=1, inplace=True)

Completamos los valores categoricos faltantes con el texto 'por_seleccionar'

    df['lead_source'].fillna('por_seleccionar', inplace=True)
    df['last_activity'].fillna('por_seleccionar', inplace=True)
    df['country'].fillna('por_seleccionar', inplace=True)
    df['specialization'].fillna('por_seleccionar', inplace=True)
    df['how_did_you_hear_about_x_education'].fillna('por_seleccionar', inplace=True)
    df['what_is_your_current_occupation'].fillna('por_seleccionar', inplace=True)
    df['what_matters_most_to_you_in_choosing_a_course'].fillna('por_seleccionar', inplace=True)
    df['lead_profile'].fillna('por_seleccionar', inplace=True)
    df['city'].fillna('por_seleccionar', inplace=True)

Completamos los valores numericos faltantes con el número 0

    df['totalvisits'].fillna(0, inplace=True)
    df['page_views_per_visit'].fillna(0, inplace=True)

Comprobamos que ya no existe ninguna columna con valores nulos:

    df.isnull().sum()

Comprobamos si los valores duplicados de la columna 'prospect_id' es igual a 0

    sum(df.duplicated(subset = 'prospect_id')) == 0

Comprobamos si los valores duplicados de la columna 'lead_number' es igual a 0
    
    sum(df.duplicated(subset = 'lead_number')) == 0

Eliminando los campos 'prospect_id' y 'lead_number', ya que tienen valores únicos

    df.drop(['prospect_id', 'lead_number'], 1, inplace = True)

Comprueba los valores posibles que existen en las variables categoricas y numericas

    categorical = ['lead_origin','lead_source','do_not_email','do_not_call','last_activity','country','specialization','how_did_you_hear_about_x_education','what_is_your_current_occupation','what_matters_most_to_you_in_choosing_a_course','search','magazine','newspaper_article','x_education_forums','newspaper','digital_advertisement','through_recommendations','receive_more_updates_about_our_courses','update_me_on_supply_chain_content','lead_profile','city','i_agree_to_pay_the_amount_through_cheque','a_free_copy_of_mastering_the_interview','last_notable_activity']
    
    numerical = ['totalvisits','total_time_spent_on_website','page_views_per_visit']

    df[categorical].nunique()

Divide los datos de entrenamiento y test (NO entrena con la columna 'churn')

    from sklearn.model_selection import train_test_split
    df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)

    df_train, df_val = train_test_split(df_train_full, test_size=0.33, random_state=1)
    y_train = df_train.converted.values
    y_val = df_val.converted.values

    del df_train['converted']
    del df_val['converted']

Hacemos una primera aproximación (Devuelve la cantidad en %)

    global_mean = df_train_full.converted.mean()
    round(global_mean, 3)

Comprobamos las variables categoricas mas correctas
    
    from sklearn.metrics import mutual_info_score

    calculate_mi = lambda col: mutual_info_score(col, df_train_full.converted)

    df_mi = df_train_full[categorical].apply(calculate_mi)
    df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')
    df_mi

Convierte los datos de entrenamiento a una lista de diccionario
    train_dict = df_train[categorical + numerical].to_dict(orient='records')
    dict(sorted(train_dict[0].items()))

Preparamos la instancia para transformar el diccionario a una matriz numerica

    from sklearn.feature_extraction import DictVectorizer

    dv = DictVectorizer(sparse=False)
    dv.fit(train_dict)

    X_train = dv.transform(train_dict)
    X_train[0]

Imprimimos lo que significa cada valor del array anterior

    dv.get_feature_names_out()

Entrenamos el modelo

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)

Realizamos predicciones

    val_dict = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dict)
    y_pred = model.predict_proba(X_val)
    y_pred

Nos quedamos con la columna de la derecha que es la que nos interesa
    
    y_pred = model.predict_proba(X_val)[:, 1]
    y_pred

Asignamos un punto de corte, es decir aquellas probabilidades que superiores al 50% serán
las que decidamos

    converted = y_pred >= 0.5
    converted

Calcular la precisión del modelo
Valores reales == valores de predicción que devuelve 0.824

    round((y_val == converted).mean(), 3)

Creamos los ficheros 'converted_predict_service.py' y 'converted_predict_app.py'.
En la consola ejecutamos:

    poetry add flask

y para realizar la prueba:

    poetry run python practica/converted_predict_app.py