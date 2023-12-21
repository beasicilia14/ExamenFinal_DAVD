
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#leo los datos
df = pd.read_csv('bank-full.csv', sep=';')


app = dash.Dash(__name__)

#layout
app.layout = html.Div([
    html.H1(children='RESUMEN DE LA CAMPAÑA', style={'color':'white', 'textAlign':'center', 'background-color':'blue'}),
    html.H3(children='Seleccione un rango de balance anual medio para ver el resumen de la campaña', style={'color':'white', 'textAlign':'center', 'background-color':'blue'}),

    # slider para ingresos
    dcc.Slider(
        id='income-slider',
        min=df['balance'].min(),
        max=df['balance'].max(),
        step=1000,
        marks={i: f'{i:,}' for i in range(int(df['balance'].min()), int(df['balance'].max())+1, 10000)},
        value=df['balance'].min(),
        tooltip={'placement': 'bottom', 'always_visible': True}
    ),
    # grafico que enseña count de la variable y.
    dcc.Graph(id='count-plot'),
    # Gráfico de barras para la importancia de las variables
    dcc.Graph(id='feature-importance-plot'),
    # gráfico que nos enseña el numero de hipotecas que se tienen 
    dcc.Graph(id='housing-plot'),
    #grafico que nos esnseña cuantos tienen un prestamo personal
    dcc.Graph(id='loan-plot'),



])

# Para que los gráficos se actualice según el input. 
@app.callback(
    [Output('count-plot', 'figure'),
     Output('feature-importance-plot', 'figure'),
     Output('housing-plot', 'figure'),
     Output('loan-plot', 'figure')], 
    [Input('income-slider', 'value')]
)
def update_plots(income_threshold):
    # Filtrar el DataFrame por ingresos mayores que el umbral seleccionado
    filtered_df = df[df['balance'] > income_threshold]

    # Crear el gráfico de barras con Plotly Express para la distribución de la respuesta
    count_plot = px.histogram(filtered_df, x='y', title=f'Distribución de la respuesta si el balance anual es mayor que: {income_threshold}',
                              labels={'y': 'Respuesta a la campaña', 'count': 'Número de personas'},
                              category_orders={'y': ['no', 'yes']})
    count_plot.update_layout(bargap=0.2)  # Ajustar el espacio entre las barras

    #Para la importancia de las variables... 
    numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    nominal_features = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    binary_features = ['default', 'housing', 'loan']

    filtered_df = pd.get_dummies(filtered_df, columns=nominal_features, drop_first=True)
    filtered_df[binary_features] = filtered_df[binary_features].replace({'no': 0, 'yes': 1})

    filtered_df['y'] = filtered_df['y'].replace({'no': 0, 'yes': 1})

    X = filtered_df.drop('y', axis=1)
    y = filtered_df['y']

#Divido en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Actualizar el modelo y obtener la importancia de las variables
    rf_model.fit(X_train, y_train)
    #Obtener la importancia de las variables
    feature_importances = rf_model.feature_importances_
    # Crear un DataFrame para visualizar la importancia de las variables
    feature_importance_df = pd.DataFrame({'Variable': X.columns, 'Importancia': feature_importances})

    # Ordenar el DataFrame por importancia de variables
    feature_importance_df = feature_importance_df.sort_values(by='Importancia', ascending=False)

    # Crear el gráfico de barras con Plotly Express para la importancia de las variables
    feature_importance_plot = px.bar(feature_importance_df, x='Importancia', y='Variable', orientation='h',
                                     title=f'Importancia de las Variables en la Predicción de y si el balance anual es mayor que: {income_threshold}',
                                     labels={'Importancia': 'Importancia', 'Variable': 'Variable'})
    

    # Gráfico de pie para la variable housing 
    housing_plot = px.pie(filtered_df, names='housing', title=f'Distribución de la variable housing si el balance anual es mayor que: {income_threshold}',
                          labels={'housing': 'Housing'}, color_discrete_sequence=['lightblue', 'lightcoral'])
    
    # Gráfico de pie para la variable loan

    loan_plot = px.pie(filtered_df, names='loan', title=f'Distribución de la variable loan si el balance anual es mayor que: {income_threshold}',
                          labels={'loan': 'Loan'}, color_discrete_sequence=['lightblue', 'lightcoral'])
    

    return count_plot, feature_importance_plot, housing_plot, loan_plot



# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
