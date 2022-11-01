from dash import html, dcc, Dash
from dash.dependencies import Input, Output
import pickle
import numpy as np
import warnings
from assets.style import TEXT_STYLE, SUB_CONTENT_STYLE, CONTENT_STYLE

warnings.filterwarnings('ignore')


# Lien : https://auriez-vous-survecu-au-titanic.herokuapp.com/

def is_survied(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    model = pickle.load(open('finalized_model_titanic.sav', 'rb'))
    x = np.array([Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]).reshape(1, 7)
    return model.predict(x)[0], model.predict_proba(x)[0][1]


app = Dash(__name__, title='Titanic')
server = app.server

# class=2, Sex=0, Age=28, SibSp=1, Parch=2, Fare=14, Embarked=0

app.layout = html.Div([
    html.Div([
        html.H1(["Auriez-vous survécu au naufrage du titanic ?"], style=TEXT_STYLE),

        html.Div([
            "Classe: ",
            dcc.RadioItems(options=[
                {'label': '1ère', 'value': '1'},
                {'label': '2ème', 'value': '2'},
                {'label': '3ème', 'value': '3'}],
                value='1',
                id='Pclass',
                style=TEXT_STYLE)
        ], style=TEXT_STYLE),
        html.Div([
            "Sexe: ",
            dcc.RadioItems(options=[
                {'label': 'Homme', 'value': '0'},
                {'label': 'Femme', 'value': '1'}],
                value='0',
                id='Sex',
                style=TEXT_STYLE)
        ], style=TEXT_STYLE),
        html.Div([
            "Age: ",
            dcc.Input(id="Age", type="number", min=0, max=99, value=30)
        ], style=TEXT_STYLE),
        html.Div([
            "Nombre de frères et sœurs/conjoints à bord: ",
            dcc.Input(id="SibSp", type="number", min=0, max=10, value=0)
        ], style=TEXT_STYLE),
        html.Div([
            "Nombre de parents/enfants à bord: ",
            dcc.Input(id="Parch", type="number", min=0, max=10, value=0)
        ], style=TEXT_STYLE),
        html.Div([
            "Prix du ticket (en livres): ",
            dcc.Input(id="Fare", type="number", min=1, max=1000, value=15)
        ], style=TEXT_STYLE),
        html.Div([
            "Port d'embarquement: ",
            dcc.RadioItems(options=[
                {'label': 'Cherbourg', 'value': '1'},
                {'label': 'Queenstown', 'value': '2'},
                {'label': 'Southampton', 'value': '0'},
            ],
                value='1',
                id='Embarked',
                style=TEXT_STYLE
            )
        ], style=TEXT_STYLE),
        html.Br(),
        html.Div(id='result'),

        html.Br(),
        html.Br(),

        html.Div(['Comment les prédictions sont faites ?'], style=TEXT_STYLE),
        html.Div(["Nous disposons des données de 891 des passagers. "
                  "Le modèle utilisé sur cette page est un KNeighborsClassifier ayant une précision de 82,68 %. "
                  "Ce dernier a été entrainé et testé avec les données réelles du Titanic. "],
                 style={'color': 'white', 'fontSize': 8}),

    ],
        style=SUB_CONTENT_STYLE)
],
    style=CONTENT_STYLE)


@app.callback(
    Output(component_id='result', component_property='children'),
    Input(component_id='Pclass', component_property='value'),
    Input(component_id='Sex', component_property='value'),
    Input(component_id='Age', component_property='value'),
    Input(component_id='SibSp', component_property='value'),
    Input(component_id='Parch', component_property='value'),
    Input(component_id='Fare', component_property='value'),
    Input(component_id='Embarked', component_property='value'),
)
def update_output_div(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    try:
        survived, percentage_survived = is_survied(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
        color = 'red' if survived == 0 else "rgba(0, 255, 0, 1)"

        return html.Div([f'Chance de survivre :{str(percentage_survived * 100)}%'],
                        style={'color': color, 'fontSize': 25})
    except Exception:
        return 'Erreur'


if __name__ == '__main__':
    app.run_server(debug=True)

