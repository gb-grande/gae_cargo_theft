# import pandas as pd
# embeddingsWithInfo = pd.read_csv("../data/embeddingsWithInfo.csv")
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import sys

#le nome do arquivo como arg
dfName = sys.argv[1]
df = pd.read_csv(dfName)

# Aplicação Dash
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Visualização dos Embeddings"),
    dcc.Graph(
        id="scatter-plot",
        config={"scrollZoom": True},
    ),
    dcc.Graph(
        id="map-plot",
        config={"scrollZoom": True},
    )
])

@app.callback(
    Output("map-plot", "figure"),
    Input("scatter-plot", "selectedData")
)
def update_map(selectedData):
    # Pegar os pontos selecionados
    if selectedData:
        points = selectedData["points"]
        selected_indices = [p["pointIndex"] for p in points]
        filtered_df = df.iloc[selected_indices]
    else:
        filtered_df = df  # Se nenhum ponto for selecionado, mostrar tudo

    mapCenter ={ 
        "lat" :-23.533773,
        "lon" : -46.625290
    }
    # Criar o mapa
    fig = px.scatter_mapbox(
        filtered_df,
        lat="latitude",
        lon="longitude",
        hover_name="node_id",
        center=mapCenter,
        zoom=10,
        height=600,
    )
    fig.update_layout(mapbox_style="carto-positron")
    return fig


@app.callback(
    Output("scatter-plot", "figure"),
    Input("map-plot", "clickData")
)
def scatter_plot(_):
    fig = px.scatter(
        df,
        x="dim0",
        y="dim1",
        hover_data=["node_id", "dim0", "dim1"]
    )
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(dragmode="select")  # Ativar modo de seleção
    return fig


app.run_server(debug=True)