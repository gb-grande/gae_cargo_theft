# import pandas as pd
# embeddingsWithInfo = pd.read_csv("../data/embeddingsWithInfo.csv")
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from dash import Dash, dcc, html, Input, Output
import sys


if len(sys.argv) != 3:
    print("Usage: python embeddingsOnMap.py pathToEmbeddings.csv pathToOcurrences.csv")

#read filename 
dfEmbeddingsName = sys.argv[1]
df = pd.read_csv(dfEmbeddingsName)

occurrencesDfName = sys.argv[2]
dfOccurrences = pd.read_csv(occurrencesDfName)



def buildSankey(filteredOccurDf):
    timeLabels = filteredOccurDf['time'].unique()
    localSubtypeLabels = filteredOccurDf['local_subtype'].unique()
    labels = np.concatenate((['total'], timeLabels, localSubtypeLabels))
    labelsIndex = dict()
    for i in range(0, len(labels)):
        labelsIndex[labels[i]] = i
    source = []
    target = []
    values = []
    vc = filteredOccurDf[['time', 'local_subtype']].value_counts()
    times = filteredOccurDf['time'].value_counts()
    #deals with total and times first
    for index, item in times.items():
        source.append(labelsIndex['total'])
        target.append(labelsIndex[index])
        values.append(item)
    
    for index, item in vc.items():
        source.append(labelsIndex[index[0]])
        target.append(labelsIndex[index[1]])
        values.append(item)
    fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = labels,
      color = "blue"
    ),
    link = dict(
      source = source, 
      target = target,
      value = values
  ))])
    return fig

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
    ),
    dcc.Graph(
        id="sankey-diag",
        config={"scrollZoom": True},
    )
])



@app.callback(
    Output("map-plot", "figure"),
    Input("scatter-plot", "selectedData")
    
)
def update_map(selectedData):
    # get selected points
    if selectedData:
        points = selectedData["points"]
        nodeIds = []
        for p in points:
            nodeIds.append(p['customdata'][0])
        print(nodeIds)
        print("-"*100)
        selected_indices = [p["pointIndex"] for p in points]  
        filtered_df = df.iloc[selected_indices]
    else:
        filtered_df = df  # if none selected, show all points

    mapCenter ={ 
        "lat" :-23.533773,
        "lon" : -46.625290
    }
    # Create map
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
    Output("sankey-diag", "figure"),
    Input("scatter-plot", "selectedData")
    
)
def update_sankey(selectedData):
    # get selected points
    if selectedData:
        points = selectedData["points"]
        selected_indices = [p["pointIndex"] for p in points]  
        filtered_df = dfOccurrences.iloc[selected_indices]
    else:
        filtered_df = dfOccurrences  # Se nenhum ponto for selecionado, mostrar tudo
    fig = buildSankey(filtered_df)
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