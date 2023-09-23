import io
from base64 import b64encode
import plotly.graph_objects as go

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import pandas as pd
buffer = io.StringIO()
stage = 'atgrasp'
rack_pcds = np.load('rack_pcd_obs_{}.npy'.format(stage))
target_pcds = np.load('mug_pcd_obs_{}.npy'.format(stage))
gripper_pcds = np.load('gripper_pcd_obs_{}.npy'.format(stage))

# df_target = pd.DataFrame({
# 'x':table_pcds[:,0], 'y':table_pcds[:,1], 'z':table_pcds[:,2]})
# df_rack = pd.DataFrame({
# 'x':rack_pcds[:,0], 'y':rack_pcds[:,1], 'z':rack_pcds[:,2]})
# fig = px.scatter_3d(df_target, x='x', y='y', z='z')
# fig.add_trace(px.Scatter(df_rack, x='x', y='y', z='z'))
# fig.update_traces(marker=dict(size=1))
# fig.show()

print(rack_pcds.shape)
pcds = [rack_pcds,target_pcds, gripper_pcds]

marker_size = 1
colours = ['red','green', 'blue','yellow']
data = []
for i in range(len(pcds)):
    trace = go.Scatter3d(
    x=pcds[i][:,0],
    y=pcds[i][:,1],
    z=pcds[i][:,2],
    mode='markers',
    marker=dict(
        size=marker_size,
        color=colours[i],     
    ))
    data.append(trace)


layout = go.Layout(
    margin=dict(
        l=None,
        r=None,
        b=None,
        t=None
    ),
    scene=dict(
        aspectmode='data'
    )
)
fig = go.Figure(data=data, layout=layout)
fig.show()


# fig.write_html(buffer)

# html_bytes = buffer.getvalue().encode()
# encoded = b64encode(html_bytes).decode()

# app = dash.Dash(__name__)
# app.layout = html.Div([
#     dcc.Graph(id="graph", figure=fig),
#     html.A(
#         html.Button("Download HTML"), 
#         id="download",
#         href="data:text/html;base64," + encoded,
#         download="plotly_graph.html"
#     )
# ])

# app.run_server(debug=True)