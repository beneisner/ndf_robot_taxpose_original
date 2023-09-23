import plotly.graph_objects as go
import numpy as np

pts = np.loadtxt(np.DataSource().open('https://raw.githubusercontent.com/plotly/datasets/master/mesh_dataset.txt'))
x, y, z = pts.T
print(type(x))
print(x.shape)
### First graph
fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z,
                   alphahull=5,
                   opacity=0.4,
                   color='cyan')])
#fig.show()
#######

data = fig._data

x = [0, 1, 0]
y = [0, 2, 3]
tvects = [x,y]
orig = [0,0,0]
df=[]
coords = [[orig, np.sum([orig, v],axis=0)] for v in tvects]

# ['circle', 'circle-open', 'square', 'square-open','diamond', 'diamond-open', 'cross', 'x']

for i,c in enumerate(coords):
    X1, Y1, Z1 = zip(c[0])
    X2, Y2, Z2 = zip(c[1])
    vector = go.Scatter3d(x = [X1[0],X2[0]],
                          y = [Y1[0],Y2[0]],
                          z = [Z1[0],Z2[0]],
                          marker = dict(size = [15,15],
                                        color = ['blue'],
                                        symbol = 'diamond',
                                        line=dict(width=500,
                                                  color='red'
                                                 )),
                          name = 'Vector'+str(i+1))
    data.append(vector)

### Second graph
fig = go.Figure(data=data)
fig.show()
########
