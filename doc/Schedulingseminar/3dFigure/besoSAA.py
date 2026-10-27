import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use('TkAgg')

# =====================================================
# PARAMETERS
# =====================================================

n_points = 20
a, b = 10, 6

mean_height = 8
height_std = mean_height / 2

np.random.seed(42)

# =====================================================
# DATA
# =====================================================

theta = np.random.uniform(0, 2*np.pi, n_points)
r = np.sqrt(np.random.uniform(0, 1, n_points))

x = a * r * np.cos(theta)
y = b * r * np.sin(theta)

z = np.random.normal(mean_height, height_std, n_points)
z = np.clip(z, 0.5, None)

avg_height = np.mean(z)

plane_x = np.linspace(-1.2*a, 1.2*a, 20)
plane_y = np.linspace(-1.2*b, 1.2*b, 20)

PX, PY = np.meshgrid(plane_x, plane_y)
PZ = np.full_like(PX, avg_height)

# =====================================================
# HELPERS
# =====================================================

def origin_trace():
    return go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=8, color='red'),
        name='Origin'
    )

def base_points_trace():
    return go.Scatter3d(
        x=x,
        y=y,
        z=np.zeros(n_points),
        mode='markers',
        marker=dict(size=4, color='blue'),
        name='Base points'
    )

def upper_points_trace():
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(size=5, color='green'),
        name='Upper points'
    )

def plane_trace():
    return go.Surface(
        x=PX,
        y=PY,
        z=PZ,
        opacity=0.35,
        colorscale=[[0,'orange'], [1,'orange']],
        showscale=False,
        name='Average plane'
    )

def radial_lines():
    traces = []

    for xi, yi in zip(x, y):
        traces.append(
            go.Scatter3d(
                x=[0, xi],
                y=[0, yi],
                z=[0, 0],
                mode='lines',
                line=dict(color='black', width=3),
                showlegend=False
            )
        )

    return traces

def vertical_lines():
    traces = []

    for xi, yi, zi in zip(x, y, z):
        traces.append(
            go.Scatter3d(
                x=[xi, xi],
                y=[yi, yi],
                z=[0, zi],
                mode='lines',
                line=dict(color='green', width=4),
                showlegend=False
            )
        )

    return traces
# =====================================================
# PHASES
# =====================================================

all_traces = [
    origin_trace(),
    base_points_trace(),
    *radial_lines(),
    upper_points_trace(),
    *vertical_lines(),
    plane_trace()
]

n_radial = n_points
n_vertical = n_points

# Trace indices:
# 0                 origin
# 1                 base points
# 2 ... 21          radial lines
# 22                upper points
# 23 ... 42         vertical lines
# 43                plane

phase_counts = [
    1,                          # Phase 1
    2,                          # Phase 2
    2 + n_radial,               # Phase 3
    3 + n_radial + n_vertical,  # Phase 4
    len(all_traces),            # Phase 5
    len(all_traces)             # Phase 6
]

frames = []

for k, count in enumerate(phase_counts, start=1):

    frame_data = []

    for i, trace in enumerate(all_traces):
        t = trace.to_plotly_json()
        t["visible"] = (i < count)
        frame_data.append(t)

    frames.append(
        go.Frame(
            name=str(k),
            data=frame_data,
            traces=list(range(len(all_traces)))
        )
    )

# =====================================================
# FIGURE
# =====================================================
fig = go.Figure(
    data=all_traces,
    frames=frames
)

# Initially show only the origin
for i in range(1, len(fig.data)):
    fig.data[i].visible = False
    
# =====================================================
# SLIDER
# =====================================================

steps = []

for k in range(1, 7):
    steps.append(
        dict(
            method="animate",
            args=[
                [str(k)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 800, "redraw": True},
                    "transition": {"duration": 400}
                }
            ],
            label=f"Phase {k}"
        )
    )

fig.update_layout(

 scene=dict(
    xaxis=dict(range=[-12, 12]),
    yaxis=dict(range=[-12, 12]),
    zaxis=dict(range=[0, 15]),
    aspectmode="cube",
    camera=dict(
       eye=dict(x=1.0, y=1.0, z=0.6)
    )
 ),

 sliders=[
    dict(
       active=0,
       currentvalue={"prefix": "Phase: "},
       steps=steps
    )
 ],

 updatemenus=[
    dict(
       type="buttons",
       showactive=False,

       buttons=[

          dict(
             label="Play",
             method="animate",
             args=[
                None,
                {
                   "frame": {
                      "duration": 1000,
                      "redraw": True
                   },
                   "fromcurrent": True
                }
             ]
          ),

          dict(
             label="Pause",
             method="animate",
             args=[
                [None],
                {
                   "mode": "immediate",
                   "frame": {
                      "duration": 0,
                      "redraw": False
                   }
                }
             ]
          )
       ]
    )
 ]
)
fig.show()