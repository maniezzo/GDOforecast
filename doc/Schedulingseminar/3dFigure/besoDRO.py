import numpy as np
import plotly.graph_objects as go

# =====================================================
# PARAMETERS
# =====================================================

n_points = 20
sample_size = 6

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

# Three independent bootstrap samples, with replacement
samples = [
    np.random.choice(n_points, sample_size, replace=True)
    for _ in range(3)
]

colors = ["green", "blue", "red"]

plane_x = np.linspace(-1.2*a, 1.2*a, 20)
plane_y = np.linspace(-1.2*b, 1.2*b, 20)

PX, PY = np.meshgrid(plane_x, plane_y)

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
        marker=dict(size=4, color='gray'),
        name='Base points'
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


def sample_traces(indices, color, sample_number):
    """
    Create vertical segments for the selected points
    and a horizontal plane at their average height.

    Repeated selections are retained when computing
    the average.
    """

    traces = []

    # Draw vertical lines for selected points
    for idx in indices:
        traces.append(
            go.Scatter3d(
                x=[x[idx], x[idx]],
                y=[y[idx], y[idx]],
                z=[0, z[idx]],
                mode='lines',
                line=dict(color=color, width=7),
                showlegend=False
            )
        )

    # Highlight selected upper points
    traces.append(
        go.Scatter3d(
            x=x[indices],
            y=y[indices],
            z=z[indices],
            mode='markers',
            marker=dict(size=6, color=color),
            name=f'Sample {sample_number}'
        )
    )

    # Average height, with repetitions
    avg = np.mean(z[indices])

    PZ = np.full_like(PX, avg)

    # Horizontal average plane
    traces.append(
        go.Surface(
            x=PX,
            y=PY,
            z=PZ,
            opacity=0.22,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            name=f'Mean {sample_number}: {avg:.2f}',
            showlegend=True,
            hovertemplate=(
                f'Sample {sample_number}<br>'
                f'Mean height: {avg:.2f}<extra></extra>'
            )
        )
    )

    return traces


# =====================================================
# BUILD ALL TRACES
# =====================================================

all_traces = []

# Phase 1: Origin
all_traces.append(origin_trace())
phase_1 = len(all_traces)

# Phase 2: Cloud of base points
all_traces.append(base_points_trace())
phase_2 = len(all_traces)

# Phase 3: Radial segments
all_traces.extend(radial_lines())
phase_3 = len(all_traces)

# Phase 4: First bootstrap sample (green)
all_traces.extend(sample_traces(samples[0], colors[0], 1))
phase_4 = len(all_traces)

# Phase 5: Second bootstrap sample (blue)
all_traces.extend(sample_traces(samples[1], colors[1], 2))
phase_5 = len(all_traces)

# Phase 6: Third bootstrap sample (red)
all_traces.extend(sample_traces(samples[2], colors[2], 3))
phase_6 = len(all_traces)

phase_counts = [
    phase_1,
    phase_2,
    phase_3,
    phase_4,
    phase_5,
    phase_6
]

# =====================================================
# FRAMES
# =====================================================

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

# Initially display only the origin
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
                    "frame": {
                        "duration": 800,
                        "redraw": True
                    },
                    "transition": {
                        "duration": 400
                    }
                }
            ],
            label=f"Phase {k}"
        )
    )

# =====================================================
# LAYOUT
# =====================================================

fig.update_layout(

    scene=dict(
        xaxis=dict(range=[-12, 12], title='X'),
        yaxis=dict(range=[-8, 8], title='Y'),
        zaxis=dict(range=[0, 18], title='Height'),

        aspectmode="cube",

        camera=dict(
            eye=dict(x=1.4, y=1.4, z=0.8)
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
                                "duration": 1200,
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
    ],

    title="Bootstrap samples and their average planes"
)

# =====================================================
# OUTPUT
# =====================================================

for k, indices in enumerate(samples):
    print(
        f"Sample {k+1}: {indices.tolist()} "
        f"Mean height = {np.mean(z[indices]):.3f}"
    )

fig.show()