__author__ = "Clément Ligneul"
__email__ = "clement.ligneul@etu.unistra.fr"

import plotly.express as px
from pandas.io.parsers import read_csv
import sys

# arguments [nb_gen, nb_plots, csv_file, title, f1, f2, f3]

if len(sys.argv) != 8:
    print(
        "Usage : "
        + sys.argv[0]
        + " <nb of generations> <nb of plots> <csv file> <title> <f1> <f2> <f3>",
        file=sys.stderr,
    )
    exit(1)

if sys.argv[4]:
    titre = sys.argv[4]
else:
    titre = "Results"

if sys.argv[5]:
    name_x = sys.argv[5]
else:
    name_x = "f1"

if sys.argv[6]:
    name_y = sys.argv[6]
else:
    name_y = "f2"

if sys.argv[7]:
    name_z = sys.argv[7]
else:
    name_z = "f3"

df = read_csv(
    sys.argv[3], header=0, delimiter=" ", usecols=[0, 1, 2], names=["f1", "f2", "f3"]
)

fig = px.scatter_3d(
    df, x="f1", y="f2", z="f3", color="f2", size="f3", size_max=18, opacity=0.7
)

fig.update_traces(marker_colorbar_title_text=name_y)

fig.update_layout(
    title=titre,
    coloraxis_colorbar=dict(
        title=name_y,
    ),
    margin=dict(l=65, r=0, b=10, t=90),
    scene=dict(
        xaxis_title=name_x,
        yaxis_title=name_y,
        zaxis_title=name_z,
        xaxis=dict(
            backgroundcolor="rgb(200,200,255)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white",
        ),
        yaxis=dict(
            backgroundcolor="rgb(200,200,255)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white",
        ),
        zaxis=dict(
            backgroundcolor="rgb(200,200,255)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white",
        ),
    ),
)

fig.write_html("/tmp/plotting/fig.html")

fig.write_image("/tmp/plotting/fig.svg", engine="kaleido", scale=2)
