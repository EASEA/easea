__author__ = "Clément Ligneul"
__email__ = "clement.ligneul@etu.unistra.fr"

from pandas.io.parsers import read_csv
import plotly.express as px
import plotly.graph_objects as go
import sys
from math import ceil

# parameters = [nb_gen, nb_plots, title, x_axis, y_axis, color]

if len(sys.argv) < 4:
    print(
        "Usage : "
        + sys.argv[0]
        + " <nb of generations> <nb of plots> <title> <x_axis> <y_axis> <[optional]color>",
        file=sys.stderr,
    )
    exit(1)

# number of generations
nb_gen = int(sys.argv[1])

# number of violin plots
nb_plots = int(sys.argv[2])

# plot title
if sys.argv[3]:
    titre = sys.argv[3]
else:
    titre = "Results"


treshold = ceil(nb_gen / nb_plots)

df = read_csv("/tmp/plotting/data.csv")

fig = go.Figure()

for i in range(1, nb_plots + 1):
    if i == nb_plots:
        n = "≤" + str((i) * treshold)
    else:
        n = "<" + str((i) * treshold)

    if len(sys.argv) == 7 and sys.argv[6] != "":
        fig.add_trace(
            go.Violin(
                y=df["BEST_FIT"][df["GEN"] < (i * treshold)][
                    df["GEN"] >= treshold * (i - 1)
                ],
                fillcolor=sys.argv[6],
                line_color="black",
                name=n,
            )
        )
    else:
        fig.add_trace(
            go.Violin(
                y=df["BEST_FIT"][df["GEN"] < (i * treshold)][
                    df["GEN"] >= treshold * (i - 1)
                ],
                fillcolor="red",
                line_color="black",
                name=n,
            )
        )

fig.update_traces(box_visible=True, meanline_visible=True, points="all")

fig.update_layout(
    title={
        "text": titre,
        "y": 0.9,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    }
)

if len(sys.argv) >= 5:
    if str(sys.argv[4]) != "f1" and str(sys.argv[4]) != "":
        fig.update_xaxes(title_text=sys.argv[4])
    else:
        fig.update_xaxes(title_text="Generations")
else:
    fig.update_xaxes(title_text="Generations")

if len(sys.argv) >= 6:
    if str(sys.argv[5]) != "f2" and str(sys.argv[5]) != "":
        fig.update_yaxes(title_text=sys.argv[5])
    else:
        fig.update_yaxes(title_text="Best Fitness")
else:
    fig.update_yaxes(title_text="Best Fitness")

fig.write_html("/tmp/plotting/fig.html")

fig.write_image("/tmp/plotting/fig.svg", engine="kaleido", scale=2)
