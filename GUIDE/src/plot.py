__author__ = "Clément Ligneul"
__email__ = "clement.ligneul@etu.unistra.fr"

from os.path import dirname, abspath, join
from sys import argv, stderr, path

# Find code directory relative to python_modules directory
# THIS_DIR = dirname(__file__)
# CODE_DIR = abspath(join(THIS_DIR, 'python_modules'))
# path.append(CODE_DIR)

# import python_modules.plotly.graph_objects as go
# from python_modules.plotly.subplots import make_subplots
# from python_modules.pandas.io.parsers import read_csv

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# from pandas.io.parsers import read_csv
from pandas.io.parsers import read_csv
from math import ceil

# parameters = [path, nb_gen, nb_plots, title, x_axis, y_axis, last_gen_only, color]
# print("ok")
# exit(0)
if len(argv) < 5:
    print(
        "Usage : "
        + argv[0]
        + "<path> <nb of generations> <nb of plots> <title> <x_axis> <y_axis> <[optional]color>",
        file=stderr,
    )
    exit(1)

# number of generations
nb_gen = int(argv[2])

# number of violin plots
nb_plots = int(argv[3])

# plot title
if argv[4]:
    title = argv[4]
else:
    title = "Results"


treshold = ceil(nb_gen / nb_plots)

df = read_csv(argv[1] + "data.csv")

if len(argv) == 9 and argv[7] == "false":
    # everything is plotted
    df2 = df

else:
    # only the best are plotted
    df2 = df.loc[df["GEN"] == (nb_gen - 1)]
    # idx = tmp["BEST_FIT"].idxmin()
    # best_run = df.iloc[idx]["RUN"]

    # df2 = df.loc[df["RUN"] == best_run]

    if title == "" or title == "Results":
        title = f"Results for the last generation"

df = df2

# fig = make_subplots(rows=4, cols=1, subplot_titles="Results")
fig = go.Figure()

for i in range(1, nb_plots + 1):
    if i == nb_plots:
        n = "≤" + str((i) * treshold)
    else:
        n = "<" + str((i) * treshold)

    if len(argv) == 9 and argv[8] != "":
        fig.add_trace(
            go.Violin(
                y=df["BEST_FIT"][df["GEN"] < (i * treshold)][
                    df["GEN"] >= treshold * (i - 1)
                ],
                fillcolor=argv[8],
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
        "text": title,
        "y": 0.9,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    },
    # height=5000
)

if len(argv) >= 6:
    if str(argv[5]) != "f1" and str(argv[5]) != "":
        fig.update_xaxes(title_text=argv[5])
    else:
        fig.update_xaxes(title_text="Generations")
else:
    fig.update_xaxes(title_text="Generations")

if len(argv) >= 8:
    if str(argv[5]) != "f2" and str(argv[6]) != "":
        fig.update_yaxes(title_text=argv[6])
    else:
        fig.update_yaxes(title_text="Best Fitness")
else:
    fig.update_yaxes(title_text="Best Fitness")

fig.write_html(argv[1] + "fig.html")

fig.write_image(argv[1] + "fig.svg", engine="kaleido", scale=2)
