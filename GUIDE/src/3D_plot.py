__author__ = "Clément Ligneul"
__email__ = "clement.ligneul@etu.unistra.fr"

from os.path import dirname, abspath, join
from sys import argv, stderr, path, exit

# Find code directory relative to python_modules directory
# THIS_DIR = dirname(__file__)
# CODE_DIR = abspath(join(THIS_DIR, 'python_modules'))
# path.append(CODE_DIR)

# import python_modules.plotly.express as px
# from pandas.io.parsers import read_csv

import plotly.express as px
from pandas.io.parsers import read_csv


# arguments [nb_gen, nb_plots, csv_file, title, f1, f2, f3, tmp dir]

if len(argv) != 9:
    print(
        "Usage : "
        + argv[0]
        + " <nb of generations> <nb of plots> <csv file> <title> <f1> <f2> <f3> <tmp_path> ",
        file=stderr,
    )
    exit(1)

if argv[4]:
    titre = argv[4]
else:
    titre = "Results"

if argv[5]:
    name_x = argv[5]
else:
    name_x = "f1"

if argv[6]:
    name_y = argv[6]
else:
    name_y = "f2"

if argv[7]:
    name_z = argv[7]
else:
    name_z = "f3"

df = read_csv(
    argv[3], header=0, delimiter=" ", usecols=[0, 1, 2], names=["f1", "f2", "f3"]
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

fig.write_html(argv[8] + "fig.html")

fig.write_image(argv[8] + "fig.svg", engine="kaleido", scale=2)
