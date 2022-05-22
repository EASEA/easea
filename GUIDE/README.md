# GUIDE (Graphical User Interface for the Development of EASEA)


This application is a graphical interface for the [EASEA](http://easea.unistra.fr/index.php/EASEA_platform) platform.

## Dependencies

This application has been designed and tested on Ubuntu 20.10 with nodejs v16.8.0

To use this program, it is necessary to have installed the most recent version of EASEA (see http://easea.unistra.fr/index.php/Downloading_EASEA)

### General dependencies

- Python (Python 3 recommended)
- [Plotly](https://plotly.com/python/) to plot graphs : 
    - With pip : `pip install plotly==5.1.0` 
    - With conda : `conda install -c plotly plotly=5.1.0`
- Pandas :
    - With pip : `pip install pandas`
- kaleido to export graphs :
    - With pip : `pip install kaleido`
    - With conda : `conda install -c conda-forge python-kaleido`
- Make, GCC v7
- CMake 3.1 or later
- Node.JS version 12 or later
- [QT5](https://www.qt.io/qt5-11) or later

### macOS dependencies

- macOS 10.10 or later (OS 64 bits only)

## Installation

- Clone this repository
- Run `sudo make install` in `GUIDE/`. To choose the location of the executable run `sudo make install LOC=<path>`
- To run the application run `guide`
- To remove the generated executable use `make uninstall`
  
---
## Features in development

- Multiple runs with intervals of population, generations etc...
- 1 plot by run
- Plot more than the best fitness
- Quick EASEA file editor (new tab)
- More run stats (number of runs above/below a certain fitness)

---
## Contact

If you have any questions you can send an email to clement.ligneul@etu.unistra.fr