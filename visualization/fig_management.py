# -*- coding: utf-8 -*-
"""
functions used to manage the creation and saving of multiple figures
A part of the code is taken from tsma package
"""
__author__ = ["Samuel Bazaz"]
__credits__ = ["Samuel Bazaz"]
__license__ = "MIT"
__version__ = "0.0.0"
__maintainer__ = ["Samuel Bazaz"]

##############
#  Packages  #
##############

import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns

from wordcloud import WordCloud


#####################
#  saves and query  #
#####################


def fig_to_image(fig, newpath_fig: str, figname: str, image_format: str = "png"):
    """Save an image of a figure if the image_format is right

    Parameters
    ----------
    fig : figure ( plotly object ),
        the figure you want to save
    newpath_fig: string,
        path of the file where you save
    figname : string,
        name of the image
    image_format : string,

    no Return
    -------
    """
    if image_format in ["pdf", "png", "jpeg"]:
        fig.write_image(
            newpath_fig + "/" + figname + "." + image_format,
            format=image_format,
            engine="kaleido",
        )


def save_fig(fig, newpath_fig: str, figname: str, save_format: str = "html") -> None:
    """Save an image if the  image_format is right and a html file of a figure
    Parameters
    ----------
    fig : figure ( plotly object ),
        the figure you want to save
    newpath_fig: string,
        path of the file where you save
    figname : string,
        name of the image
    image_format : string,
    no Return
    -------
    """
    if save_format == "html":
        fig.write_html(newpath_fig + "/" + figname + ".html")
    else:
        fig_to_image(fig, newpath_fig, figname, save_format)


def read_from_html(filepath: str):
    """Read an html file at filepath, for instance a figure
    Parameters
    ----------
    filepath : string,
        the path of the file
    Returns
    -------
    ... : html object,
        ex: plotly figure
    """
    # read the html file
    with open(filepath) as f:
        html = f.read()
    # convert in json object
    call_arg_str = re.findall(r"Plotly\.newPlot\((.*)\)", html[-(2**16) :])[0]
    call_args = json.loads(f"[{call_arg_str}]")
    # convert the json object into plotly figure
    plotly_json = {"data": call_args[1], "layout": call_args[2]}
    return pio.from_json(json.dumps(plotly_json))
