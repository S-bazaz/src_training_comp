# -*- coding: utf-8 -*-
"""

"""
__author__ = ["Samuel Bazaz"]
__credits__ = ["Samuel Bazaz"]
__license__ = "MIT"
__version__ = "0.0.0"
__maintainer__ = ["Samuel Bazaz"]

##############
#  Packages  #
##############
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

#################
#  raw dataviz  #
#################


def my_wordcloud(series: pd.Series, size: int = 10, title: str = "") -> plt.Figure:
    """
    Crée et retourne une visualisation de type nuage de mots à partir d'une
    série donnée.

    Args:
        series (pd.Series): La série à partir de laquelle
            créer le nuage de mots.
        size (int, optional): La taille de la figure générée (en pouces).
            Défaut à 10.
        title (str, optional): Le titre de la figure générée. Défaut à "".

    Returns:
        plt.figure: La figure du nuage de mots généré.
    """
    fig = plt.figure()
    text = " ".join(series.astype(str))
    wordcloud = WordCloud(background_color="white", max_words=70).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title(title)
    fig = plt.gcf()
    fig.set_size_inches(size, size)
    return fig


def nan_bars(df: pd.DataFrame) -> go.Figure:
    """
    Crée un graphique en barres pour afficher le pourcentage de valeurs
    manquantes pour chaque variable dans un DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame pour lequel on veut visualiser
            le pourcentage de non-réponses.

    Returns:
        go.Figure: Le graphique en barres Plotly
            montrant le pourcentage de non-réponses pour chaque variable.
    """
    part_nan = df.isnull().mean()
    perc_nan = pd.DataFrame(100 * part_nan.to_numpy())
    perc_nan.index = part_nan.index
    fig = px.bar(perc_nan, template="plotly_white")
    fig.update_layout(
        title="<b> % de non réponses </b>",
        xaxis_title="variable",
        yaxis_title="%",
    )
    return fig


def var_hist(
    df: pd.DataFrame,
    ncol: int = 5,
    facet_row_spacing: float = 0.2,
    histnorm: str = "percent",
) -> go.Figure:
    """
    Affiche les histogrammes des variables catégorielles.

    Args:
        df (pd.DataFrame): Le dataframe qui contient les données.
        ncol (int, optional): Le nombre de colonnes dans la grille.
            Par défaut 5.
        facet_row_spacing (float, optional): Espace entre les rangées.
            Par défaut 0.2.
        histnorm (str, optional): La normalisation de l'histogramme.
            Peut être "percent" ou "probability".
            Par défaut "percent".

    Returns:
        go.Figure: Le graphique Plotly contenant les histogrammes
            des variables catégorielles.
    """
    df2 = pd.melt(df)
    fig = px.histogram(
        df2,
        x="value",
        color="variable",
        facet_col="variable",
        facet_col_wrap=ncol,
        facet_row_spacing=facet_row_spacing,
        facet_col_spacing=0.04,
        hover_name="variable",
        histnorm=histnorm,
        template="plotly_white",
    )

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_layout(
        title="<b> Distributions des variables catégorielles </b>",
    )
    return fig


def var_box(df, ncol: int = 5) -> go.Figure:
    """
    Crée une figure contenant des boîtes à moustaches pour chaque variable
    catégorielle dans le DataFrame.

    Args:
        df (DataFrame): Le DataFrame contenant les variables catégorielles.
        ncol (int, optional): Le nombre de colonnes dans la figure. Par défaut 5.

    Returns:
        go.Figure: La figure contenant
        les boîtes à moustaches.
    """
    df2 = pd.melt(df)
    fig = px.box(
        df2,
        x="variable",
        y="value",
        color="variable",
        facet_col="variable",
        facet_col_wrap=ncol,
        facet_row_spacing=0.2,
        template="plotly_white",
    )

    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_xaxes(showticklabels=False)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(
        title="<b> Boites à moustaches des variables catégorielles </b>",
        xaxis_title="",
    )
    return fig


def length_hist(
    df: pd.DataFrame, ncol: int = 5, facet_row_spacing: float = 0.2
) -> go.Figure:
    """
    Crée un histogramme représentant la distribution des longueurs
    des variables d'un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame à utiliser.
        ncol (int, optionnel): Nombre de colonnes pour les sous-graphiques.
            Par défaut, 5.
        facet_row_spacing (float, optionnel): Espacement vertical
            entre les sous-graphiques. Par défaut, 0.2.

    Returns:
        go.Figure: Figure représentant l'histogramme.
    """
    length = df.applymap(lambda x: len(str(x)), na_action="ignore")
    fig = var_hist(length, ncol=ncol, facet_row_spacing=facet_row_spacing)
    fig.update_layout(
        title="<b> Distributions des longueurs des variables</b>",
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    return fig


def my_heatmap(df, xvar, yvar, zvar, palette="Viridis"):
    fig = go.Figure(
        data=go.Heatmap(z=df[zvar], x=df[xvar], y=df[yvar], colorscale=palette)
    )
    fig.update_layout(autosize=False, width=800, height=500, font={"size": 12})
    return fig


#############
#  queries  #
#############


def query_model(
    df: pd.DataFrame, var: str = "pcs", algo: str = "fasttext"
) -> pd.DataFrame:
    """Effectue une requête sur le DataFrame pour filtrer les données
    selon la variable de codification et algorithmes spécifiés.

        Args:
            df (pd.DataFrame): Le DataFrame à filtrer.
            var (str, optional): La valeur de la variable à filtrer. Par défaut, "pcs".
            algo (str, optional): La valeur de l'algorithme à filtrer. Par défaut, "fasttext".

        Returns:
            pd.DataFrame: Le sous-ensemble du DataFrame filtré selon les conditions spécifiées.
    """
    return df.query(
        f"var_main == '{var}' and var_test == '{var}' and algo_main == '{algo}' and algo_test == '{algo}'"
    )


def query_metric(
    df: pd.DataFrame, decision: str = "bestproba", metric: str = "accuracy"
) -> pd.DataFrame:
    """Effectue une requête sur le DataFrame pour filtrer les données selon la décision et la métrique spécifiées.

    Args:
        df (pd.DataFrame): Le DataFrame à filtrer.
        decision (str, optional): La valeur de la décision à filtrer. Par défaut, "bestproba".
        metric (str, optional): La valeur de la métrique à filtrer. Par défaut, "accuracy".

    Returns:
        pd.DataFrame: Le sous-ensemble du DataFrame filtré selon les conditions spécifiées.
    """
    return df.query(f"decision == '{decision}' and metric == '{metric}'")


def query_model_and_metric(
    df: pd.DataFrame,
    var: str = "pcs",
    algo: str = "fasttext",
    decision: str = "bestproba",
    metric: str = "accuracy",
) -> pd.DataFrame:
    """Effectue une double requête sur le DataFrame pour filtrer les données selon les critères spécifiés.

    Args:
        df (pd.DataFrame): Le DataFrame initial à filtrer.
        var (str, optional): La valeur de la variable à filtrer. Par défaut, "pcs".
        algo (str, optional): La valeur de l'algorithme à filtrer. Par défaut, "fasttext".
        decision (str, optional): La valeur de la décision à filtrer. Par défaut, "bestproba".
        metric (str, optional): La valeur de la métrique à filtrer. Par défaut, "accuracy".

    Returns:
        pd.DataFrame: Le sous-ensemble final du DataFrame filtré selon les critères spécifiés.
    """
    return query_metric(query_model(df, var, algo), decision, metric)


def query_enquete_test(
    df: pd.DataFrame, enquete: str = "rp", version: str = "anc"
) -> pd.DataFrame:
    """Effectue une requête sur le DataFrame pour filtrer les données
    en fonction des valeurs spécifiées pour enquete et version.

    Args:
        df (pd.DataFrame): Le DataFrame à filtrer.
        enquete (str, optional): La valeur de l'enquête à filtrer. Par défaut, "rp".
        version (str, optional): La valeur de la version à filtrer. Par défaut, "anc".

    Returns:
        pd.DataFrame: Le sous-ensemble du DataFrame filtré selon les critères spécifiés.
    """
    return df.query(f"enquete_test == '{enquete}' and version_test == '{version}'")


##################
#  constructors  #
##################


def join_main_info(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne au DataFrame en combinant
    les valeurs des colonnes enquete_main, version_main, subset_main et config_main.
        df (pd.DataFrame): Le DataFrame auquel ajouter la nouvelle colonne.

    Returns:
        pd.DataFrame: Le DataFrame avec la nouvelle colonne ajoutée.
    """
    df["Main:enquete|version|subset"] = df[
        ["enquete_main", "version_main", "subset_main"]
    ].agg("|".join, axis=1)
    # df["Main:enquete|version|subset|config"] = df[["enquete_main", "version_main", "subset_main", "config_main"]].agg(
    # "|".join, axis=1)


def get_color_palette(
    df: pd.DataFrame,
    palette: str = "Spectrale",
    varname_for_color: str = "Main:enquete|version|subset|config",
) -> List[str]:
    """Renvoie une palette de couleurs hexadécimales basée sur les valeurs uniques de la colonne spécifiée du DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame contenant la colonne utilisée pour déterminer la palette de couleurs.
        palette (str, optional): Le nom de la palette de couleurs à utiliser. Par défaut, "Spectrale".
        varname_for_color (str, optional): Le nom de la colonne utilisée pour déterminer les valeurs uniques et la taille de la palette. Par défaut, "Main:enquete|version|subset|config".

    Returns:
        List[str]: Une liste de couleurs hexadécimales représentant la palette de couleurs.
    """
    return sns.color_palette(palette, df[varname_for_color].nunique()).as_hex()


def my_cross_test_box_plot(df_queried: pd.DataFrame, palette: str) -> px.box:
    """Crée un diagramme boîte à moustache interactif à partir d'un DataFrame avec des facettes et une palette de couleurs personnalisée.


    Args:
        df_queried (pd.DataFrame): Le DataFrame contenant les données pour le diagramme en boîte.
        palette (str): Le nom de la palette de couleurs à utiliser.

    Returns:
        px.box: Le diagramme créé avec Plotly Express.
    """
    color = get_color_palette(
        df_queried, palette, varname_for_color="Main:enquete|version|subset"
    )

    grouped_df = (
        df_queried.groupby(
            ["config_test", "subset_test", "Main:enquete|version|subset"]
        )
        .size()
        .to_frame("count")
    )

    print(grouped_df)
    df_queried = pd.merge(
        df_queried,
        grouped_df,
        on=["config_test", "subset_test", "Main:enquete|version|subset"],
        how="left",
    )

    fig = px.box(
        df_queried.rename(
            columns={"value": "Test", "Main:enquete|version|subset": " "}
        ),
        x=" ",
        y="Test",
        # facet_row = "data_id_main",
        color=" ",
        # facet_col="subset_test",
        # facet_row="config_test",
        facet_col="config_test",
        facet_row="subset_test",
        # facet_row_spacing=0.1,
        # template="plotly_white",
        template="plotly_dark",
        color_discrete_sequence=color,
        hover_data=["count"],
        points=False,
    )

    # fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_yaxes(showticklabels=True)
    fig.update_xaxes(showticklabels=False)
    # fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


###############
#  iterators  #
###############


def pcs_fasttext_iterator_box_plots(
    dct_paths: Dict[str, str],
    df: pd.DataFrame,
    decision: str = "bestproba",
    metric: str = "accuracy",
    palette: str = "Spectral",
    figid: str = "0",
    width: int = 1000,
    height: int = 800,
) -> None:
    """Génère des diagrammes en boîte interactifs pour les modèles PCS FastText.

    La fonction extrait les données spécifiques au modèle PCS FastText à partir du DataFrame.
    Elle génère ensuite des diagrammes boîte à moustache interactifs pour chaque combinaison unique d'enquête de test et de version.

    Args:
        dct_paths (Dict[str, str]): Un dictionnaire contenant les chemins de fichiers nécessaires.
        df (pd.DataFrame): Le DataFrame contenant les données des modèles PCS FastText.
        decision (str, optional): La décision à filtrer. Par défaut, "bestproba".
        metric (str, optional): La métrique à filtrer. Par défaut, "accuracy".
        palette (str, optional): Le nom de la palette de couleurs à utiliser. Par défaut, "Spectral".
        figid (str, optional): L'identifiant du diagramme en boîte. Par défaut, "0".
        width (int, optional): La largeur du diagramme en boîte. Par défaut, 1000.
        height (int, optional): La hauteur du diagramme en boîte. Par défaut, 800.

    Returns:
        None
    """
    df_q1 = query_model_and_metric(
        df,
        var="pcs",
        algo="fasttext",
        decision=decision,
        metric=metric,
    )
    #print(df_q1.iloc[0])
    # elimination des erreurs de collecte
    # df_q1 = df_q1.query(" enquete_main != 'epi-mdg-rp-srcv-vrs' ")
    # df_q1 = df_q1.query(" enquete_main != 'epi-mdg-rp-srcv-vrs' and subset_main != 'PROFAISa' ")
    join_main_info(df_q1)
    subtitle = f"<br><sup>Model: PCSfasttext</sup> <sup>Decision: {decision}</sup>"
    title = f"<b>Boites à moustaches {metric}</b>{subtitle}"

    for index, row in (
        df_q1[["enquete_test", "version_test"]].drop_duplicates().iterrows()
    ):
        enquete = row["enquete_test"]
        version = row["version_test"]
        df_q2 = query_enquete_test(df_q1, enquete=enquete, version=version)
        # df_q2 = df_q2.query(f" enquete_main == '{enquete}'")

        enquete = enquete.upper()
        fig = my_cross_test_box_plot(df_q2, palette)
        fig.update_layout(
            margin=dict(r=10, b=100, l=10, t=100),
            title=f"{title} <sup>Enquete test: {enquete}{version}</sup>",
            legend_title="Main",
            font={"size": 11},
            width=width,
            height=height,
        )

        figpath = Path(dct_paths["07_reporting"]).joinpath(
            f"crosstest_boxplot_{enquete}{version}_{metric}_v{figid}.html"
        )
        figpath = str(figpath)
        print(f"Saving : {figpath}")
        fig.write_html(figpath)


# ############
# #  metrics #
# ############
#
# def cross_test_metrics_cloud(
#     df, palette="Spectral", symbol="model", width=1600, height=1000
# ):
#     color = sns.color_palette(palette, df["metric"].nunique()).as_hex()
#
#     n_ticks = df[["data_id_main", "data_id_test"]].nunique()
#     fig = px.scatter_3d(
#         df,
#         x="data_id_main",
#         y="data_id_test",
#         z="value",
#         color="metric",
#         symbol=symbol,
#         opacity=0.4,
#         color_discrete_sequence=color,
#     )
#     fig.update_layout(
#         title="<b>Cross test metrics</b>",
#         autosize=False,
#         width=width,
#         height=height,
#         font={"size": 12},
#         margin=dict(r=50, b=100, l=10, t=10),
#         scene=dict(
#             xaxis=dict(
#                 nticks=2 * int(n_ticks["data_id_main"]),
#                 backgroundcolor="rgb(220, 220, 230)",
#                 tickfont=dict(color="purple", size=10, family="Old Standard TT, serif"),
#             ),
#             yaxis=dict(
#                 nticks=2 * int(n_ticks["data_id_test"]),
#                 backgroundcolor="rgb(230, 220,230)",
#                 tickfont=dict(color="blue", size=10, family="Old Standard TT, serif"),
#             ),
#             xaxis_title="<b>Main</b>",
#             yaxis_title="<b>Test</b>",
#             zaxis_title="<b>Performance</b>",
#         ),
#     )
#     return fig
