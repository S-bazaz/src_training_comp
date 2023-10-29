# -*- coding: utf-8 -*-
"""
functions used to create or manipulate dataframes
"""
__author__ = ["Samuel Bazaz"]
__credits__ = ["Samuel Bazaz"]
__license__ = "MIT"
__version__ = "0.0.0"
__maintainer__ = ["Samuel Bazaz"]

##############
#  Packages  #
##############

import re
import pandas as pd
import numpy as np
from typing import Dict, Union, List, Tuple, Any


######################
#  data exploration  #
######################


def df_nan_percent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le pourcentage de valeurs manquantes dans chaque colonne d'un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contenant les données.

    Returns:
        pd.DataFrame: DataFrame contenant le pourcentage de valeurs manquantes pour chaque colonne.
    """
    part_nan = df.isnull().mean()
    perc_nan = pd.DataFrame(100 * part_nan.to_numpy())
    perc_nan.index = part_nan.index
    return perc_nan


##########################
#  Reformat for loading  #
##########################


def df_sas7bdat_reformat(df):  # Deprecated si on utilise bien pyreadstat pour les fichiers SAS
    """
    Reformate les données dans un DataFrame pandas issu d'un fichier SAS7BDAT.

    Args:
        df (pd.DataFrame): Le DataFrame pandas contenant les données à reformater.

    Returns:
        pd.DataFrame: Le DataFrame pandas avec les données reformattées.
    """
    for col in df:
        notnullserie = df[col].loc[~df[col].isnull()]
        if len(notnullserie) > 0:
            notnull_val = notnullserie.iloc[0]
            if isinstance(notnull_val, bytes):
                df[col] = df[col].str.decode("ISO-8859-1")
    return df


#############
#  replace  #
#############
#

def df_nan_harmonization(
    df: pd.DataFrame, lst_value: List[str] = ["*", "", " ", "nan"], lst_match: List[str] = [r"NAN"]
) -> pd.DataFrame:
    """
    Harmonise les valeurs NaN dans un DataFrame en remplaçant les valeurs spécifiées.

    Args:
        df (pd.DataFrame): DataFrame à harmoniser.
        lst_value (List[str], optional): Liste des valeurs à remplacer par NaN.
        lst_match (List[str], optional): Liste des motifs de correspondance
            pour les valeurs à remplacer par NaN.

    Returns:
        pd.DataFrame: DataFrame harmonisé avec les valeurs NaN.

    """
    for val in lst_value:
        df = df.replace(val, np.nan)

    for pattern in lst_match:

        def aux_replace(value):
            if isinstance(value, str):
                if re.match(pattern, value, flags=0):
                    return np.nan
            return value

        df = df.applymap(aux_replace)
    return df


##############
#  Renaming  #
##############


def remove_special_caract(k: str) -> str:
    """
    Supprime les caractères spéciaux d'une chaîne de caractères.

    Args:
        k (str): Chaîne de caractères à traiter.

    Returns:
        str: Chaîne de caractères sans caractères spéciaux.
    """
    return re.sub("[?ï»¿]+", "", k)


def complete_mapping(df: pd.DataFrame, mapping: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complète un mapping en ajoutant les clés manquantes du DataFrame
    et en supprimant les caractères spéciaux des nouvelles clés.

    Args:
        df (pd.DataFrame): DataFrame contenant les clés à vérifier.
        mapping (Dict[str, Any]): Mapping existant à compléter.

    Returns:
        Dict[str, Any]: Mapping complété avec les nouvelles clés du DataFrame.
    """
    map2 = {}
    for k in df:
        if k in mapping:
            map2[k] = mapping[k]
        else:
            map2[k] = remove_special_caract(k)
    return map2


def df_rename_cols(df: pd.DataFrame, mapping: Dict[str, Any]) -> pd.DataFrame:
    """
    Renomme les colonnes d'un DataFrame en utilisant un mapping
    complété par une élimination des caractères spéciaux

    Args:
        df (pd.DataFrame): DataFrame à renommer.
        mapping (Dict[str, Any]): Mapping des nouveaux noms de colonnes.

    Returns:
        pd.DataFrame: DataFrame avec les colonnes renommées.
    """
    if mapping is None:
        return df

    return df.rename(columns=complete_mapping(df, mapping))


############
#  Typing  #
############


def change_typing_from_dct(df: pd.DataFrame, dct_typing: Dict[str, Any]) -> None:
    """
    Change le type de données des colonnes d'un DataFrame
    en utilisant un dictionnaire de typage.

    Args:
        df (pd.DataFrame): DataFrame dont les types de données doivent être modifiés.
        dct_typing (Dict[str, Any]): Dictionnaire de typage des colonnes.

    Returns:
        None
    """
    for col, tp in dct_typing.items():
        df[col] = df[col].astype(tp)


####################
#  visualizations  #
####################

# TO DO
