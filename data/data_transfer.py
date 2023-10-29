# -*- coding: utf-8 -*-
"""
functions used to transfer data : raw data Import, shared files management
"""
__author__ = ["Samuel Bazaz"]
__credits__ = ["Samuel Bazaz"]
__license__ = "MIT"
__version__ = "0.0.0"
__maintainer__ = ["Samuel Bazaz"]

##############
#  Packages  #
##############

import os
import sys
import s3fs
import pandas as pd
from pathlib import Path
from shutil import copy2
from typing import Dict, Any, Tuple, Optional, Union


##################
#      Imports     #
##################

root_path = Path(os.path.abspath(__file__)).parents[2]
sys.path.insert(0, str(root_path))

from src.data.paths_fetcher import get_config_path, get_paths_yaml_path, my_walk
from src.data.loading_saving import s3_filesystem, load_yaml, load_dfs, load_df
from src.data.basictypes_proc import substring_from_keys_list


######################
#  data Import  #
######################


def get_configured_paths(
    doc_raw_paths: Dict[str, Any], doc_config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Récupère les chemins configurés à partir des chemins bruts et de la configuration.

    Args:
        doc_raw_paths (Dict[str, Any]): Les chemins bruts sous forme de dictionnaire.
        doc_config (Dict[str, Any]): La configuration sous forme de dictionnaire.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]:
        Un tuple contenant les chemins configurés locaux et S3.
    """
    dct_paths_local = substring_from_keys_list(doc_raw_paths["local"], doc_config)
    dct_paths_s3 = substring_from_keys_list(doc_raw_paths["s3"], doc_config)
    return dct_paths_local, dct_paths_s3


def get_dct_sep(doc_config: Dict[str, Any]) -> Dict[str, str]:
    """
    Récupère les séparateurs de colonne configurés à partir de la configuration.

    Args:
        doc_config (Dict[str, Any]): La configuration sous forme de dictionnaire.

    Returns:
        Dict[str, str]: Un dictionnaire contenant les séparateurs de colonne configurés.
    """
    if doc_config is None:
        return {}
    return {k: v["sep"] for k, v in doc_config.items()}


def import_raw_data(
    root: Union[str, Path], mode: str = "aus"
) -> Dict[str, pd.DataFrame]:
    """
    Importe les données brutes à partir des chemins configurés.

    Args:
        root (Union[str, Path]): Le chemin racine du projet.
        mode (str, optional): Le mode d'Import des données. Par défaut, "aus".

    Returns:
        Dict[str, pd.DataFrame]: Un dictionnaire contenant les DataFrames importés,
            avec les noms des fichiers en tant que clés.
    """

    yaml_raw_path = get_paths_yaml_path(root, mode, "raw")
    yaml_dfs_config = get_config_path(root, "2_loading_config")

    doc_raw = load_yaml(yaml_raw_path)
    doc_config = load_yaml(yaml_dfs_config)

    dct_paths_local, dct_paths_s3 = get_configured_paths(doc_raw, doc_config)
    dct_sep = get_dct_sep(doc_config)

    dct_data = {}
    dct_data.update(load_dfs(dct_paths_local, dct_sep, None))

    if doc_raw["s3"] is not None:
        dct_data.update(load_dfs(dct_paths_s3, dct_sep, s3_filesystem()))
    return dct_data


def get_1_raw_path_and_fsmode(
    name: str, root: Union[str, Path], mode: str = "aus"
) -> Tuple[Union[str, None], Union[str, None]]:
    """
    Obtient le chemin d'accès brut et le mode de stockage du fichier en fonction du nom donné.

    Args:
        name (str): Nom du fichier.
        root (Union[str, Path]): Répertoire racine du projet.
        mode (str, optional): Mode de stockage. Par défaut, "aus".

    Returns:
        tuple[Union[str, None], Union[str, None]]: Chemin d'accès brut et mode de stockage du fichier, respectivement.

    """
    yaml_raw_path = get_paths_yaml_path(root, mode, "raw")
    doc = load_yaml(yaml_raw_path)

    if name in doc["local"]:
        return doc["local"][name], "local"

    if doc["s3"] is None:
        return None, None

    if name in doc["s3"]:
        return doc["s3"][name], "s3"
    return None, None


def load_1_raw_df(
    name: str, root: Union[str, Path], csv_sep: str, special_sep: str, mode: str = "aus"
) -> Union[Dict[str, pd.DataFrame], None]:
    """
    Charge un fichier brut en tant que DataFrame en fonction du nom donné.

    Args:
        name (str): Nom du fichier.
        root (Union[str, Path]): Répertoire racine du projet.
        csv_sep (str): Séparateur pour les fichiers CSV.
        special_sep (str): Séparateur spécial pour les autres types de fichiers.
        mode (str, optional): Mode de stockage. Par défaut, "aus".

    Returns:
        Union[Dict[str, pd.DataFrame], None]: Un dictionnaire contenant le nom du fichier et le DataFrame correspondant,
                                              ou None si le fichier n'a pas été trouvé.

    """

    path, fsmode = get_1_raw_path_and_fsmode(name, root, mode)
    if path is None:
        return None

    fs = None
    if mode == "s3":
        fs = s3_filesystem()

    return load_df(path, fs, csv_sep=csv_sep, special_sep=special_sep)


###############################
#  transfer repo shared files #
###############################


def my_copy(
    from_path: str,
    to_path: str,
    fs_from: Optional[s3fs.S3FileSystem],
    fs_to: Optional[s3fs.S3FileSystem],
) -> None:
    """
    Copie un fichier d'un emplacement à un autre en utilisant les systèmes de fichiers spécifiés.

    Args:
        from_path (str): Chemin du fichier source.
        to_path (str): Chemin de destination.
        fs_from (Optional[s3fs.S3FileSystem]): Système de fichiers source.
            Si spécifié, le fichier sera téléchargé à partir de ce système de fichiers.
        fs_to (Optional[s3fs.S3FileSystem]): Système de fichiers de destination.
            Si spécifié, le fichier sera copié vers ce système de fichiers.
            Si à la fois `fs_from` et `fs_to` sont spécifiés, la copie sera effectuée directement entre les systèmes de fichiers.

    Returns:
        None
    """
    if fs_from is not None:
        fs_from.download(from_path, to_path)

    elif fs_to is not None:
        fs_to.put(from_path, to_path)

    else:
        copy2(from_path, to_path)


def pull_saving_files(root: str, mode: str = "aus") -> None:
    """
    Télécharge les fichiers de sauvegarde à partir de l'emplacement spécifié.

    Args:
        root (str): Chemin racine.
        mode (str, optional): Mode de sauvegarde. Par défaut, "aus".

    Returns:
        None
    """
    fs_from = None
    if mode == "s3":
        fs_from = s3_filesystem()

    yaml_prod_path = get_paths_yaml_path(root, mode, "prod")
    doc = load_yaml(yaml_prod_path)
    dct = doc[mode]

    for k, path in dct.items():
        for root, dirs, files in my_walk(path, fs_from):
            for file_name in files:
                print(file_name)
                file_path = os.path.join(root, file_name)
                my_copy(file_path, doc["local"][k], fs_from, None)


def push_saving_files(root: Union[str, Path], mode: str = "aus") -> None:
    """
    Copie les fichiers de sauvegarde vers une destination spécifiée.
    !!!ne fonctionne pas avec S3 pour l'instant!!!

    Args:
        root (Union[str, Path]): Chemin racine du répertoire.
        mode (str, optional): Mode de sauvegarde. Par défaut, "aus".

    Returns:
        None
    """
    fs_to = None
    if mode == "s3":
        fs_to = s3_filesystem()

    yaml_prod_path = get_paths_yaml_path(root, mode, "prod")
    doc = load_yaml(yaml_prod_path)

    for k, path in doc.items():
        for root, dirs, files in my_walk(path, fs_to):
            for file_name in files:
                print(file_name)
                file_path = os.path.join(root, file_name)

                my_copy(doc["local"][k], file_path, fs_to, None)
