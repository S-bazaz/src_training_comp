# -*- coding: utf-8 -*-
"""
functions used to load and save : dataframes, dictionaries, figures ...
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
import yaml
import s3fs
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union, BinaryIO


##################
#      Imports     #
##################

root_path = Path(os.path.abspath(__file__)).parents[2]
sys.path.insert(0, str(root_path))

from src.data.dataframe_proc import df_sas7bdat_reformat


###########
#  YAML   #
###########


def load_yaml(path: str) -> Dict:
    """
    Charge et retourne la configuration à partir d'un fichier YAML.

    Args:
        path (str): Chemin du fichier de configuration YAML.

    Returns:
        dict: Configuration chargée à partir du fichier YAML.
    """
    return yaml.safe_load(open(path, "r"))


def save_yaml(newpath: str, doc: dict, comment: str = None) -> None:
    """
    Crée un nouveau fichier de configuration YAML à partir d'un dictionnaire.

    Args:
        newpath (str): Chemin du nouveau fichier de configuration YAML à créer.
        doc (dict): Dictionnaire contenant les informations de configuration à sauvegarder.
        comment (str) : commentaire à ajouter au début du fichier, sert à indiquer les performances obtenues à l'issue du gridsearch
    """
    with open(newpath, "w") as f:
        if comment:
            f.write("# " + comment + "\n")
        yaml.dump(doc, f)


def init_model_config(path: str) -> None:
    """
    Initialise un fichier de configuration de modèle avec des paramètres vides.

    Args:
        path (str): Chemin du fichier de configuration.

    Returns:
        None
    """
    if not os.path.exists(path):
        print(f"init yaml {path}")
        with open(path, "w") as f:
            yaml.dump(
                {"preprocessing": {}, "train": {}, "predict": {}, "evaluate": {}}, f
            )


##################################
#  filesystem and file loading   #
##################################


def s3_filesystem(
    key_id: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID", default=None),
    acces_key: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY", default=None),
    token: Optional[str] = os.getenv("AWS_SESSION_TOKEN", default=None),
    endpoint: Optional[str] = "https://"
    + str(os.getenv("AWS_S3_ENDPOINT", default=None)),
) -> Optional[s3fs.S3FileSystem]:
    """
    Renvoie un objet S3FileSystem utilisé pour interagir avec le système de
    fichiers S3 d'AWS.

    Args:
        key_id (str, optionnel): L'ID de la clé d'accès AWS.
            Par défaut, il est obtenu à partir de la variable d'environnement
            `AWS_ACCESS_KEY_ID`.
        acces_key (str, optionnel): La clé d'accès AWS.
            Par défaut, elle est obtenue à partir de la variable d'environnement `
            AWS_SECRET_ACCESS_KEY`.
        token (str, optionnel): Le jeton de session AWS.
            Par défaut, il est obtenu à partir de la variable d'environnement
            `AWS_SESSION_TOKEN`.
        endpoint (str, optionnel): L'URL de l'endpoint S3.
            Par défaut, il est obtenu à partir de la variable d'environnement
            `AWS_S3_ENDPOINT`.

    Returns:
        s3fs.S3FileSystem: Un objet S3FileSystem
        utilisé pour interagir avec le système de fichiers S3 d'AWS.
        Si au moins l'un des paramètres requis (endpoint, key_id, acces_key, token) est manquant,
        None est retourné.
    """

    if any(elem is None for elem in [endpoint, key_id, acces_key, token]):
        return None

    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": endpoint},
        key=key_id,
        secret=acces_key,
        token=token,
    )


def get_file(
    path: str,
    cloud_fs: s3fs.S3FileSystem = None,
    extension: str = "parquet",
    encoding: str = "utf-8",
) -> Union[BinaryIO, s3fs.S3File]:
    """
    Renvoie un objet de type BinaryIO ou S3File pour le chemin donné en paramètre.

    Args:
        path (str): Le chemin vers le fichier.
        cloud_fs (s3fs.S3FileSystem, optionnel):
            L'objet S3FileSystem pour le stockage sur le cloud.
            Par défaut, il est initialisé avec les clés d'accès
            et le jeton de session définis dans les variables d'environnement.
        extension (str, optionnel): L'extension du fichier. Par défaut, "parquet".
        encoding (str, optionnel): L'encodage du fichier. Par défaut, "utf-8".

    Returns:
        Union[BinaryIO, s3fs.S3File]: L'objet pour accéder au fichier.
    """
    mode = "rb"
    if extension not in ["parquet", "sas7bdat"]:
        mode = "r"

    if cloud_fs is None:
        if encoding != "" and mode != "rb":
            return open(path, mode=mode, encoding=encoding)
        else:
            return open(path, mode=mode)
    else:
        return cloud_fs.open(path, mode=mode)


########################
#  dataframe loading   #
########################


def load_df(
    path: str,
    file_system: Union[s3fs.S3FileSystem, None] = None,
    csv_sep: str = ";",
    special_sep: str = "\t",
    encoding: str = "",
) -> pd.DataFrame:
    """
    Charge un fichier CSV, Parquet ou SAS7BDAT en tant que DataFrame pandas.
    Uniquement des parquet si on part bien des bases intermédiaires

    Args:
        path (str): Le chemin du fichier à charger.
        file_system (s3fs.S3FileSystem, optionnel): Le système de fichiers cloud à utiliser.
            Par défaut, il n'est pas spécifié ce qui revient à être en local.
        csv_sep (str, optionnel): Le séparateur de colonne pour les fichiers CSV.
            Par défaut, ";".
        special_sep (str, optionnel): Le séparateur de colonne pour les fichiers spéciaux.
            Par défaut, "\t".
        encoding (str, optionnel): L'encodage du fichier. Par défaut, "".

    Returns:
        pd.DataFrame: Le DataFrame pandas chargé à partir du fichier.
    """

    extension = path.split(".")[-1]
    f = get_file(path, file_system, extension, encoding=encoding)
    print(f)

    if extension == "csv":
        return pd.read_csv(f, sep=csv_sep)

    elif extension == "parquet":
        return pd.read_parquet(f)

    elif extension == "sas7bdat":
        df = pd.read_sas(f, format="sas7bdat")  # utiliser pyreadstat à la place
        return df_sas7bdat_reformat(df)
    else:
        return pd.read_csv(path, sep=special_sep)


def load_dfs(
    dct_path: Dict[str, str],
    dct_sep: Dict[str, str] = {},
    file_system: Union[s3fs.S3FileSystem, None] = None,
    default_sep: str = ";",
    special_sep: str = "\t",
    encoding: str = "",
) -> Dict[str, pd.DataFrame]:
    """
    Charge plusieurs fichiers en tant que DataFrames pandas.

    Args:
        dct_path (Dict[str, str]):
            Un dictionnaire contenant les noms de fichiers en tant que clés
            et les chemins des fichiers en tant que valeurs.
        dct_sep (Dict[str, str], optionnel):
            Un dictionnaire contenant les séparateurs de colonne spécifiques par fichier.
            Les clés correspondent aux noms de fichiers et les valeurs aux séparateurs.
            Par défaut, un dictionnaire vide.
        file_system (s3fs.S3FileSystem, optionnel):
            Le système de fichiers cloud à utiliser.
            Par défaut, il n'est pas spécifié ce qui revient à être en local.
        default_sep (str, optionnel):
            Le séparateur de colonne par défaut pour les fichiers CSV.
            Par défaut, ";".
        special_sep (str, optionnel):
            Le séparateur de colonne pour les fichiers spéciaux.
            Par défaut, "\t".
        encoding (str, optionnel):
            L'encodage des fichiers. Par défaut, "" (mode auto)

    Returns:
        Dict[str, pd.DataFrame]:
            Un dictionnaire contenant les DataFrames chargés à partir des fichiers.
            Les clés correspondent aux noms de fichiers et les valeurs aux DataFrames pandas.
    """

    dct = {}

    for file_name, path in dct_path.items():
        if file_name in dct_sep:
            csv_sep = dct_sep[file_name]
            special = dct_sep[file_name]
        else:
            csv_sep = default_sep
            special = special_sep
        dct[file_name] = load_df(
            path,
            file_system=file_system,
            csv_sep=csv_sep,
            special_sep=special,
            encoding=encoding,
        )
    return dct


##############
#  figures   #
##############

# TO DO
