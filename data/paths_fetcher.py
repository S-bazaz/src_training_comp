# -*- coding: utf-8 -*-
"""
functions used to fetch paths for Imports and exportations
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
from pathlib import Path
from typing import Dict, Union, List, Tuple, Any


##################
#      Imports     #
##################

root_path = Path(os.path.abspath(__file__)).parents[2]
sys.path.insert(0, str(root_path))

from src.data.loading_saving import s3_filesystem, load_yaml, save_yaml
from src.idcodif import IdData, IdCodif, get_idcodif


##################
#  YAML paths  #
##################


def get_config_path(root: str, filename: str) -> str:
    """
    Construit et retourne le chemin du fichier de configuration YAML.

    Args:
        root (str or Path): Répertoire racine à partir duquel construire le chemin.
        filename (str): Nom du fichier de configuration (sans extension).

    Returns:
        str: Chemin complet du fichier de configuration YAML.
    """
    if isinstance(root, str):
        pathobj = Path(root)
    else:
        pathobj = root

    pathobj = pathobj.joinpath("data_config", f"{filename}.yaml")
    return str(pathobj)


def get_paths_yaml_path(root: Union[str, Path], mode: str, type: str = "raw") -> str:
    """
    Retourne le chemin du fichier YAML contenant
    les chemins des données spécifiées ou les chemins des fichiers de sauvegardes.

    Args:
        root (Union[str, Path]): Le chemin racine.
        mode (str): Le mode spécifique.
        type (str, optionnel): Le type de données. Par défaut, "raw".

    Returns:
        str: Le chemin du fichier YAML.
    """
    if isinstance(root, str):
        root = Path(root)

    temp = root.joinpath("data", "paths", mode, f"{type}_data_paths.yaml")
    return str(temp)


def get_saving_paths(root: Union[str, Path], mode: str = "aus") -> Dict[str, str]:
    """
    Récupère les chemins de sauvegarde à partir du fichier YAML de configuration.

    Args:
        root (Union[str, Path]): Le répertoire racine.
        mode (str, optionnel): Le mode de sauvegarde à récupérer.
            Par défaut, "aus".

    Returns:
        Dict[str, str]: Un dictionnaire contenant les chemins de sauvegarde locaux.
            Les clés correspondent aux noms de fichiers et les valeurs aux chemins de sauvegarde.
    """
    yaml_prod_path = get_paths_yaml_path(root, mode, "prod")
    doc = load_yaml(yaml_prod_path)
    return doc["local"]


def get_model_specific_config_path(dct_paths: Dict[str, str], id: IdCodif,
                                   lst_prefix_ext=["temp", "max", "gs"][::-1]) -> str:
    """
    Renvoie le chemin du fichier de configuration spécifique du model.

    Construit le chemin du fichier de configuration spécifique au modèle en fonction de l'ID du modèle,
    puis vérifie si le répertoire correspondant existe. S'il n'existe pas, le répertoire est créé.
    Ensuite, génère le nom du fichier de configuration en utilisant le nom du sous-ensemble de données et
    l'extension appropriés en fonction de l'ID de configuration.

    Args:
        dct_paths (Dict[str, str]): Un dictionnaire contenant les chemins de fichiers nécessaires.
        id (IdCodif): L'ID du modèle.
        lst_prefix_ext (List[str], optional): Une liste contenant les préfixes d'extension de configuration
            dans l'ordre inverse de priorité. Par défaut, ["temp", "max", "gs"].

    Returns:
        str: Le chemin complet du fichier de configuration spécifique au modèle.
    """

    pathobj = Path(dct_paths["04_model_hyper_para"])
    pathobj = pathobj.joinpath(id.model.var, id.model.algo)
    pathobj = pathobj.joinpath(id.data.enquete, id.data.version)

    if not os.path.exists(str(pathobj)):
        os.makedirs(str(pathobj))

    subset = str(id.data.subset)
    config = str(id.config)
    ext = config[-1]
    for prefix in lst_prefix_ext:
        if prefix in config:
            ext = f"{prefix}{ext}"

    if "eggs" in config:
        ext.replace("gs", "eggs")
    return str(pathobj.joinpath(f"{subset}_{ext}.yaml"))

def get_model_config_paths(dct_paths: Dict[str, str], id: IdCodif) -> Tuple[str, str]:
    """
    Renvoie les chemins des fichiers de configuration global et spécifique au modèle.

    Construit le chemin du fichier de configuration global en fonction de la variable et de l'algorithme
    du modèle spécifiés dans l'ID du modèle. Ensuite, utilise la fonction `get_model_specific_config_path`
    pour obtenir le chemin du fichier de configuration spécifique du modèle.

    Args:
        dct_paths (Dict[str, str]): Un dictionnaire contenant les chemins de fichiers nécessaires.
        id (IdCodif): L'ID du modèle.

    Returns:
        Tuple[str, str]: Un tuple contenant le chemin du fichier de configuration global et le chemin
            du fichier de configuration spécifique au modèle.
    """
    var = id.model.var
    algo = id.model.algo

    pathobj = Path(dct_paths["data_config"])
    pathobj = pathobj.joinpath(f"4_model_general_config_{var}_{algo}.yaml")
    global_path = str(pathobj)

    specific_path = get_model_specific_config_path(dct_paths, id)

    return global_path, specific_path


def get_model_config_path_from_iddata(
    dct_paths: Dict[str, str],
    iddata: IdData,
    var: str = "pcs",
    algo: str = "fasttext",
    config: str = "0",
) -> str:
    """
    Obtient le chemin du fichier de configuration du modèle à partir des informations d'identification des données.

    Args:
        dct_paths (Dict[str, str]): Dictionnaire des chemins de fichiers.
        iddata (IdData): Objet d'identification des données.
        var (str, optional): Variable du modèle. Par défaut "pcs".
        algo (str, optional): Algorithme du modèle. Par défaut "fasttext".
        config (str, optional): Version de configuration du modèle. Par défaut "0".

    Returns:
        str: Chemin du fichier de configuration du modèle.
    """
    idcodif = get_idcodif(iddata, var=var, algo=algo, config=config)
    return get_model_specific_config_path(dct_paths, idcodif)


def get_variables_config_path(root: Any, idcodif: IdCodif) -> str:
    """
    Obtient le chemin du fichier de configuration des variables
    à partir de la racine et de l'identification de la codification.

    Args:
        root (Any): Racine du chemin.
        idcodif (IdCodif): Objet d'identification du codage.

    Returns:
        str: Chemin du fichier de configuration des variables.
    """
    var = idcodif.model.var.lower()
    return get_config_path(root, f"3_variables_config_{var}")

def get_analyses_data_path(
    dct_paths: Dict[str, str], name: str = "crossmetrics", var: str = "pcs", idbase: str = "0"
) -> str:
    """
    Renvoie le chemin du fichier de données d'analyse.

    Construit le chemin du fichier de données d'analyse en fonction du nom, de la variable et de l'identifiant de base
    spécifiés. Le chemin est construit en utilisant le répertoire spécifié dans `dct_paths["06_analyses_data"]`.

    Args:
        dct_paths (Dict[str, str]): Un dictionnaire contenant les chemins de fichiers nécessaires.
        name (str, optional): Le nom du fichier de données d'analyse. Par défaut, "crossmetrics".
        var (str, optional): La variable. Par défaut, "pcs".
        idbase (str, optional): L'identifiant de base. Par défaut, "0".

    Returns:
        str: Le chemin du fichier de données d'analyse.
    """
    pathobj = Path(dct_paths["06_analyses_data"])
    pathobj = pathobj.joinpath(f"{var}_{name}_{idbase}.parquet")
    return str(pathobj)

####################
#  paths and trees #
####################

def add_path_to_nodes(dct: Dict[str, Union[Dict, None]], accu_path: Path) -> None:
    """
    Ajoute un chemin aux nœuds d'un dictionnaire récursif.

    Args:
        dct (Dict[str, Union[Dict, None]]): Le dictionnaire à modifier.
        accu_path (Path): Le chemin accumulé à ajouter aux nœuds.

    Returns:
        None
    """
    if dct is not None:
        for k in dct:
            if dct[k] is not None:
                add_path_to_nodes(dct[k], accu_path.joinpath(k))
                dct[k]["path"] = str(accu_path.joinpath(k))
            else:
                dct[k] = {"path": str(accu_path.joinpath(k))}

        if "path" not in dct:
            dct["path"] = str(accu_path)


def get_leaves_with_path(
    dct: Dict[str, Union[Dict, None]], accu: List[str] = []
) -> None:
    """
    Récupère les chemins des feuilles d'un dictionnaire récursif.

    Args:
        dct (Dict[str, Union[Dict, None]]): Le dictionnaire à parcourir.
        accu (List[str], optionnel): La liste accumulatrice pour stocker les chemins.
        Par défaut, une liste vide.

    Returns:
        None
    """
    if dct is not None:
        child = list(dct)
        child.remove("path")
        if len(child) > 0:
            for k in child:
                get_leaves_with_path(dct[k], accu)
        else:
            accu.append(dct["path"])


def aux_accu_leaves(
    tree: Dict[str, Union[Dict, None]], mode: str, accu: List[str], accu_repo: List[str]
) -> None:
    """
    Fonction auxiliaire pour récupèrer les chemins des feuilles
    de deux arbres de dictionnaires avec différents modes.

    Args:
        tree (Dict[str, Union[Dict, None]]):
            L'arbre de dictionnaires contenant les modes et les répertoires.
        mode (str):
            Le mode spécifique à parcourir dans l'arbre ( s3 ou aus )
        accu (List[str]):
            La liste accumulatrice pour stocker les chemins des feuilles du mode spécifique.
        accu_repo (List[str]):
            La liste accumulatrice pour stocker les chemins des feuilles du répertoire spécifique.

    Returns:
        None
    """
    get_leaves_with_path(tree[f"{mode}"], accu=accu)
    get_leaves_with_path(tree[f"repo_{mode}"], accu=accu_repo)


def get_leaves_by_mode(
    tree: Dict[str, Union[Dict, None]], mode: str = "aus"
) -> Tuple[List[str], Union[List[str], Dict]]:
    """
    Récupère les chemins des feuilles d'un arbre de dictionnaires
    en fonction du mode spécifié.

    Args:
        tree (Dict[str, Union[Dict, None]]):
            L'arbre de dictionnaires contenant les modes et les répertoires.
        mode (str, optionnel):
            Le mode spécifique à parcourir dans l'arbre. Par défaut, "aus".

    Returns:
        Tuple[List[str], Union[List[str], Dict]]: Les feuilles du mode spécifié et les feuilles S3 (si le mode n'est pas "aus").
    """
    leaves = []
    leaves_s3 = []

    if mode == "aus":
        aux_accu_leaves(tree, mode, leaves, leaves)
    else:
        leaves_s3 = {}
        aux_accu_leaves(tree, mode, leaves_s3, leaves)

    return leaves, leaves_s3


################################
#  paths fetcher for raw data  #
################################


def save_path_where_right_format(
    file_path: str, formats: List[str], dct_accu: Dict[str, str]
) -> None:
    """
    Vérifie si le format du fichier correspond à ceux spécifiés dans la liste `formats`,
    puis ajoute le chemin du fichier au dictionnaire `dct_accu` en utilisant le nom de fichier sans extension comme clé.

    Args:
        file_path (str): Le chemin du fichier.
        formats (List[str]): Une liste de formats valides.
        dct_accu (Dict[str, str]): Le dictionnaire pour stocker les chemins des fichiers au format correct.

    Returns:
        None
    """
    pathobj = Path(file_path)
    ext = pathobj.suffix[1:]

    if ext in formats:
        print(file_path)
        dct_accu[pathobj.stem] = file_path


def my_walk(path: str, fs: Union[None, s3fs.S3FileSystem] = None):
    """
    Parcourt récursivement un répertoire
    et retourne un générateur contenant les chemins des fichiers et répertoires.

    Args:
        path (str): Le chemin du répertoire à parcourir.
        fs (Union[None, s3fs.S3FileSystem], optionnel): L'objet S3FileSystem pour le stockage sur le cloud.
            Par défaut, None.

    Returns:
        Generator: Un générateur contenant les chemins des fichiers et répertoires.
    """
    if fs is None:
        return os.walk(path)
    else:
        return fs.walk(path)


def get_all_data_from_path(
    path: str,
    formats: List[str],
    dct_accu: Dict[str, str],
    fs: Union[None, s3fs.S3FileSystem] = None,
) -> None:
    """
    Parcourt récursivement un répertoire,
    et sauvegarde les chemins des fichiers correspondant aux formats spécifiés dans le dictionnaire `dct_accu`.

    Args:
        path (str): Le chemin du répertoire à parcourir.
        formats (List[str]): Une liste de formats valides.
        dct_accu (Dict[str, str]): Le dictionnaire pour stocker les chemins des fichiers au format correct.
        fs (Union[None, s3fs.S3FileSystem], optionnel): L'objet S3FileSystem pour le stockage sur le cloud.
            Par défaut, None.

    Returns:
        None
    """
    for root, dirs, files in my_walk(path, fs):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            save_path_where_right_format(file_path, formats, dct_accu)


def get_all_data_from_paths(
    paths: List[str],
    formats: List[str],
    dct_accu: Dict[str, str],
    fs: Union[None, s3fs.S3FileSystem] = None,
) -> None:
    """
    Parcourt récursivement plusieurs répertoires
     et sauvegarde les chemins des fichiers correspondant
     aux formats spécifiés dans le dictionnaire `dct_accu`.

    Args:
        paths (List[str]):
            Une liste de chemins de répertoires à parcourir.
        formats (List[str]):
            Une liste de formats valides.
        dct_accu (Dict[str, str]):
            Le dictionnaire pour stocker les chemins des fichiers au format correct.
        fs (Union[None, s3fs.S3FileSystem], optionnel):
            L'objet S3FileSystem pour le stockage sur le cloud.
            Par défaut, None.

    Returns:
        None
    """
    for path in paths:
        get_all_data_from_path(path, formats, dct_accu, fs=fs)


def get_selected_data_files(doc: Dict, mode: str = "aus") -> Dict[str, Dict[str, str]]:
    """
    Récupère tous les chemins des fichiers de données sélectionnés
    selon le mode spécifié.

    Args:
        doc (Dict): Le document contenant les informations sur les données.
        mode (str, optionnel): Le mode spécifique. Par défaut, "aus".

    Returns:
        Dict[str, Dict[str, str]]: Un dictionnaire contenant les fichiers de données sélectionnés,
        organisés par type ("local" et "s3").
    """

    formats = doc["raw_data_formats"]

    leaves, leaves_s3 = get_leaves_by_mode(doc["raw_data"], mode=mode)
    data_files, data_files_s3 = {}, {}

    if mode == "s3":
        get_all_data_from_paths(leaves_s3, formats, data_files_s3, fs=s3_filesystem())
    get_all_data_from_paths(leaves, formats, data_files, fs=None)
    return {"local": data_files, "s3": data_files_s3}


######################################
#  saving files fetcher and creation #
######################################


def paths_list_to_dict(lst: List[str]) -> Dict[str, str]:
    """
    Convertit une liste de chemins en un dictionnaire
    où les clés sont les noms des fichiers (sans extension)
    et les valeurs sont les chemins complets.

    Args:
        lst (List[str]): La liste de chemins.

    Returns:
        Dict[str, str]: Le dictionnaire résultant.
    """
    return {Path(path).stem: path for path in lst}


def get_all_selected_saving_files(
    doc: Dict, mode: str = "aus"
) -> Dict[str, Dict[str, str]]:
    """
    Récupère tous les fichiers de sauvegarde sélectionnés selon le mode spécifié.

    Args:
        doc (Dict): Le document contenant les informations sur les données.
        mode (str, optionnel): Le mode spécifique. Par défaut, "aus".

    Returns:
        Dict[str, Dict[str, str]]: Un dictionnaire contenant
            les fichiers de sauvegarde sélectionnés,
            organisés par type ("local" et "s3").
    """
    leaves, leaves_s3 = get_leaves_by_mode(doc["prod_data"], mode=mode)
    return {"local": paths_list_to_dict(leaves), "s3": paths_list_to_dict(leaves_s3)}


def create_files_from_tree_with_path(dct: Dict[str, Dict]) -> None:
    """
    Crée des fichiers à partir d'un dictionnaire d'arborescence avec des chemins.

    Args:
        dct (Dict[str, Dict]): Le dictionnaire d'arborescence avec des chemins.

    Returns:
        None
    """
    if dct is not None:
        for k in dct:
            if k != "path":
                os.makedirs(dct[k]["path"], exist_ok=True)
                create_files_from_tree_with_path(dct[k])


####################################################
#  YAMLs of data and saving files paths creation   #
####################################################


def add_paths_and_create_files(
    root: Union[str, Path], doc: Dict[str, Dict], mode: str = "aus"
) -> None:
    """
    Ajoute les chemins et crée les fichiers à partir du dictionnaire donné.
    Seul les fichiers sur s3 ne sont pas créés

    Args:
        root (Union[str, Path]): La racine du chemin.
        doc (Dict[str, Dict]): Le dictionnaire contenant les données à traiter.
        mode (str, optionnel): Le mode de traitement des données.
            Par défaut, "aus".

    Returns:
        None
    """
    if isinstance(root, str):
        root = Path(root)

    if mode == "aus":
        for s in ["raw_data", "prod_data"]:
            add_path_to_nodes(doc[s]["aus"], Path(""))
            add_path_to_nodes(doc[s]["repo_aus"], root)

            create_files_from_tree_with_path(doc[s]["aus"])
            create_files_from_tree_with_path(doc[s]["repo_aus"])
    else:
        for s in ["raw_data", "prod_data"]:
            add_path_to_nodes(doc[s]["s3"], Path(""))
            add_path_to_nodes(doc[s]["repo_s3"], root)

            create_files_from_tree_with_path(doc[s]["repo_s3"])


def create_data_paths_yamls(root: Union[str, Path], mode: str = "aus") -> None:
    """
    Crée les fichiers YAML des chemins de données.

    Args:
        root (Union[str, Path]): La racine du chemin.
        mode (str, optionnel): Le mode de création des fichiers YAML.
            Par défaut, "aus".

    Returns:
        None
    """
    config_path = get_config_path(root, "1_files_config")
    doc = load_yaml(config_path)

    add_paths_and_create_files(root, doc, mode=mode)

    doc_raw = get_selected_data_files(doc, mode=mode)
    doc_prod = get_all_selected_saving_files(doc, mode=mode)

    yaml_raw_path = get_paths_yaml_path(root, mode, "raw")
    yaml_prod_path = get_paths_yaml_path(root, mode, "prod")

    save_yaml(yaml_raw_path, doc_raw)
    save_yaml(yaml_prod_path, doc_prod)
