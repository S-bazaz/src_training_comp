# -*- coding: utf-8 -*-
"""
functions used manipulate basic types : string, list, set, dictionary ...
It can be to encode or decode strings or create special mapping ...
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
import re
from typing import Dict, Any, List, Tuple, Optional, Union


##########
#  list  #
##########


def setop_to_list(
    lst1: List[Union[str, int]], lst2: List[Union[str, int]], op: str = "union"
) -> List[Union[str, int]]:
    """
    Effectue une opération ensembliste sur deux listes
    et renvoie le résultat sous forme de liste.

    Args:
        lst1 (List[Union[str, int]]): Première liste.
        lst2 (List[Union[str, int]]): Deuxième liste.
        op (str, optional): Opération ensembliste à effectuer.
            Les valeurs valides sont "union" (par défaut) et "intersection".

    Returns:
        List[Union[str, int]]: Résultat de l'opération ensembliste sous forme de liste.
    """
    set1 = set(lst1)
    set2 = set(lst2)
    if op == "union":
        return list(set1.union(set2))
    return list(set1.intersection(set2))


################
#  dictionary  #
################


def substring_from_keys_list(dct: Dict[str, Any], lst: List[str]) -> Dict[str, Any]:
    """
    Récupère une sous-section d'un dictionnaire
    en ne conservant que les clés présentes dans une liste donnée.

    Args:
        dct (Dict[str, Any]): Le dictionnaire source.
        lst (List[str]): La liste des clés à conserver.

    Returns:
        Dict[str, Any]:
            Un nouveau dictionnaire contenant uniquement
            les paires clé-valeur correspondant aux clés de la liste.
    """
    if dct is None:
        return {}
    return {k: v for k, v in dct.items() if k in lst}


##########################
#  recursive dictionary  #
##########################


def get_varnames_from_subconfig(
    doc: Dict[str, Union[None, List[str], Dict[str, Union[None, List[str]]]]],
    sub_config_dct: Dict[str, List[str]],
) -> List[str]:
    """
    Récupère les noms de variables à partir d'un sous-arbre de configuration spécifiée dans un document.

    Args:
        doc (dict): Document contenant les configurations.
        sub_config_dct (dict): Dictionnaire spécifiant les sous-configurations à considérer.

    Returns:
        list: Liste des noms de variables récupérés à partir de la sous-configuration spécifiée.

    """
    varnames = []
    if doc is None:
        return varnames
    for k, v in sub_config_dct.items():
        dct = doc[k]
        for lstname in v:
            temp = dct[lstname]
            if temp is not None:
                if isinstance(temp, list):
                    varnames.extend(temp)
                else:
                    varnames.append(temp)
    varnames = list(set(varnames))
    return varnames


def add_config_to_doc(
    doc: Dict[str, Any],
    val: Union[List[str], Any],
    part: str,
    k: str,
    mode: str = "union",
) -> None:
    """
    Ajoute une configuration à un document existant.

    Args:
        doc (dict): Document contenant les configurations.
        val (list or dict): Configuration à ajouter.
        part (str): Partie du document à laquelle ajouter la configuration.
        k (str): Clé de la configuration à ajouter.
        mode (str, optional): Mode d'ajout de la configuration. Valeurs possibles : "union" (par défaut),
            "intersection". Si le mode est "union", les valeurs existantes et les nouvelles valeurs sont
            combinées en une liste. Si le mode est "intersection", seules les valeurs communes sont conservées.

    Returns:
        None

    """
    if k in doc[part]:
        if isinstance(val, list):
            doc[part][k] = setop_to_list(val, doc[part][k], op=mode)

        elif isinstance(val, dict):
            if len(val) > len(doc[part][k]):
                doc[part][k] = val

    elif mode == "union":
        doc[part][k] = val


def merge_config_to_doc(
    doc: Dict[str, Any], config: Dict[str, Any], mode: str = "union"
) -> Dict[str, Union[Dict[str, Union[list, dict]], None]]:
    """
    Fusionne une configuration dans un document existant.

    Args:
        doc (dict or None): Document existant ou None.
        config (dict): Configuration à fusionner dans le document.
        mode (str, optional): Mode de fusion. Valeurs possibles : "union" (par défaut), "intersection".
            Si le mode est "union", les valeurs existantes et les nouvelles valeurs sont combinées en une liste.
            Si le mode est "intersection", seules les valeurs communes sont conservées.

    Returns:
        dict: Document résultant de la fusion.

    """
    if doc is None:
        doc = config.copy()
    else:
        for part in config:
            for k in config[part]:
                add_config_to_doc(doc, config[part][k], part, k, mode=mode)
    return doc


####################################
# pcs identification decomposition #
####################################

# def get_var_from_model_id(model_id):
#     return re.findall(r"[A-Z]+", model_id)[0]
#
# def get_proftype_from_data_id(data_id):
#     return re.findall(r"PROF[AIS]+", data_id)[0][4:]

# def get_all_from_data_id(dct_data, dct_df, dct_config, data_id, model_id_pref="PCSfasttext"):
#     res = {}
#
#     res["df"] = dct_df[data_id]
#
#     if data_id in dct_config:
#         config_version = dct_config[data_id]
#     else:
#         config_version = 0
#
#     model_id = f"{model_id_pref}_{data_id}"
#     res["model_id"] = model_id
#     res["config_path"] = os.path.join(
#         dct_data["04_model_hyper_para"], f"{model_id}_{config_version}.yaml"
#     )
#
#     res["enquet"] = re.findall(r"[A-Z]+", data_id)[0]
#     res["core"] = re.findall(r"[^A-Z]+", data_id)[0]
#
#     res["prof"] = get_proftype_from_data_id(data_id)
#
#     return res
