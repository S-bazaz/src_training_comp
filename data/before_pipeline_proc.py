# -*- coding: utf-8 -*-
"""
functions used to create the final databases used as inputs to the model pipelines
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
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


##################
#      Imports     #
##################

root_path = Path(os.path.abspath(__file__)).parents[2]
sys.path.insert(0, str(root_path))

from src.idcodif import IdData, IdCodif
from src.data.paths_fetcher import (
    get_config_path,
    get_model_specific_config_path,
    get_variables_config_path,
)
from src.data.loading_saving import load_yaml, save_yaml, init_model_config
from src.data.dataframe_proc import (
    df_rename_cols,
    change_typing_from_dct,
    df_nan_harmonization,
)
from src.data.basictypes_proc import (
    setop_to_list,
    get_varnames_from_subconfig,
    merge_config_to_doc,
)


###############################
#  renaming, typing, queries  #
###############################


def extract_from_dataconfig(
    doc: Dict[str, Any],
    key: str,
    lst: List[str] = ["enquete", "version", "mapping", "queries", "typing_for_queries"],
) -> Tuple[Any, ...]:
    """
    Extrait les valeurs correspondant à certaines clés d'un dictionnaire de configuration.

    Args:
        doc (Dict[str, Any]): Dictionnaire de configuration.
        key (str): Clé principale du dictionnaire.
        lst (List[str], optional): Liste des clés à extraire. Par défaut, ["newname", "mapping", "queries", "typing_for_queries"].

    Returns:
        Tuple[Any, ...]: Tuple contenant les valeurs correspondantes aux clés extraites.
    """
    res = [doc[key][s] for s in lst]
    return tuple(res)


def add_queries_to_accu(
    df: Any,
    enquete: str,
    version: str,
    queries: Dict[str, str],
    dct_accu: Dict[str, Any],
) -> None:
    if queries is None:
        subset = re.findall(r"PROF[AIS]+", version)[0]
        version = version.replace(subset, "")
        iddata = IdData(enquete=enquete, version=version, subset=subset)
        dct_accu[iddata] = df
    else:
        for subset, q in queries.items():
            iddata = IdData(enquete=enquete, version=version, subset=subset)
            dct_accu[iddata] = df.query(q)


def drop_empty_dfs(accu: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supprime les DataFrames vides d'un dictionnaire accumulatif.

    Args:
        accu (Dict[str, Any]): Dictionnaire accumulatif contenant les DataFrames.

    Returns:
        Dict[str, Any]: Dictionnaire accumulatif avec les DataFrames non vides.
    """
    res = {}
    for data_id, df in accu.items():
        if len(df.index) == 0:
            print(f"!!!{data_id} is empty!!!")
        else:
            res[data_id] = df
    return res


def raw_data_renaming_nan_queries(
    root: str, dct_df: Dict[str, Any], debug_printing: bool = False
) -> Dict[str, Any]:
    """
    Effectue le renommage,
    la gestion des valeurs manquantes
    et l'application des requêtes sur les données brutes.

    Args:
        root (str): Chemin racine du projet.
        dct_df (Dict[str, Any]): Dictionnaire contenant les DataFrames brutes.
        debug_printing (bool, optional): Activation de l'affichage de débogage. Par défaut False.

    Returns:
        Dict[str, Any]: Dictionnaire contenant les DataFrames traitées.
    """
    accu = {}
    yaml_path = get_config_path(root, "2_loading_config")
    doc = load_yaml(yaml_path)
    for basename in doc:
        if basename in dct_df:
            if debug_printing:
                print(f"----------{basename} ------------")
            df = dct_df[basename]

            (
                enquete,
                version,
                mapping,
                queries,
                typing_for_queries,
            ) = extract_from_dataconfig(
                doc,
                basename,
                ["enquete", "version", "mapping", "queries", "typing_for_queries"],
            )
            # nan values
            df = df_nan_harmonization(df)
            # renaming
            df = df_rename_cols(df, mapping)
            # ajout nom de l'enquete pour les procédures de bootstrap du train
            df["enquete"] = enquete
            df["enquete_volumes_egaux"] = enquete

            if debug_printing:
                print(list(df))

            if typing_for_queries is not None:
                # to avoid typing and query error based on nullable type and nan value type conversion
                for col_query in typing_for_queries:
                    df.dropna(how='all', axis=0, subset=col_query, inplace=True)
                if "PROFA_STATUT" in df:
                    print(df["PROFA_STATUT"])
                if "PROFPR_STATUT" in df:
                    print(df["PROFPR_STATUT"])

                change_typing_from_dct(df, typing_for_queries)

            # queries
            add_queries_to_accu(df, enquete, version, queries, accu)

    return drop_empty_dfs(accu)


######################
#  columns creation  #
######################


def get_repr_to_idata(dct_df: Dict[str, Any]) -> Dict[str, IdData]:
    """
    Convertit un dictionnaire de données en un dictionnaire d'objets IdData correspondants.

    Args:
        dct_df (Dict[str, Any]): Dictionnaire de données.

    Returns:
        Dict[str, IdData]: Dictionnaire d'objets IdData correspondants.
    """
    return {k.to_str(): k for k in dct_df}


def unify_col_by_prof(
    dct_df: Dict[str, Any],
    repr_to_iddata: Dict[str, IdData],
    data_id_prefix: str,
    var: str,
    dct_proflinks: Dict[str, str] = {"A": "A", "I": "IS", "S": "IS"},
) -> None:
    """
    Unifie la colonne 'var' des données en utilisant les profils (A, I, S).

    Args:
        dct_df (Dict[str, Any]): Dictionnaire de données.
        repr_to_iddata (Dict[str, IdData]): Dictionnaire des objets IdData correspondants.
        data_id_prefix (str): Préfixe de l'ID de données.
        var (str): Nom de la colonne à unifier.
        dct_proflinks (Dict[str, str], optional): Dictionnaire de correspondance des profils.
            Par défaut, {"A": "A", "I": "IS", "S": "IS"}.

    Returns:
        None
    """
    for cat in ["A", "I", "S"]:
        data_id = f"{data_id_prefix}PROF{cat}"
        if data_id in repr_to_iddata:
            ext = dct_proflinks[cat]
            iddata = repr_to_iddata[data_id]
            if f"{var}_PROF{ext}" in dct_df[iddata]:
                dct_df[iddata][var] = dct_df[iddata][f"{var}_PROF{ext}"]
                print(f"Change {var}_PROF{ext} to {var}")
            if f"{ext}{var}" in dct_df[iddata]:
                dct_df[iddata][var] = dct_df[iddata][f"{ext}{var}"]
                print(f"Change {ext}{var} to {var}")


###################
# config checking #
###################

def check_config_and_adapt(dct_paths: Dict[str, str], dct_id: Dict[Any, Any], delete = False) -> None:
    """Vérifie l'existence des fichiers de configuration des modèles et adapte les paramètres si nécessaire.

    La fonction parcourt les paires d'`iddata` et `idcodif` dans le dictionnaire `dct_id` et vérifie si le fichier
    de configuration du modèle correspondant existe dans le chemin spécifié par `dct_paths`. Si un fichier de
    configuration n'existe pas, la fonction supprime le préfixe "gs"
    de la valeur de `idcodif.config` pour revenir à un paramétrage non optimisé

    Args:
        dct_paths (Dict[str, str]): Dictionnaire des chemins des fichiers de configuration des modèles.
        dct_id (Dict[Any, Any]): Dictionnaire des paires d'`iddata` et `idcodif`.

    Returns:
        None
    """
    for iddata, idcodif in dct_id.copy().items():
        path = get_model_specific_config_path(dct_paths, idcodif)

        if not os.path.exists(path):
            print(idcodif)
            if delete:
                print("config doesn't exist -> not tested")
                del dct_id[iddata]
            else:
                print("config doesn't exist -> not optimize configuration")
                dct_id[iddata].config = idcodif.config.replace("gs", "")



#####################
# columns selection #
#####################


def add_columns_to_doc(
    accu: Dict[str, Any], doc_config: Dict[str, Any], config_path: str
) -> None:
    """
    Ajoute les colonnes détectées au document de configuration.

    Args:
        accu (dict): Accumulateur contenant les colonnes détectées.
        doc_config (dict):document de configuration.
        config_path (str): Chemin vers la configuration.

    Returns:
        None

    """
    an_update_is_needed = False
    for col in set(accu):
        if doc_config["typing"] is None:
            doc_config["typing"] = {col: None}
            an_update_is_needed = True

        elif col not in doc_config["typing"]:
            doc_config["typing"][col] = None
            an_update_is_needed = True

    if an_update_is_needed:
        print("---------WARNING--------")
        print("a new column is detected")
        print("please select its type in :")
        print(config_path)


def select_columns_and_set_typing_config(
    root: str,
    dct_paths: Dict[str, str],
    dct_df: Dict[str, pd.DataFrame],
    dct_idcodif: Dict[str, IdData],
) -> None:
    """
    Sélectionne les colonnes spécifiées dans le fichier de configuration des variables
    et met à jour le fichier de configuration des types.

    Args:
        root (str): Répertoire racine.
        dct_paths (Dict[str, str]): Dictionnaire des chemins.
        dct_df (Dict[str, pd.DataFrame]): Dictionnaire des DataFrames.
        dct_idcodif (Dict[str, IdData]): Dictionnaire des informations d'identification.

    Returns:
        None
    """
    config_path = None
    accu = []
    print("______Selection of columns and typing__________")
    for iddata in dct_df:
        idcodif = dct_idcodif[iddata]
        print("Select columns ", idcodif)

        if config_path is None:
            config_path = get_variables_config_path(root, idcodif)
            doc_config = load_yaml(config_path)
            config_vars = doc_config["columns"]

        model_config_path = get_model_specific_config_path(dct_paths, idcodif)
        init_model_config(model_config_path)
        doc = load_yaml(model_config_path)

        cols = get_varnames_from_subconfig(doc, config_vars)
        dct_df[iddata] = dct_df[iddata][cols]

        accu.extend(cols)

    add_columns_to_doc(accu, doc_config, config_path)
    save_yaml(config_path, doc_config)


##########
# typing #
##########


def harmonize_types(
    root: str, dct_df: Dict[str, pd.DataFrame], var: str = "pcs"
) -> None:
    """
    Harmonise les types des colonnes des DataFrames en fonction de la configuration.

    Args:
        root (str): Chemin racine.
        dct_df (dict): Dictionnaire contenant les DataFrames.
        var (str, optional): Variable utilisée. Par défaut, "pcs".

    Returns:
        None

    """
    config_path = get_config_path(root, f"3_variables_config_{var}")
    typing = load_yaml(config_path)["typing"]
    for df in dct_df.values():
        for col in df:
            # print("harmonise type : ", col)
            df[col] = df[col].astype(typing[col])


####################
# list for merging #
####################


def merge_all_and_tcm_by_prof(dct_df: Dict[str, Any]) -> List[List[str]]:
    """
    Génère les combinaisons de table et config pour les fusions de données

    Args:
        dct_df (Dict[str, Any]): Dictionnaire des DataFrames de données.

    Returns:
        List[List[str]]: Listes des identifiants de données à fusionner.
    """

    lst_selection = []
    accu_all = []
    accu_tcm = []
    for type_prof in ["A", "I", "S"]:
        for data_id in dct_df:
            lst_type = data_id.subset.upper().replace("PROF", "")
            lst_type = list(lst_type)
            if type_prof in lst_type:
                accu_all.append(data_id)
                enquete = data_id.enquete.upper()
                if enquete != "RP" and enquete != "EEC":  # Uniquement RP ? Uniquement pour PROFA ?
                    accu_tcm.append(data_id)
        lst_selection.append(accu_all.copy())
        lst_selection.append(accu_tcm.copy())
        accu_all = []
        accu_tcm = []
    return lst_selection

def merge_all_by_prof_except_rp_profa(dct_df: Dict[str, Any]) -> List[List[str]]:
    """
    Génère les combinaisons de table et config pour les fusions de données

    Args:
        dct_df (Dict[str, Any]): Dictionnaire des DataFrames de données.

    Returns:
        List[List[str]]: Listes des identifiants de données à fusionner.
    """

    lst_selection = []
    accu_all = []

    for type_prof in ["A", "I", "S"]:
        if type_prof == "A":
            accu_except_rp_profa = []
        for data_id in dct_df:
            lst_type = data_id.subset.upper().replace("PROF", "")
            lst_type = list(lst_type)
            enquete = data_id.enquete.upper()
            if type_prof in lst_type:
                accu_all.append(data_id)
                if type_prof == "A" and enquete != "RP":
                    accu_except_rp_profa.append(data_id)


        lst_selection.append(accu_all.copy())
        if type_prof == "A":
            lst_selection.append(accu_except_rp_profa.copy())

        accu_all = []
    return lst_selection


###########
# merging #
###########


def get_dct_info_data(
    iddata: str,
    dct_paths: Dict[str, str],
    dct_df: Dict[str, Any],
    dct_idcodif: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Obtient les informations sur les données à partir de leurs identifiants.

    Args:
        iddata (str): Identifiant des données.
        dct_paths (Dict[str, str]): Dictionnaire des chemins de fichiers.
        dct_df (Dict[str, Any]): Dictionnaire des DataFrames de données.
        dct_idcodif (Dict[str, Any]): Dictionnaire des configurations d'identification des données.

    Returns:
        Dict[str, Any]: Dictionnaire contenant les informations sur les données.
    """
    dct = {"df": dct_df[iddata].copy(), "id": dct_idcodif[iddata].copy()}
    dct["doc"] = load_yaml(get_model_specific_config_path(dct_paths, dct["id"]))
    return dct


def merge_update_accu(dct0: Dict[str, Any], accu: Dict[str, Any], mode: str) -> None:
    """
    Fusionne et met à jour les informations dans le dictionnaire d'accumulation.

    Args:
        dct0 (Dict[str, Any]): Dictionnaire contenant les informations à fusionner.
        accu (Dict[str, Any]): Dictionnaire d'accumulation à mettre à jour.
        mode (str): Mode de fusion pour la configuration.

    Returns:
        None
    """
    dct = dct0.copy()
    accu["doc"] = merge_config_to_doc(accu["doc"], dct["doc"], mode=mode)

    # create merge weights

    ## Ca marche mais c'est trop lent

    # weights_test = dct["doc"]["preprocessing"]["weights_test"]
    # weights_score = dct["doc"]["evaluate"]["weights_score"]
    # if weights_score:
    #     weights = weights_score
    # else:
    #     weights = weights_test
    # if weights:
    #     temp_weights = dct["df"][weights].copy()
    #     # nan values -> median
    #     temp_weights = temp_weights.fillna(temp_weights.median(skipna=True))
    #     # normalise
    #     # print(temp_weights)
    #     temp_weights.apply(lambda x: x / float(temp_weights.sum()))
    # else:
    #     temp_weights = 1/(dct["df"].shape[0])
    #
    # dct["df"]["merge_weights"] = temp_weights



    if accu["df"] is None:
        accu["df"] = dct["df"]
    else:
        accu["df"] = pd.merge(accu["df"], dct["df"], how="outer")

    if accu["id"] is None:
        accu["id"] = dct["id"]
        accu["id"].data_info_as_accu()
    else:
        enquete, version, subset = dct["id"].get_data_info()
        accu["id"].append_to_data_info_as_accu(enquete, version, subset)


def get_cols_for_intersection_with_prio(
    root: str, idcodif: Any, accu: Dict[str, Any], withprio=True
) -> Tuple:
    doc_config = load_yaml(get_variables_config_path(root, idcodif))
    config_vars = doc_config["columns"]
    df = accu["df"]
    cols = get_varnames_from_subconfig(accu["doc"], config_vars)
    if withprio:
        lst_prio = doc_config["prio_for_merge"]
        inter_prio = setop_to_list(lst_prio, list(df), op="intersection")
        cols = setop_to_list(inter_prio, cols, op="union")
        return cols, inter_prio
    return cols, []


def no_doubl_and_sort(lst: List) -> List:
    """
    Supprime les doublons d'une liste et la trie.

    Args:
        lst (List): Liste d'éléments.

    Returns:
        List: Liste sans doublons, triée.
    """
    lst2 = list(set(lst))
    lst2.sort()
    return lst2


def no_prof_lst_subset(lst_subset: List[str]) -> List[str]:
    """
    Supprime le préfixe 'PROF' de chaque élément d'une liste de sous-ensembles.

    Args:
        lst_subset (List[str]): Liste de sous-ensembles.

    Returns:
        List[str]: Liste des sous-ensembles sans le préfixe 'PROF'.
    """
    lst = [s.replace("PROF", "") for s in lst_subset]
    s = "".join(lst)
    return list(s)


def no_doubl_no_prof_and_sort(lst_subset: List[str]) -> List[str]:
    """
    Supprime les doublons, le préfixe 'PROF' et trie une liste de sous-ensembles.

    Args:
        lst_subset (List[str]): Liste de sous-ensembles.

    Returns:
        List[str]: Liste des sous-ensembles sans doublons, sans le préfixe 'PROF' et triée.
    """
    return no_doubl_and_sort(no_prof_lst_subset(lst_subset))


def codif_id_accu_mode_to_normal(idcodif: IdCodif, merge_mode: str) -> IdCodif:
    """
    Convertit le format des informations d'identification (idcodif) d'un accumulateur
    en un format normalisé pour la fusion en mode spécifié (merge_mode).

    Args:
        idcodif (IdCodif): Informations d'identification de l'accumulateur.
        merge_mode (str): Mode de fusion.

    Returns:
        IdCodif: Informations d'identification converties au format normalisé.
    """

    lst_enquete, lst_version, lst_subset = idcodif.get_data_info()
    enquete = "-".join(no_doubl_and_sort(lst_enquete))
    version = "-".join(no_doubl_and_sort(lst_version)) + merge_mode
    subset = "PROF" + "".join(no_doubl_no_prof_and_sort(lst_subset))
    idcodif.update_data_info(enquete, version, subset)
    return idcodif


def add_merge_to_dct_and_save_yaml(
    df: pd.DataFrame,
    idcodif: IdData,
    doc: dict,
    dct_df: Dict[str, pd.DataFrame],
    dct_paths: Dict[str, str],
) -> None:
    """
    Ajoute la fusion de données (df) avec les informations d'identification (idcodif)
    au dictionnaire dct_df et sauvegarde le document (doc) au chemin spécifié.

    Args:
        df (pd.DataFrame): Données fusionnées.
        idcodif (IdData): Informations d'identification de la fusion.
        doc (dict): Document YAML à sauvegarder.
        dct_df (Dict[str, pd.DataFrame]): Dictionnaire des DataFrames.
        dct_paths (Dict[str, str]): Dictionnaire des chemins.

    Returns:
        None
    """
    dct_df[idcodif.data] = df
    new_config_path = get_model_specific_config_path(dct_paths, idcodif)
    if not os.path.exists(new_config_path):
        save_yaml(new_config_path, doc)


def get_prio_features_by_types(lst_prio):
    cat_features = []
    txt_features = []
    for var in lst_prio:
        if var.startswith("lib"):
            txt_features.append(var)
        else:
            cat_features.append(var)
    return cat_features, txt_features


def merge_datasets(
    root: str,
    dct_paths: Dict[str, str],
    dct_df: Dict[str, pd.DataFrame],
    dct_idcodif: Dict[str, IdData],
    lst_data_id: List[str],
    mode: Optional[str] = "intersection",
) -> None:
    """
    Fusionne plusieurs ensembles de données en fonction du mode spécifié.

    Args:
        root (str): Répertoire racine.
        dct_paths (Dict[str, str]): Dictionnaire des chemins.
        dct_df (Dict[str, pd.DataFrame]): Dictionnaire des DataFrames.
        dct_idcodif (Dict[str, IdData]): Dictionnaire des informations d'identification.
        lst_data_id (List[str]): Liste des identifiants de données à fusionner.
        mode (Optional[str]): Mode de fusion ("intersection" par défaut).

    Returns:
        None
    """
    if len(lst_data_id) > 1:
        accu = {"df": None, "id": None, "doc": None}

        for iddata in lst_data_id:
            dct = get_dct_info_data(iddata, dct_paths, dct_df, dct_idcodif)

            merge_update_accu(dct, accu, mode=mode)

        df, id, doc = accu.values()
        if mode != "union":
            if mode == "intersection":
                # add prio columns
                cols, inter_prio = get_cols_for_intersection_with_prio(root, id, accu)
                cat_features, txt_features = get_prio_features_by_types(inter_prio)
                doc["preprocessing"]["cat_features"].extend(cat_features)
                doc["preprocessing"]["txt_features"].extend(txt_features)
            else:
                cols, inter_prio = get_cols_for_intersection_with_prio(
                    root, id, accu, withprio=False
                )
            if cols == []:
                pass
            # attention aux doublons
            doc["preprocessing"]["cat_features"] = list(
                set(doc["preprocessing"]["cat_features"])
            )
            doc["preprocessing"]["txt_features"] = list(
                set(doc["preprocessing"]["txt_features"])
            )

            doc["preprocessing"]["var_testability"] = None
            doc["preprocessing"]["weights_train"] = None
            doc["preprocessing"]["weights_test"] = None
            doc["evaluate"]["weights_score"] = None

            # doc["evaluate"]["weights_score"] = "merge_weights"
            # if "merge_weights" not in cols:
            #     cols.append("merge_weights")

            df = df[cols]

        id = codif_id_accu_mode_to_normal(id, mode)
        add_merge_to_dct_and_save_yaml(df, id, doc, dct_df, dct_paths)


def merges_datasets(
    root: str,
    dct_paths: Dict[str, str],
    dct_df: Dict[str, pd.DataFrame],
    dct_idcodif: Dict[str, IdData],
    lst_groups: List[List[str]],
    mode: Optional[str] = "union",
) -> None:
    """
    Fusionne plusieurs ensembles de données en groupes en fonction du mode spécifié.

    Args:
        root (str): Répertoire racine.
        dct_paths (Dict[str, str]): Dictionnaire des chemins.
        dct_df (Dict[str, pd.DataFrame]): Dictionnaire des DataFrames.
        dct_idcodif (Dict[str, IdData]): Dictionnaire des informations d'identification.
        lst_groups (List[List[str]]): Liste des groupes d'identifiants de données à fusionner.
        mode (Optional[str]): Mode de fusion ("union" par défaut).

    Returns:
        None
    """
    if len(lst_groups) > 0:
        for lst_data_id in lst_groups:
            merge_datasets(
                root,
                dct_paths,
                dct_df,
                dct_idcodif,
                lst_data_id,
                mode=mode,
            )
