# -*- coding: utf-8 -*-
"""
class used to identify :
the model
the train set
the test set
the configuration
the seed
"""

__author__ = ["Samuel Bazaz"]
__credits__ = ["Samuel Bazaz"]
__license__ = "MIT"
__version__ = "0.0.0"
__maintainer__ = ["Samuel Bazaz"]

import os
import pandas as pd
from typing import Optional, Tuple, Dict, Any


class IdModel:
    """La classe IdModel représente un modèle d'identification avec deux attributs : var et algo"""

    def __init__(self, var: str = "pcs", algo: str = "fasttext") -> None:
        self.var = var
        self.algo = algo

    def copy(self) -> "IdModel":
        return IdModel(var=self.var, algo=self.algo)

    def to_str(self) -> str:
        var_maj = self.var.upper()
        algo_min = self.algo.lower()
        return f"{var_maj}{algo_min}"

    def __repr__(self) -> str:
        return self.to_str()


class IdData:
    """La classe IdData représente l'identifiant d'un ensemble de données avec trois attributs : enquete, version et subset."""

    def __init__(self, enquete: str, version: str, subset: str) -> None:
        self.enquete = enquete
        self.version = version
        self.subset = subset

    def copy(self) -> "IdData":
        return IdData(enquete=self.enquete, version=self.version, subset=self.subset)

    def to_str(self) -> str:
        enq_maj = self.enquete.upper()
        vers_min = self.version.lower()
        sub_maj = self.subset.upper()
        return f"{enq_maj}{vers_min}{sub_maj}"

    def __repr__(self) -> str:
        return self.to_str()

    def get_prof(self) -> str:
        """
        Retourne le profil en majuscules en supprimant le préfixe "PROF" de l'attribut subset.

        Returns:
            str: Le profil en majuscules.
        """
        return self.subset.upper().replace("PROF", "")


class IdCodif:
    def __init__(
        self,
        seed: int,
        model: Optional[IdModel] = None,
        data: Optional[IdData] = None,
        config: Optional[str] = None,
    ) -> None:
        self.seed = seed
        self.model = model
        self.data = data
        self.config = config

    def copy(self) -> "IdCodif":
        if self.model is None:
            model2 = None
        else:
            model2 = self.model.copy()

        if self.data is None:
            data2 = None
        else:
            data2 = self.data.copy()

        return IdCodif(model=model2, data=data2, config=self.config, seed=self.seed)

    def get_data_id_configured(
        self, obj: Optional[IdData], config: Optional[str]
    ) -> Optional[str]:
        """
        Retourne l'identifiant de données configuré en ajoutant le suffixe "config" suivi du nom de la configuration.

        Args:
            obj (Optional[IdData]): L'objet IdData.
            config (Optional[str]): Le nom de la configuration.

        Returns:
            Optional[str]: L'identifiant de données configuré.
        """
        if obj is None:
            return None
        data_id = obj.to_str()
        return f"{data_id}config{config}"

    def to_str(self, give_model: bool = True, give_seed: bool = True) -> str:
        seed = self.seed
        modelobj = self.model

        if give_model:
            if modelobj is None:
                model_str = ""
            else:
                model_str = modelobj.to_str()

        if self.data is None:
            data_str = ""
        else:
            data_str = self.get_data_id_configured(obj=self.data, config=self.config)

        if give_model:
            pip_str = f"{model_str}_{data_str}"
        else:
            pip_str = data_str

        if give_seed:
            return f"{pip_str}s{seed}"
        else:
            return pip_str

    def __repr__(self) -> str:
        return self.to_str(True, True)

    def get_data_info(self) -> Tuple[str, str, str]:
        """
        Retourne les informations sur les données : l'enquête, la version et le sous-ensemble.

        Returns:
            Tuple[str, str, str]: Les informations sur les données.
        """
        enquete = self.data.enquete
        version = self.data.version
        subset = self.data.subset
        return enquete, version, subset

    def update_data_info(self, enquete: str, version: str, subset: str) -> None:
        """
        Met à jour les informations sur les données : l'enquête, la version et le sous-ensemble.

        Args:
            enquete (str): L'enquête à mettre à jour.
            version (str): La version à mettre à jour.
            subset (str): Le sous-ensemble à mettre à jour.

        Returns:
            None
        """
        self.data.enquete = enquete
        self.data.version = version
        self.data.subset = subset

    def data_info_as_accu(self) -> None:
        """
        Met à jour les informations sur les données en tant qu'accumulateur.

        Les informations actuelles sur l'enquête, la version et le sous-ensemble sont utilisées
        pour mettre à jour les informations de l'accumulateur.

        Returns:
            None
        """
        enquete, version, subset = self.get_data_info()
        self.update_data_info([enquete], [version], [subset])

    def append_to_data_info_as_accu(
        self, enquete: str, version: str, subset: str
    ) -> None:
        """
        Ajoute les informations sur les données à l'accumulateur.

        Les nouvelles informations sur l'enquête, la version et le sous-ensemble sont ajoutées
        aux informations existantes de l'accumulateur.

        Args:
            enquete (str): Nouvelle enquête à ajouter.
            version (str): Nouvelle version à ajouter.
            subset (str): Nouveau sous-ensemble à ajouter.

        Returns:
            None
        """
        lst_enquete, lst_version, lst_subset = self.get_data_info()
        lst_enquete.append(enquete)
        lst_version.append(version)
        lst_subset.append(subset)

    def to_dct(self) -> Dict[str, Any]:
        """
        Convertit l'objet IdCodif en un dictionnaire contenant ses attributs.

        Returns:
            Dict[str, Any]: Dictionnaire des attributs de l'objet IdCodif.
        """
        dct = self.model.__dict__.copy()
        dct.update(self.data.__dict__)
        dct["config"] = self.config
        dct["seed"] = self.seed
        return dct.copy()


def get_idcodif(
    iddata: IdData,
    var: str = "pcs",
    algo: str = "fasttext",
    config: str = "0",
    seed: int = 0,
) -> IdCodif:
    """
    Crée et retourne un objet IdCodif avec les paramètres spécifiés.

    Args:
        iddata (IdData): Objet IdData représentant les données.
        var (str): Variable à encoder (par défaut "pcs").
        algo (str): Algorithme d'encodage (par défaut "fasttext").
        config (str): Configuration spécifique (par défaut "0").
        seed (int): Graine aléatoire (par défaut 0).

    Returns:
        IdCodif: Objet IdCodif créé.
    """
    idmodel = IdModel(var=var, algo=algo)
    return IdCodif(seed=seed, model=idmodel, data=iddata, config=config)


# bequille à modifier, seulement pour run exceptionnel


def get_dct_idcodif(
    dct_df: Dict[str, pd.DataFrame],
    dct_config: Dict[str, str],
    var: str = "pcs",
    algo: str = "fasttext",
    seed: int = 0,
) -> Dict[str, IdCodif]:
    """
    Crée et retourne un dictionnaire de IdCodif en fonction des données et des configurations spécifiées.

    Args:
        dct_df (Dict[str, pd.DataFrame]): Dictionnaire des DataFrames.
        dct_config (Dict[str, str]): Dictionnaire des configurations.
        var (str): Variable à encoder (par défaut "pcs").
        algo (str): Algorithme d'encodage (par défaut "fasttext").
        seed (int): Graine aléatoire (par défaut 0).

    Returns:
        Dict[str, IdCodif]: Dictionnaire des objets IdCodif.
    """
    res = {}
    for iddata, conf in dct_config.items():
        if iddata in dct_df:
            res[iddata] = get_idcodif(iddata, var, algo, conf, seed)

    return res


# test---
# idmodel = IdModel(var = "pcs", algo="fasttext")
# iddata = IdData(enquete="rp", version="anc", subset ="profa" )
# idcodif = IdCodif(seed=0, model = idmodel, data = iddata, config="0")

# dct = {}
# dct[iddata] = 1
# dct[idcodif] = 2
# print(dct)
# idata = IdData(enquete="rp", version="anc", subset ="profa" )
#
# dct = get_dct_idcodif([idata], {idata:"0"})
# print(dct[idata].config)
