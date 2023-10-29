# -*- coding: utf-8 -*-
"""
This is a Python script defining a Classifier abstract class for machine 
learning models evaluation.

The Classifier class is a child of the mlflow.pyfunc.PythonModel class
"""

__author__ = ["Samuel Bazaz"]
__credits__ = ["Samuel Bazaz"]
__license__ = "MIT"
__version__ = "0.0.0"
__maintainer__ = ["Samuel Bazaz"]

##############
#  Packages  #
##############

# import mlflow
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pathlib import Path
from abc import abstractmethod, ABC
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from sklearn.metrics import confusion_matrix


##################
#      Imports     #
##################

root_path = Path(os.path.abspath(__file__)).parents[2]
sys.path.insert(0, str(root_path))
from src.idcodif import IdCodif


####################
#  Abstract class  #
####################


# class Classifier(mlflow.pyfunc.PythonModel):
class Classifier(ABC):
    """
    ML flow connection and evaluation of the model prediction
    """

    def __init__(self, lst_metrics) -> None:
        """
        Constructor for the Evaluator class.
        """
        self.lst_metrics = lst_metrics

    @abstractmethod
    def load_context(self, context):
        """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
        Args:
            context: MLflow context where the model artifact is stored.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, context, model_input):
        raise NotImplementedError()

    def _select_metrics(
        self, lst_selection: List[str]
    ) -> Dict[str, Callable[[Any, Any], float]]:
        """
        Cette fonction sélectionne les métriques à utiliser pour l'évaluation d'un modèle.

        Args:
        lst_selection (List[str]): Une liste de noms de métriques à sélectionner.

        Returns:
        Dict[str, Callable[[Any, Any], float]]: Un dictionnaire contenant les métriques sélectionnées, avec leur nom en tant que clé et la fonction de métrique en tant que valeur.
        """
        lst_metrics = self.lst_metrics

        def name(f):
            return f.__name__.replace("_score", "")

        if lst_selection == []:
            return {name(f): f for f in lst_metrics}
        return {name(f): f for f in lst_metrics if name(f) in lst_selection}

    @staticmethod
    def _add_decision_and_score(
        df: pd.DataFrame, f: Callable, kwargs: Dict
    ) -> pd.DataFrame:
        """
        Ajoute une colonne de prédictions et une colonne de scores à un DataFrame.

        Args:
            df (pd.DataFrame): DataFrame d'entrée.
            f (Callable): Fonction pour générer les prédictions et les scores.
            kwargs (Dict): Arguments supplémentaires à passer à la fonction.

        Returns:
            pd.DataFrame: DataFrame avec les colonnes de prédictions et de scores ajoutées.
        """

        name = f.__name__.replace("_decision_and_score_", "")
        aux_replace_str = lambda row: f(row, **kwargs)
        df[f"pred_{name}"], df[f"score_{name}"] = zip(
            *(df[["model_codes", "model_values"]].apply(aux_replace_str, axis=1))
        )

    @staticmethod
    def _my_metric(
        x: Any,
        y: Any,
        fmetric: Callable,
        metricname: str,
        sample_weights: Optional[Any] = None,
    ) -> Any:
        """
        Calcule la métrique personnalisée en utilisant la fonction de métrique spécifiée.

        Args:
            x (Any): Première valeur à comparer.
            y (Any): Deuxième valeur à comparer.
            fmetric (Callable): Fonction de métrique à utiliser.
            metricname (str): Nom de la métrique.
            sample_weights (Optional[Any], optional): Poids des échantillons. Par défaut None.

        Returns:
            Any: Métrique calculée.
        """

        if metricname in ["recall", "precision", "jaccard", "f1"]:
            return fmetric(x, y, average="micro", sample_weight=sample_weights)
        return fmetric(x, y, sample_weight=sample_weights)

    def _evaluation_on_sub_code(
        self,
        y_true: Any,
        y_pred: Any,
        lst_pos: List[int],
        fmetric: Callable,
        metricname: str,
        sample_weights: Optional[Any],
    ) -> Any:
        """
        Évalue la métrique sur les sous-codes spécifiques des prédictions et des étiquettes réelles.

        Args:
            y_true (Any): Étiquettes réelles.
            y_pred (Any): Prédictions.
            lst_pos (List[int]): Liste des positions des sous-codes.
            fmetric (Callable): Fonction de métrique d'évaluation.
            metricname (str): Nom de la métrique.
            sample_weights (Optional[Any]): Poids des échantillons (optionnel).

        Returns:
            Any: Résultat de la métrique d'évaluation.
        """
        suby_true = self._sub_code(y_true, lst_pos)
        suby_pred = self._sub_code(y_pred, lst_pos)
        return self._my_metric(
            suby_true, suby_pred, fmetric, metricname, sample_weights
        )

    @staticmethod
    def _add_metric_to_accu(
        accu: Dict[str, Any], name: str, value: Any
    ) -> Dict[str, Any]:
        """
        Ajoute une métrique et sa valeur à un dictionnaire d'accumulation.

        Args:
            accu (Dict[str, Any]): Dictionnaire d'accumulation.
            name (str): Nom de la métrique.
            value (Any): Valeur de la métrique.

        Returns:
            Dict[str, Any]: Dictionnaire d'accumulation mis à jour.
        """
        accu["metric"].append(name)
        accu["value"].append(value)
        return accu.copy()

    def _get_eval_and_add(
        self,
        y_true: Any,
        y_pred: Any,
        fmetric: Callable,
        metricname: str,
        sample_weights: Any,
    ) -> Callable:
        """
        Renvoie une fonction qui évalue une métrique sur des prédictions et des valeurs cibles
        et ajoute le résultat à un dictionnaire d'accumulation.

        Args:
            y_true (Any): Valeurs cibles.
            y_pred (Any): Prédictions.
            fmetric (Callable): Fonction de métrique à utiliser.
            metricname (str): Nom de la métrique.
            sample_weights (Any): Poids d'échantillon (optionnel).

        Returns:
            Callable: Fonction d'évaluation et d'ajout à l'accumulation.
        """

        def eval_and_add(
            accu,
            lst_pos,
            name_to_add,
        ):
            value = self._evaluation_on_sub_code(
                y_true=y_true,
                y_pred=y_pred,
                lst_pos=lst_pos,
                fmetric=fmetric,
                metricname=metricname,
                sample_weights=sample_weights,
            )
            accu = self._add_metric_to_accu(accu, name_to_add, value)
            return accu

        return eval_and_add

    @staticmethod
    def _sub_code(y: pd.Series, lst: List[int]) -> pd.Series:
        """
        Applique une transformation aux éléments d'une série y en utilisant une liste d'indices lst pour sélectionner certaines parties des éléments.

        Args:
            y (Series): Série contenant les éléments à transformer.
            lst (List[int]): Liste d'indices pour la sélection des parties des éléments.

        Returns:
            Series: Série contenant les éléments transformés.
        """
        return y.apply(lambda x: "".join(np.array(list(x))[lst]) if x != "NC" else x)

    @staticmethod
    def _to_int(lst: List[str]) -> List[int]:
        """
        Convertit une liste de chaînes de caractères en une liste d'entiers.

        Args:
            lst (List[str]): Liste de chaînes de caractères à convertir.

        Returns:
            List[int]: Liste d'entiers convertis.
        """
        return list(map(int, lst))

    def _get_hierarchic_positions(self, group: str, lst_pos: List[int]) -> List[int]:
        """
        Retourne les positions hiérarchiques communes entre un groupe donné et une liste de positions.

        Args:
            group (str): Groupe au format "group_x_y_z" où x, y et z sont des nombres.
            lst_pos (List[int]): Liste de positions.

        Returns:
            List[int]: Liste des positions hiérarchiques communes.
        """
        lst_pos_h = group.split("_")[1:]
        lst_pos_h = self._to_int(lst_pos_h)
        return list(set(lst_pos).intersection(set(lst_pos_h)))

    def _add_hierarchic_metrics(
        self,
        accu: Dict[str, Any],
        lst_pos: Any,
        hierarchical_config: Dict[str, float],
        metricname: str,
        eval_and_add: Callable,
    ) -> Dict[str, Any]:
        """
        Ajoute des métriques hiérarchiques à un dictionnaire d'accumulation
        en utilisant une configuration pour la pondération.

        Args:
            accu (Dict[str, Any]): Dictionnaire d'accumulation.
            lst_pos (Any): Liste des positions.
            hierarchical_config (Dict[str, float]): Pondération pour la métrique hiérarchique.
            metricname (str): Nom de la métrique.
            eval_and_add (Callable): Fonction d'évaluation et d'ajout à l'accumulation.

        Returns:
            Dict[str, Any]: Dictionnaire d'accumulation mis à jour.
        """
        to_str = lambda lst: list(map(str, lst))

        hierarchic_name = f"{metricname}"
        hierarchic_value = 0
        sum_pond = 0

        for group, pond in hierarchical_config.items():
            lst_pos_h = self._get_hierarchic_positions(group, lst_pos)

            if lst_pos_h != []:
                groupname = ".".join(to_str(lst_pos_h))
                sub_name = f"{metricname}_{groupname}"

                accu = eval_and_add(accu=accu, lst_pos=lst_pos_h, name_to_add=sub_name)

                hierarchic_name += f"_{groupname}p{pond}"
                hierarchic_value += accu["value"][-1]
                sum_pond += pond

        # moyenne pondérée
        hierarchic_value /= sum_pond
        accu = self._add_metric_to_accu(accu, hierarchic_name, hierarchic_value)
        return accu

    def _get_df_metrics_one_decision(
        self,
        y_pred: Any,
        y_true: Any,
        dct_metrics: Dict[str, Callable],
        sample_weights: Any,
        lst_pos: Any,
        hierarchical_config: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Renvoie un DataFrame contenant les métriques pour une décision donnée.

        Args:
            y_pred (Any): Prédictions.
            y_true (Any): Valeurs cibles.
            dct_metrics (Dict[str, Callable]): Dictionnaire des fonctions de métriques.
            sample_weights (Any): Poids d'échantillon (optionnel).
            lst_pos (Any): Liste des positions.
            hierarchical_config (Dict[str, float]): Pondération pour la métrique hiérarchique.

        Returns:
            pd.DataFrame: DataFrame contenant les métriques.
        """
        res = {"metric": [], "value": []}
        for metricname, fmetric in dct_metrics.items():
            eval_and_add = self._get_eval_and_add(
                y_true=y_true,
                y_pred=y_pred,
                fmetric=fmetric,
                metricname=metricname,
                sample_weights=sample_weights,
            )
            res = eval_and_add(res, lst_pos, metricname)
            res = self._add_hierarchic_metrics(
                accu=res,
                lst_pos=lst_pos,
                hierarchical_config=hierarchical_config,
                metricname=metricname,
                eval_and_add=eval_and_add,
            )
        return pd.DataFrame(res)

    def _get_f_scoring(
        self,
        dct_metrics: Dict[str, Callable],
        sample_weights: Any,
        lst_pos: List[int],
        hierarchical_config: Dict[str, Any],
    ) -> Callable[[Any, Any], pd.DataFrame]:
        """
        Retourne une fonction d'évaluation des performances
        qui calcule plusieurs métriques à partir des prédictions.

        Args:
            dct_metrics (Dict[str, Callable]): Dictionnaire des métriques avec leurs fonctions correspondantes.
            sample_weights (Any): Poids d'échantillon.
            lst_pos (List[int]): Positions spécifiées.
            hierarchical_config (Dict[str, Any]): Pondération pour les métrics hiérarchiques.

        Returns:
            Callable[[Any, Any], pd.DataFrame]: Fonction d'évaluation des performances.
        """

        def f_scoring(y_true, y_pred):
            return self._get_df_metrics_one_decision(
                y_pred,
                y_true,
                dct_metrics,
                sample_weights,
                lst_pos,
                hierarchical_config,
            )

        return f_scoring

    def _get_df_metrics_all_decisions(
        self,
        y_true: Any,
        pred_cols: List[str],
        df_test: pd.DataFrame,
        dct_metrics: Dict[str, Callable],
        sample_weights: Any,
        lst_pos: List[int],
        hierarchical_config: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Calcule les métriques pour toutes les décisions prédites dans le DataFrame.

        Args:
            y_true (Any): Valeurs réelles.
            pred_cols (List[str]): Colonnes des prédictions.
            df_test (pd.DataFrame): DataFrame de test.
            dct_metrics (Dict[str, Callable]): Dictionnaire des métriques avec leurs fonctions correspondantes.
            sample_weights (Any): Poids d'échantillon.
            lst_pos (List[int]): Positions spécifiées.
            hierarchical_config (Dict[str, Any]): Configuration hiérarchique.

        Returns:
            pd.DataFrame: DataFrame contenant les métriques pour toutes les décisions.
        """

        f_scoring = self._get_f_scoring(
            dct_metrics, sample_weights, lst_pos, hierarchical_config
        )
        frames = []
        for ypred in pred_cols:
            y_pred = df_test[ypred].copy()
            df_metrics = f_scoring(y_true, y_pred)
            predname = ypred.replace("pred_", "")
            df_metrics["decision"] = predname
            frames.append(df_metrics.copy())

        return pd.concat(frames, keys=["decision", "metric", "value"])

    @staticmethod
    def _get_pred_cols(df_test: pd.DataFrame) -> pd.DataFrame:
        """
        Retourne les colonnes du DataFrame df_test qui commencent par "pred_".

        Args:
            df_test (DataFrame): DataFrame contenant les prédictions.

        Returns:
            DataFrame: DataFrame contenant les colonnes commençant par "pred_".
        """
        return df_test.filter(like="pred_")

    @staticmethod
    def _get_sample_weights(
        df_test: pd.DataFrame,
        weights_col: str,
    ) -> Optional[pd.Series]:
        """
        Calcule les poids d'échantillonnage en fonction de la colonne de poids spécifiée dans le DataFrame df_test.

        Args:
            df_test (DataFrame): DataFrame contenant les données de test.
            weights_col (str): Nom de la colonne de poids dans le DataFrame.

        Returns:
            Optional[Series]: Series contenant les poids d'échantillonnage ou None si les poids ne sont pas utilisés.
        """
        sample_weights = None
        if weights_col:
            sample_weights = (
                df_test[weights_col]
                .fillna(1)
                .astype(int)
            )
        return sample_weights

    @staticmethod
    def _ids_to_id_dct(
        id_main: Optional[IdCodif], id_test: Optional[IdCodif]
    ) -> Dict[str, Any]:
        """
        Crée un dictionnaire à partir d'objets IdData.

        Args:
            id_main (Optional[IdCodif]): Objet IdCodif principal.
            id_test (Optional[IdCodif]): Objet IdCodif de test (non nul pour un Cross test).

        Returns:
            Dict[str, Any]: Dictionnaire contenant les informations des objets IdCodif.
        """
        if id_main is None:
            return {}
        dctmain = id_main.to_dct()
        if id_test is None:
            return dctmain.copy()
        else:
            dcttest = id_test.to_dct()
            dctmain = {f"{k}_main": v for k, v in dctmain.items()}
            dctmain["seed"] = dctmain["seed_main"]

            dcttest = {f"{k}_test": v for k, v in dcttest.items()}
            dctmain.update(dcttest)
            del dctmain["seed_main"]
            del dctmain["seed_test"]

            return dctmain.copy()

    @staticmethod
    def _add_constants_to_df(dct: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des constantes en tant que colonnes supplémentaires à un DataFrame.

        Args:
            dct (Dict[str, Any]): Dictionnaire contenant les constantes à ajouter.
            df (DataFrame): DataFrame auquel ajouter les constantes.

        Returns:
            DataFrame: DataFrame avec les constantes ajoutées en tant que colonnes supplémentaires.
        """
        df2 = df.copy()
        for k, v in dct.items():
            df2[k] = v
        return df2

    @staticmethod
    def _mlflow_df_metrics_log(df_metrics: pd.DataFrame) -> None:
        """
        Enregistre les métriques d'un DataFrame dans MLflow.

        Args:
            df_metrics (DataFrame): DataFrame contenant les métriques à enregistrer.

        Returns:
            None
        """
        df2 = df_metrics.copy()
        df2["temp"] = (
            df2.copy().drop(["value"], axis=1).astype(str).agg("|".join, axis=1)
        )
        df2 = df2.reset_index()
        for i in df2.index:
            # mlflow.log_metric(df2.loc[i, "temp"], df2.loc[i, "value"])
            print(df2.loc[i, "temp"], df2.loc[i, "value"])

    def evaluate(
        self,
        df_test: pd.DataFrame,
        ytrue: str,
        kwargs_eval: dict,
        track: bool = True,
        id_main: Optional[IdCodif] = None,
        id_test: Optional[IdCodif] = None,
    ) -> Dict[str, Union[int, float]]:
        """
        Évalue les prédictions d'un modèle sur un ensemble de données de test.

        Args:
            df_test (pd.DataFrame): DataFrame contenant les données de test.
            ytrue (str): Nom de la colonne contenant les vraies étiquettes.
            kwargs_eval (dict): Dictionnaire des arguments d'évaluation.
            track (bool, optional): Indique si les métriques doivent être suivies avec MLflow. Par défaut True.
            id_main (Optional[IdCodif]): Objet d'identification principale. Par défaut None.
            id_test (Optional[IdCodif]): Objet d'identification de test. Par défaut None.

        Returns:
            Dict[str, Union[int, float]]: Dictionnaire des métriques évaluées.
        """
        hierarchical_config = kwargs_eval["hierarchical"]
        lst_selection = kwargs_eval["metrics"]
        lst_pos = kwargs_eval["positions"]
        weights_col = kwargs_eval["weights_score"]

        y_true = df_test[ytrue].copy()

        sample_weights = self._get_sample_weights(
           df_test, weights_col
        )
        dct_metrics = self._select_metrics(lst_selection)
        pred_cols = self._get_pred_cols(df_test)

        df_metrics_all_decisions = self._get_df_metrics_all_decisions(
            y_true,
            pred_cols,
            df_test,
            dct_metrics,
            sample_weights,
            lst_pos,
            hierarchical_config,
        )

        id_dct = self._ids_to_id_dct(id_main, id_test)
        df_metrics = self._add_constants_to_df(id_dct, df_metrics_all_decisions)

        if track:
            self._mlflow_df_metrics_log(df_metrics)
        return df_metrics

    @staticmethod
    def track_fig(fig: go.Figure, fig_name: str = "fig", session_id: int = 0) -> None:
        """
        Enregistre la figure dans MLflow.

        Args:
            fig (go.Figure): la figure à enregistrer.
            fig_name (str, optionnel): le nom de la figure. Par défaut "fig".
            session_id (int, optionnel): l'identifiant de la session. Par défaut 0.

        Returns:
            None
        """
        mlflow.log_figure(fig, f"{fig_name}_{session_id}.html")

    @staticmethod
    def confusion_matrix(
        df_test: pd.DataFrame,
        ytrue: str,
        ypred: str,
        track: bool = True,
        session_id: int = 0,
    ) -> go.Figure:
        """
        Calcule la matrice de confusion pour les prédictions.

        Args:
            df_test (pd.DataFrame): Le dataframe de test contenant les valeurs réelles et prédites.
            ytrue (str): Le nom de la colonne contenant les vraies valeurs.
            ypred (str): Le nom de la colonne contenant les prédictions.
            track (bool): Si vrai, enregistre la figure dans MLFlow. (default True)
            session_id (int): Identifiant de la session. (default 0)

        Returns:
            go.Figure: La figure de la matrice de confusion.
        """
        # labels = sorted(pd.unique(df_test[ytrue + ypred].copy().astype(str)))
        labels = sorted(list(np.unique(np.array(df_test[ytrue + ypred]))))
        fig = go.Figure(
            data=go.Heatmap(
                z=confusion_matrix(
                    df_test[ytrue],
                    df_test[ypred],
                    labels=labels,
                    normalize="true",
                ),
                x=labels,
                y=labels,
                colorscale="Viridis",
            )
        )
        fig.update_layout(title=f"<b>Matrice de confusion</b> session {session_id}")

        if track:
            mlflow.log_figure(fig, f"confusion_matrix_{session_id}.html")
        return fig
