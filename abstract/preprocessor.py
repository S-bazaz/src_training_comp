import unicodedata
import random
import pandas as pd

from abc import ABC, abstractmethod
from random import Random
from typing import Tuple, List, Union, Dict, Optional
from sklearn.model_selection import train_test_split

####################
#  Abstract class  #
####################


class Preprocessor(ABC):
    """
    Preprocessor class.
    """

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """preprocessing final method for features transformation
        Args:
            df (pd.DataFrame): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            pd.DataFrame: _description_
        """
        raise NotImplementedError()

    @abstractmethod
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """preprocessing final method for train set and test set decomposition

        Args:
            df (pd.DataFrame): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: _description_
        """
        raise NotImplementedError()

    @abstractmethod
    def _drop_wrong_labels(self, df: pd.DataFrame, kwargs):
        raise NotImplementedError()

    @staticmethod
    def _standardize_string(
        st: str, stopwords: List[str] = [], stemmer: callable = lambda lst: lst
    ) -> str:
        """
        Standardise une chaîne de caractères en appliquant plusieurs étapes de traitement.

        Args:
            st (str): Chaîne de caractères à standardiser.
            stopwords (List[str]): Liste des mots vides à supprimer.
            stemmer (callable): Fonction de stemming à appliquer.

        Returns:
            str: Chaîne de caractères standardisée.
        """
        # no accents
        st = unicodedata.normalize("NFD", st)
        st = st.encode("ascii", "ignore")
        st = str(st.decode("utf-8")).replace("'", " ")
        # lower for stopword filtration
        st = st.lower()
        # decomposition for word transformations
        lst = st.split()
        # no stopwords and stemmerization
        st = " ".join([stemmer(w) for w in lst if w not in stopwords])
        # upper
        st = st.upper()
        return st

    def _standardize_txt_labels(self, df, txt_ft, stopwords, stemmer):
        """
        Standardise les étiquettes textuelles dans un DataFrame.

        Args:
            df (pd.DataFrame): DataFrame contenant les étiquettes à standardiser.
            txt_ft (str): Nom de la colonne contenant les étiquettes textuelles.
            stopwords (list): Liste des mots vides à supprimer.
            stemmer (nltk.stem.Stemmer): Objet de stemming.

        Returns:
            None
        """
        f_step = self._standardize_string
        f_aux = lambda st: f_step(st, stopwords=stopwords, stemmer=stemmer)
        df[txt_ft] = df[txt_ft].applymap(f_aux)

    def _fill_na(
        self,
        df: pd.DataFrame,
        cat_ft: str,
        txt_ft: str,
        cat_value: Union[int, float] = -1,
        txt_value: str = "VIDE",
        cat_numerical: bool = False,
    ) -> None:
        """
        Remplit les valeurs manquantes dans un DataFrame.

        Args:
            df (pd.DataFrame): DataFrame contenant les données.
            cat_ft (str): Nom de la colonne catégorielle.
            txt_ft (str): Nom de la colonne textuelle.
            cat_value (Union[int, float]): Valeur de remplacement pour les valeurs manquantes dans la colonne catégorielle.
            txt_value (str): Valeur de remplacement pour les valeurs manquantes dans la colonne textuelle.
            cat_numerical (bool): Indique si la colonne catégorielle est de type numérique.

        Returns:
            None
        """

        df[txt_ft] = df[txt_ft].fillna(txt_value).astype(str)
        if cat_numerical:
            df[cat_ft] = df[cat_ft].astype(float).fillna(cat_value)
        else:
            df[cat_ft] = df[cat_ft].astype(str).fillna(txt_value)

    @staticmethod
    def _data_separation_1enquete(
        x: pd.DataFrame,
        y: pd.DataFrame,
        seed: int,
        proportion: float,
        proportion2: Union[float, None] = None,
        accu: Dict[str, pd.DataFrame] = {},
    ) -> Dict[str, pd.DataFrame]:
        """
        Sépare les données en ensembles d'entraînement et de test.

        Args:
            x (pd.DataFrame): Données d'entrée.
            y (pd.DataFrame): Données cibles.
            seed (int): Graine aléatoire pour la reproductibilité.
            proportion (float): Proportion des données à utiliser pour l'ensemble de test.
            proportion2 (Union[float, None]): Proportion des données restantes à utiliser pour un second ensemble de test.
            accu (Dict[str, pd.DataFrame]): Dictionnaire pour stocker les ensembles de données séparés.

        Returns:
            Dict[str, pd.DataFrame]: Dictionnaire contenant les ensembles de données séparés.
        """

        def add_concat_to_accu(x, y, n):
            accu[n] = pd.concat([y, x], axis=1)

        x1, x2, y1, y2 = train_test_split(
            x,
            y,
            test_size=proportion,
            random_state=seed,
            shuffle=True,
        )
        add_concat_to_accu(x1, y1, "1")
        if proportion2 is not None:
            x2, x3, y2, y3 = train_test_split(
                x2,
                y2,
                test_size=proportion2,
                random_state=seed,
                shuffle=True,
            )
            add_concat_to_accu(x3, y3, "3")
        add_concat_to_accu(x2, y2, "2")
        return accu

    def _data_separation(
            self,
            x: pd.DataFrame,
            y: pd.DataFrame,
            seed: int,
            proportion: float,
            proportion2: Union[float, None] = None,
            weights_train: Union[str, None] = None,
            accu: Dict[str, pd.DataFrame] = {},
    ) -> Dict[str, pd.DataFrame]:
        """
        Sépare les données en ensembles d'entraînement et de test.

        Sépare les données `x` et `y` en ensembles d'entraînement et de test en utilisant la méthode
        de séparation spécifiée par les paramètres `proportion` et `proportion2`. Si la variable `weights_train`
        est spécifiée, les données sont stratifiées en fonction de cette variable.

        Args:
            x (pd.DataFrame): Le DataFrame contenant les caractéristiques des données.
            y (pd.DataFrame): Le DataFrame contenant les étiquettes des données.
            seed (int): La graine aléatoire pour la reproductibilité.
            proportion (float): La proportion des données à utiliser pour l'ensemble d'entraînement.
            proportion2 (float, optional): La proportion des données à utiliser pour l'ensemble de test.
                Si non spécifié, les données restantes après la création de l'ensemble d'entraînement sont utilisées.
            weights_train (str, optional): La variable de poids pour la stratification des données.
                Par défaut, None.
            accu (Dict[str, pd.DataFrame], optional): Un dictionnaire pour accumuler les ensembles d'entraînement
                et de test. Par défaut, {}.

        Returns:
            Dict[str, pd.DataFrame]: Un dictionnaire contenant les ensembles d'entraînement et de test.
        """
        if weights_train is None:
            return self._data_separation_1enquete(
                x, y, seed, proportion, proportion2, accu
            )

        # here to handle split by strat
        lst_enquete = x[weights_train].unique()
        # print("list enquête \n", lst_enquete)

        if len(lst_enquete) == 1:
            return self._data_separation_1enquete(
                x, y, seed, proportion, proportion2, accu
            )

        else:
            dct_split_accu = {"1": [], "2": []}
            x.reset_index()
            y.index = x.index
            # print("index x \n", x.index)
            # print("index y \n", y.index)
            for enquete in lst_enquete:
                xq = x.query(f"enquete == '{enquete}'")

                #yq = y.loc[xq.index, :]
                yq = y.loc[xq.index]
                dct_split_enq = self._data_separation_1enquete(
                    xq, yq, seed, proportion, proportion2, accu
                )
                dct_split_accu["1"].append(dct_split_enq["1"])
                dct_split_accu["2"].append(dct_split_enq["2"])

            return {
                "1": pd.concat(dct_split_accu["1"]),
                "2": pd.concat(dct_split_accu["2"]),
            }

    @staticmethod
    def __deterministic_split(x_raw: pd.DataFrame, y_raw: pd.DataFrame, seed: str) -> Dict[str, pd.DataFrame]:
        """
        Divise les données en ensembles d'entraînement et de test de façon déterministes.

        Divise les données `x_raw` et `y_raw` en ensembles d'entraînement et de test en utilisant
        la variable `seed` pour la séparation. Les données d'entraînement sont celles où la valeur
        de `seed` est "train", et les données de test sont celles où la valeur de `seed` est "test".

        Args:
            x_raw (pd.DataFrame): Le DataFrame contenant les caractéristiques des données.
            y_raw (pd.DataFrame): Le DataFrame contenant les étiquettes des données.
            seed (str): La variable de séparation utilisée pour diviser les données.

        Returns:
            Dict[str, pd.DataFrame]: Un dictionnaire contenant les ensembles d'entraînement et de test.
        """
        df = pd.concat([y_raw, x_raw], axis=1)
        df_train: pd.DataFrame = df.query(f"{seed} == 'train'")
        df_test: pd.DataFrame = df.query(f"{seed} == 'test'")
        return {"train": df_train, "test": df_test}

    def __random_split_before_resampling(
            self,
            x_raw: pd.DataFrame,
            y_raw: pd.DataFrame,
            seed: str,
            train_prop: float,
            validation_prop: float,
            var_testability: str,
            weights_train: Optional[pd.Series] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Divise les données en ensembles d'entraînement, de validation et de test en utilisant
        un tirage aléatoire conditionné par la variable `var_testability`.

        Si `var_testability` est None, toutes les données sont utilisées pour l'entraînement sans contrainte.
        Sinon, seules les données où `var_testability` est "train" sont utilisées pour l'entraînement,
        et les données où `var_testability` est "test" sont utilisées pour le test. Ensuite, un tirage
        aléatoire est effectué sur les données de test pour former les ensembles de validation et de test.

        Args:
            x_raw (pd.DataFrame): Le DataFrame contenant les caractéristiques des données.
            y_raw (pd.DataFrame): Le DataFrame contenant les étiquettes des données.
            seed (str): La graine utilisée pour le tirage aléatoire.
            train_prop (float): La proportion des données à utiliser pour l'entraînement.
            validation_prop (float): La proportion des données de test à utiliser pour la validation.
            var_testability (str): La variable indiquant la testabilité des données ("train" ou "test").
            weights_train (Optional[pd.Series], optional): Les poids d'entraînement pour l'échantillonnage stratifié.
                Par défaut, None.

        Returns:
            Dict[str, pd.DataFrame]: Un dictionnaire contenant les ensembles d'entraînement, de validation et de test.
        """
        if var_testability is None:
            # Si toutes les données ont des poids de test
            # on n'a pas de contraintes sur les deux tirages

            return self._data_separation(
                x_raw,
                y_raw,
                seed=seed,
                proportion=1 - train_prop,
                proportion2=validation_prop,
                weights_train=weights_train,
            )

        else:
            # Sinon on ne récupère les données de test et de validation que là ou il y en a.
            df = pd.concat([y_raw, x_raw], axis=1)
            df_train: pd.DataFrame = df.query(f"{var_testability} == 'train'")
            df_test: pd.DataFrame = df.query(f"{var_testability} == 'test'")
            dct = {"1": df_train, "2": df_test}

            # Puis on réalise un tirage si l'on veut des données de validation
            if validation_prop:
                dct2 = self._data_separation(
                    df_test.iloc[:, 1:],
                    df_test.iloc[:, 0],
                    seed,
                    proportion=validation_prop,
                    accu=dct,
                    weights_train=weights_train,
                )
                dct["2"] = dct2["1"].copy()
                dct["3"] = dct2["2"].copy()
            return dct

    @staticmethod
    def _resampling(
            df: pd.DataFrame,
            r_gen: Random,
            resampling_mode: Optional[str] = "enquete",
            size: Optional[int] = None,
            size_multiplier: Optional[float] = None,
            max_size: int = 100000,
            weights_var: Optional[str] = None,
            bootstrap: bool = False,
    ) -> pd.DataFrame:
        """
        Effectue l'échantillonnage des données du DataFrame en utilisant différentes méthodes.

        Rééchantillonne les données du DataFrame en fonction des paramètres spécifiés. L'échantillonnage
        peut être effectué en utilisant différentes méthodes, telles que la stratification par enquête ou
        l'échantillonnage bootstrap(tirage avec remise).

        Args:
            df (pd.DataFrame): Le DataFrame contenant les données à échantillonner.
            r_gen (Random): L'objet Random utilisé pour l'échantillonnage.
            resampling_mode (str, optional): Le mode d'échantillonnage. Par défaut, "enquete".
            size (int, optional): La taille de l'échantillon à extraire. Si spécifié, `size_multiplier` sera ignoré.
            size_multiplier (float, optional): Le multiplicateur de taille de l'échantillon par rapport à la taille initiale.
                Ignoré si `size` est spécifié. Par défaut, None.
            max_size (int, optional): La taille maximale de l'échantillon. Par défaut, 100000.
            weights_var (str, optional): La variable de poids à utiliser pour l'échantillonnage stratifié.
                Par défaut, None.
            bootstrap (bool, optional): Indique si l'échantillonnage doit être effectué en utilisant la méthode bootstrap.
                Par défaut, False.

        Returns:
            pd.DataFrame: Le DataFrame rééchantillonné.
        """
        df.reset_index(inplace=True)
        # resampling size
        n1 = df.shape[0]
        if size:
            n2 = size
        else:
            n2 = int(min(n1 * size_multiplier, max_size))

        # weights
        if weights_var is None or weights_var not in df:
            weights = None
        else:
            weights = df[weights_var].fillna(1)

        def f_resample(indexes0, size):
            if bootstrap:
                return r_gen.choices(indexes0, k=size, weights=weights)
            elif weights is None:
                return r_gen.sample(indexes0, k=size)
            return df.loc[indexes0].sample(weights=weights, n=size).index

        # indexes
        if resampling_mode:
            df_par_enquete = df.copy().groupby(resampling_mode)
            dct_enquetes = {
                enquete: donnees_enquete.index.tolist()
                for enquete, donnees_enquete in df_par_enquete
            }

            if resampling_mode == "enquete_volumes_egaux":
                if bootstrap:
                    n_ego = int(n2 / len(dct_enquetes))
                else:
                    n_ego = min([len(donnees_enquete) for donnees_enquete in dct_enquetes.values()])
                # print(f"n_ego : {n_ego}")

            # Effectuer un échantillonnage stratifié en utilisant les index stockés dans le dictionnaire
            indexes = []
            for donnees_enquete in dct_enquetes.values():
                if resampling_mode == "enquete_volumes_egaux":
                    taille_sample_enquete = n_ego
                else:
                    taille_sample_enquete = int(n2 * (len(donnees_enquete) / n1))
                    # print(f"n_enquete : {taille_sample_enquete}")

                # print("taille_sample_enquete ", taille_sample_enquete)
                indexes.extend(f_resample(indexes0=donnees_enquete, size=taille_sample_enquete))
        else:
            indexes = f_resample(indexes0=list(range(n1)), size=n2)
        res = df.iloc[indexes, :].copy()
        res.reset_index(inplace=True)
        # print(res.shape)
        return res

    def __get_f_aux_resampling(
            self,
            dct: Dict[str, pd.DataFrame],
            seed: int,
            resampling_mode: Optional[str],
            weights_train: Optional[pd.Series],
            weights_test: Optional[pd.Series],
            size: Optional[int],
            size_multiplier: Optional[float],
            max_df_size: int = 1000000,
            bootstrap: bool = False
    ) -> callable:
        """
        Retourne une fonction auxiliaire pour effectuer le sous-échantillonnage des données.

        Si `size` ou `size_multiplier` est spécifié, la fonction retourne une fonction auxiliaire
        pour effectuer le sous-échantillonnage en utilisant la méthode `_resampling`. Sinon, elle
        retourne simplement une fonction identité qui renvoie les données sans modification.

        Args:
            dct (Dict[str, pd.DataFrame]): Un dictionnaire contenant les ensembles de données à sous-échantillonner.
            seed (int): La graine utilisée pour le tirage aléatoire.
            resampling_mode (Optional[str]): Le mode de sous-échantillonnage à appliquer. Par défaut, None.
            weights_train (Optional[pd.Series]): Les poids d'entraînement pour l'échantillonnage stratifié.
                Par défaut, None.
            weights_test (Optional[pd.Series]): Les poids de test pour l'échantillonnage stratifié. Par défaut, None.
            size (Optional[int]): La taille spécifique du sous-échantillon à extraire. Par défaut, None.
            size_multiplier (Optional[float]): Le multiplicateur de taille du sous-échantillon à extraire.
                Par défaut, None.
            max_df_size (int): La taille maximale autorisée pour le sous-échantillon. Par défaut, 1000000.
            bootstrap (bool): Indique si le sous-échantillonnage doit être effectué avec remise. Par défaut, False.

        Returns:
            callable: La fonction auxiliaire pour effectuer le sous-échantillonnage des données.
        """
        # cas avec resampling
        if size or size_multiplier:
            r_gen = random.Random(seed)

            def f_aux_resampling(k):
                # cas train
                if k == "1":
                    weights_var = weights_train
                # cas test
                else:
                    weights_var = weights_test

                return self._resampling(
                    dct[k],
                    r_gen,
                    resampling_mode=resampling_mode,
                    size=size,
                    size_multiplier=size_multiplier,
                    max_size=max_df_size,
                    weights_var=weights_var,
                    bootstrap = bootstrap
                )

            return f_aux_resampling
        # cas sans resampling
        else:
            return lambda k: dct[k]

    def _split_data(
            self,
            x_raw: pd.DataFrame,
            y_raw: pd.DataFrame,
            seed: Union[int, str] = 0,
            resampler: Optional[object] = None,
            var_testability: Optional[str] = None,
            validation_prop: Optional[float] = None,
            train_prop: float = 0.8,
            resampling_mode: str = "enquete",
            weights_train: Optional[pd.Series] = None,
            weights_test: Optional[pd.Series] = None,
            size: Optional[int] = None,
            size_multiplier: float = 1,
            max_df_size: int = 10000,
            bootstrap: bool = False
    ) -> dict:
        """
        Sépare les données d'entraînement, de test et de validation en effectuant éventuellement un sous-échantillonnage.

        Les données sont d'abord divisées en ensembles de train, de test et de validation en fonction des colonnes de données
        spécifiées. Ensuite, si un objet `resampler` est spécifié, les données sont sous-échantillonnées en utilisant cet objet.
        Enfin, un sous-échantillonnage supplémentaire est effectué en fonction des tailles respectives des ensembles de train,
        de test et de validation, ou simplement pour obtenir des tailles suffisantes pour chaque ensemble.

        Args:
            x_raw (pd.DataFrame): Le DataFrame contenant les données d'entrée.
            y_raw (pd.DataFrame): Le DataFrame contenant les étiquettes de sortie.
            seed (Union[int, str], optional): La graine utilisée pour le tirage aléatoire. Par défaut, 0.
            resampler (Optional[object], optional): L'objet utilisé pour effectuer le sous-échantillonnage des données.
                Par défaut, None.
            var_testability (Optional[str], optional): La colonne indiquant la testabilité des données. Par défaut, None.
            validation_prop (Optional[float], optional): La proportion des données à inclure dans l'ensemble de validation.
                Par défaut, None.
            train_prop (float, optional): La proportion des données à inclure dans l'ensemble d'entraînement. Par défaut, 0.8.
            resampling_mode (str, optional): Le mode de sous-échantillonnage à appliquer. Par défaut, "enquete".
            weights_train (Optional[pd.Series], optional): Les poids d'entraînement pour l'échantillonnage stratifié.
                Par défaut, None.
            weights_test (Optional[pd.Series], optional): Les poids de test pour l'échantillonnage stratifié. Par défaut, None.
            size (Optional[int], optional): La taille spécifique du sous-échantillon à extraire. Par défaut, None.
            size_multiplier (float, optional): Le multiplicateur de taille du sous-échantillon à extraire. Par défaut, 1.
            max_df_size (int, optional): La taille maximale autorisée pour le sous-échantillon. Par défaut, 10000.
            bootstrap (bool, optional): Indique si le sous-échantillonnage doit être effectué avec remise. Par défaut, False.

        Returns:
            dict: Un dictionnaire contenant les ensembles de train, de test et de validation.
        """

        # cas déterministe -> choix selon colonne de bool
        # ex: seed = data_category
        if isinstance(seed, str):
            return self.__deterministic_split(x_raw, y_raw, seed)

        # On resample en fonction du code à estimer
        # ex : RandomOverSampler, SMOTE, ADASYN, resampler
        if resampler is not None:
            x_raw, y_raw = self.resampler.fit_resample(x_raw, y_raw)

        # On tire les données de train, test et validation
        dct = self.__random_split_before_resampling(
            x_raw=x_raw,
            y_raw=y_raw,
            seed=seed,
            train_prop=train_prop,
            validation_prop=validation_prop,
            var_testability=var_testability,
            weights_train=weights_train,
        )
        # On resample en fonction des tailles respectives des bases
        # Ou alors simplement pour avoir des tailles de test, train et validation suffisants
        keymap = {"1": "train", "2": "test", "3": "valid"}

        f_aux_resampling = self.__get_f_aux_resampling(
            dct=dct,
            seed=seed,
            resampling_mode=resampling_mode,
            weights_train=weights_train,
            weights_test=weights_test,
            size=size,
            size_multiplier=size_multiplier,
            max_df_size=max_df_size,
            bootstrap=bootstrap
            )
        return {keymap[k]: f_aux_resampling(k) for k in dct}
