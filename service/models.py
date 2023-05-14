import dill
import pickle
import typing as tp
import numpy as np
from zipfile import ZipFile
from pydantic import BaseModel
from rectools.dataset import Dataset
from rectools.models import ImplicitALSWrapperModel
from service.settings import get_config

config = get_config()


class Error(BaseModel):
    error_key: str
    error_message: str
    error_loc: tp.Optional[tp.Any] = None


class BaseRecModel:
    def get_reco(self, user_id: int, k_recs: int) -> tp.List[int]:
        pass


class DummyModel(BaseRecModel):
    def __init__(self) -> None:
        pass

    def get_reco(self, user_id: int, k_recs: int) -> tp.List[int]:
        return list(range(k_recs))


class PopularModel(BaseRecModel):
    def __init__(self) -> None:
        with ZipFile(config.zip_models_path, 'r') as models:
            self.pop_model = dill.load(models.open(config.pop_model))

    def get_reco(self, user_id: int, k_recs: int = 10) -> tp.List[int]:
        return list(self.pop_model.recommend(k_recs))


class KNNModel(BaseRecModel):
    def __init__(self) -> None:
        with ZipFile(config.zip_models_path, 'r') as models:
            self.knn_model = dill.load(models.open(config.knn_model))
            self.pop_model = dill.load(models.open(config.pop_model))
            self.emb_maps = pickle.load(models.open(config.emb_maps))
            self.users_list = set(self.emb_maps['user_id_map'].keys())

    def get_reco(self, user_id: int, k_recs: int = 10) -> tp.List[int]:
        """
        сначала проводится проверка холодный ли юзер,
        есть ли он в списке юзеров из трейна
        если да - то модель KNN выдает рекомендации,
        если нет - то выдаем ему популярное
        """
        if user_id in self.users_list:
            try:
                recs = self.knn_model.similar_items(user_id)
                if recs:
                    recs = [x[0] for x in recs if not np.isnan(x[0])]

                    if len(recs) < k_recs:
                        pop = self.pop_model.recommend(k_recs)
                        recs.extend(pop[:k_recs])
                        recs = list(dict.fromkeys(recs))
                        recs = recs[:k_recs]

                    return recs
            except Exception:
                return list(self.pop_model.recommend(k_recs))

        return list(self.pop_model.recommend(k_recs))


class ALSModel(BaseRecModel):
    def __init__(self) -> None:
        with ZipFile(config.zip_models_path, 'r') as models:
            self.dataset = pickle.load(models.open(config.dataset))
            self.als_model = pickle.load(models.open(config.als_model))
            self.pop_model = dill.load(models.open(config.pop_model))

        self.user_id_idx = self.dataset.user_id_map.to_internal
        self.item_id_idx = self.dataset.item_id_map.to_internal
        self.item_idx_id = self.dataset.item_id_map.to_external
        self.csr_matrix = self.dataset.get_user_item_matrix()
        self.users = set(self.user_id_idx.keys())

    def get_reco(self, user_id: int, k_recs: int = 10) -> tp.List[int]:
        if user_id in self.users:
            user_id = self.user_id_idx[user_id]
            recs = self.als_model.model.recommend(
                userid=user_id,
                user_items=self.csr_matrix,
                N=k_recs,
                filter_already_liked_items=True
            )
            final_recs = [self.item_idx_id[item_idx] for item_idx, _ in recs]
            return final_recs
        return list(self.pop_model.recommend(k_recs))

    def get_explain_reco(self, user_id: int, item_id: int) -> tp.Tuple[
        float, int]:
        total_score, top_contributions, _ = self.als_model.model.explain(
            userid=self.user_id_idx[user_id],
            user_items=self.csr_matrix,
            itemid=self.item_id_idx[item_id],
            N=1
        )

        top_contributions = self.item_idx_id[top_contributions[0][0]]

        return total_score, top_contributions


class LightFMModel(BaseRecModel):
    def __init__(self) -> None:
        with ZipFile(config.zip_models_path, 'r') as models:
            self.emb_maps = pickle.load(models.open(config.emb_maps))
            self.pop_model = dill.load(models.open(config.pop_model))
            self.knows_items = pickle.load(models.open(config.knows_items))
            self.users = set(self.emb_maps['user_id_map'].keys())

    def get_reco(self, user_id: int, k_recs: int = 10) -> tp.List[int]:
        """
        check if user is in users list
        if true - return lightfm recs
        if false - return popular recs
        """
        if user_id in self.users:
            scores = self.emb_maps['user_embeddings'][
                     self.emb_maps['user_id_map'][user_id], :] \
                .dot(self.emb_maps['item_embeddings'].T)

            filter_items = self.knows_items[user_id]
            additional_N = len(filter_items) if user_id \
                                                in self.knows_items else 0

            total_k = k_recs + additional_N
            unsorted_recs = scores.argpartition(-total_k)[-total_k:]
            unsorted_recs_score = scores[unsorted_recs]

            recs = unsorted_recs[(-unsorted_recs_score).argsort()]
            final_recs = [self.emb_maps['item_id_inv_map'][item]
                          for item in recs if item not in filter_items]
            return final_recs[:k_recs]
        return list(self.pop_model.recommend(k_recs))


ALL_MODELS = {'lightfm_model': LightFMModel(), 'knn_model': KNNModel(),
              'als_model': ALSModel()}


def get_models() -> tp.Dict[str, BaseRecModel]:
    return ALL_MODELS
