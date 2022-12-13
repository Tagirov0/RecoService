import pickle
import typing as tp
import zipfile

import dill
import numpy as np
from pydantic import BaseModel

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


class KNNModel(BaseRecModel):
    def __init__(self) -> None:
        self.unzip_model = zipfile.ZipFile(config.zip_models_path, 'r')
        self.knn_model = dill.load(self.unzip_model.open(config.knn_model))
        self.pop_model = dill.load(self.unzip_model.open(config.pop_model))
        self.users_list = pickle.load(self.unzip_model.open(config.users_list))

    def get_reco(self, user_id: int, k_recs: int = 10) -> tp.List[int]:
        """
        сначала проводится проверка холодный ли юзер,
        есть ли он в списке юзеров из трейна
        если да - то модель KNN выдает рекомендации,
        если нет - то выдаем ему популярное
        """
        if user_id in self.users_list:
            recs = self.knn_model.similar_items(user_id)
            if recs:
                recs = [x[0] for x in recs if not np.isnan(x[0])]

                if len(recs) < k_recs:
                    pop = self.pop_model.recommend(k_recs)
                    recs.extend(pop[:k_recs])
                    recs = list(dict.fromkeys(recs))
                    recs = recs[:k_recs]

                return recs
        return list(self.pop_model.recommend(k_recs))


class LightFMModel(BaseRecModel):
    def __init__(self) -> None:
        self.unzip_model = zipfile.ZipFile(config.zip_models_path, 'r')
        self.emb_maps = pickle.load(self.unzip_model.open(config.emb_maps))
        self.pop_model = dill.load(self.unzip_model.open(config.pop_model))

    def get_reco(self, user_id: int, k_recs: int = 10) -> tp.List[int]:
        """
        check if user is in users list
        if true - return lightfm recs
        if false - return popular recs
        """
        emb_users_list = self.emb_maps['user_id_map'].index

        if user_id in emb_users_list:
            output = self.emb_maps['user_embeddings'][
                    self.emb_maps['user_id_map'][user_id], :]\
                .dot(self.emb_maps['item_embeddings'].T)
            recs = (-output).argsort()[:10]
            return [self.emb_maps['item_id_map'][item_id]for item_id in recs]
        return list(self.pop_model.recommend(k_recs))


ALL_MODELS = {'dummy_model': DummyModel(),
              'knn_model': KNNModel(),
              'lightfm_model': LightFMModel()}


def get_models() -> tp.Dict[str, BaseRecModel]:
    return ALL_MODELS
