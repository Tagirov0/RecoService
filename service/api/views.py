import traceback
import pickle
from typing import List

from fastapi import APIRouter, Depends, FastAPI, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from service.api.exceptions import (
    ModelNotFoundError,
    NotAuthorizedError,
    UserNotFoundError
)
from service.log import app_logger
from service.models import get_models
from service.settings import get_config


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


class ExplainResponse(BaseModel):
    p: int
    explanation: str


MODELS = get_models()
router = APIRouter()
config = get_config()
token_bearer = HTTPBearer(auto_error=False)


async def get_api_key(
    token: HTTPAuthorizationCredentials = Security(token_bearer),
) -> str:
    if not token:
        raise NotAuthorizedError(
            error_message="Missing bearer token",
        )
    return token.credentials


def check_api_key(expected: str, actual: str) -> None:
    if expected != actual:
        raise NotAuthorizedError(
            error_message="Invalid token",
        )


@router.get(
    path="/explain/{model_name}/{user_id}/{item_id}",
    tags=["Explanations"],
    response_model=ExplainResponse,
    responses={404: {"description": "Model, user or item not found"},
               401: {"description": "Not authorized"}},
)
async def explain(request: Request, model_name: str, user_id: int,
                  item_id: int) -> ExplainResponse:
    if model_name != 'als_model':
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")
    else:
        model = MODELS[model_name]
    explain_data = pickle.load(open(config.explain_data, 'rb'))
    popular, item_titles = explain_data.values()

    if user_id > 10 ** 9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")
    elif item_id not in set(item_titles.keys()):
        raise ItemNotFoundError(error_message=f"Item {item_id} not found")
    elif user_id in model.users and model.item_id_idx.get(item_id):
        score, contributor = model.get_explain_reco(user_id, item_id)
        p = round(score * 100)
        explanation = f'Рекомендуем тем, кому нравится ' \
                      f'«{item_titles[contributor]}»'
    else:
        p = round((1 - (popular.index(item_id) + 1) / len(popular)) * 99)
        explanation = f'«{item_titles[item_id]}» входит в топ {100 - p}% ' \
                      f'самых просматриваемых фильмов'

    return ExplainResponse(p=p, explanation=explanation)


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    ''' Health check '''
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={404: {"description": "User or model not found"},
               401: {"description": "Not authorized"}},
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int
) -> RecoResponse:
    ''' Get recommendations for a user '''
    app_logger.info(f"API_KEY: {request.app.state.api_key}")

    if user_id > 10 ** 9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    try:
        model = MODELS[model_name]
        reco_list = model.get_reco(user_id, request.app.state.k_recs)
    except KeyError:
        raise ModelNotFoundError(
            error_message=f"Model {model_name} not found")
    return RecoResponse(user_id=user_id, items=reco_list)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
