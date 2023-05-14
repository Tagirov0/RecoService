from http import HTTPStatus

from requests.structures import CaseInsensitiveDict
from starlette.testclient import TestClient

from service.settings import ServiceConfig

GET_RECO_PATH = "/reco/{model_name}/{user_id}"
GET_EXPLAIN_PATH = "/explain/{model_name}/{user_id}/{item_id}"


def test_health(
    client: TestClient,
) -> None:
    with client:
        response = client.get("/health")
    assert response.status_code == HTTPStatus.OK


def test_get_explain_cold_users(client: TestClient) -> None:
    user_id = 151
    item_id = 65
    path = GET_EXPLAIN_PATH.format(model_name="als_model",
                                   user_id=user_id,
                                   item_id=item_id)
    with client:
        response = client.get(path)
    assert response.status_code == HTTPStatus.OK
    response_json = response.json()
    assert response_json["p"] == 38
    assert response_json["explanation"] == "«Грязная ложь» входит в топ 62% " \
                                           "самых просматриваемых фильмов"


def test_get_explain_warm_users(client: TestClient) -> None:
    user_id = 387
    item_id = 15297
    path = GET_EXPLAIN_PATH.format(model_name="als_model",
                                   user_id=user_id,
                                   item_id=item_id)
    with client:
        response = client.get(path)
    assert response.status_code == HTTPStatus.OK
    response_json = response.json()
    assert response_json["p"] == 33
    assert response_json["explanation"] == "Рекомендуем тем, кому нравится " \
                                           "«Жестокий Стамбул»"


def test_get_explain_for_unknown_user(client: TestClient) -> None:
    user_id = 10**10
    item_id = 1
    path = GET_EXPLAIN_PATH.format(model_name="als_model",
                                   user_id=user_id,
                                   item_id=item_id)
    with client:
        response = client.get(path)
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "user_not_found"


def test_get_explain_for_unknown_item(client: TestClient) -> None:
    user_id = 1
    item_id = 10**10
    path = GET_EXPLAIN_PATH.format(model_name="als_model",
                                   user_id=user_id,
                                   item_id=item_id)
    with client:
        response = client.get(path)
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "item_not_found"


def test_get_explain_for_unknown_model(client: TestClient) -> None:
    user_id = 1
    item_id = 1
    path = GET_EXPLAIN_PATH.format(model_name="some_model",
                                   user_id=user_id,
                                   item_id=item_id)
    with client:
        response = client.get(path)
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "model_not_found"


def test_get_reco_success(client: TestClient,) -> None:
    user_id = 123
    path = GET_RECO_PATH.format(model_name="knn_model", user_id=user_id)
    with client:
        response = client.get(path)
    assert response.status_code == HTTPStatus.OK
    response_json = response.json()
    assert response_json["user_id"] == user_id
    assert len(response_json["items"]) == 10
    assert all(isinstance(item_id, int) for item_id in response_json["items"])


def test_get_reco_for_unknown_user(client: TestClient,) -> None:
    user_id = 10**10
    path = GET_RECO_PATH.format(model_name="knn_model", user_id=user_id)
    with client:
        response = client.get(path)
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "user_not_found"


def test_get_reco_for_unknown_model(client: TestClient,) -> None:
    user_id = 1
    path = GET_RECO_PATH.format(model_name="some_model", user_id=user_id)
    with client:
        response = client.get(path)
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "model_not_found"
