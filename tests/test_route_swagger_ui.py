import unittest
from http import HTTPStatus

from fastapi.testclient import TestClient

from privatellm.main import app


class FastApiRouteSwaggerTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root(self):
        response = self.client.get("/")
        assert response.status_code == HTTPStatus.NOT_FOUND

    def test_swagger_ui(self):
        response = self.client.get("/docs")
        assert response.status_code == HTTPStatus.OK

    def test_openapi_json(self):
        response = self.client.get("/openapi.json")
        assert response.status_code == HTTPStatus.OK
        assert response.json()["info"]["title"] == "TextTitan"
