import unittest

from fastapi.testclient import TestClient

from privatellm.main import app


class FastApiRouteSwaggerTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root(self):
        response = self.client.get("/")
        assert response.status_code == 404

    def test_swagger_ui(self):
        response = self.client.get("/docs")
        assert response.status_code == 200

    def test_openapi_json(self):
        response = self.client.get("/openapi.json")
        assert response.status_code == 200
        assert response.json()["info"]["title"] == "TextTitan"
