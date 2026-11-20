import unittest
import time

from app import create_app


class AuthLoginTest(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self):
        if getattr(self.app, "mongo_client", None) is not None:
            self.app.mongo_client.close()

    def test_default_investigator_login_works(self):
        response = self.client.post(
            "/api/auth/login",
            json={"username": "investigator", "password": "demo123"},
        )

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertTrue(payload["access_token"])
        self.assertEqual(payload["user"]["username"], "investigator")
        self.assertEqual(payload["user"]["role"], "investigator")

    def test_invalid_credentials_and_missing_fields_have_auth_statuses(self):
        invalid = self.client.post(
            "/api/auth/login",
            json={"username": "investigator", "password": "not-the-password"},
        )
        missing = self.client.post("/api/auth/login", json={"username": "investigator"})

        self.assertEqual(invalid.status_code, 401)
        self.assertEqual(missing.status_code, 400)

    def test_database_unavailable_is_not_reported_as_bad_credentials(self):
        self.app.db = None
        state = self.app.extensions["mongo_connection_state"]
        state.update(connected=False, last_check=time.monotonic(), error_type="ServerSelectionTimeoutError")

        response = self.client.post(
            "/api/auth/login",
            json={"username": "investigator", "password": "not-checked"},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["code"], "database_unavailable")
