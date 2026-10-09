import unittest

from app import create_app


class AuthLoginTest(unittest.TestCase):
    def test_default_investigator_login_works(self):
        app = create_app()
        app.config["TESTING"] = True

        with app.test_client() as client:
            response = client.post(
                "/api/auth/login",
                json={"username": "investigator", "password": "demo123"},
            )

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertEqual(payload["user"]["username"], "investigator")
        self.assertEqual(payload["user"]["role"], "investigator")
