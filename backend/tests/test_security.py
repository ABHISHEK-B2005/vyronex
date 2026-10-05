import unittest

from jose import JWTError

from core.security import create_access_token, decode_access_token, hash_password, verify_password


class PasswordSecurityTests(unittest.TestCase):
    def test_password_hash_is_salted_and_verifiable(self):
        first = hash_password("correct horse battery staple")
        second = hash_password("correct horse battery staple")

        self.assertNotEqual(first, second)
        self.assertTrue(verify_password("correct horse battery staple", first))
        self.assertFalse(verify_password("wrong password", first))

    def test_malformed_hash_fails_closed(self):
        self.assertFalse(verify_password("anything", "not-a-valid-hash"))


class TokenSecurityTests(unittest.TestCase):
    def test_token_contains_subject_and_role(self):
        secret = "test-secret-that-is-long-enough-for-hmac"
        token = create_access_token("user-1", "athlete", secret)

        claims = decode_access_token(token, secret)
        self.assertEqual(claims["sub"], "user-1")
        self.assertEqual(claims["role"], "athlete")

    def test_wrong_secret_rejects_token(self):
        token = create_access_token("user-1", "gym", "test-secret-that-is-long-enough-for-hmac")

        with self.assertRaises(JWTError):
            decode_access_token(token, "another-secret-that-is-long-enough")

    def test_short_secret_is_rejected(self):
        with self.assertRaises(ValueError):
            create_access_token("user-1", "gym", "short")


if __name__ == "__main__":
    unittest.main()