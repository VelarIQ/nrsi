"""PQC Key Exchange Test Suite.

Validates the Iron-Clad Shield 5-Layer Stack:
  L1: ML-KEM-1024 (FIPS 203) key encapsulation
  L2: X25519 classical ECDH
  L3: Hybrid KDF (HKDF-SHA256 / SHAKE-256)
  L4: CRYSTALS-Dilithium (FIPS 204) signatures
  L5: AES-256-GCM session key derivation

Covers both liboqs-backed and hash-based fallback paths.

Usage:
    python -m pytest tests/test_pqc_keyexchange.py -v
"""

from __future__ import annotations

import os
import time
import unittest

from nrsip.pqc_keyexchange import (
    DilithiumProvider,
    HybridKDF,
    IronCladHandshake,
    KEMKeyPair,
    MLKEMProvider,
    SessionKeys,
    SigningKeyPair,
    X25519Provider,
    _cryptography_available,
    _oqs_available,
)


# ───────────────────────────────────────────────────────────────────
# 1. DilithiumProvider
# ───────────────────────────────────────────────────────────────────


class TestDilithiumKeypair(unittest.TestCase):
    """Dilithium keypair generation and algorithm labelling."""

    def setUp(self):
        self.provider = DilithiumProvider()

    def test_generate_returns_signing_keypair(self):
        kp = self.provider.generate_keypair()
        self.assertIsInstance(kp, SigningKeyPair)
        self.assertIsInstance(kp.public_key, bytes)
        self.assertIsInstance(kp.secret_key, bytes)
        self.assertGreater(len(kp.public_key), 0)
        self.assertGreater(len(kp.secret_key), 0)

    def test_algorithm_label_reflects_backend(self):
        kp = self.provider.generate_keypair()
        if _oqs_available:
            self.assertEqual(kp.algorithm, "Dilithium3")
        else:
            self.assertEqual(kp.algorithm, "Dilithium3-fallback")

    def test_keypairs_are_unique(self):
        kp1 = self.provider.generate_keypair()
        kp2 = self.provider.generate_keypair()
        self.assertNotEqual(kp1.public_key, kp2.public_key)
        self.assertNotEqual(kp1.secret_key, kp2.secret_key)


class TestDilithiumSignVerify(unittest.TestCase):
    """Dilithium sign / verify / rejection paths."""

    def setUp(self):
        self.provider = DilithiumProvider()
        self.kp = self.provider.generate_keypair()

    def test_sign_returns_bytes(self):
        sig = self.provider.sign(self.kp.secret_key, b"hello NRS")
        self.assertIsInstance(sig, bytes)
        self.assertGreater(len(sig), 0)

    def test_different_messages_yield_different_signatures(self):
        s1 = self.provider.sign(self.kp.secret_key, b"alpha")
        s2 = self.provider.sign(self.kp.secret_key, b"bravo")
        self.assertNotEqual(s1, s2)

    def test_signing_same_message_both_verify(self):
        msg = b"stable"
        s1 = self.provider.sign(self.kp.secret_key, msg)
        s2 = self.provider.sign(self.kp.secret_key, msg)
        self.assertTrue(self.provider.verify(self.kp.public_key, msg, s1))
        self.assertTrue(self.provider.verify(self.kp.public_key, msg, s2))

    @unittest.skipUnless(_oqs_available, "liboqs required for real sign/verify roundtrip")
    def test_sign_verify_roundtrip_oqs(self):
        msg = b"authenticated NRSIP message"
        sig = self.provider.sign(self.kp.secret_key, msg)
        self.assertTrue(self.provider.verify(self.kp.public_key, msg, sig))

    @unittest.skipUnless(_oqs_available, "liboqs required")
    def test_tampered_signature_rejected_oqs(self):
        msg = b"provenance record"
        sig = bytearray(self.provider.sign(self.kp.secret_key, msg))
        sig[0] ^= 0xFF
        self.assertFalse(self.provider.verify(self.kp.public_key, msg, bytes(sig)))

    @unittest.skipUnless(_oqs_available, "liboqs required")
    def test_wrong_message_rejected_oqs(self):
        sig = self.provider.sign(self.kp.secret_key, b"original")
        self.assertFalse(self.provider.verify(self.kp.public_key, b"tampered", sig))

    def test_garbage_signature_rejected(self):
        """Random bytes must never verify regardless of backend."""
        garbage = os.urandom(64)
        result = self.provider.verify(self.kp.public_key, b"msg", garbage)
        self.assertFalse(result)


# ───────────────────────────────────────────────────────────────────
# 2. HybridKDF
# ───────────────────────────────────────────────────────────────────


class TestHybridKDF(unittest.TestCase):
    """Hybrid KDF combining PQC + classical shared secrets."""

    def setUp(self):
        self.kdf = HybridKDF()
        self.pqc = os.urandom(32)
        self.classical = os.urandom(32)

    def test_derive_returns_session_keys(self):
        sk = self.kdf.derive(self.pqc, self.classical)
        self.assertIsInstance(sk, SessionKeys)

    def test_session_key_is_32_bytes(self):
        sk = self.kdf.derive(self.pqc, self.classical)
        self.assertEqual(len(sk.session_key), 32)

    def test_mac_key_is_32_bytes(self):
        sk = self.kdf.derive(self.pqc, self.classical)
        self.assertEqual(len(sk.mac_key), 32)

    def test_iv_is_12_bytes(self):
        sk = self.kdf.derive(self.pqc, self.classical)
        self.assertEqual(len(sk.iv), 12)

    def test_algorithm_is_aes256gcm(self):
        sk = self.kdf.derive(self.pqc, self.classical)
        self.assertEqual(sk.algorithm, "AES-256-GCM")

    def test_deterministic_with_fixed_salt(self):
        salt = b"fixed-test-salt"
        sk1 = self.kdf.derive(self.pqc, self.classical, salt=salt)
        sk2 = self.kdf.derive(self.pqc, self.classical, salt=salt)
        self.assertEqual(sk1.session_key, sk2.session_key)
        self.assertEqual(sk1.mac_key, sk2.mac_key)
        self.assertEqual(sk1.iv, sk2.iv)

    def test_different_salts_different_keys(self):
        sk1 = self.kdf.derive(self.pqc, self.classical, salt=b"salt-A")
        sk2 = self.kdf.derive(self.pqc, self.classical, salt=b"salt-B")
        self.assertNotEqual(sk1.session_key, sk2.session_key)

    def test_different_secrets_different_keys(self):
        other_pqc = os.urandom(32)
        sk1 = self.kdf.derive(self.pqc, self.classical, salt=b"s")
        sk2 = self.kdf.derive(other_pqc, self.classical, salt=b"s")
        self.assertNotEqual(sk1.session_key, sk2.session_key)

    def test_none_salt_accepted(self):
        sk = self.kdf.derive(self.pqc, self.classical, salt=None)
        self.assertIsInstance(sk, SessionKeys)

    def test_kdf_label_matches_backend(self):
        sk = self.kdf.derive(self.pqc, self.classical)
        if _cryptography_available:
            self.assertEqual(sk.kdf, "HKDF-SHA256")
        else:
            self.assertEqual(sk.kdf, "SHA256-chain-fallback")

    def test_pqc_used_flag(self):
        sk = self.kdf.derive(self.pqc, self.classical)
        self.assertEqual(sk.pqc_used, _oqs_available)

    def test_classical_used_flag(self):
        sk = self.kdf.derive(self.pqc, self.classical)
        self.assertEqual(sk.classical_used, _cryptography_available)

    def test_derived_at_ms_is_recent(self):
        before = int(time.time() * 1000)
        sk = self.kdf.derive(self.pqc, self.classical)
        after = int(time.time() * 1000)
        self.assertGreaterEqual(sk.derived_at_ms, before)
        self.assertLessEqual(sk.derived_at_ms, after)

    def test_session_key_mac_key_iv_are_distinct(self):
        sk = self.kdf.derive(self.pqc, self.classical)
        self.assertNotEqual(sk.session_key, sk.mac_key)
        self.assertNotEqual(sk.session_key, sk.iv)


# ───────────────────────────────────────────────────────────────────
# 3. MLKEMProvider (ML-KEM-1024 / Kyber)
# ───────────────────────────────────────────────────────────────────


class TestMLKEMKeypair(unittest.TestCase):
    """ML-KEM keypair generation and algorithm labels."""

    def setUp(self):
        self.kem = MLKEMProvider()

    def test_generate_returns_kem_keypair(self):
        kp = self.kem.generate_keypair()
        self.assertIsInstance(kp, KEMKeyPair)
        self.assertIsInstance(kp.public_key, bytes)
        self.assertIsInstance(kp.secret_key, bytes)

    def test_algorithm_label(self):
        kp = self.kem.generate_keypair()
        if _oqs_available:
            self.assertEqual(kp.algorithm, "ML-KEM-1024")
            self.assertEqual(kp.security_level, 5)
        else:
            self.assertEqual(kp.algorithm, "ML-KEM-1024-fallback")
            self.assertEqual(kp.security_level, 0)

    def test_keypairs_unique(self):
        kp1 = self.kem.generate_keypair()
        kp2 = self.kem.generate_keypair()
        self.assertNotEqual(kp1.public_key, kp2.public_key)
        self.assertNotEqual(kp1.secret_key, kp2.secret_key)


class TestMLKEMEncapDecap(unittest.TestCase):
    """ML-KEM encapsulate / decapsulate correctness and fallback."""

    def setUp(self):
        self.kem = MLKEMProvider()
        self.kp = self.kem.generate_keypair()

    def test_encapsulate_returns_ct_and_ss(self):
        ct, ss = self.kem.encapsulate(self.kp.public_key)
        self.assertIsInstance(ct, bytes)
        self.assertIsInstance(ss, bytes)
        self.assertGreater(len(ct), 0)
        self.assertGreater(len(ss), 0)

    def test_decapsulate_returns_bytes(self):
        ct, _ = self.kem.encapsulate(self.kp.public_key)
        ss_dec = self.kem.decapsulate(self.kp.secret_key, ct)
        self.assertIsInstance(ss_dec, bytes)
        self.assertGreater(len(ss_dec), 0)

    @unittest.skipUnless(_oqs_available, "liboqs required for correct KEM roundtrip")
    def test_encap_decap_roundtrip_oqs(self):
        ct, ss_enc = self.kem.encapsulate(self.kp.public_key)
        ss_dec = self.kem.decapsulate(self.kp.secret_key, ct)
        self.assertEqual(ss_enc, ss_dec)

    def test_encapsulate_is_nondeterministic(self):
        """Consecutive encapsulations produce different shared secrets."""
        _, ss1 = self.kem.encapsulate(self.kp.public_key)
        _, ss2 = self.kem.encapsulate(self.kp.public_key)
        self.assertNotEqual(ss1, ss2)

    def test_decapsulate_is_deterministic(self):
        """Same (sk, ct) always decapsulates to the same shared secret."""
        ct, _ = self.kem.encapsulate(self.kp.public_key)
        d1 = self.kem.decapsulate(self.kp.secret_key, ct)
        d2 = self.kem.decapsulate(self.kp.secret_key, ct)
        self.assertEqual(d1, d2)


# ───────────────────────────────────────────────────────────────────
# 4. X25519Provider
# ───────────────────────────────────────────────────────────────────


class TestX25519Provider(unittest.TestCase):
    """X25519 classical ECDH: keypair gen, shared secret derivation."""

    def setUp(self):
        self.x25519 = X25519Provider()

    def test_generate_returns_pub_priv_tuple(self):
        pub, priv = self.x25519.generate_keypair()
        self.assertIsInstance(pub, bytes)
        self.assertIsInstance(priv, bytes)
        self.assertGreater(len(pub), 0)
        self.assertGreater(len(priv), 0)

    def test_keypairs_are_unique(self):
        pub_a, priv_a = self.x25519.generate_keypair()
        pub_b, priv_b = self.x25519.generate_keypair()
        self.assertNotEqual(pub_a, pub_b)
        self.assertNotEqual(priv_a, priv_b)

    def test_derive_shared_returns_bytes(self):
        pub_a, priv_a = self.x25519.generate_keypair()
        pub_b, priv_b = self.x25519.generate_keypair()
        shared = self.x25519.derive_shared(priv_a, pub_b)
        self.assertIsInstance(shared, bytes)
        self.assertGreater(len(shared), 0)

    @unittest.skipUnless(_cryptography_available, "cryptography lib required for ECDH agreement")
    def test_shared_secret_agreement(self):
        """Both sides derive the same shared secret (Diffie-Hellman property)."""
        pub_a, priv_a = self.x25519.generate_keypair()
        pub_b, priv_b = self.x25519.generate_keypair()
        shared_ab = self.x25519.derive_shared(priv_a, pub_b)
        shared_ba = self.x25519.derive_shared(priv_b, pub_a)
        self.assertEqual(shared_ab, shared_ba)

    def test_shared_secret_key_length(self):
        pub, priv = self.x25519.generate_keypair()
        pub2, _ = self.x25519.generate_keypair()
        shared = self.x25519.derive_shared(priv, pub2)
        self.assertEqual(len(shared), 32)


# ───────────────────────────────────────────────────────────────────
# 5. IronCladHandshake — Full Session Key Derivation
# ───────────────────────────────────────────────────────────────────


class TestIronCladHandshake(unittest.TestCase):
    """Full Iron-Clad 5-layer handshake: init, exchange, finalize."""

    def setUp(self):
        self.hs = IronCladHandshake()

    def test_server_init_has_all_keys(self):
        init = self.hs.server_init()
        for key in ("kem_public", "kem_secret", "x25519_public", "x25519_private"):
            self.assertIn(key, init)
            self.assertIsInstance(init[key], bytes)
            self.assertGreater(len(init[key]), 0)

    def test_client_exchange_produces_session_keys(self):
        srv = self.hs.server_init()
        cli = self.hs.client_exchange(srv["kem_public"], srv["x25519_public"])
        self.assertIn("session_keys", cli)
        self.assertIsInstance(cli["session_keys"], SessionKeys)
        self.assertEqual(len(cli["session_keys"].session_key), 32)

    def test_client_exchange_returns_kem_ciphertext(self):
        srv = self.hs.server_init()
        cli = self.hs.client_exchange(srv["kem_public"], srv["x25519_public"])
        self.assertIn("kem_ciphertext", cli)
        self.assertIsInstance(cli["kem_ciphertext"], bytes)

    def test_server_finalize_returns_session_keys(self):
        srv = self.hs.server_init()
        cli = self.hs.client_exchange(srv["kem_public"], srv["x25519_public"])
        server_sk = self.hs.server_finalize(
            srv["kem_secret"], cli["kem_ciphertext"],
            srv["x25519_private"], cli["x25519_public"],
        )
        self.assertIsInstance(server_sk, SessionKeys)
        self.assertEqual(len(server_sk.session_key), 32)
        self.assertEqual(len(server_sk.mac_key), 32)
        self.assertEqual(len(server_sk.iv), 12)

    @unittest.skipUnless(
        _oqs_available and _cryptography_available,
        "Full key agreement requires both liboqs and cryptography",
    )
    def test_handshake_matching_session_keys(self):
        """With real PQC + ECDH, both sides must derive identical session keys."""
        srv = self.hs.server_init()
        cli = self.hs.client_exchange(srv["kem_public"], srv["x25519_public"])
        server_sk = self.hs.server_finalize(
            srv["kem_secret"], cli["kem_ciphertext"],
            srv["x25519_private"], cli["x25519_public"],
        )
        client_sk = cli["session_keys"]
        self.assertEqual(server_sk.session_key, client_sk.session_key)
        self.assertEqual(server_sk.mac_key, client_sk.mac_key)
        self.assertEqual(server_sk.iv, client_sk.iv)

    def test_handshake_under_200ms(self):
        t0 = time.monotonic()
        srv = self.hs.server_init()
        cli = self.hs.client_exchange(srv["kem_public"], srv["x25519_public"])
        self.hs.server_finalize(
            srv["kem_secret"], cli["kem_ciphertext"],
            srv["x25519_private"], cli["x25519_public"],
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        self.assertLess(elapsed_ms, 200, f"Handshake took {elapsed_ms:.1f}ms")

    def test_successive_handshakes_yield_different_keys(self):
        """Ephemeral keys → each session is unique (forward secrecy)."""
        def _run():
            srv = self.hs.server_init()
            cli = self.hs.client_exchange(srv["kem_public"], srv["x25519_public"])
            return cli["session_keys"].session_key

        self.assertNotEqual(_run(), _run())


# ───────────────────────────────────────────────────────────────────
# 6. Capability / Algorithm Negotiation
# ───────────────────────────────────────────────────────────────────


class TestCapabilities(unittest.TestCase):
    """Capability reporting and algorithm negotiation contract."""

    def setUp(self):
        self.caps = IronCladHandshake().capabilities()

    def test_required_keys_present(self):
        required = {
            "oqs_available", "cryptography_available",
            "kem_algorithm", "classical_algorithm",
            "signature_algorithm", "kdf", "cipher",
            "security_level", "nist_compliant", "cnsa_2_0",
        }
        self.assertTrue(required.issubset(self.caps.keys()))

    def test_cipher_always_aes256gcm(self):
        self.assertEqual(self.caps["cipher"], "AES-256-GCM")

    def test_oqs_flag_consistent(self):
        self.assertEqual(self.caps["oqs_available"], _oqs_available)
        self.assertEqual(self.caps["cryptography_available"], _cryptography_available)

    def test_security_level_reflects_oqs(self):
        if _oqs_available:
            self.assertEqual(self.caps["security_level"], 5)
            self.assertTrue(self.caps["nist_compliant"])
            self.assertTrue(self.caps["cnsa_2_0"])
        else:
            self.assertEqual(self.caps["security_level"], 0)
            self.assertFalse(self.caps["nist_compliant"])
            self.assertFalse(self.caps["cnsa_2_0"])

    def test_algorithm_labels_reflect_fallback(self):
        if _oqs_available:
            self.assertEqual(self.caps["kem_algorithm"], "ML-KEM-1024")
            self.assertEqual(self.caps["signature_algorithm"], "Dilithium3")
        else:
            self.assertIn("fallback", self.caps["kem_algorithm"])
            self.assertIn("fallback", self.caps["signature_algorithm"])

    def test_kdf_label(self):
        if _cryptography_available:
            self.assertEqual(self.caps["kdf"], "HKDF-SHA256")
        else:
            self.assertEqual(self.caps["kdf"], "SHA256-chain")

    def test_classical_algorithm_label(self):
        if _cryptography_available:
            self.assertEqual(self.caps["classical_algorithm"], "X25519")
        else:
            self.assertIn("fallback", self.caps["classical_algorithm"])


# ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main()
