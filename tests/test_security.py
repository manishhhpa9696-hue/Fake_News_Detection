"""
Unit Tests for Security, SSRF Prevention, and Input Sanitization.
"""

import pytest
from src.security import is_safe_ip, validate_url_security, sanitize_input_text


def test_is_safe_ip_blocks_private():
    assert is_safe_ip("127.0.0.1") is False
    assert is_safe_ip("10.0.0.5") is False
    assert is_safe_ip("192.168.1.1") is False
    assert is_safe_ip("172.16.0.1") is False
    assert is_safe_ip("169.254.169.254") is False  # AWS metadata service IP
    assert is_safe_ip("::1") is False


def test_is_safe_ip_allows_public():
    # Public DNS IPs (Cloudflare / Google)
    assert is_safe_ip("1.1.1.1") is True
    assert is_safe_ip("8.8.8.8") is True


def test_validate_url_security_blocks_schemes():
    is_safe, msg, _ = validate_url_security("file:///etc/passwd")
    assert is_safe is False
    assert "Unsupported scheme" in msg

    is_safe, msg, _ = validate_url_security("ftp://ftp.example.com/file")
    assert is_safe is False


def test_validate_url_security_blocks_localhost():
    is_safe, msg, _ = validate_url_security("http://localhost:8080/admin")
    assert is_safe is False

    is_safe, msg, _ = validate_url_security("http://127.0.0.1:5000/api")
    assert is_safe is False


def test_sanitize_input_text():
    oversized = "a" * 60000
    sanitized = sanitize_input_text(oversized)
    assert len(sanitized) == 50000
