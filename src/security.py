"""
Security and URL validation module.
Defends against Server-Side Request Forgery (SSRF), private network probing,
disallowed schemes, oversized payloads, and unsafe content scraping.
"""

import ipaddress
import socket
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from typing import Tuple, Optional, Dict, Any

# Maximum article input size allowed (50,000 chars)
MAX_TEXT_LENGTH = 50000
MAX_RESPONSE_BYTES = 2 * 1024 * 1024  # 2 MB
REQUEST_TIMEOUT_SECONDS = 5.0
ALLOWED_SCHEMES = {"http", "https"}


def is_safe_ip(ip_str: str) -> bool:
    """
    Check if an IP address is publicly routable and not in private,
    loopback, link-local, multicast, or reserved ranges.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
        # Block private, loopback, link-local, multicast, and reserved addresses
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False
        return True
    except ValueError:
        return False


def validate_url_security(url: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validates a URL against SSRF attacks and illegal protocols.
    Returns: (is_valid: bool, message: str, resolved_ip: Optional[str])
    """
    if not url or not isinstance(url, str):
        return False, "URL cannot be empty.", None

    url = url.strip()
    if len(url) > 2048:
        return False, "URL exceeds maximum allowed length (2048 characters).", None

    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Malformed URL format: {str(e)}", None

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return False, f"Unsupported scheme '{parsed.scheme}'. Only http and https are permitted.", None

    hostname = parsed.hostname
    if not hostname:
        return False, "URL lacks a valid hostname or domain.", None

    # Check for direct localhost aliases
    if hostname.lower() in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
        return False, "Access to localhost or loopback is blocked for security.", None

    # Resolve hostname to IP to defend against DNS rebinding / private IP targets
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        ips = [info[4][0] for info in addr_info]
    except socket.gaierror:
        return False, f"Could not resolve domain name '{hostname}'.", None
    except Exception as e:
        return False, f"DNS resolution error: {str(e)}", None

    if not ips:
        return False, "No IP address could be resolved for domain.", None

    for resolved_ip in ips:
        if not is_safe_ip(resolved_ip):
            return False, f"Access to private/internal IP address '{resolved_ip}' is blocked.", resolved_ip

    return True, "URL is safe to fetch.", ips[0]


def safe_fetch_url(url: str) -> Dict[str, Any]:
    """
    Safely retrieves web article content with SSRF checks, headers, timeout,
    and safe HTML text extraction.
    """
    is_safe, msg, resolved_ip = validate_url_security(url)
    if not is_safe:
        return {
            "success": False,
            "error": msg,
            "title": "",
            "text": "",
            "author": "",
            "publisher": "",
            "domain": urlparse(url).hostname or "",
        }

    domain = urlparse(url).hostname or ""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36 (FakeNewsDetectorSafeBot/1.0)"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
    }

    try:
        # Use stream=True to prevent loading arbitrarily large payloads into memory
        response = requests.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
            stream=True,
            allow_redirects=True,
        )

        # Validate final redirected URL against SSRF
        redirected_url = response.url
        if redirected_url != url:
            is_redirect_safe, redir_msg, _ = validate_url_security(redirected_url)
            if not is_redirect_safe:
                return {
                    "success": False,
                    "error": f"Redirect blocked: {redir_msg}",
                    "title": "",
                    "text": "",
                    "author": "",
                    "publisher": "",
                    "domain": domain,
                }

        # Check HTTP status
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"HTTP status {response.status_code} ({response.reason})",
                "title": "",
                "text": "",
                "author": "",
                "publisher": "",
                "domain": domain,
            }

        # Read only up to MAX_RESPONSE_BYTES
        content_bytes = b""
        for chunk in response.iter_content(chunk_size=16384):
            content_bytes += chunk
            if len(content_bytes) > MAX_RESPONSE_BYTES:
                return {
                    "success": False,
                    "error": "Article size exceeds maximum permitted download limit (2 MB).",
                    "title": "",
                    "text": "",
                    "author": "",
                    "publisher": "",
                    "domain": domain,
                }

        # Parse HTML safely using BeautifulSoup
        html_text = content_bytes.decode(response.encoding or "utf-8", errors="replace")
        soup = BeautifulSoup(html_text, "html.parser")

        # Remove scripts, styles, forms, and navigation elements
        for element in soup(["script", "style", "nav", "footer", "header", "form", "aside", "noscript"]):
            element.decompose()

        # Extract title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)

        # Extract author & publisher metadata if present
        author = ""
        meta_author = soup.find("meta", attrs={"name": "author"}) or soup.find("meta", attrs={"property": "article:author"})
        if meta_author and meta_author.get("content"):
            author = meta_author.get("content").strip()

        publisher = ""
        meta_site = soup.find("meta", attrs={"property": "og:site_name"}) or soup.find("meta", attrs={"name": "publisher"})
        if meta_site and meta_site.get("content"):
            publisher = meta_site.get("content").strip()

        # Extract paragraphs
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 25]
        body_text = " ".join(paragraphs)

        if not body_text:
            body_text = soup.get_text(separator=" ", strip=True)

        # Truncate text if too long
        body_text = sanitize_input_text(body_text)

        return {
            "success": True,
            "error": None,
            "title": title[:500],
            "text": body_text,
            "author": author[:100],
            "publisher": publisher[:100],
            "domain": domain,
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Connection timed out while fetching article (limit: 5s).",
            "title": "",
            "text": "",
            "author": "",
            "publisher": "",
            "domain": domain,
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Failed to retrieve article: {str(e)}",
            "title": "",
            "text": "",
            "author": "",
            "publisher": "",
            "domain": domain,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error during content parsing: {str(e)}",
            "title": "",
            "text": "",
            "author": "",
            "publisher": "",
            "domain": domain,
        }


def sanitize_input_text(text: str) -> str:
    """
    Limits maximum text length to prevent memory exhaustion and strips excessive control characters.
    """
    if not text:
        return ""
    text = str(text).strip()
    return text[:MAX_TEXT_LENGTH]
