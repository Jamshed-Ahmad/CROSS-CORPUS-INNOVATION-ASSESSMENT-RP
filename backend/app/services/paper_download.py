import re
import socket
import ipaddress
from urllib.parse import urlparse, urljoin
from typing import Generator, Tuple
import requests
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from app.logging_config import logger

MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
REQUEST_TIMEOUT_SECONDS = 15

DISALLOWED_HOSTS = {
    "localhost", "127.0.0.1", "::1", "0.0.0.0", "metadata.google.internal",
    "instance-data", "169.254.169.254"
}

def validate_url_security(url: str) -> str:
    """
    Strictly validates external URL to protect against SSRF, local file access,
    internal network probing, and malicious redirects.
    """
    if not url or not isinstance(url, str):
        raise HTTPException(status_code=400, detail="Invalid or empty URL provided.")
    
    url = url.strip()
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse URL: {e}")

    # 1. Protocol check
    if parsed.scheme.lower() not in ("http", "https"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported protocol '{parsed.scheme}'. Only HTTP and HTTPS are permitted."
        )

    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(status_code=400, detail="URL missing valid hostname.")

    hostname_lower = hostname.lower()
    if hostname_lower in DISALLOWED_HOSTS or hostname_lower.endswith(".local") or hostname_lower.endswith(".internal"):
        logger.warning(f"SSRF Attempt blocked for host: {hostname}")
        raise HTTPException(status_code=400, detail="Access to local or internal addresses is strictly prohibited.")

    # 2. DNS resolution and IP verification
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    try:
        addr_info = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        logger.warning(f"DNS resolution failure for host {hostname}: {e}")
        raise HTTPException(status_code=400, detail=f"Could not resolve host '{hostname}'.")

    for entry in addr_info:
        sockaddr = entry[4]
        ip_str = sockaddr[0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
            if (
                ip_obj.is_private or
                ip_obj.is_loopback or
                ip_obj.is_reserved or
                ip_obj.is_link_local or
                ip_obj.is_multicast or
                ip_obj.is_unspecified
            ):
                logger.warning(f"SSRF attempt blocked for IP: {ip_str} (Host: {hostname})")
                raise HTTPException(
                    status_code=400,
                    detail="Access to internal, private, or loopback network addresses is strictly prohibited."
                )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid IP address encountered during host resolution.")

    return url

def sanitize_filename(title: str, default: str = "research_paper") -> str:
    """Sanitizes paper title for safe use in Content-Disposition header."""
    if not title:
        return default
    # Replace invalid filename characters
    clean = re.sub(r'[\\/*?:"<>|]', "", title)
    clean = re.sub(r'\s+', "_", clean).strip("._-")
    if not clean:
        clean = default
    return clean[:80]

def stream_verified_paper_pdf(url: str, paper_title: str) -> StreamingResponse:
    """
    Safely fetches and streams a legitimate full-text research paper PDF.
    Validates URL against SSRF, follows verified redirects, and verifies PDF magic bytes.
    """
    validated_url = validate_url_security(url)
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 (Cross-Corpus Academic Download Service)",
        "Accept": "application/pdf,application/octet-stream,*/*;q=0.8"
    }

    # Handle up to 3 redirects with validation of each target URL
    current_url = validated_url
    response = None
    for hop in range(4):
        try:
            response = session.get(
                current_url,
                headers=headers,
                stream=True,
                timeout=REQUEST_TIMEOUT_SECONDS,
                allow_redirects=False
            )
        except requests.exceptions.Timeout:
            logger.error(f"Paper download timed out for URL: {current_url}")
            raise HTTPException(status_code=504, detail="The remote paper repository timed out while retrieving the PDF.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Paper download network error for URL {current_url}: {e}")
            raise HTTPException(status_code=502, detail=f"Network error connecting to paper repository: {str(e)}")

        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location")
            if not location:
                raise HTTPException(status_code=502, detail="Remote repository redirected without a valid Location header.")
            # Resolve relative redirects
            redirect_url = urljoin(current_url, location)
            current_url = validate_url_security(redirect_url)
            response.close()
            continue
        break

    if not response or response.status_code >= 400:
        status_code = response.status_code if response else 500
        if status_code in (401, 403):
            raise HTTPException(
                status_code=403,
                detail="Direct download is restricted by publisher authentication or paywall. Please use 'View Source' to access the publication landing page."
            )
        raise HTTPException(
            status_code=status_code,
            detail=f"The publisher repository returned HTTP status {status_code}. Paper PDF could not be downloaded directly."
        )

    # Inspect first chunk to verify PDF header signature (%PDF-)
    chunk_iterator = response.iter_content(chunk_size=8192)
    try:
        first_chunk = next(chunk_iterator)
    except StopIteration:
        raise HTTPException(status_code=422, detail="The remote repository returned an empty file.")

    # Check for PDF magic bytes within the first 1024 bytes
    if b"%PDF-" not in first_chunk[:1024]:
        # Check if remote server returned HTML (e.g. login/paywall notice or landing page)
        first_chunk_str = first_chunk[:512].decode("utf-8", errors="ignore").lower()
        if "<html" in first_chunk_str or "<!doctype" in first_chunk_str:
            raise HTTPException(
                status_code=422,
                detail="The paper link returned an HTML landing or paywall page rather than an accessible PDF document. Please use 'View Source'."
            )
        raise HTTPException(
            status_code=422,
            detail="The returned file does not appear to be a valid PDF document."
        )

    safe_name = sanitize_filename(paper_title) + ".pdf"

    def content_stream() -> Generator[bytes, None, None]:
        total_streamed = len(first_chunk)
        yield first_chunk
        try:
            for chunk in chunk_iterator:
                if chunk:
                    total_streamed += len(chunk)
                    if total_streamed > MAX_PDF_SIZE_BYTES:
                        logger.warning(f"Paper download exceeded size limit ({MAX_PDF_SIZE_BYTES} bytes): {paper_title}")
                        break
                    yield chunk
        finally:
            response.close()

    return StreamingResponse(
        content_stream(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}"',
            "Cache-Control": "no-cache"
        }
    )
