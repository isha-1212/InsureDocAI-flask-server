"""
ML Client for Django Backend
Handles communication between Django and Flask ML server
Optimized for low-latency Supabase fetch + ML extraction.
"""

import asyncio
import hashlib
import logging
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class MLClient:
    """Client to communicate with Flask ML server"""
    
    def __init__(self, ml_server_url: str = None):
        self.ml_server_url = (ml_server_url or os.environ.get('ML_SERVER_URL', 'http://127.0.0.1:5001')).rstrip('/')
        self.timeout = int(os.environ.get('ML_CLIENT_TIMEOUT', 30))
        self.extract_timeout = int(os.environ.get('ML_EXTRACT_TIMEOUT', 300))
        self.download_timeout = int(os.environ.get('ML_DOWNLOAD_TIMEOUT', 60))
        self.max_retries = int(os.environ.get('ML_DOWNLOAD_RETRIES', 2))
        self.cache_dir = os.environ.get('ML_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'mediclaim_cache'))
        self.cache_ttl = int(os.environ.get('ML_CACHE_TTL_SECONDS', 3600))
        self.max_connections = int(os.environ.get('ML_HTTP_MAX_CONNECTIONS', 50))
        self.max_keepalive = int(os.environ.get('ML_HTTP_MAX_KEEPALIVE', 20))
        # Default on so Django can recover from stale/missing local ML server processes.
        self.auto_start = os.environ.get('ML_AUTO_START_SERVER', '1') == '1'
        self.startup_wait_seconds = int(os.environ.get('ML_SERVER_STARTUP_WAIT_SECONDS', 20))
        self._ml_process: Optional[subprocess.Popen] = None
        os.makedirs(self.cache_dir, exist_ok=True)

        self._sync_client = httpx.Client(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=self.max_connections, max_keepalive_connections=self.max_keepalive),
        )

    def _iter_server_urls(self) -> List[str]:
        urls = [self.ml_server_url]
        if "localhost" in self.ml_server_url:
            urls.append(self.ml_server_url.replace("localhost", "127.0.0.1"))
        elif "127.0.0.1" in self.ml_server_url:
            urls.append(self.ml_server_url.replace("127.0.0.1", "localhost"))
        # Preserve order and remove duplicates
        seen = set()
        ordered = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                ordered.append(u)
        return ordered
    
    def health_check(self) -> bool:
        """Check if ML server is running"""
        if self._ping_health():
            return True

        logger.warning("ML server health check failed")
        if self.auto_start:
            self._try_start_ml_server()
            deadline = time.time() + self.startup_wait_seconds
            while time.time() < deadline:
                if self._ping_health():
                    logger.info("ML server became healthy after auto-start")
                    return True
                time.sleep(1)
        return False

    def _ping_health(self) -> bool:
        for base_url in self._iter_server_urls():
            try:
                response = self._sync_client.get(f"{base_url}/health", timeout=5)
                if response.status_code == 200:
                    if base_url != self.ml_server_url:
                        logger.info("Switching ML server URL to reachable endpoint: %s", base_url)
                        self.ml_server_url = base_url
                    return True
            except Exception:
                continue
        return False

    def _try_start_ml_server(self) -> None:
        # If we already started a process and it is still alive, don't start again.
        if self._ml_process and self._ml_process.poll() is None:
            return

        try:
            ml_dir = os.path.dirname(__file__)
            server_script = os.path.join(ml_dir, 'ml_flask_server.py')
            if not os.path.exists(server_script):
                logger.error(f"ML server script not found: {server_script}")
                return

            creationflags = 0
            if os.name == 'nt':
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

            self._ml_process = subprocess.Popen(
                [sys.executable, server_script],
                cwd=ml_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
                start_new_session=(os.name != 'nt'),
            )
            logger.info("Started ML Flask server process")
        except Exception as e:
            logger.error(f"Failed to auto-start ML server: {str(e)}")
    
    def extract_single_document(self, file_path: str, document_type: str) -> Dict:
        """
        Extract data from a single document
        
        Args:
            file_path (str): Path to the document file
            document_type (str): Type of document
            
        Returns:
            dict: Extraction results
        """
        try:
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")
            
            logger.info(f"Extracting {document_type} from {file_path}")
            
            with open(file_path, 'rb') as file:
                files = {'file': file}
                data = {'document_type': document_type}

                response = self._sync_client.post(
                    f"{self.ml_server_url}/extract",
                    files=files,
                    data=data,
                    timeout=self.extract_timeout
                )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return result['extracted_fields']
                else:
                    raise Exception(result.get('error', 'Unknown error'))
            else:
                error_msg = f"ML server returned status {response.status_code}"
                try:
                    error_detail = response.json().get('error', '')
                    if error_detail:
                        error_msg += f": {error_detail}"
                except:
                    pass
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"Error extracting document: {str(e)}")
            raise
    
    def extract_batch_documents(self, documents: List[Dict]) -> Dict:
        """
        Extract data from multiple documents
        
        Args:
            documents (list): List of document info dicts with 'file_url' and 'document_type'
            
        Returns:
            dict: Batch extraction results
        """
        try:
            logger.info(f"Extracting batch of {len(documents)} documents")
            
            data = {'documents': documents}
            
            response = self._sync_client.post(
                f"{self.ml_server_url}/extract/batch",
                json=data,
                timeout=self.extract_timeout  # Explicit extraction timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return result
                else:
                    raise Exception(result.get('error', 'Unknown error'))
            else:
                error_msg = f"ML server returned status {response.status_code}"
                try:
                    error_detail = response.json().get('error', '')
                    if error_detail:
                        error_msg += f": {error_detail}"
                except:
                    pass
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"Error in batch extraction: {str(e)}")
            raise
    
    def extract_from_supabase_url(self, file_url: str, document_type: str) -> Dict:
        """
        Send signed URL directly to ML server for extraction.
        
        Args:
            file_url (str): Supabase storage URL
            document_type (str): Type of document
            
        Returns:
            dict: Extraction results
        """
        try:
            payload = {'file_url': file_url, 'document_type': document_type}
            response = None
            last_conn_err = None
            for base_url in self._iter_server_urls():
                try:
                    response = self._sync_client.post(
                        f"{base_url}/extract/url",
                        json=payload,
                        timeout=self.extract_timeout
                    )
                    if response.status_code == 404:
                        response = self._sync_client.post(
                            f"{base_url}/extract/url/",
                            json=payload,
                            timeout=self.extract_timeout
                        )
                    if base_url != self.ml_server_url:
                        logger.info("Switching ML server URL to reachable endpoint: %s", base_url)
                        self.ml_server_url = base_url
                    break
                except httpx.HTTPError as conn_err:
                    last_conn_err = conn_err
                    continue
            if response is None:
                if last_conn_err:
                    raise last_conn_err
                raise Exception("Unable to reach ML server")

            # Backward compatibility: older ML server may not expose /extract/url yet.
            if response.status_code == 404:
                logger.warning(
                    "ML server extract-url endpoint not found at %s. "
                    "This usually means an older ML server process is running. "
                    "Falling back to legacy file download flow.",
                    self.ml_server_url,
                )
                return self._legacy_extract_from_supabase_url(file_url, document_type)

            if response.status_code != 200:
                error_msg = f"ML server returned status {response.status_code}"
                try:
                    error_detail = response.json().get('details') or response.json().get('error', '')
                    if error_detail:
                        error_msg += f": {error_detail}"
                except Exception:
                    pass
                raise Exception(error_msg)

            result = response.json()
            if result.get('success'):
                return result.get('extracted_fields', {})
            raise Exception(result.get('error', 'Unknown error'))
                
        except Exception as e:
            logger.error(f"Error extracting from Supabase URL {file_url}: {str(e)}")
            raise

    async def extract_from_supabase_url_async(self, file_url: str, document_type: str) -> Dict:
        """
        Async: Send signed URL directly to ML server for extraction.
        """
        try:
            payload = {'file_url': file_url, 'document_type': document_type}
            response = None
            last_conn_err = None
            for base_url in self._iter_server_urls():
                try:
                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(self.timeout),
                        limits=httpx.Limits(
                            max_connections=self.max_connections,
                            max_keepalive_connections=self.max_keepalive
                        ),
                    ) as async_client:
                        response = await async_client.post(
                            f"{base_url}/extract/url",
                            json=payload,
                            timeout=self.extract_timeout
                        )

                        if response.status_code == 404:
                            response = await async_client.post(
                                f"{base_url}/extract/url/",
                                json=payload,
                                timeout=self.extract_timeout
                            )
                    if base_url != self.ml_server_url:
                        logger.info("Switching ML server URL to reachable endpoint: %s", base_url)
                        self.ml_server_url = base_url
                    break
                except httpx.HTTPError as conn_err:
                    last_conn_err = conn_err
                    continue
            if response is None:
                if last_conn_err:
                    raise last_conn_err
                raise Exception("Unable to reach ML server")

            # Backward compatibility: older ML server may not expose /extract/url yet.
            if response.status_code == 404:
                logger.warning(
                    "ML server extract-url endpoint not found at %s. "
                    "This usually means an older ML server process is running. "
                    "Falling back to legacy file download flow.",
                    self.ml_server_url,
                )
                return await asyncio.to_thread(self._legacy_extract_from_supabase_url, file_url, document_type)

            if response.status_code != 200:
                error_msg = f"ML server returned status {response.status_code}"
                try:
                    payload = response.json()
                    error_detail = payload.get('details') or payload.get('error', '')
                    if error_detail:
                        error_msg += f": {error_detail}"
                except Exception:
                    pass
                raise Exception(error_msg)

            result = response.json()
            if result.get('success'):
                return result.get('extracted_fields', {})
            raise Exception(result.get('error', 'Unknown error'))
        except Exception as e:
            logger.error(f"Error extracting from Supabase URL {file_url}: {str(e)}")
            raise

    def _legacy_extract_from_supabase_url(self, file_url: str, document_type: str) -> Dict:
        """
        Legacy fallback: download signed URL content, then send file to /extract.
        """
        cached = self._get_cached_file_path(file_url)
        if cached:
            return self.extract_single_document(cached, document_type)

        temp_path = self._get_temp_download_path(file_url)
        self._stream_download_sync(file_url, temp_path)
        self._promote_cache_file(file_url, temp_path)
        return self.extract_single_document(self._get_cache_path(file_url), document_type)
    
    def _get_file_extension(self, url: str) -> str:
        """Get file extension from URL"""
        try:
            path = url.split('?')[0]  # Remove query parameters
            ext = os.path.splitext(path)[1]
            return ext if ext else '.jpg'  # Default to jpg
        except:
            return '.jpg'

    def _cache_key(self, url: str) -> str:
        normalized = url.split('?', 1)[0]  # Strip token for cache key
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def _get_cache_path(self, url: str) -> str:
        ext = self._get_file_extension(url)
        return os.path.join(self.cache_dir, f"{self._cache_key(url)}{ext}")

    def _get_temp_download_path(self, url: str) -> str:
        return f"{self._get_cache_path(url)}.tmp"

    def _get_cached_file_path(self, url: str) -> Optional[str]:
        path = self._get_cache_path(url)
        if not os.path.exists(path):
            return None
        if self.cache_ttl <= 0:
            return path
        try:
            age = time.time() - os.path.getmtime(path)
            if age <= self.cache_ttl:
                return path
        except Exception:
            pass
        return None

    def _promote_cache_file(self, url: str, temp_path: str) -> None:
        final_path = self._get_cache_path(url)
        os.replace(temp_path, final_path)

    def _stream_download_sync(self, url: str, dest_path: str) -> None:
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._sync_client.stream("GET", url, timeout=self.download_timeout) as response:
                    response.raise_for_status()
                    with open(dest_path, "wb") as f:
                        for chunk in response.iter_bytes():
                            if chunk:
                                f.write(chunk)
                return
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2 ** attempt))
        raise last_err

    async def _stream_download_async(self, url: str, dest_path: str) -> None:
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout),
                    limits=httpx.Limits(
                        max_connections=self.max_connections,
                        max_keepalive_connections=self.max_keepalive
                    ),
                ) as async_client:
                    async with async_client.stream("GET", url, timeout=self.download_timeout) as response:
                        response.raise_for_status()
                        with open(dest_path, "wb") as f:
                            async for chunk in response.aiter_bytes():
                                if chunk:
                                    f.write(chunk)
                return
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))
        raise last_err


# Global ML client instance
ml_client = MLClient()


def extract_document_via_ml_server(file_path: str, document_type: str) -> Dict:
    """
    Helper function to extract document data via ML server
    """
    return ml_client.extract_single_document(file_path, document_type)


def extract_from_supabase_via_ml_server(file_url: str, document_type: str) -> Dict:
    """
    Helper function to extract from Supabase URL via ML server
    """
    return ml_client.extract_from_supabase_url(file_url, document_type)
