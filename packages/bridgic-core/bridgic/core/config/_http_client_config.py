"""
HTTP client configuration types and utilities.

This module provides serializable configuration types and utility functions
for creating HTTP clients (both sync and async).
"""

from typing import Optional, Dict, TypedDict, Literal, Union, Any
import httpx


class HttpClientTimeoutConfig(TypedDict, total=False):
    """
    Configuration for HTTP client timeout settings.
    
    Attributes
    ----------
    connect : Optional[float]
        Timeout for establishing a connection (seconds).
    read : Optional[float]
        Timeout for reading data (seconds).
    write : Optional[float]
        Timeout for writing data (seconds).
    pool : Optional[float]
        Timeout for acquiring a connection from the pool (seconds).
    """
    connect: Optional[float]
    read: Optional[float]
    write: Optional[float]
    pool: Optional[float]


class HttpClientAuthConfig(TypedDict, total=False):
    """
    Configuration for HTTP client authentication.
    
    Attributes
    ----------
    type : Literal["basic", "bearer"]
        The type of authentication.
    username : Optional[str]
        Username for basic auth (required if type is "basic").
    password : Optional[str]
        Password for basic auth (required if type is "basic").
    token : Optional[str]
        Bearer token (required if type is "bearer").
    """
    type: Literal["basic", "bearer"]
    username: Optional[str]
    password: Optional[str]
    token: Optional[str]


class HttpClientConfig(TypedDict, total=False):
    """
    Serializable configuration for creating an HTTP client.
    
    This configuration can be serialized and deserialized, allowing the framework
    to recreate HTTP clients after serialization.
    
    Attributes
    ----------
    headers : Optional[Dict[str, str]]
        HTTP headers to include with all requests.
    timeout : Optional[HttpClientTimeoutConfig]
        Timeout configuration for the HTTP client.
    auth : Optional[HttpClientAuthConfig]
        Authentication configuration for the HTTP client.
    """
    headers: Optional[Dict[str, str]]
    timeout: Optional[HttpClientTimeoutConfig]
    auth: Optional[HttpClientAuthConfig]


def create_http_client_from_config(
    config: Optional[HttpClientConfig],
    is_async: bool = True,
) -> Optional[Union[httpx.AsyncClient, httpx.Client]]:
    """
    Create an HTTP client (sync or async) from a serializable configuration.
    
    Parameters
    ----------
    config : Optional[HttpClientConfig]
        The HTTP client configuration. If None, returns None.
    is_async : bool
        If True, creates an `httpx.AsyncClient`. If False, creates an `httpx.Client`.
        Defaults to True.
    
    Returns
    -------
    Optional[Union[httpx.AsyncClient, httpx.Client]]
        The created HTTP client, or None if config is None.
    
    Raises
    ------
    ValueError
        If the auth configuration is invalid (missing required fields or unsupported type).
    
    Example
    -------
    >>> # Create async client with custom headers
    >>> config = {
    ...     "headers": {"Authorization": "Bearer token123"}
    ... }
    >>> client = create_http_client_from_config(config, is_async=True)
    >>> 
    >>> # Create async client with custom timeout
    >>> config = {
    ...     "timeout": {
    ...         "connect": 10.0,
    ...         "read": 60.0
    ...     }
    ... }
    >>> client = create_http_client_from_config(config, is_async=True)
    """
    # Return None when config is None.
    if config is None:
        return None

    # Extract configuration
    headers = config.get("headers")
    timeout_config = config.get("timeout")
    auth_config = config.get("auth")

    # Build timeout object if configured
    timeout = None
    if timeout_config:
        timeout = httpx.Timeout(
            connect=timeout_config.get("connect"),
            read=timeout_config.get("read"),
            write=timeout_config.get("write"),
            pool=timeout_config.get("pool"),
        )

    # Build auth object if configured
    auth = None
    if auth_config:
        auth_type = auth_config.get("type")
        if auth_type == "basic":
            username = auth_config.get("username")
            password = auth_config.get("password")
            if username is None or password is None:
                raise ValueError(
                    "Basic auth requires both 'username' and 'password' in http_client_config['auth']."
                )
            auth = httpx.BasicAuth(username=username, password=password)
        elif auth_type == "bearer":
            token = auth_config.get("token")
            if token is None:
                raise ValueError(
                    "Bearer auth requires 'token' in http_client_config['auth']."
                )
            # Bearer token is typically passed via headers.
            # So we add it to headers if not already present.
            if headers is None:
                headers = {}
            if "Authorization" not in headers:
                headers["Authorization"] = f"Bearer {token}"
        else:
            raise ValueError(
                f"Unsupported auth type: {auth_type}. Supported types: 'basic', 'bearer'."
            )

    # Build client kwargs
    client_kwargs: Dict[str, Any] = {
        "follow_redirects": True,
    }
    
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    
    if headers is not None:
        client_kwargs["headers"] = headers
    
    if auth is not None:
        client_kwargs["auth"] = auth

    # Create the appropriate client type
    if is_async:
        return httpx.AsyncClient(**client_kwargs)
    else:
        return httpx.Client(**client_kwargs)

