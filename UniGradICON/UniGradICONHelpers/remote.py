"""Client of the UniGradICON registration server (UniGradICONServer/server.py)."""

import json
import logging
import os
import time
import uuid
from contextlib import ExitStack

import requests

CONNECT_TIMEOUT = 5
REQUEST_TIMEOUT = 30
UPLOAD_TIMEOUT = 600  # volumes can be large and connections slow
# NB: requests applies the connect timeout also while it sends the request, so an upload to a server that stops
# reading for a moment (e.g. busy with other clients' uploads) failed after CONNECT_TIMEOUT. Transfers get more.
TRANSFER_CONNECT_TIMEOUT = 60
# waits before retrying a transfer that failed on the connection, not on the server's answer
TRANSFER_RETRY_DELAYS = (2, 5, 10)


class ServerError(Exception):
  pass


class RegistrationServerClient:
  def __init__(self, url, token=None):
    self.url = url.strip().rstrip("/")
    if "://" not in self.url:
      self.url = "http://" + self.url
    self.token = token or None

  def _describe(self, error):
    """What went wrong with the connection, in words that point at the cause."""
    text = str(error).lower()
    if isinstance(error, requests.ConnectTimeout):
      return (f"The registration server at {self.url} did not accept the connection in time "
              "(network or firewall?).")
    if isinstance(error, requests.ConnectionError):
      if "refused" in text:
        return f"Nothing is listening at {self.url}: is the registration server running?"
      if "name or service not known" in text or "nodename nor servname" in text or "failed to resolve" in text:
        return f"Cannot find the registration server's host in {self.url}."
      if "aborted" in text or "reset" in text or "timed out" in text or "broken pipe" in text:
        return f"The connection to the registration server at {self.url} was interrupted during the transfer."
      return f"Cannot reach the registration server at {self.url}."
    return f"The registration server at {self.url} did not answer in time."

  def _request(self, method, path, timeout=REQUEST_TIMEOUT, connectTimeout=CONNECT_TIMEOUT, retryDelays=(),
               filePaths=None, **kwargs):
    """Send a request; connection failures are retried after each of retryDelays (seconds).

    filePaths ({field: path}) are opened for every attempt, as a failed attempt may have read them partly.
    """
    headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
    for attempt in range(len(retryDelays) + 1):
      try:
        with ExitStack() as files:
          if filePaths:
            kwargs["files"] = {
              field: (os.path.basename(filePath), files.enter_context(open(filePath, "rb")))
              for field, filePath in filePaths.items()
            }
          response = requests.request(method, self.url + path, headers=headers,
                                      timeout=(connectTimeout, timeout), **kwargs)
        break
      except (requests.ConnectionError, requests.Timeout) as e:
        description = self._describe(e)
        if attempt == len(retryDelays):
          raise ServerError(description) from e
        delay = retryDelays[attempt]
        logging.warning(f"{description} Retrying in {delay} s ({attempt + 1} of {len(retryDelays)}).")
        time.sleep(delay)
    if response.status_code == 401:
      raise ServerError("The registration server refused the access token.")
    if not response.ok:
      try:
        detail = response.json().get("detail")
      except ValueError:
        detail = response.text
      raise ServerError(f"Registration server error {response.status_code}: {detail}")
    return response

  def info(self, timeout=REQUEST_TIMEOUT):
    info = self._request("GET", "/info", timeout=timeout).json()
    if info.get("server") != "UniGradICON":
      raise ServerError(f"{self.url} is not a UniGradICON registration server.")
    return info

  def submit(self, fixed_path, moving_path, params, fixed_mask_path=None):
    """Upload the two images, and the fixed image mask if there is one; returns the job status."""
    paths = {"fixed": fixed_path, "moving": moving_path}
    if fixed_mask_path:
      paths["fixed_mask"] = fixed_mask_path
    # Reuse the submission UUID on every retry, including when only the response was lost.
    return self._request("POST", "/jobs", timeout=UPLOAD_TIMEOUT, connectTimeout=TRANSFER_CONNECT_TIMEOUT,
                         retryDelays=TRANSFER_RETRY_DELAYS, filePaths=paths,
                         data={"params": json.dumps(params), "request_id": uuid.uuid4().hex}).json()

  def status(self, job_id):
    return self._request("GET", f"/jobs/{job_id}").json()

  def download(self, job_id, result, path):
    """Save a result of a finished job, ``transform`` or ``warped``, to path."""
    response = self._request("GET", f"/jobs/{job_id}/{result}", timeout=UPLOAD_TIMEOUT,
                             connectTimeout=TRANSFER_CONNECT_TIMEOUT, retryDelays=TRANSFER_RETRY_DELAYS)
    with open(path, "wb") as f:
      f.write(response.content)
    return path

  def cancel(self, job_id):
    return self._request("DELETE", f"/jobs/{job_id}").json()
