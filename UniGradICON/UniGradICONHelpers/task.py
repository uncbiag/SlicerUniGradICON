"""Background execution for the interactive UI. Never accesses Qt or MRML nodes."""

import json
import os
import queue
import shutil
import signal
import subprocess
import threading
import time

from . import registration
from .remote import RegistrationServerClient, ServerError

LOCAL_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "local_worker.py")


class RegistrationTask:
  def __init__(self, images, settings, directory, weights_dir, executable, url=None, token=None):
    self.images = images  # SimpleITK snapshots, made on Slicer's main thread
    self.settings = dict(settings)
    self.directory = directory
    self.weights_dir = weights_dir
    self.executable = executable
    self.url = url
    self.token = token
    self.events = queue.Queue()
    self.cancelled = threading.Event()
    self.abandoned = threading.Event()
    self.finished = threading.Event()
    self.thread = None
    self._process = None

  def start(self):
    self.thread = threading.Thread(target=self._run, name="UniGradICON-task", daemon=True)
    self.thread.start()

  def cancel(self):
    self.cancelled.set()
    # The local process watches this independently of Python's GIL and ITK.
    try:
      with open(os.path.join(self.directory, "cancel"), "w"):
        pass
    except FileNotFoundError:
      pass
    process = self._process
    if process is not None and process.poll() is None:
      self._signal_process(process)

  def check_cancelled(self):
    if self.cancelled.is_set():
      raise registration.RegistrationCancelled()

  def abandon(self):
    """Cancel without requiring a live GUI to consume the terminal event."""
    self.abandoned.set()
    self.cancel()
    if self.thread is None or self.finished.is_set():
      self.cleanup()

  def report(self, **status):
    self.events.put(("status", status))

  def cleanup(self):
    shutil.rmtree(self.directory, ignore_errors=True)

  def _run(self):
    try:
      import SimpleITK as sitk
      self.report(phase="Preparing images")
      paths = {}
      for role, image in self.images.items():
        self.check_cancelled()
        paths[role] = os.path.join(self.directory, role + ".nrrd")
        sitk.WriteImage(image, paths[role], True)
      self.images = None  # release the snapshots before inference
      self.check_cancelled()
      transform, warped = self._remote(paths) if self.url else self._local(paths)
      self.check_cancelled()
      self.report(phase="Loading results")
      if transform.endswith(".tfm"):
        # Parsing a large text displacement transform in MRML blocks Qt. Convert
        # it off the main thread; Slicer can then load the binary transform quickly.
        binary_transform = os.path.join(self.directory, "result.h5")
        sitk.WriteTransform(sitk.ReadTransform(transform), binary_transform)
        transform = binary_transform
      warped_image = sitk.ReadImage(warped) if warped else None
      self.check_cancelled()
      self.events.put(("result", (transform, warped_image)))
    except registration.RegistrationCancelled:
      self.cleanup()
      self.events.put(("cancelled", None))
    except Exception as error:
      self.cleanup()
      if self.cancelled.is_set():
        if self.url:
          self.events.put(("warning", f"Server cancellation could not be confirmed: {error}"))
        self.events.put(("cancelled", None))
      else:
        self.events.put(("error", f"{type(error).__name__}: {error}"))
    finally:
      self.images = None
      self.finished.set()
      if self.cancelled.is_set() or self.abandoned.is_set():
        self.cleanup()

  def _local(self, paths):
    arguments = dict(self.settings)
    arguments["model_name"] = arguments.pop("model")
    arguments.update(fixed_path=paths["fixed"], moving_path=paths["moving"],
                     weights_dir=self.weights_dir, out_dir=self.directory, transform_format="h5")
    manifest_path = os.path.join(self.directory, "job.json")
    with open(manifest_path, "w") as file:
      json.dump({"arguments": arguments, "parent_pid": os.getpid()}, file)
    with open(os.path.join(self.directory, "worker.log"), "w") as log:
      process = subprocess.Popen([self.executable, LOCAL_WORKER_SCRIPT, manifest_path], stdout=log, stderr=log,
                                 start_new_session=os.name != "nt")
      self._process = process
      status = {}
      try:
        while True:
          if self.cancelled.is_set() and process.poll() is None:
            self._stop_process(process)
            self.check_cancelled()
          try:
            with open(os.path.join(self.directory, "status.json")) as file:
              latest = json.load(file)
            if latest != status:
              status = latest
              self.report(**status)
          except FileNotFoundError:
            pass
          if process.poll() is not None:
            break
          time.sleep(0.1)
        self.check_cancelled()
        # Read once more after exit so a fast job cannot hide its final update.
        try:
          with open(os.path.join(self.directory, "status.json")) as file:
            status = json.load(file)
        except FileNotFoundError:
          raise RuntimeError("Local registration could not start. Check the local backend installation or use remote processing.")
        if status.get("state") != "done" or process.returncode != 0:
          raise RuntimeError(status.get("error") or "Local registration process stopped unexpectedly")
        return status["transform"], status.get("warped")
      finally:
        if process.poll() is None:
          self._stop_process(process)
        self._process = None

  @staticmethod
  def _signal_process(process):
    # PythonSlicer is a launcher: stop its entire process group/tree so no GPU
    # worker survives cancellation, including inside a C++ call that holds the GIL.
    if os.name == "nt":
      subprocess.Popen(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
      try:
        os.killpg(process.pid, signal.SIGTERM)
      except ProcessLookupError:
        pass

  @staticmethod
  def _stop_process(process):
    RegistrationTask._signal_process(process)
    try:
      process.wait(timeout=5)
    except subprocess.TimeoutExpired:
      if os.name != "nt":
        try:
          os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
          pass
      else:
        process.kill()
      process.wait(timeout=5)

  def _remote(self, paths):
    client = RegistrationServerClient(self.url, self.token)
    job_id = None
    try:
      self.report(phase="Connecting")
      info = client.info()
      if self.settings["device"] not in info["devices"]:
        raise ValueError(f"The server does not offer {self.settings['device']}. Select an available device.")
      self.check_cancelled()
      self.report(phase="Uploading images")
      job_id = client.submit(paths["fixed"], paths["moving"], self.settings)["id"]
      failures = 0
      while True:
        self.check_cancelled()
        try:
          status = client.status(job_id)
          failures = 0
        except ServerError:
          failures += 1
          if failures >= 3:
            raise
          self.report(phase="Reconnecting to server")
          if self.cancelled.wait(1):
            self.check_cancelled()
          continue
        state = status["state"]
        if state == "done":
          break
        if state == "failed":
          raise RuntimeError(status.get("error") or "The server registration failed")
        if state == "cancelled":
          raise registration.RegistrationCancelled()
        phase = (f"Queued ({status.get('queuePosition', 0)} ahead)" if state == "queued"
                 else status.get("phase") or "Registering")
        self.report(phase=phase, **(status.get("progress") or {}))
        if self.cancelled.wait(0.25):
          self.check_cancelled()
      self.report(phase="Downloading results")
      transform = client.download(job_id, "transform", os.path.join(self.directory, "transform.tfm"))
      self.check_cancelled()
      warped = None
      if self.settings["warp"]:
        warped = client.download(job_id, "warped", os.path.join(self.directory, "warped.nrrd"))
      return transform, warped
    finally:
      if job_id:
        try:
          response = client.cancel(job_id) or {}
          if self.cancelled.is_set() and response.get("state") == "cancelling":
            self.events.put(("warning", "Server cancellation was requested; its current step may still be finishing."))
        except ServerError as error:
          self.events.put(("warning", f"Could not remove server job {job_id}: {error}"))
