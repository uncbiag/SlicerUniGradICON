"""Interactive task regressions without Qt, MRML, model weights or a GPU."""

import json
from pathlib import Path
import queue
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from UniGradICONHelpers import registration, task, local_worker


def make_task(tmp_path, warp=True, remote=True):
  sitk = pytest.importorskip("SimpleITK")
  directory = tmp_path / "job"
  directory.mkdir()
  images = {role: sitk.Image([3, 3, 3], sitk.sitkFloat32) for role in ("fixed", "moving")}
  settings = dict(fixed_modality="MRI", moving_modality="MRI", model="unigradicon",
                  loss="LNCC", io_steps=0, device="CPU", warp=warp)
  return task.RegistrationTask(images, settings, str(directory), str(tmp_path), sys.executable,
                               url="http://testserver" if remote else None)


def terminal_event(job, timeout=10):
  events = []
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    try:
      event = job.events.get(timeout=0.1)
    except queue.Empty:
      continue
    events.append(event)
    if event[0] in ("result", "error", "cancelled"):
      job.thread.join(timeout=2)
      assert not job.thread.is_alive()
      return events
  raise AssertionError(f"No terminal event: {events}")


@pytest.mark.parametrize("warp", [True, False])
def test_remote_task_downloads_requested_results_before_deleting_job(tmp_path, monkeypatch, warp):
  sitk = pytest.importorskip("SimpleITK")
  calls = []
  class Client:
    def __init__(self, *args):
      pass
    def info(self):
      return {"devices": ["CPU"]}
    def submit(self, fixed, moving, settings):
      assert Path(fixed).exists() and Path(moving).exists()
      assert settings["warp"] == warp
      calls.append("submit")
      return {"id": "job"}
    def status(self, job_id):
      return {"state": "done"}
    def download(self, job_id, kind, path):
      calls.append(kind)
      if kind == "transform":
        sitk.WriteTransform(sitk.AffineTransform(3), path)
      else:
        sitk.WriteImage(sitk.Image([3, 3, 3], sitk.sitkFloat32), path)
      return path
    def cancel(self, job_id):
      calls.append("delete")
  monkeypatch.setattr(task, "RegistrationServerClient", Client)
  job = make_task(tmp_path, warp)
  try:
    job.start()
    events = terminal_event(job)
    assert events[-1][0] == "result", events
    assert calls == ["submit", "transform"] + (["warped"] if warp else []) + ["delete"]
    assert events[-1][1][1] is not None if warp else events[-1][1][1] is None
    phases = [data["phase"] for kind, data in events if kind == "status"]
    assert phases == ["Preparing images", "Connecting", "Uploading images", "Downloading results", "Loading results"]
  finally:
    job.cleanup()


def test_cancelling_during_upload_cancels_accepted_job_and_preserves_outputs(tmp_path, monkeypatch):
  entered = threading.Event()
  release = threading.Event()
  calls = []
  class Client:
    def __init__(self, *args):
      pass
    def info(self):
      return {"devices": ["CPU"]}
    def submit(self, *args):
      entered.set()
      assert release.wait(5)
      return {"id": "accepted"}
    def cancel(self, job_id):
      calls.append(job_id)
  monkeypatch.setattr(task, "RegistrationServerClient", Client)
  job = make_task(tmp_path)
  job.start()
  assert entered.wait(5)
  job.cancel()
  release.set()
  events = terminal_event(job)
  assert events[-1][0] == "cancelled", events
  assert calls == ["accepted"]
  assert not Path(job.directory).exists()


def test_failed_upload_cleans_up_and_reports_recoverable_error(tmp_path, monkeypatch):
  class Client:
    def __init__(self, *args):
      pass
    def info(self):
      raise task.ServerError("Nothing is listening; start the registration server")
  monkeypatch.setattr(task, "RegistrationServerClient", Client)
  job = make_task(tmp_path)
  job.start()
  events = terminal_event(job)
  assert events[-1][0] == "error"
  assert "start the registration server" in events[-1][1]
  assert not Path(job.directory).exists()


def test_local_cancellation_stops_launcher_and_worker_process_group(tmp_path, monkeypatch):
  script = tmp_path / "busy_worker.py"
  script.write_text('''import json, os, sys, time, subprocess
folder = os.path.dirname(sys.argv[1])
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
with open(os.path.join(folder, 'child.pid'), 'w') as f: f.write(str(child.pid))
with open(os.path.join(folder, 'status.tmp'), 'w') as f: json.dump({'phase': 'Registering'}, f)
os.replace(os.path.join(folder, 'status.tmp'), os.path.join(folder, 'status.json'))
time.sleep(60)
''')
  monkeypatch.setattr(task, "LOCAL_WORKER_SCRIPT", str(script))
  job = make_task(tmp_path, remote=False)
  job.start()
  child_path = Path(job.directory) / "child.pid"
  deadline = time.monotonic() + 5
  while not child_path.exists():
    assert time.monotonic() < deadline
    time.sleep(0.01)
  child_pid = int(child_path.read_text())
  start = time.monotonic()
  job.cancel()
  events = terminal_event(job)
  assert events[-1][0] == "cancelled", events
  assert time.monotonic() - start < 5
  assert not Path(job.directory).exists()
  # A terminated grandchild can briefly remain a zombie until its parent is reaped.
  if sys.platform.startswith("linux"):
    stat = Path(f"/proc/{child_pid}/stat")
    assert not stat.exists() or stat.read_text().split()[2] == "Z"


def test_local_worker_reports_backend_phases_and_result(monkeypatch):
  def register_files(**kwargs):
    kwargs["phase_callback"]("Resampling")
    kwargs["call_back"](5, 2, 3.0, 4.0)
    return "transform.tfm", None
  monkeypatch.setattr(registration, "register_files", register_files)
  events = []
  local_worker.run({"arguments": {}}, events.append)
  assert events == [{"phase": "Resampling"},
                    {"phase": "Registering", "total": 5, "step": 2, "elapsed": 3.0, "remaining": 4.0},
                    {"state": "done", "transform": "transform.tfm", "warped": None}]


def test_abandoned_completed_result_does_not_need_gui_to_clean_up(tmp_path):
  job = make_task(tmp_path)
  (Path(job.directory) / "result.h5").write_bytes(b"result")
  job.thread = threading.Thread(target=lambda: None)
  job.thread.start()
  job.thread.join()
  job.finished.set()
  job.abandon()
  assert not Path(job.directory).exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group escalation")
def test_local_stop_escalates_if_worker_ignores_terminate(monkeypatch):
  import signal
  import subprocess
  signals = []
  class Process:
    pid = 999999
    calls = 0
    def wait(self, timeout):
      self.calls += 1
      if self.calls == 1:
        raise subprocess.TimeoutExpired("worker", timeout)
      return 0
  monkeypatch.setattr(task.os, "killpg", lambda pid, signum: signals.append(signum))
  task.RegistrationTask._stop_process(Process())
  assert signals == [signal.SIGTERM, signal.SIGKILL]
