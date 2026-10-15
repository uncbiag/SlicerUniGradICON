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


def test_remote_masks_labels_and_overlapping_segment_propagation(tmp_path, monkeypatch):
  import SimpleITK as sitk
  import numpy as np
  submissions = []
  class Client:
    def __init__(self, *args):
      pass
    def info(self):
      return {"devices": ["CPU"], "features": ["segmentationGuidance"]}
    def submit(self, fixed, moving, settings, **uploads):
      submissions.append((settings, uploads))
      assert set(uploads) == {"fixed_mask_path", "fixed_segmentation_path", "moving_segmentation_path"}
      for path in uploads.values():
        assert sitk.ReadImage(path).GetSize() == (3, 3, 3)
      return {"id": "guided"}
    def status(self, job_id):
      return {"state": "done"}
    def download(self, job_id, kind, path):
      assert kind == "transform"
      sitk.WriteTransform(sitk.TranslationTransform(3, [1, 0, 0]), path)
      return path
    def cancel(self, job_id):
      pass
  monkeypatch.setattr(task, "RegistrationServerClient", Client)
  job = make_task(tmp_path, warp=False)
  labels = np.zeros((3, 3, 3), np.uint8)
  labels[:, :, 1] = 1
  for key in ("fixed_mask", "fixed_segmentation", "moving_segmentation", "propagation_0", "propagation_1"):
    job.images[key] = sitk.GetImageFromArray(labels)
  job.settings.update(dice_loss_weight=0.1, io_steps=1)
  try:
    job.start()
    result = terminal_event(job)[-1]
    assert result[0] == "result", result
    assert len(result[1][2]) == 2
    expected = np.zeros_like(labels)
    expected[:, :, 0] = 1
    for image in result[1][2]:
      np.testing.assert_array_equal(sitk.GetArrayFromImage(image), expected)
    assert submissions[0][0]["dice_loss_weight"] == 0.1
  finally:
    job.cleanup()


def test_guidance_is_not_silently_ignored_by_older_server(tmp_path, monkeypatch):
  class Client:
    def __init__(self, *args):
      pass
    def info(self):
      return {"devices": ["CPU"]}
    def submit(self, *args, **kwargs):
      raise AssertionError("Must not submit Dice guidance to an older server")
  monkeypatch.setattr(task, "RegistrationServerClient", Client)
  job = make_task(tmp_path)
  job.settings["dice_loss_weight"] = 0.1
  job.start()
  error = terminal_event(job)[-1]
  assert error[0] == "error" and "does not support segmentation-guided IO" in error[1]
  assert not Path(job.directory).exists()


@pytest.mark.parametrize("mode", ["input", "loss"])
def test_remote_masking_mode_uploads_both_masks(tmp_path, monkeypatch, mode):
  import SimpleITK as sitk
  seen = []
  class Client:
    def __init__(self, *args):
      pass
    def info(self):
      return {"devices": ["CPU"], "features": ["inputMasking", "lossFunctionMasking"]}
    def submit(self, fixed, moving, settings, **uploads):
      assert settings["input_masking"] == (mode == "input")
      assert settings["loss_function_masking"] == (mode == "loss")
      assert set(uploads) == {"fixed_mask_path", "moving_mask_path"}
      assert all(sitk.ReadImage(path).GetSize() == (3, 3, 3) for path in uploads.values())
      seen.append(settings)
      return {"id": "masked"}
    def status(self, job_id):
      return {"state": "done"}
    def download(self, job_id, kind, path):
      sitk.WriteTransform(sitk.Transform(3, sitk.sitkIdentity), path)
      return path
    def cancel(self, job_id):
      pass
  monkeypatch.setattr(task, "RegistrationServerClient", Client)
  job = make_task(tmp_path, warp=False)
  job.images.update(fixed_mask=sitk.Image([3, 3, 3], sitk.sitkUInt8) + 1,
                    moving_mask=sitk.Image([3, 3, 3], sitk.sitkUInt8) + 1)
  job.settings.update(input_masking=mode == "input", loss_function_masking=mode == "loss", io_steps=1)
  try:
    job.start()
    result = terminal_event(job)[-1]
    assert result[0] == "result", result
    assert len(seen) == 1
  finally:
    job.cleanup()


@pytest.mark.parametrize("mode", ["input", "loss"])
def test_remote_masking_rejects_server_that_cannot_honor_mode(tmp_path, monkeypatch, mode):
  class Client:
    def __init__(self, *args):
      pass
    def info(self):
      return {"devices": ["CPU"]}
  monkeypatch.setattr(task, "RegistrationServerClient", Client)
  job = make_task(tmp_path)
  job.settings.update(input_masking=mode == "input", loss_function_masking=mode == "loss", io_steps=1)
  job.start()
  result = terminal_event(job)[-1]
  assert result[0] == "error" and "does not support" in result[1]
  assert not Path(job.directory).exists()


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


def test_multilabel_does_not_submit_to_server_that_would_binarize_labels(tmp_path, monkeypatch):
  class Client:
    def __init__(self, *args):
      pass
    def info(self):
      return {"devices": ["CPU"], "features": ["segmentationGuidance"]}
    def submit(self, *args, **kwargs):
      raise AssertionError("Do not upload multi-label inputs to an older server")
  monkeypatch.setattr(task, "RegistrationServerClient", Client)
  job = make_task(tmp_path)
  job.settings.update(dice_loss_weight=10, multilabel_segmentation=True, io_steps=1)
  job.start()
  result = terminal_event(job)[-1]
  assert result[0] == "error" and "preserve multiple segmentation labels" in result[1]
