"""Tests of the registration server with a fake registration, so they need no PyTorch.

    pip install fastapi python-multipart httpx pytest
    pytest test_server.py
"""

import json
import os
import threading
import time
import concurrent.futures
from contextlib import ExitStack
from pathlib import Path
import sys
import uuid
import pytest

from fastapi.testclient import TestClient

import server
from UniGradICONHelpers import registration
from UniGradICONHelpers import remote

PARAMS = {
  "fixed_modality": "MRI", "moving_modality": "MRI", "model": "unigradicon",
  "loss": "LNCC", "io_steps": 3, "device": "CPU",
}


@pytest.fixture(autouse=True)
def managed_clients(monkeypatch):
  """Exercise the lifespan and stop worker/cleanup threads after each test."""
  original = TestClient
  with ExitStack() as clients:
    monkeypatch.setattr(sys.modules[__name__], "TestClient",
                        lambda *args, **kwargs: clients.enter_context(original(*args, **kwargs)))
    yield


class FakeRegistration:
  """Writes small result files; can be held at a step until released."""

  def __init__(self, fail=False):
    self.fail = fail
    self.release = threading.Event()
    self.release.set()
    self.calls = []
    self.masks = []

  def __call__(self, fixed_path, moving_path, fixed_modality, moving_modality, model_name, loss, io_steps,
               device, weights_dir, out_dir, call_back=None, warp=True, fixed_mask_path=None):
    self.calls.append((open(fixed_path, "rb").read(), open(moving_path, "rb").read(), io_steps, device))
    self.masks.append(open(fixed_mask_path, "rb").read() if fixed_mask_path else None)
    for step in range(io_steps):
      call_back(io_steps, step, float(step), float(io_steps - step))
      self.release.wait(5)
    if self.fail:
      raise RuntimeError("images do not overlap")
    transform_path = os.path.join(out_dir, "transform.tfm")
    open(transform_path, "w").write("transform")
    if not warp:
      return transform_path, None
    warped_path = os.path.join(out_dir, "warped.nrrd")
    open(warped_path, "wb").write(b"warped")
    return transform_path, warped_path


def client_for(fake, token=None):
  return TestClient(server.create_app(token=token, register=fake, devices=["CPU"]))


def submit(client, params=PARAMS, headers=None, fixed_name="fixed.nrrd", request_id=None):
  data = {"params": json.dumps(params)}
  if request_id is not None:
    data["request_id"] = request_id
  return client.post(
    "/jobs",
    files={"fixed": (fixed_name, b"FIXED"), "moving": ("moving.nii.gz", b"MOVING")},
    data=data,
    headers=headers or {},
  )


def wait_for(client, job_id, states, headers=None, timeout=5):
  deadline = time.time() + timeout
  while time.time() < deadline:
    status = client.get(f"/jobs/{job_id}", headers=headers or {}).json()
    if status["state"] in states:
      return status
    time.sleep(0.02)
  raise AssertionError(f"job {job_id} never reached {states}, last {status}")


def test_info_lists_devices_and_models():
  info = client_for(FakeRegistration()).get("/info").json()
  assert info["devices"] == ["CPU"]
  assert set(info["models"]) == set(registration.MODEL_WEIGHTS)


def test_guidance_requires_labels_supported_backend_and_positive_io(monkeypatch):
  monkeypatch.setattr(server, "guidance_available", lambda: True)
  captured = []
  def guided_register(**kwargs):
    captured.append(kwargs)
    assert Path(kwargs["fixed_segmentation_path"]).read_bytes() == b"FIXED_LABEL"
    assert Path(kwargs["moving_segmentation_path"]).read_bytes() == b"MOVING_LABEL"
    assert kwargs["dice_loss_weight"] == 0.25
    return FakeRegistration()(**{key: value for key, value in kwargs.items()
                               if key not in ("fixed_segmentation_path", "moving_segmentation_path", "dice_loss_weight")})
  client = client_for(guided_register)
  assert "segmentationGuidance" in client.get("/info").json()["features"]
  params = dict(PARAMS, dice_loss_weight=0.25)
  files = {"fixed": ("f.nrrd", b"FIXED"), "moving": ("m.nrrd", b"MOVING"),
           "fixed_segmentation": ("f_seg.nrrd", b"FIXED_LABEL"),
           "moving_segmentation": ("m_seg.nrrd", b"MOVING_LABEL")}
  response = client.post("/jobs", files=files, data={"params": json.dumps(params)})
  assert response.status_code == 200, response.text
  assert wait_for(client, response.json()["id"], ("done", "failed"))["state"] == "done"
  assert len(captured) == 1
  without_moving = {key: value for key, value in files.items() if key != "moving_segmentation"}
  assert client.post("/jobs", files=without_moving, data={"params": json.dumps(params)}).status_code == 400
  assert client.post("/jobs", files=files, data={"params": json.dumps(dict(params, io_steps=0))}).status_code == 400
  assert client.post("/jobs", files=files, data={"params": json.dumps(PARAMS)}).status_code == 400
  monkeypatch.setattr(server, "guidance_available", lambda: False)
  assert "segmentationGuidance" not in client.get("/info").json()["features"]
  assert client.post("/jobs", files=files, data={"params": json.dumps(params)}).status_code == 400


@pytest.mark.parametrize("weight", [-1, 101, True, None, "0.1", float("nan"), float("inf")])
def test_invalid_dice_weight_is_rejected(weight):
  from fastapi import HTTPException
  with pytest.raises(HTTPException) as error:
    server.validate_params(dict(PARAMS, dice_loss_weight=weight), ["CPU"])
  assert error.value.status_code == 400


@pytest.mark.parametrize("mode", ["input", "loss"])
def test_masking_modes_validate_uploads_and_reach_worker(mode, monkeypatch):
  monkeypatch.setattr(server, "loss_masking_available", lambda: True)
  captured = []
  def masked_register(**kwargs):
    captured.append(kwargs)
    assert Path(kwargs["fixed_mask_path"]).read_bytes() == b"FIXED_MASK"
    assert Path(kwargs["moving_mask_path"]).read_bytes() == b"MOVING_MASK"
    assert kwargs["input_masking"] == (mode == "input")
    assert kwargs["loss_function_masking"] == (mode == "loss")
    return FakeRegistration()(**{key: value for key, value in kwargs.items()
                               if key not in ("moving_mask_path", "input_masking", "loss_function_masking")})
  client = client_for(masked_register)
  info = client.get("/info").json()
  assert "inputMasking" in info["features"] and "lossFunctionMasking" in info["features"]
  params = dict(PARAMS, input_masking=mode == "input", loss_function_masking=mode == "loss")
  files = {"fixed": ("f.nrrd", b"FIXED"), "moving": ("m.nrrd", b"MOVING"),
           "fixed_mask": ("f_mask.nrrd", b"FIXED_MASK"), "moving_mask": ("m_mask.nrrd", b"MOVING_MASK")}
  response = client.post("/jobs", files=files, data={"params": json.dumps(params)})
  assert response.status_code == 200, response.text
  assert wait_for(client, response.json()["id"], ("done", "failed"))["state"] == "done"
  assert len(captured) == 1
  missing = {key: value for key, value in files.items() if key != "moving_mask"}
  assert client.post("/jobs", files=missing, data={"params": json.dumps(params)}).status_code == 400
  monkeypatch.setattr(server, "loss_masking_available", lambda: False)
  if mode == "loss":
    assert client.post("/jobs", files=files, data={"params": json.dumps(params)}).status_code == 400
    assert client.post("/jobs", files=files, data={"params": json.dumps(dict(params, io_steps=0))}).status_code == 400


@pytest.mark.parametrize("field", ["input_masking", "loss_function_masking"])
@pytest.mark.parametrize("value", ["true", 1, [], {}])
def test_masking_flags_are_strict_booleans(field, value):
  from fastapi import HTTPException
  with pytest.raises(HTTPException) as error:
    server.validate_params(dict(PARAMS, **{field: value}), ["CPU"])
  assert error.value.status_code == 400


def test_job_runs_and_returns_results():
  fake = FakeRegistration()
  client = client_for(fake)
  job = submit(client).json()
  status = wait_for(client, job["id"], ("done",))
  assert status["progress"]["total"] == 3
  assert fake.calls == [(b"FIXED", b"MOVING", 3, "CPU")]
  assert client.get(f"/jobs/{job['id']}/transform").content == b"transform"
  assert client.get(f"/jobs/{job['id']}/warped").content == b"warped"


def test_fixed_mask_is_passed_on():
  fake = FakeRegistration()
  client = client_for(fake)
  job = client.post(
    "/jobs",
    files={"fixed": ("fixed.nrrd", b"FIXED"), "moving": ("moving.nrrd", b"MOVING"),
           "fixed_mask": ("mask.nrrd", b"MASK")},
    data={"params": json.dumps(PARAMS)},
  ).json()
  wait_for(client, job["id"], ("done",))
  job_without = submit(client).json()
  wait_for(client, job_without["id"], ("done",))
  assert fake.masks == [b"MASK", None]
  assert client.post(
    "/jobs",
    files={"fixed": ("fixed.nrrd", b"F"), "moving": ("moving.nrrd", b"M"), "fixed_mask": ("mask.png", b"X")},
    data={"params": json.dumps(PARAMS)},
  ).status_code == 400


def test_transform_only_job_has_no_warped_image():
  client = client_for(FakeRegistration())
  job = submit(client, dict(PARAMS, warp=False)).json()
  wait_for(client, job["id"], ("done",))
  assert client.get(f"/jobs/{job['id']}/transform").content == b"transform"
  assert client.get(f"/jobs/{job['id']}/warped").status_code == 404
  assert submit(client, dict(PARAMS, warp="no")).status_code == 400


def test_failure_is_reported():
  client = client_for(FakeRegistration(fail=True))
  job = submit(client).json()
  status = wait_for(client, job["id"], ("failed",))
  assert "images do not overlap" in status["error"]
  assert client.get(f"/jobs/{job['id']}/warped").status_code == 409


def test_running_job_can_be_cancelled():
  fake = FakeRegistration()
  fake.release.clear()  # hold the job at its first step
  client = client_for(fake)
  job = submit(client).json()
  wait_for(client, job["id"], ("running",))
  assert client.delete(f"/jobs/{job['id']}").json()["state"] == "cancelling"
  fake.release.set()
  deadline = time.time() + 5
  while client.get(f"/jobs/{job['id']}").status_code != 404:
    assert time.time() < deadline, "cancelled job was never removed"
    time.sleep(0.02)


def test_queued_jobs_report_their_position():
  fake = FakeRegistration()
  fake.release.clear()
  client = client_for(fake)
  first = submit(client).json()
  wait_for(client, first["id"], ("running",))
  second = submit(client).json()
  third = submit(client).json()
  assert client.get(f"/jobs/{third['id']}").json()["queuePosition"] == 1
  assert client.delete(f"/jobs/{second['id']}").json()["state"] == "removed"
  assert client.get(f"/jobs/{third['id']}").json()["queuePosition"] == 0
  fake.release.set()
  wait_for(client, third["id"], ("done",))
  assert len(fake.calls) == 2  # the cancelled one never ran


def test_jobs_run_concurrently_on_each_gpu():
  fake = FakeRegistration()
  fake.release.clear()
  client = TestClient(server.create_app(register=fake, devices=["CPU", "GPU"], gpus=2, jobs_per_gpu=2))
  info = client.get("/info").json()
  assert (info["gpus"], info["jobsPerGpu"]) == (2, 2)
  gpu_jobs = [submit(client, dict(PARAMS, device="GPU")).json() for _ in range(5)]
  cpu_job = submit(client).json()
  for job in gpu_jobs[:4] + [cpu_job]:
    wait_for(client, job["id"], ("running",))
  assert client.get(f"/jobs/{gpu_jobs[4]['id']}").json()["queuePosition"] == 0  # CPU jobs do not count
  fake.release.set()
  for job in gpu_jobs + [cpu_job]:
    wait_for(client, job["id"], ("done",))
  assert len(fake.calls) == 6


def test_invalid_parameters_are_rejected():
  client = client_for(FakeRegistration())
  assert submit(client, dict(PARAMS, device="GPU")).status_code == 400
  assert submit(client, dict(PARAMS, io_steps=-1)).status_code == 400
  assert submit(client, dict(PARAMS, fixed_modality="PET")).status_code == 400
  assert submit(client, fixed_name="fixed.png").status_code == 400


@pytest.mark.parametrize("params", [[], None, "string", 42, True])
def test_non_object_parameters_are_rejected(params):
  client = client_for(FakeRegistration())
  response = submit(client, params)
  assert response.status_code == 400
  assert response.json()["detail"] == "'params' must be a JSON object"


def test_token_is_required_when_set():
  client = client_for(FakeRegistration(), token="secret")
  assert client.get("/info").status_code == 401
  assert client.get("/info", headers={"Authorization": "Bearer wrong"}).status_code == 401
  headers = {"Authorization": "Bearer secret"}
  assert client.get("/info", headers=headers).status_code == 200
  job = submit(client, headers=headers).json()
  wait_for(client, job["id"], ("done",), headers=headers)


def test_retry_after_lost_response_runs_registration_once(tmp_path, monkeypatch):
  fake = FakeRegistration()
  client = client_for(fake)
  image = tmp_path / "image.nrrd"
  image.write_bytes(b"IMAGE")
  responses = []
  keys = []

  def request(method, url, **kwargs):
    keys.append(kwargs["data"]["request_id"])
    response = client.request(method, url, files=kwargs["files"], data=kwargs["data"])
    response.ok = response.status_code < 400
    responses.append(response.json())
    if len(responses) == 1:
      raise remote.requests.ReadTimeout("response lost after server accepted the upload")
    return response

  monkeypatch.setattr(remote.requests, "request", request)
  monkeypatch.setattr(remote, "TRANSFER_RETRY_DELAYS", (0,))
  result = remote.RegistrationServerClient("http://testserver").submit(str(image), str(image), PARAMS)
  wait_for(client, result["id"], ("done",))
  assert keys[0] == keys[1]
  assert responses[0]["id"] == responses[1]["id"] == result["id"]
  assert len(fake.calls) == 1


def test_concurrent_duplicate_submissions_enqueue_only_one_job(monkeypatch):
  fake = FakeRegistration()
  client = client_for(fake)
  barrier = threading.Barrier(2)
  copy = server.shutil.copyfileobj
  folders = []

  def overlap(source, target):
    folders.append(Path(target.name).parent)
    copy(source, target)
    barrier.wait(timeout=5)

  monkeypatch.setattr(server.shutil, "copyfileobj", overlap)
  key = uuid.uuid4().hex
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(submit, client, request_id=key) for _ in range(2)]
    jobs = [future.result(timeout=10).json() for future in futures]
  assert jobs[0]["id"] == jobs[1]["id"]
  wait_for(client, jobs[0]["id"], ("done",))
  assert len(fake.calls) == 1
  assert sum(path.exists() for path in set(folders)) == 1
  assert submit(client, dict(PARAMS, io_steps=4), request_id=key).status_code == 409
  assert submit(client, request_id="invalid").status_code == 400


def test_idle_server_expires_finished_jobs_and_removes_files(monkeypatch):
  monkeypatch.setattr(server, "JOB_LIFETIME_SECONDS", 0.05)
  monkeypatch.setattr(server, "JOB_CLEANUP_INTERVAL_SECONDS", 0.01)
  fake = FakeRegistration()
  folders = []
  copy = server.shutil.copyfileobj

  def record_folder(source, target):
    folders.append(Path(target.name).parent)
    copy(source, target)

  monkeypatch.setattr(server.shutil, "copyfileobj", record_folder)
  client = client_for(fake)
  job = submit(client).json()
  wait_for(client, job["id"], ("done",))
  deadline = time.monotonic() + 5
  while client.get(f"/jobs/{job['id']}").status_code != 404:
    assert time.monotonic() < deadline
    time.sleep(0.01)
  assert all(not path.exists() for path in folders)


def test_expiry_preserves_running_and_queued_jobs(monkeypatch):
  monkeypatch.setattr(server, "JOB_LIFETIME_SECONDS", 0)
  monkeypatch.setattr(server, "JOB_CLEANUP_INTERVAL_SECONDS", 0.01)
  fake = FakeRegistration()
  fake.release.clear()
  client = client_for(fake)
  first = submit(client).json()
  wait_for(client, first["id"], ("running",))
  second = submit(client).json()
  time.sleep(0.05)
  assert client.get(f"/jobs/{first['id']}").json()["state"] == "running"
  assert client.get(f"/jobs/{second['id']}").json()["state"] == "queued"
  client.delete(f"/jobs/{second['id']}")
  client.delete(f"/jobs/{first['id']}")
  fake.release.set()


@pytest.mark.parametrize("name", ["image.png", "image.mhd", "image.nhdr"])
def test_invalid_file_names_allocate_no_job_folder(name, monkeypatch):
  client = client_for(FakeRegistration())
  def unwanted(*args, **kwargs):
    raise AssertionError("rejected uploads must not allocate a directory")
  monkeypatch.setattr(server.tempfile, "mkdtemp", unwanted)
  response = submit(client, fixed_name=name)
  assert response.status_code == 400
  if name.endswith((".mhd", ".nhdr")):
    assert "companion data files" in response.json()["detail"]


def test_partial_upload_failure_removes_job_folder(monkeypatch):
  client = client_for(FakeRegistration())
  folders = []
  def fail(source, target):
    folders.append(Path(target.name).parent)
    target.write(b"partial")
    raise OSError("disk full")
  monkeypatch.setattr(server.shutil, "copyfileobj", fail)
  with pytest.raises(OSError, match="disk full"):
    submit(client)
  assert folders and all(not path.exists() for path in folders)


@pytest.mark.parametrize("value", [True, 1.5, 3.0, None, float("inf"), float("nan"), [], {}])
def test_io_steps_reject_non_integer_values(value):
  response = submit(client_for(FakeRegistration()), dict(PARAMS, io_steps=value))
  assert response.status_code == 400
  assert "integer" in response.json()["detail"]


@pytest.mark.parametrize("name,value", [("model", []), ("model", {}), ("device", []), ("loss", {})])
def test_invalid_choice_types_are_client_errors(name, value):
  response = submit(client_for(FakeRegistration()), dict(PARAMS, **{name: value}))
  assert response.status_code == 400


def test_multilabel_feature_negotiation_and_parameter_validation(monkeypatch):
  monkeypatch.setattr(server, "guidance_available", lambda: True)
  info = client_for(FakeRegistration()).get("/info").json()
  assert "multiLabelSegmentation" in info["features"]
  values = dict(PARAMS, io_steps=1, dice_loss_weight=10, multilabel_segmentation=True)
  assert server.validate_params(values, ["CPU"])["multilabel_segmentation"] is True
  for flag, weight in [("yes", 10), (1, 10), (True, 0)]:
    with pytest.raises(server.HTTPException, match="multilabel_segmentation"):
      server.validate_params(dict(values, multilabel_segmentation=flag, dice_loss_weight=weight), ["CPU"])
