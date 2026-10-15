"""Backend regressions; run with pytest from the repository root.

The optimization tests need torch, icon_registration and itk. Download tests
use fake transfers and need none of those packages or model weights.
"""

import concurrent.futures
import multiprocessing
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from UniGradICONHelpers import registration


def test_upload_closes_already_open_files_when_another_input_is_missing(tmp_path, monkeypatch):
  from UniGradICONHelpers import remote
  image = tmp_path / "fixed.nrrd"
  image.write_bytes(b"IMAGE")
  opened = []
  original_open = open

  def track(path, mode):
    stream = original_open(path, mode)
    opened.append(stream)
    return stream

  monkeypatch.setattr(remote, "open", track, raising=False)
  with pytest.raises(FileNotFoundError):
    remote.RegistrationServerClient("http://unused").submit(
      str(image), str(tmp_path / "missing.nrrd"), {})
  assert len(opened) == 1 and opened[0].closed


@pytest.mark.parametrize("value", [0.0, 7.0])
def test_constant_mri_has_clear_validation_error(value):
  itk = pytest.importorskip("itk")
  np = pytest.importorskip("numpy")
  from UniGradICONHelpers import icon_helper
  image = itk.image_from_array(np.full((4, 4, 4), value, dtype=np.float32))
  with pytest.raises(ValueError, match="MRI image has no usable intensity range"):
    icon_helper.preprocess(image, "MRI")


def test_mri_normalization_preserves_existing_percentile_and_geometry():
  itk = pytest.importorskip("itk")
  np = pytest.importorskip("numpy")
  from UniGradICONHelpers import icon_helper
  values = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
  image = itk.image_from_array(values)
  image.SetOrigin((13, -7, 42))
  image.SetSpacing((0.5, 1.5, 2.5))
  normalized = icon_helper.preprocess(image, "MRI")
  np.testing.assert_allclose(itk.array_from_image(normalized), np.minimum(values, 989) / 989, atol=1e-7)
  assert tuple(normalized.GetOrigin()) == tuple(image.GetOrigin())
  assert tuple(normalized.GetSpacing()) == tuple(image.GetSpacing())


def test_single_voxel_mri_reports_intensity_error_instead_of_invalid_quantile():
  itk = pytest.importorskip("itk")
  np = pytest.importorskip("numpy")
  from UniGradICONHelpers import icon_helper
  image = itk.image_from_array(np.ones((1, 1, 1), dtype=np.float32))
  with pytest.raises(ValueError, match="MRI image has no usable intensity range"):
    icon_helper.preprocess(image, "MRI")


@pytest.mark.parametrize("modality", ["MRI", "CT/CBCT"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_voxels_are_rejected_even_outside_percentile_range(modality, value):
  itk = pytest.importorskip("itk")
  np = pytest.importorskip("numpy")
  from UniGradICONHelpers import icon_helper
  data = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
  data.flat[-1] = value
  with pytest.raises(ValueError, match="non-finite intensities"):
    icon_helper.preprocess(itk.image_from_array(data), modality)


def test_empty_mask_has_clear_validation_error():
  itk = pytest.importorskip("itk")
  np = pytest.importorskip("numpy")
  from UniGradICONHelpers import icon_helper
  image = itk.image_from_array(np.arange(64, dtype=np.float32).reshape(4, 4, 4))
  mask = itk.image_from_array(np.zeros((4, 4, 4), dtype=np.float32))
  with pytest.raises(ValueError, match="no intensity variation"):
    icon_helper.preprocess(image, "MRI", segmentation=mask)


def test_itk_resampling_preserves_linear_field_on_fixed_grid():
  itk = pytest.importorskip("itk")
  np = pytest.importorskip("numpy")
  z, y, x = np.indices((8, 7, 6))
  values = 2 * (-10 + 1.5 * x) + 3 * (-10 + 2 * y) - 0.5 * (-10 + 2.5 * z) + 17
  moving = itk.image_from_array(values.astype(np.float64))
  moving.SetOrigin((-10, -10, -10))
  moving.SetSpacing((1.5, 2, 2.5))
  fixed = itk.image_from_array(np.zeros((2, 4, 3), dtype=np.int16))
  fixed.SetOrigin((-6, -5, -3))
  transform = itk.TranslationTransform[itk.D, 3].New()
  transform.SetOffset((-1, 0.5, -1.5))
  output = registration.resample_moving_image(moving, fixed, transform)
  z, y, x = np.indices((2, 4, 3))
  expected = 2 * (-6 + x - 1) + 3 * (-5 + y + 0.5) - 0.5 * (-3 + z - 1.5) + 17
  actual = itk.array_from_image(output)
  np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)
  assert actual.dtype == np.dtype(np.float32)
  assert tuple(output.GetOrigin()) == tuple(fixed.GetOrigin())
  assert tuple(output.GetSpacing()) == tuple(fixed.GetSpacing())
  np.testing.assert_array_equal(np.array(output.GetDirection()), np.array(fixed.GetDirection()))


def test_transform_only_registration_skips_itk_resampling(monkeypatch):
  pytest.importorskip("itk")
  pytest.importorskip("torch")
  pytest.importorskip("icon_registration")
  from UniGradICONHelpers import icon_helper
  monkeypatch.setattr(icon_helper, "preprocess", lambda image, modality, **kwargs: image)
  monkeypatch.setattr(icon_helper, "register_pair", lambda *args, **kwargs: ("transform", None))

  def unwanted(*args):
    raise AssertionError("transform-only registration must not resample")

  monkeypatch.setattr(registration, "resample_moving_image", unwanted)
  assert registration.register_images(None, "fixed", "moving", "MRI", "MRI", warp=False) == ("transform", None)


def test_overlapping_weight_downloads_publish_complete_files(tmp_path, monkeypatch):
  barrier = threading.Barrier(2)
  downloads = []

  def download(url, path):
    downloads.append(path)
    Path(path).write_bytes(b"complete weights")
    barrier.wait(timeout=5)

  monkeypatch.setattr(registration.urllib.request, "urlretrieve", download)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(registration.ensure_weights, str(tmp_path), "unigradicon") for _ in range(2)]
    paths = [f.result(timeout=10) for f in futures]
  assert paths[0] == paths[1]
  assert len(set(downloads)) == 2
  assert Path(paths[0]).read_bytes() == b"complete weights"
  assert list(tmp_path.glob("*.part")) == []


def _download_in_process(folder, barrier, results):
  def download(url, path):
    Path(path).write_bytes(b"complete weights")
    barrier.wait(timeout=10)

  registration.urllib.request.urlretrieve = download
  try:
    results.put((registration.ensure_weights(folder, "unigradicon"), None))
  except Exception as error:
    results.put((None, repr(error)))


def test_weight_downloads_from_separate_workers(tmp_path):
  context = multiprocessing.get_context("spawn")
  barrier = context.Barrier(2)
  results = context.Queue()
  processes = [context.Process(target=_download_in_process, args=(str(tmp_path), barrier, results))
               for _ in range(2)]
  try:
    for process in processes:
      process.start()
    received = [results.get(timeout=20) for _ in processes]
    for process in processes:
      process.join(timeout=5)
      assert process.exitcode == 0
    assert all(error is None for _, error in received), received
    assert Path(received[0][0]).read_bytes() == b"complete weights"
    assert list(tmp_path.glob("*.part")) == []
  finally:
    for process in processes:
      if process.is_alive():
        process.terminate()
        process.join()


def test_failed_weight_download_is_cleaned_up_and_can_be_retried(tmp_path, monkeypatch):
  def fail(url, path):
    Path(path).write_bytes(b"incomplete")
    raise OSError("interrupted transfer")

  monkeypatch.setattr(registration.urllib.request, "urlretrieve", fail)
  with pytest.raises(OSError, match="interrupted transfer"):
    registration.ensure_weights(str(tmp_path), "unigradicon")
  assert list(tmp_path.iterdir()) == []

  monkeypatch.setattr(registration.urllib.request, "urlretrieve",
                      lambda url, path: Path(path).write_bytes(b"complete"))
  path = registration.ensure_weights(str(tmp_path), "unigradicon")
  assert Path(path).read_bytes() == b"complete"


@pytest.mark.parametrize("failure", ["cancel", "error", None])
def test_optimization_restores_weights_for_the_next_registration(failure):
  torch = pytest.importorskip("torch")
  pytest.importorskip("icon_registration")
  pytest.importorskip("itk")
  from UniGradICONHelpers import icon_helper

  class TinyModel(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, a, b):
      return (self.weight.square(),)

  model = TinyModel()
  original = model.weight.detach().clone()
  baseline = model(None, None)[0].detach().clone()
  calls = []

  def progress(total, step, elapsed, remaining):
    calls.append(step)
    if step == 1 and failure:
      assert not torch.equal(model.weight.detach(), original)
      if failure == "cancel":
        raise registration.RegistrationCancelled()
      raise RuntimeError("registration failed")

  if failure:
    exception = registration.RegistrationCancelled if failure == "cancel" else RuntimeError
    with pytest.raises(exception):
      icon_helper.finetune_execute(model, None, None, 2, progress)
  else:
    icon_helper.finetune_execute(model, None, None, 2, progress)
  assert calls == [0, 1]
  assert torch.equal(model.weight.detach(), original)
  assert torch.equal(model(None, None)[0].detach(), baseline)


@pytest.mark.parametrize("fail", [False, True])
def test_registration_scopes_deterministic_convolutions(monkeypatch, fail):
  torch = pytest.importorskip("torch")
  pytest.importorskip("icon_registration")
  pytest.importorskip("itk")
  from UniGradICONHelpers import icon_helper

  enabled = torch.backends.cudnn.enabled
  tf32 = torch.backends.cudnn.allow_tf32

  def run(*args):
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert torch.backends.cudnn.enabled == enabled
    assert torch.backends.cudnn.allow_tf32 == tf32
    if fail:
      raise RuntimeError("registration failed")
    return "result"

  monkeypatch.setattr(icon_helper, "_register_pair", run)
  with torch.backends.cudnn.flags(enabled=enabled, benchmark=True, deterministic=False, allow_tf32=tf32):
    if fail:
      with pytest.raises(RuntimeError, match="registration failed"):
        icon_helper.register_pair(None, None, None)
    else:
      assert icon_helper.register_pair(None, None, None) == "result"
    assert not torch.backends.cudnn.deterministic
    assert torch.backends.cudnn.benchmark
