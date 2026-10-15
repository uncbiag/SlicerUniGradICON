"""uniGradICON registration without Slicer, shared by the module and the registration server.

Both use the same registration code and models, whether running inside Slicer or on a server.
"""

import os
import queue
import tempfile
import urllib.request

INPUT_SHAPE = [1, 1, 175, 175, 175]

# model name -> (weights file name, download URL)
MODEL_WEIGHTS = {
  "unigradicon": (
    "unigradicon_weights.pth",
    "https://github.com/uncbiag/uniGradICON/releases/download/unigradicon_weights/Step_2_final.trch",
  ),
  "multigradicon": (
    "multigradicon_weights.pth",
    "https://github.com/uncbiag/uniGradICON/releases/download/multigradicon_weights/Step_2_final.trch",
  ),
}

DEVICE_CPU = "CPU"
DEVICE_GPU = "GPU"

_models = {}  # (model name, device) -> network, loaded once


class RegistrationCancelled(Exception):
  """Raised from the progress callback to stop a registration between optimization steps."""


def available_devices():
  import torch
  return [DEVICE_CPU, DEVICE_GPU] if torch.cuda.is_available() else [DEVICE_CPU]


def gpu_count():
  import torch
  return torch.cuda.device_count()


def available_models(weights_dir):
  return [name for name, (file_name, _) in MODEL_WEIGHTS.items()
          if os.path.exists(os.path.join(weights_dir, file_name))]


def ensure_weights(weights_dir, model_name, log=print):
  """Path of the weights of a model, downloaded first when they are missing."""
  file_name, url = MODEL_WEIGHTS[model_name]
  path = os.path.join(weights_dir, file_name)
  if not os.path.exists(path):
    os.makedirs(weights_dir, exist_ok=True)
    log(f"Downloading {model_name} weights from {url}")
    # Each worker owns its download. Publish only a complete file, so another
    # process can safely load the weights while concurrent downloads finish.
    descriptor, partial_path = tempfile.mkstemp(prefix=file_name + ".", suffix=".part", dir=weights_dir)
    os.close(descriptor)
    try:
      urllib.request.urlretrieve(url, partial_path)
      os.replace(partial_path, path)
    finally:
      if os.path.exists(partial_path):
        os.remove(partial_path)
  return path


def load_model(weights_dir, model_name, loss, device, phase_callback=None):
  """The network of a model on a device, with the similarity loss set; loaded once per pair."""
  import torch
  import icon_registration as icon
  from UniGradICONHelpers import icon_helper

  torch_device = "cuda" if device == DEVICE_GPU else "cpu"
  key = (model_name, torch_device)
  model = _models.get(key)
  if model is None:
    if phase_callback and model_name not in available_models(weights_dir):
      phase_callback("Downloading model weights")
    weights_path = ensure_weights(weights_dir, model_name)
    if phase_callback:
      phase_callback("Loading model")
    model = icon_helper.make_network(INPUT_SHAPE, include_last_step=True, loss_fn=icon.LNCC(sigma=5), device=torch_device)
    model.regis_net.load_state_dict(torch.load(weights_path, map_location=torch_device, weights_only=True))
    model.to(torch_device)
    model.device = torch_device
    _models[key] = model
  model.similarity = icon_helper.make_sim(loss)
  model.eval()
  return model


def binary_mask_on(mask, reference):
  """A mask as 0/1 floats on the grid of the reference image: nonzero labels count as inside."""
  import itk
  import numpy as np

  if (tuple(mask.GetLargestPossibleRegion().GetSize()) != tuple(reference.GetLargestPossibleRegion().GetSize())
      or not np.allclose(mask.GetOrigin(), reference.GetOrigin())
      or not np.allclose(mask.GetSpacing(), reference.GetSpacing())
      or not np.allclose(np.array(mask.GetDirection()), np.array(reference.GetDirection()))):
    mask = itk.resample_image_filter(
      mask,
      interpolator=itk.NearestNeighborInterpolateImageFunction.New(mask),
      use_reference_image=True,
      reference_image=reference,
    )
  binary = itk.image_from_array((itk.array_from_image(mask) != 0).astype(np.float32))
  binary.CopyInformation(reference)
  return binary


def register_images(model, fixed, moving, fixed_modality, moving_modality, io_steps=0, call_back=None, warp=True,
                    fixed_mask=None, phase_callback=None):
  """Register two itk images. Returns the moving-to-fixed transform and the warped moving image,
  which is None when ``warp`` is off. A ``fixed_mask`` limits the fixed image to the region of interest."""
  from UniGradICONHelpers import icon_helper

  if fixed_mask is not None:
    fixed_mask = binary_mask_on(fixed_mask, fixed)
  if phase_callback:
    phase_callback("Preparing images")
  moving_input = icon_helper.preprocess(moving, moving_modality)
  fixed_input = icon_helper.preprocess(fixed, fixed_modality, segmentation=fixed_mask)
  if phase_callback:
    phase_callback("Registering")
  phi_AB, _ = icon_helper.register_pair(
    model,
    moving_input,
    fixed_input,
    finetune_steps=None if io_steps == 0 else io_steps,
    call_back=call_back,
  )
  if not warp:
    return phi_AB, None
  if phase_callback:
    phase_callback("Resampling")
  return phi_AB, resample_moving_image(moving, fixed, phi_AB)


def resample_moving_image(moving, fixed, transform):
  """ITK linear resampling of the original moving image onto the fixed grid."""
  import itk

  moving = itk.CastImageFilter[type(moving), itk.Image[itk.F, 3]].New()(moving)
  interpolator = itk.LinearInterpolateImageFunction.New(moving)
  warped = itk.resample_image_filter(
    moving,
    transform=transform,
    interpolator=interpolator,
    use_reference_image=True,
    reference_image=fixed,
  )
  return warped


def process_exists(pid):
  """Whether a process is still running; os.getppid() does not tell on Windows."""
  if os.name == "nt":
    import ctypes
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
      return False
    exit_code = ctypes.c_ulong()
    kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
    kernel32.CloseHandle(handle)
    return exit_code.value == STILL_ACTIVE
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    pass  # exists, owned by somebody else
  return True


def serve_registrations(requests, messages, cancel_flag, gpu=None):
  """Loop of the registration server's worker process: one registration at a time.

  Takes keyword arguments of register_files from ``requests`` (None stops it) and reports on
  ``messages``: ("progress", {...}), ("done", (transform_path, warped_path)), ("cancelled", None)
  or ("failed", message). A set ``cancel_flag`` stops a registration at its next step.
  It also stops when the server is gone, so a killed server does not leave it behind.
  ``gpu`` is the index of the only GPU it uses.
  """
  if gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # before PyTorch is imported
  server = os.getppid()
  while True:
    try:
      arguments = requests.get(timeout=1)
    except queue.Empty:
      if not process_exists(server):
        return
      continue
    if arguments is None:
      return

    def call_back(total, step, elapsed=None, remaining=None):
      if cancel_flag.value:
        raise RegistrationCancelled()
      messages.put(("progress", {"total": total, "step": step, "elapsed": elapsed, "remaining": remaining}))

    def phase_callback(phase):
      messages.put(("phase", phase))

    try:
      messages.put(("done", register_files(**arguments, call_back=call_back, phase_callback=phase_callback)))
    except RegistrationCancelled:
      messages.put(("cancelled", None))
    except Exception as e:
      messages.put(("failed", f"{type(e).__name__}: {e}"))


def register_files(fixed_path, moving_path, fixed_modality, moving_modality, model_name, loss, io_steps, device,
                   weights_dir, out_dir, call_back=None, warp=True, fixed_mask_path=None, phase_callback=None,
                   transform_format="tfm"):
  """Register two image files; writes a TFM or HDF5 transform and, if ``warp``, warped.nrrd.

  Returns their paths, None for the warped image when it was not made.
  """
  import itk

  if transform_format not in ("tfm", "h5"):
    raise ValueError("Transform format must be tfm or h5")
  if phase_callback:
    phase_callback("Loading model")
  model = load_model(weights_dir, model_name, loss, device, phase_callback=phase_callback)
  fixed = itk.imread(fixed_path)
  moving = itk.imread(moving_path)
  fixed_mask = itk.imread(fixed_mask_path) if fixed_mask_path else None
  phi_AB, warped = register_images(model, fixed, moving, fixed_modality, moving_modality, io_steps, call_back, warp,
                                   fixed_mask, phase_callback=phase_callback)
  if phase_callback:
    phase_callback("Saving results")
  transform_path = os.path.join(out_dir, "transform." + transform_format)
  itk.transformwrite([phi_AB], transform_path)
  if warped is None:
    return transform_path, None
  warped_path = os.path.join(out_dir, "warped.nrrd")
  itk.imwrite(warped, warped_path, compression=True)
  return transform_path, warped_path
