"""Masking modes, optional upstream Dice/loss masking, and label propagation."""
import inspect
import math


UNIGRADICON_VERSION = "1.0.6"
UNIGRADICON_REQUIREMENT = f"unigradicon=={UNIGRADICON_VERSION}"
MINIMUM_ICON_VERSION = "1.1.9"
DEFAULT_DICE_WEIGHT = 10.0


GUIDANCE_REQUIREMENT = (
  f"Segmentation-guided IO needs uniGradICON {UNIGRADICON_VERSION} with Dice-loss support. "
  f"Install {UNIGRADICON_REQUIREMENT} on the processing computer and restart Slicer or the registration server. "
  "Ordinary registration, input masking, and segmentation propagation do not need Dice support."
)


def backend_packages_current():
  from importlib import metadata
  from packaging.version import Version, InvalidVersion
  try:
    return (metadata.version("unigradicon") == UNIGRADICON_VERSION
            and Version(metadata.version("icon_registration")) >= Version(MINIMUM_ICON_VERSION))
  except (metadata.PackageNotFoundError, InvalidVersion):
    return False


def label_values(values):
  import numpy as np
  values = np.asarray(values)
  if (not np.isfinite(values).all() or np.any(values < 0)
      or np.any(values > 2147483647) or np.any(values != np.floor(values))):
    raise ValueError("Segmentation labels must be finite, nonnegative integers up to 2147483647.")
  return values.astype(np.uint32)


def compact_label_pair(first, second):
  import numpy as np
  first, second = label_values(first), label_values(second)
  shared = np.intersect1d(np.unique(first), np.unique(second))
  shared = shared[shared != 0]
  if not len(shared):
    raise ValueError("Dice guidance requires at least one shared non-background label on the image grids.")
  def remap(values):
    result = np.zeros(values.shape, np.float32)
    for index, label in enumerate(shared, 1):
      result[values == label] = index
    return result
  return remap(first), remap(second)


def validate_weight(weight):
  if (isinstance(weight, bool) or not isinstance(weight, (int, float))
      or not math.isfinite(weight) or not 0 <= weight <= 100):
    raise ValueError("Dice loss weight must be a finite number between 0 and 100.")
  return weight


def validate_masking(input_masking=None, loss_function_masking=False, io_steps=None):
  if not isinstance(loss_function_masking, bool) or (input_masking is not None and not isinstance(input_masking, bool)):
    raise ValueError("Masking options must be true or false.")
  if loss_function_masking and io_steps is not None and io_steps <= 0:
    raise ValueError("Loss-function masking requires at least one IO step.")


def guidance_backend():
  try:
    import unigradicon
    if "dice_loss_weight" in inspect.signature(unigradicon.make_network).parameters:
      return unigradicon
  except (ImportError, AttributeError, ValueError):
    pass
  raise RuntimeError(GUIDANCE_REQUIREMENT)


def guidance_available():
  try:
    guidance_backend()
    return True
  except RuntimeError:
    return False


def masking_backend():
  try:
    import unigradicon
    import icon_registration as icon
    if ({"loss_function_masking", "dice_loss_weight"} <= inspect.signature(unigradicon.make_network).parameters.keys()
        and all("mask_B" in inspect.signature(loss.__call__).parameters
                for loss in (icon.LNCC, icon.SquaredLNCC, icon.MINDSSC))):
      return unigradicon
  except (ImportError, AttributeError, ValueError):
    pass
  raise RuntimeError(f"Loss-function masking needs uniGradICON {UNIGRADICON_VERSION} with loss_function_masking support "
                     f"and icon_registration >= {MINIMUM_ICON_VERSION}. Install {UNIGRADICON_REQUIREMENT} "
                     "on the processing computer and restart Slicer or the registration server.")


def loss_masking_available():
  try:
    masking_backend()
    return True
  except RuntimeError:
    return False


MASKING_MODES = {"none": "None", "input": "Input masking", "loss": "Loss-function masking"}


def masking_mode(options):
  mode = options.get("maskingMode", "input" if options.get("mask") else "none")
  if mode not in MASKING_MODES:
    raise ValueError("Choose None, Input masking, or Loss-function masking.")
  return mode


def registration_options(options):
  mode = masking_mode(options)
  settings = {}
  if mode != "none":
    settings.update(input_masking=mode == "input", loss_function_masking=mode == "loss")
  if options.get("guidance"):
    settings["dice_loss_weight"] = options["weight"]
    settings["multilabel_segmentation"] = True
  return settings


def require_server_features(info, settings):
  features = info.get("features", [])
  if settings.get("multilabel_segmentation") and "multiLabelSegmentation" not in features:
    raise ValueError("This server does not preserve multiple segmentation labels. Update its extension server code.")
  if settings.get("dice_loss_weight") and "segmentationGuidance" not in features:
    raise ValueError("This server does not support segmentation-guided IO. Update its extension and uniGradICON backend.")
  if settings.get("loss_function_masking") and "lossFunctionMasking" not in features:
    raise ValueError("This server does not support loss-function masking. Update its extension and uniGradICON backend.")
  if settings.get("input_masking") and "inputMasking" not in features:
    raise ValueError("This server does not support input masking of both images. Update its server code.")


def propagate(paths, reference, transform_path, check_cancelled):
  import SimpleITK as sitk
  transform = sitk.ReadTransform(transform_path)
  images = []
  for path in paths:
    check_cancelled()
    image = path if isinstance(path, sitk.Image) else sitk.ReadImage(path, sitk.sitkUInt8)
    images.append(sitk.Resample(image, reference, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8))
  return images
