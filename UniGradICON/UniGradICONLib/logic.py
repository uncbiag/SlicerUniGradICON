"""Scene/parameter-node adapter for the shared registration backend and batch API."""
import json
import logging
import os

import numpy as np
import qt
import SimpleITK as sitk
import slicer
import vtk
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

from UniGradICONHelpers.segmentation import (
  DEFAULT_DICE_WEIGHT, UNIGRADICON_VERSION, UNIGRADICON_REQUIREMENT, backend_packages_current,
)

from .settings import (
  PresetManager, REMOTE_PROCESSING_SETTING, SERVER_URL_SETTING,
  SERVER_TOKENS_SETTING_GROUP, WEIGHTS_DIRECTORY,
)


def _checkPyTorchNumPyCompatibility():
  """Offer a NumPy downgrade when PyTorch reports a known NumPy ABI mismatch."""
  try:
    import torch
  except ImportError as e:
    error_msg = str(e)
    if "_ARRAY_API not found" in error_msg or "numpy.core.multiarray" in error_msg:
      import numpy as np
      if np.__version__.startswith("2."):
        if not slicer.util.confirmOkCancelDisplay(
          "NumPy 2.x is incompatible with the installed PyTorch.\n\n"
          "Do you want to downgrade NumPy to a compatible version?\n"
          "You will need to restart 3D Slicer after the change.",
          windowTitle="UniGradICON - NumPy compatibility"
        ):
          raise RuntimeError(
            "UniGradICON requires NumPy < 2 for PyTorch compatibility. "
            "You can run 'slicer.util.pip_install(\"numpy<2\")' in the Python Console and restart Slicer."
          )
        slicer.util.pip_install("numpy<2")
        slicer.util.infoDisplay(
          "NumPy has been downgraded for PyTorch compatibility.\n"
          "Please restart 3D Slicer for the changes to take effect.",
          windowTitle="UniGradICON"
        )
        raise RuntimeError(
          "NumPy was downgraded for PyTorch compatibility. "
          "Please restart 3D Slicer."
        )
    raise


class UniGradICONLogic(ScriptedLoadableModuleLogic):
  """Registration API for MRML scenes, usable without the module widget."""

  INPUT_VOLUME_REFS = {"fixed": "FixedVolume", "moving": "MovingVolume"}
  OUTPUT_TRANSFORM_REF = "OutputTransform"
  OUTPUT_VOLUME_REF = "OutputVolume"
  # LOSS_PARAM, IO_PARAM and DEVICE_PARAM are the parameter names (not values) stored in scenes;
  # kept as they are for compatibility with existing scenes and older versions
  LOSS_PARAM = "LNCC"
  SETTINGS_JSON_PARAM = "StagesJson"  # stored name from SlicerANTs; kept for scene compatibility
  STAGES_JSON_PARAM = SETTINGS_JSON_PARAM  # old constant name, for scripts
  IO_PARAM = "0"
  DEVICE_PARAM = "CPU"
  REMOTE_PROCESSING_PARAM = "RemoteProcessing"
  SERVER_URL_PARAM = "ServerUrl"
  CREATE_VOLUME_PARAM = "CreateTransformedVolume"
  SEGMENTATION_REFS = {"fixed": "FixedSegmentation", "moving": "MovingSegmentation"}
  MASK_REFS = {"fixedMask": "FixedMaskSegmentation", "movingMask": "MovingMaskSegmentation"}
  PROPAGATION_REF = "PropagationSegmentation"
  OUTPUT_SEGMENTATION_REF = "OutputSegmentation"

  def __init__(self, weights_location=None):
    ScriptedLoadableModuleLogic.__init__(self)
    self.weights_location = weights_location or WEIGHTS_DIRECTORY
    # the network is loaded on the first local registration, so the module works without
    # PyTorch when it registers on a server
    self.modelName = "unigradicon"
    self.loss = "LNCC"
    self._localBackendReady = False
    self._backendRestartRequired = False
    self.cancelRequested = False

  def ensureLocalBackend(self):
    """Install/check local dependencies. The shared backend downloads the selected model on demand."""
    if self._backendRestartRequired:
      raise RuntimeError("The registration backend was updated. Restart Slicer to load the new packages.")
    if self._localBackendReady:
      return

    try:
      import PyTorchUtils
    except ModuleNotFoundError:
      raise RuntimeError("This module requires PyTorch extension. Install it from the Extensions Manager.")

    minimumTorchVersion = "1.12"
    torchLogic = PyTorchUtils.PyTorchUtilsLogic()
    if not torchLogic.torchInstalled():
      print("PyTorch Python package is required. Installing... (it may take several minutes)")
      if torchLogic.installTorch(askConfirmation=True, torchVersionRequirement=f">={minimumTorchVersion}") is None:
        raise RuntimeError("PyTorch installation was cancelled or failed. Install it using PyTorch Util.")
    else:
      from packaging import version
      if version.parse(torchLogic.torch.__version__) < version.parse(minimumTorchVersion):
        raise RuntimeError(f"PyTorch {torchLogic.torch.__version__} is too old. "
                           f"Install PyTorch >= {minimumTorchVersion} using PyTorch Util.")
    _checkPyTorchNumPyCompatibility()
    if not backend_packages_current():
      if not slicer.util.confirmOkCancelDisplay(
          f"The local backend requires uniGradICON {UNIGRADICON_VERSION}. Click OK to install or update it."):
        raise RuntimeError(f"Install {UNIGRADICON_REQUIREMENT} to use local registration.")
      import sys
      restart_needed = any(name in sys.modules for name in ("unigradicon", "icon_registration"))
      slicer.util.pip_install(UNIGRADICON_REQUIREMENT)
      from importlib import invalidate_caches
      invalidate_caches()
      self._backendRestartRequired = restart_needed
      if restart_needed:
        raise RuntimeError("The registration backend was updated. Restart Slicer to load the new packages.")
    try:
      import unigradicon  # noqa: F401
      import icon_registration  # noqa: F401
      import itk  # noqa: F401
      import UniGradICONHelpers.icon_helper  # noqa: F401
    except (ImportError, AttributeError) as error:
      raise RuntimeError(f"The local backend is incomplete. Reinstall {UNIGRADICON_REQUIREMENT} "
                         "and restart Slicer.") from error
    self._localBackendReady = True

  def setDefaultParameters(self, parameterNode):
    """Initialize defaults and migrate legacy input IDs to MRML references."""
    presetParameters = PresetManager().getPresetParametersByName()
    newSet = not parameterNode.GetParameter(self.SETTINGS_JSON_PARAM)
    if newSet:
      parameterNode.SetParameter(self.SETTINGS_JSON_PARAM, json.dumps(presetParameters))
    if not parameterNode.GetParameter("MaskingMode"):
      parameterNode.SetParameter("MaskingMode", "input" if parameterNode.GetParameter("UseFixedSegmentMask") == "true" else "none")
    # Older scenes store input IDs only in JSON. Promote them to MRML references
    # so future scene imports can remap them along with the output references.
    imageSettings = json.loads(parameterNode.GetParameter(self.SETTINGS_JSON_PARAM))["image"]
    for role, referenceRole in self.INPUT_VOLUME_REFS.items():
      if not parameterNode.GetNodeReferenceID(referenceRole):
        parameterNode.SetNodeReferenceID(referenceRole, imageSettings.get(role) or "")
    if not parameterNode.GetNodeReference(self.OUTPUT_TRANSFORM_REF):
      parameterNode.SetNodeReferenceID(self.OUTPUT_TRANSFORM_REF, "")
    if not parameterNode.GetNodeReference(self.OUTPUT_VOLUME_REF):
      parameterNode.SetNodeReferenceID(self.OUTPUT_VOLUME_REF, "")
    if not parameterNode.GetParameter(self.CREATE_VOLUME_PARAM):
      parameterNode.SetParameter(self.CREATE_VOLUME_PARAM,
                                 "true" if newSet or parameterNode.GetNodeReference(self.OUTPUT_VOLUME_REF) else "false")
    if not parameterNode.GetParameter(self.LOSS_PARAM):
      parameterNode.SetParameter(self.LOSS_PARAM, str(presetParameters["modelSettings"]["loss"]))
    if not parameterNode.GetParameter(self.IO_PARAM):
      parameterNode.SetParameter(self.IO_PARAM, str(presetParameters["modelSettings"]["io_steps"]))
    if not parameterNode.GetParameter(self.DEVICE_PARAM):
      parameterNode.SetParameter(self.DEVICE_PARAM, self.deviceName(presetParameters["modelSettings"]["device"]))
    # where to register: the last choice made in the module is the default for a new node
    settings = qt.QSettings()
    if not parameterNode.GetParameter(self.REMOTE_PROCESSING_PARAM):
      parameterNode.SetParameter(self.REMOTE_PROCESSING_PARAM,
                                 "true" if settings.value(REMOTE_PROCESSING_SETTING) == "true" else "false")
    if not parameterNode.GetParameter(self.SERVER_URL_PARAM):
      parameterNode.SetParameter(self.SERVER_URL_PARAM, settings.value(SERVER_URL_SETTING) or "")

  def isRemoteProcessing(self, parameterNode):
    return parameterNode.GetParameter(self.REMOTE_PROCESSING_PARAM) == "true"

  @staticmethod
  def _serverTokenKey(url):
    from urllib.parse import quote
    from UniGradICONHelpers import remote
    # one address however it was typed, and no slashes, which QSettings reads as groups
    return f"{SERVER_TOKENS_SETTING_GROUP}/{quote(remote.RegistrationServerClient(url).url, safe='')}"

  @classmethod
  def serverToken(cls, url):
    """The token remembered for a server: in this computer's Slicer settings, never in a scene."""
    if not url:
      return ""
    return qt.QSettings().value(cls._serverTokenKey(url)) or ""

  @classmethod
  def setServerToken(cls, url, token):
    if not url:
      return
    if token:
      qt.QSettings().setValue(cls._serverTokenKey(url), token)
    else:
      qt.QSettings().remove(cls._serverTokenKey(url))

  def registerVolumesUsingParameterNode(self, parameterNode, fixedVolumeNode, movingVolumeNode, outputTransformNode,
                                        fixedMaskNode=None, initialTransformNode=None, logCallback=None):
    """Register two volumes with the settings of a parameter node, in this Slicer or on its server.

    For callers without the module's GUI, such as other modules. Writes the transform
    that resamples the moving volume onto the fixed one into outputTransformNode, and blocks until
    done. A fixedMaskNode (labelmap) limits the fixed image to its nonzero region.
    With masking or Dice guidance, harden any initial transform on the moving image
    and its masks/segmentations before calling, rather than passing initialTransformNode.
    """
    import time
    from UniGradICONHelpers import remote
    log = logCallback or logging.info
    self.cancelRequested = False
    parameters = self.createProcessParameters(parameterNode)
    parameters['outputSettings']['transform'] = outputTransformNode
    parameters['outputSettings']['volume'] = None
    parameters['segmentationSettings']['propagate'] = False
    from UniGradICONHelpers.segmentation import masking_mode
    options = parameters['segmentationSettings']
    if initialTransformNode is not None and (options['guidance'] or masking_mode(options) != "none"):
      raise ValueError("For masking or Dice guidance, first apply and harden the initial transform on the moving image "
                       "and its masks/segmentations, then register without an initialTransformNode.")
    temporaryNodes = []
    try:
      movingInput = movingVolumeNode
      if initialTransformNode is not None:
        movingInput = slicer.modules.volumes.logic().CloneVolume(
          slicer.mrmlScene, movingVolumeNode, movingVolumeNode.GetName() + "_initialized")
        temporaryNodes.append(movingInput)
        movingInput.SetAndObserveTransformNodeID(initialTransformNode.GetID())
        slicer.vtkSlicerTransformLogic().hardenTransform(movingInput)
      parameters['image']['fixed'] = fixedVolumeNode
      parameters['image']['moving'] = movingInput

      start = time.time()
      if self.isRemoteProcessing(parameterNode):
        url = parameterNode.GetParameter(self.SERVER_URL_PARAM)
        if not url:
          raise ValueError("Remote processing is selected, but no server address is set.")
        client = remote.RegistrationServerClient(url, self.serverToken(url))
        log(f"Registering {movingVolumeNode.GetName()} to {fixedVolumeNode.GetName()} on {client.url}...")
        lastState = []

        def logStatus(status):
          if status["state"] not in lastState:
            lastState.append(status["state"])
            log(f"  {status['state']} on the server")

        registration = RemoteRegistration(self, client, parameters, statusCallback=logStatus, fixedMask=fixedMaskNode)
        registration.start()
        error, cancelled = registration.waitForCompletion(lambda: self.cancelRequested)
        if cancelled:
          raise ValueError("User requested cancel.")
        if error:
          raise RuntimeError(f"Registration on {client.url} failed: {error}")
      else:
        log(f"Registering {movingVolumeNode.GetName()} to {fixedVolumeNode.GetName()} in this Slicer...")
        self.ensureLocalBackend()

        def logProgress(total, step, elapsed=None, remaining=None):
          if step:
            log(f"  instance optimization step {step}/{total}")

        try:
          self.process(**parameters, call_back=logProgress, fixedMask=fixedMaskNode)
        except Exception as e:
          from UniGradICONHelpers import registration
          if isinstance(e, registration.RegistrationCancelled):
            raise ValueError("User requested cancel.")
          raise

      if initialTransformNode is not None:
        # the result maps the initialized moving volume: put the initial transform under it
        initialCopy = slicer.mrmlScene.AddNewNodeByClass(initialTransformNode.GetClassName())
        temporaryNodes.append(initialCopy)
        initialCopy.CopyContent(initialTransformNode)
        initialCopy.SetAndObserveTransformNodeID(outputTransformNode.GetID())
        slicer.vtkSlicerTransformLogic().hardenTransform(initialCopy)
        outputTransformNode.SetAndObserveTransformToParent(initialCopy.GetTransformToParent())
      log(f"  done in {time.time() - start:.0f} s")
    finally:
      for node in temporaryNodes:
        slicer.mrmlScene.RemoveNode(node)

  def inputVolumeNode(self, parameterNode, role):
    referenceRole = self.INPUT_VOLUME_REFS[role]
    if parameterNode.GetNodeReferenceID(referenceRole):
      return parameterNode.GetNodeReference(referenceRole)
    # Support callers that still populate StagesJson directly, and legacy scenes.
    settings = json.loads(parameterNode.GetParameter(self.SETTINGS_JSON_PARAM))
    nodeID = settings["image"].get(role)
    scene = parameterNode.GetScene() or slicer.mrmlScene
    return scene.GetNodeByID(nodeID) if nodeID else None

  def createProcessParameters(self, paramNode):
    parameters = json.loads(paramNode.GetParameter(self.SETTINGS_JSON_PARAM))
    for role in self.INPUT_VOLUME_REFS:
      node = self.inputVolumeNode(paramNode, role)
      parameters['image'][role] = node.GetID() if node else None

    parameters['outputSettings'] = {}
    parameters['outputSettings']['transform'] = paramNode.GetNodeReference(self.OUTPUT_TRANSFORM_REF)
    parameters['outputSettings']['volume'] = paramNode.GetNodeReference(self.OUTPUT_VOLUME_REF)
    parameters['outputSettings']['loss'] = paramNode.GetParameter(self.LOSS_PARAM)

    parameters['generalSettings'] = {}
    parameters['generalSettings']['io_steps'] = int(paramNode.GetParameter(self.IO_PARAM))
    parameters['generalSettings']['device'] = str(paramNode.GetParameter(self.DEVICE_PARAM))

    parameters['segmentationSettings'] = self.segmentationSettings(paramNode)

    return parameters

  def segmentationSettings(self, node):
    settings = {
      "maskingMode": node.GetParameter("MaskingMode") or
        ("input" if node.GetParameter("UseFixedSegmentMask") == "true" else "none"),
      "guidance": node.GetParameter("UseSegmentationGuidance") == "true",
      "propagate": node.GetParameter("PropagateSegmentation") == "true",
      "weight": float(node.GetParameter("DiceLossWeight") or DEFAULT_DICE_WEIGHT),
      "output": node.GetNodeReference(self.OUTPUT_SEGMENTATION_REF),
    }
    for role, reference in self.SEGMENTATION_REFS.items():
      settings[role] = node.GetNodeReference(reference)
    separate_inputs = node.GetParameter("SegmentationInputsVersion") in ("2", "3")
    for role, reference in self.MASK_REFS.items():
      source_role = role.removesuffix("Mask")
      settings[role] = node.GetNodeReference(reference) if separate_inputs else settings[source_role]
    settings["propagationSource"] = (node.GetNodeReference(self.PROPAGATION_REF)
                                      if separate_inputs else settings["moving"])
    return settings

  def itk2sitk(self, itk_image):
    import itk
    sitkImage = sitk.GetImageFromArray(itk.GetArrayFromImage(itk_image))

    sitkImage.SetOrigin(tuple(itk_image.GetOrigin()))
    sitkImage.SetSpacing(tuple(itk_image.GetSpacing()))
    sitkImage.SetDirection(itk.GetArrayFromMatrix(itk_image.GetDirection()).flatten())

    return sitkImage

  def sitk2itk(self, sitk_image):
    import itk
    itk_image = itk.GetImageFromArray(sitk.GetArrayFromImage(sitk_image))
    image_dimension = 3

    itk_image.SetOrigin(sitk_image.GetOrigin())
    itk_image.SetSpacing(sitk_image.GetSpacing())
    itk_image.SetDirection(itk.GetMatrixFromArray(np.reshape(np.array(sitk_image.GetDirection()), [image_dimension]*2)))

    return itk_image

  def changeModelSettings(self, presetParameters):
    self.modelName = presetParameters['modelSettings']['model']
    self.loss = presetParameters['modelSettings']['loss']

  # the modalities the preset files carry; they never matched the module's lists, which stayed at MRI
  PRESET_MODALITIES = ("mri", "ct")

  @staticmethod
  def modalityName(modality):
    """MRI or CT/CBCT for a modality in any spelling the module or uniGradICON uses, None if unknown."""
    name = str(modality).strip().upper()
    if name in ("MRI", "MR"):
      return "MRI"
    if name in ("CT", "CBCT", "CT/CBCT"):
      return "CT/CBCT"
    return None

  @classmethod
  def modalityNames(cls, fixedModality, movingModality):
    """The fixed and moving modality, as the module and uniGradICON's preprocessing name them.

    Settings still holding the presets' untouched "mri"/"ct" run as MRI/MRI, as they always did: the
    module's modality lists did not have those names and stayed at MRI. Unknown names count as MRI.
    """
    if (fixedModality, movingModality) == cls.PRESET_MODALITIES:
      return "MRI", "MRI"
    names = []
    for modality in (fixedModality, movingModality):
      name = cls.modalityName(modality)
      if name is None:
        logging.warning(f"Unknown modality {modality!r}; registering it as MRI.")
        name = "MRI"
      names.append(name)
    return tuple(names)

  @staticmethod
  def deviceName(device):
    """CPU or GPU, as the module and the server name them; the presets say "cpu"."""
    return "GPU" if str(device).strip().upper() in ("GPU", "CUDA") else "CPU"

  def registrationSettings(self, image, outputSettings, modelSettings=None, generalSettings=None):
    """What a registration runs with, the same for this Slicer and a server."""
    fixedModality, movingModality = self.modalityNames(image['modality-fixed'], image['modality-moving'])
    return {
      "fixed_modality": fixedModality,
      "moving_modality": movingModality,
      "model": (modelSettings or {}).get('model') or self.modelName,
      "loss": outputSettings.get('loss') or self.loss,
      "io_steps": int(generalSettings["io_steps"]),
      "device": self.deviceName(generalSettings['device']),
      # without an output volume there is no need to resample the moving image
      "warp": outputSettings.get('volume') is not None,
    }

  def loadResults(self, outputSettings, transformPath, warpedImage=None):
    """Put a registration result into the output transform node, and volume node if there is one."""
    import sitkUtils
    transformNode = outputSettings['transform']
    outputVolumeNode = outputSettings.get('volume')
    if transformNode is None:
      raise ValueError("An output transform is required.")
    stagedTransform = self.temporaryNode("vtkMRMLTransformNode", "UniGradICON result staging")
    stagedVolume = None
    try:
      storageNode = stagedTransform.CreateDefaultStorageNode()
      storageNode.SetFileName(transformPath)
      if not storageNode.ReadData(stagedTransform):
        raise RuntimeError("The registration transform could not be loaded. Run again with new outputs.")
      if outputVolumeNode is not None:
        if warpedImage is None:
          raise ValueError("The requested transformed volume was not returned.")
        stagedVolume = self.temporaryVolume("UniGradICON volume staging")
        sitkUtils.PushVolumeToSlicer(warpedImage, stagedVolume)
      # Validate both results before replacing either existing output. Results use
      # world coordinates; retaining an old parent would apply that transform twice.
      transformNode.CopyContent(stagedTransform)
      transformNode.SetAndObserveTransformNodeID(None)
      if stagedVolume is not None:
        outputVolumeNode.CopyOrientation(stagedVolume)
        outputVolumeNode.SetAndObserveImageData(stagedVolume.GetImageData())
        outputVolumeNode.SetAndObserveTransformNodeID(None)
        outputVolumeNode.CreateDefaultDisplayNodes()
    finally:
      if stagedVolume is not None:
        self.removeTemporaryVolume(stagedVolume)
      slicer.mrmlScene.RemoveNode(stagedTransform)

  @staticmethod
  def temporaryNode(nodeClass, name):
    node = getattr(slicer, nodeClass)()
    node.SetName(name)
    node.SetHideFromEditors(True)
    node.SetSaveWithScene(False)
    return slicer.mrmlScene.AddNode(node)

  @classmethod
  def temporaryVolume(cls, name):
    return cls.temporaryNode("vtkMRMLScalarVolumeNode", name)

  @staticmethod
  def removeTemporaryVolume(node):
    owned = (node, node.GetDisplayNode(), node.GetStorageNode())
    for item in owned:
      if item is not None and item.GetScene() == slicer.mrmlScene:
        slicer.mrmlScene.RemoveNode(item)

  @staticmethod
  def snapshotVolume(node):
    """Copy scalar voxels with their world geometry, including linear parent transforms."""
    import sitkUtils
    if isinstance(node, str):
      node = slicer.util.getNode(node)
    if node is None or node.GetScene() != slicer.mrmlScene:
      raise ValueError("An input image was removed. Select the images and run again.")
    parent = node.GetParentTransformNode()
    if parent is not None and not parent.IsTransformToWorldLinear():
      raise ValueError(f"Harden the nonlinear parent transform of {node.GetName()} before registration.")
    image = sitkUtils.PullVolumeFromSlicer(node)
    if parent is not None:
      local = vtk.vtkMatrix4x4()
      world = vtk.vtkMatrix4x4()
      node.GetIJKToRASMatrix(local)
      parent.GetMatrixTransformToWorld(world)
      local = np.array([[local.GetElement(row, column) for column in range(4)] for row in range(4)])
      world = np.array([[world.GetElement(row, column) for column in range(4)] for row in range(4)])
      ijkToLPS = np.diag([-1, -1, 1, 1]) @ world @ local
      spacing = np.linalg.norm(ijkToLPS[:3, :3], axis=0)
      if np.any(spacing <= 0) or abs(np.linalg.det(ijkToLPS[:3, :3])) < 1e-12:
        raise ValueError("An input parent transform has singular geometry. Harden or correct it first.")
      image.SetOrigin(tuple(ijkToLPS[:3, 3]))
      image.SetSpacing(tuple(spacing))
      image.SetDirection(tuple((ijkToLPS[:3, :3] / spacing).flatten()))
    return image

  def snapshotRegistrationInputs(self, image, options, fixedMask=None):
    from .segmentation import snapshot_inputs
    snapshots = {role: self.snapshotVolume(image[role]) for role in ("fixed", "moving")}
    snapshot_inputs(snapshots, options)
    if fixedMask is not None and "fixed_mask" not in snapshots:
      snapshots["fixed_mask"] = self.snapshotVolume(fixedMask)
    return snapshots

  def process(self, image, outputSettings, modelSettings=None, generalSettings=None, wait_for_completion=False, call_back=None,
              fixedMask=None, segmentationSettings=None):
    """Register in this Slicer, with the dictionaries createProcessParameters makes.

    :param image: fixed and moving volume nodes (or IDs) and their modalities ('modality-fixed', 'modality-moving')
    :param outputSettings: output transform node, optional output volume node, and the similarity loss
    :param modelSettings: the model ('unigradicon' or 'multigradicon'), as in the preset files
    :param generalSettings: instance optimization steps ('io_steps') and device ('CPU' or 'GPU')
    :param wait_for_completion: unused; the registration always runs to completion
    :param call_back: progress callback(total, step[, elapsed, remaining])
    :param fixedMask: optional labelmap that limits the fixed image to its nonzero region
    :param segmentationSettings: optional dictionary from segmentationSettings(parameterNode),
      with mask/guidance sources, flags and propagation output
    """

    import itk
    from UniGradICONHelpers import registration

    settings = self.registrationSettings(image, outputSettings, modelSettings, generalSettings)
    from UniGradICONHelpers.segmentation import masking_mode
    options = segmentationSettings or {}
    mode = masking_mode(options)
    if (options.get("guidance") or mode == "loss") and settings["io_steps"] <= 0:
      raise ValueError("Dice label matching and loss-function masking require at least one IO step.")
    snapshots = self.snapshotRegistrationInputs(image, options, fixedMask)
    weight = options.get("weight", 0) if options.get("guidance") else 0
    extra = {"dice_loss_weight": weight} if weight else {}
    if mode == "loss":
      extra["loss_function_masking"] = True
    model = registration.load_model(self.weights_location, settings["model"], settings["loss"], settings["device"], **extra)
    fixed, moving = (self.sitk2itk(snapshots[role]) for role in ("fixed", "moving"))
    guidance = {role: self.sitk2itk(snapshots[role]) for role in
                ("fixed_mask", "moving_mask", "fixed_segmentation", "moving_segmentation") if role in snapshots}
    if mode != "none":
      guidance.update(input_masking=mode == "input", loss_function_masking=mode == "loss")

    def progress(*args):
      # a cancel request stops the registration at its next instance optimization step
      if self.cancelRequested:
        raise registration.RegistrationCancelled()
      if call_back is not None:
        call_back(*args)

    io_steps = settings["io_steps"]
    if io_steps != 0 and call_back is not None:
      call_back(io_steps, 0)
    phi_AB, warped_moving_image = registration.register_images(
      model, fixed, moving, settings["fixed_modality"], settings["moving_modality"], io_steps, progress,
      warp=settings["warp"], **guidance)
    if io_steps != 0 and call_back is not None:
      call_back(io_steps, io_steps)

    import tempfile
    with tempfile.TemporaryDirectory(prefix="UniGradICON-results-", dir=slicer.app.temporaryPath) as directory:
      transformPath = os.path.join(directory, "transform.h5")
      itk.transformwrite([phi_AB], transformPath)
      from .segmentation import load_results
      propagated = None
      if options.get("propagate"):
        from UniGradICONHelpers.segmentation import propagate
        labels = [value for key, value in snapshots.items() if key.startswith("propagation_")]
        def check_cancelled():
          if self.cancelRequested:
            raise registration.RegistrationCancelled()
        propagated = propagate(labels, snapshots["fixed"], transformPath, check_cancelled)
      load_results(self, outputSettings, transformPath,
                   self.itk2sitk(warped_moving_image) if warped_moving_image is not None else None,
                   options, propagated)


class RemoteRegistration:
  """A registration on a server: uploads the inputs, polls the job and loads the results."""

  POLL_INTERVAL_MS = 1000
  MAX_POLL_FAILURES = 10  # a busy server may miss a status request now and then

  def __init__(self, logic, client, parameters, statusCallback=None, finishedCallback=None, fixedMask=None):
    self.logic = logic
    self.fixedMask = fixedMask
    self.error = None
    self.cancelled = False
    self.client = client
    self.parameters = parameters
    self.statusCallback = statusCallback
    self.finishedCallback = finishedCallback  # (error or None, cancelled)
    self.jobId = None
    self.finished = False
    self.pollFailures = 0
    self.workDir = None
    self.timer = qt.QTimer()
    self.timer.setInterval(self.POLL_INTERVAL_MS)
    self.timer.timeout.connect(self.poll)

  def start(self):
    try:
      self._uploadInputs()
    except Exception as error:
      self._finish(str(error))
      raise

  def _uploadInputs(self):
    import tempfile
    self.workDir = tempfile.mkdtemp(prefix="UniGradICON-", dir=slicer.app.temporaryPath)
    options = self.parameters.get("segmentationSettings") or {}
    self.snapshots = self.logic.snapshotRegistrationInputs(self.parameters["image"], options, self.fixedMask)
    paths = {}
    for role, image in self.snapshots.items():
      paths[role] = os.path.join(self.workDir, role + ".nrrd")
      sitk.WriteImage(image, paths[role], True)
    settings = self.logic.registrationSettings(
      self.parameters["image"], self.parameters["outputSettings"],
      self.parameters.get("modelSettings"), self.parameters["generalSettings"])
    from UniGradICONHelpers.segmentation import registration_options, require_server_features
    additions = registration_options(options)
    settings.update(additions)
    if additions:
      require_server_features(self.client.info(), settings)
    uploads = {role + "_path": paths[role] for role in ("moving_mask", "fixed_segmentation", "moving_segmentation") if role in paths}
    self.jobId = self.client.submit(paths["fixed"], paths["moving"], settings, paths.get("fixed_mask"), **uploads)["id"]
    self.timer.start()

  def poll(self):
    if self.finished:
      return
    from UniGradICONHelpers import remote
    try:
      try:
        status = self.client.status(self.jobId)
        self.pollFailures = 0
      except remote.ServerError:
        self.pollFailures += 1
        if self.pollFailures < self.MAX_POLL_FAILURES:
          return
        raise
      if status["state"] == "done":
        self.timer.stop()
        transformPath = self.client.download(self.jobId, "transform", os.path.join(self.workDir, "transform.tfm"))
        outputSettings = self.parameters["outputSettings"]
        warpedImage = None
        if outputSettings.get("volume") is not None:
          warpedPath = self.client.download(self.jobId, "warped", os.path.join(self.workDir, "warped.nrrd"))
          warpedImage = sitk.ReadImage(warpedPath)
        from .segmentation import load_results
        options = self.parameters.get("segmentationSettings") or {}
        propagated = None
        if options.get("propagate"):
          from UniGradICONHelpers.segmentation import propagate
          labels = [value for key, value in self.snapshots.items() if key.startswith("propagation_")]
          propagated = propagate(labels, self.snapshots["fixed"], transformPath, lambda: None)
        load_results(self.logic, outputSettings, transformPath, warpedImage, options, propagated)
        self._finish(None)
      elif status["state"] == "failed":
        self._finish(status.get("error") or "unknown error")
      elif status["state"] == "cancelled":
        self._finish(None, cancelled=True)
      elif self.statusCallback:
        self.statusCallback(status)
    except remote.ServerError as e:
      self._finish(str(e))
    except Exception as e:
      logging.exception("Loading the registration result failed")
      self._finish(f"{type(e).__name__}: {e}")

  def cancel(self):
    if self.finished:
      return
    self._finish(None, cancelled=True)

  def waitForCompletion(self, cancelRequested=None):
    """Block until the job is finished, keeping Slicer responsive; returns (error, cancelled)."""
    import time
    while not self.finished:
      if cancelRequested is not None and cancelRequested():
        self.cancel()
        break
      slicer.app.processEvents()  # runs the polling timer
      time.sleep(0.05)
    return self.error, self.cancelled

  def _finish(self, error, cancelled=False):
    if self.finished:
      return
    import shutil
    from UniGradICONHelpers import remote
    self.timer.stop()
    self.snapshots = None
    self.finished = True
    self.error = error
    self.cancelled = cancelled
    if self.jobId:
      try:
        self.client.cancel(self.jobId)
      except remote.ServerError as removalError:
        logging.warning("Results or cancellation completed, but server job cleanup failed: %s", removalError)
      self.jobId = None
    if self.workDir:
      shutil.rmtree(self.workDir, ignore_errors=True)
    if self.finishedCallback:
      self.finishedCallback(error, cancelled)
