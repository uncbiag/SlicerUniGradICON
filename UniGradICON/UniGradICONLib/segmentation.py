"""Slicer segmentation controls, world-coordinate snapshots, and staged outputs."""
import numpy as np
import SimpleITK as sitk
import slicer
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from UniGradICONHelpers.segmentation import DEFAULT_DICE_WEIGHT, MASKING_MODES, masking_mode


def snapshot_segment(node, segment_id, allow_empty=False):
  if node is None or node.GetScene() != slicer.mrmlScene:
    raise ValueError("Select a segmentation in the current scene.")
  if not segment_id or node.GetSegmentation().GetSegment(segment_id) is None:
    raise ValueError("Select a mask from the chosen segmentation.")
  parent = node.GetParentTransformNode()
  if parent is not None and not parent.IsTransformToWorldLinear():
    raise ValueError("Harden nonlinear segmentation parent transforms before registration.")
  oriented = slicer.vtkOrientedImageData()
  if not slicer.vtkSlicerSegmentationsModuleLogic.GetSegmentBinaryLabelmapRepresentation(node, segment_id, oriented):
    raise ValueError("The segment could not be converted to a binary labelmap.")
  scalars = oriented.GetPointData().GetScalars()
  if scalars is None:
    if allow_empty:
      return sitk.Image([1, 1, 1], sitk.sitkUInt8)
    raise ValueError("The selected mask is empty.")
  label = node.GetSegmentation().GetSegment(segment_id).GetLabelValue()
  data = (vtk_to_numpy(scalars).reshape(tuple(reversed(oriented.GetDimensions()))) == label).astype(np.uint8)
  if not data.any():
    if allow_empty:
      return sitk.Image([1, 1, 1], sitk.sitkUInt8)
    raise ValueError("The selected mask is empty.")
  matrix = vtk.vtkMatrix4x4()
  oriented.GetImageToWorldMatrix(matrix)
  ras = slicer.util.arrayFromVTKMatrix(matrix)
  lps = np.diag([-1, -1, 1, 1]) @ ras
  extent = oriented.GetExtent()
  origin = lps @ np.array([extent[0], extent[2], extent[4], 1])
  spacing = np.linalg.norm(lps[:3, :3], axis=0)
  if np.any(spacing <= 0) or abs(np.linalg.det(lps[:3, :3])) < 1e-12:
    raise ValueError("The segmentation has singular image geometry.")
  image = sitk.GetImageFromArray(data)
  image.SetOrigin(tuple(origin[:3]))
  image.SetSpacing(tuple(spacing))
  image.SetDirection(tuple((lps[:3, :3] / spacing).flatten()))
  return image


def segment_on_reference(node, segment_id, reference, allow_empty=False):
  image = sitk.Resample(snapshot_segment(node, segment_id, allow_empty), reference, sitk.Transform(3, sitk.sitkIdentity),
                        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
  if not allow_empty and not sitk.GetArrayViewFromImage(image).any():
    raise ValueError("The selected mask does not overlap its image. Check the segmentation and image geometry.")
  return image


def binary_mask_on_reference(node, reference):
  """Use a labelmap's nonzero voxels or the union of a segmentation's segments."""
  if node is None or node.GetScene() != slicer.mrmlScene:
    raise ValueError("Select a mask in the current scene.")
  identity = sitk.Transform(3, sitk.sitkIdentity)
  if node.IsA("vtkMRMLLabelMapVolumeNode"):
    from .logic import UniGradICONLogic
    image = UniGradICONLogic.snapshotVolume(node) != 0
    mask = sitk.Resample(image, reference, identity, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
  elif node.IsA("vtkMRMLSegmentationNode"):
    mask = sitk.Image(reference.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(reference)
    segments = node.GetSegmentation()
    for index in range(segments.GetNumberOfSegments()):
      mask |= segment_on_reference(node, segments.GetNthSegmentID(index), reference, allow_empty=True)
  else:
    raise ValueError("Choose a binary labelmap or segmentation as the mask.")
  if not sitk.GetArrayViewFromImage(mask).any():
    raise ValueError("The selected mask is empty or does not overlap its image.")
  return mask


def guidance_classes(options):
  sources = [options.get(role) for role in ("fixed", "moving")]
  if any(node is None or node.GetScene() != slicer.mrmlScene for node in sources):
    raise ValueError("Select fixed and moving segmentations for IO guidance.")
  if all(node.IsA("vtkMRMLLabelMapVolumeNode") for node in sources):
    from UniGradICONHelpers.segmentation import label_values
    sets = [set(np.unique(label_values(slicer.util.arrayFromVolume(node)))) - {0} for node in sources]
    shared = sorted(sets[0] & sets[1])
    pairs = [{"label": int(value), "name": f"Label {value}"} for value in shared]
  elif all(node.IsA("vtkMRMLSegmentationNode") for node in sources):
    mappings = []
    for source in sources:
      segmentation = source.GetSegmentation()
      mapping = {}
      for index in range(segmentation.GetNumberOfSegments()):
        segment_id = segmentation.GetNthSegmentID(index)
        name = segmentation.GetSegment(segment_id).GetName().strip()
        if not name or name.casefold() in mapping:
          raise ValueError("IO segmentation names must be nonempty and unique within each segmentation.")
        mapping[name.casefold()] = (segment_id, name)
      mappings.append(mapping)
    shared = sorted(set(mappings[0]) & set(mappings[1]))
    pairs = [{"label": index, "name": mappings[0][name][1],
              "fixedSegment": mappings[0][name][0], "movingSegment": mappings[1][name][0]}
             for index, name in enumerate(shared, 1)]
  else:
    raise ValueError("For multi-label IO, select two labelmaps with matching values or two segmentations with matching names.")
  if not pairs:
    raise ValueError("No matching non-background labels or segment names were found for IO guidance.")
  return pairs


def guidance_images(images, options):
  pairs = guidance_classes(options)
  from .logic import UniGradICONLogic
  from UniGradICONHelpers.segmentation import label_values
  for role in ("fixed", "moving"):
    source, reference = options[role], images[role]
    if source.IsA("vtkMRMLLabelMapVolumeNode"):
      original = UniGradICONLogic.snapshotVolume(source)
      data = label_values(sitk.GetArrayFromImage(original))
      data[~np.isin(data, [pair["label"] for pair in pairs])] = 0
      original_labels = sitk.GetImageFromArray(data)
      original_labels.CopyInformation(original)
      result = sitk.Resample(original_labels, reference, sitk.Transform(3, sitk.sitkIdentity),
                             sitk.sitkNearestNeighbor, 0, sitk.sitkUInt32)
    else:
      data = np.zeros(tuple(reversed(reference.GetSize())), np.uint32)
      for pair in pairs:
        segment = sitk.GetArrayFromImage(segment_on_reference(source, pair[role + "Segment"], reference, allow_empty=True)) != 0
        if np.any((data != 0) & segment):
          raise ValueError("Matched IO segments overlap. Make the labels mutually exclusive or use a labelmap with explicit label values.")
        data[segment] = pair["label"]
      result = sitk.GetImageFromArray(data)
      result.CopyInformation(reference)
    images[role + "_segmentation"] = result
  present = [set(np.unique(sitk.GetArrayViewFromImage(images[role + "_segmentation"]))) for role in ("fixed", "moving")]
  usable = present[0] & present[1] - {0}
  pairs = [pair for pair in pairs if pair["label"] in usable]
  if not pairs:
    raise ValueError("No matching segmentation labels overlap both image grids.")
  options["labelPairs"] = pairs


def stage_segmentation(logic, images, metadata):
  """Build every segment before replacing the requested output node."""
  import sitkUtils
  if len(images) != len(metadata):
    raise ValueError("The transformed segmentation is incomplete.")
  staged = slicer.vtkMRMLSegmentationNode()
  volume = logic.temporaryNode("vtkMRMLLabelMapVolumeNode", "UniGradICON segment staging")
  try:
    for image, segment in zip(images, metadata):
      sitkUtils.PushVolumeToSlicer(image, volume)
      oriented = slicer.vtkSlicerSegmentationsModuleLogic.CreateOrientedImageDataFromVolumeNode(volume)
      new_segment = slicer.vtkSegment()
      new_segment.SetName(segment["name"])
      new_segment.SetColor(segment["color"])
      new_segment.AddRepresentation(slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName(), oriented)
      if not staged.GetSegmentation().AddSegment(new_segment, segment["id"]):
        raise ValueError("A transformed segment could not be created.")
    return staged
  finally:
    logic.removeTemporaryVolume(volume)


def load_results(logic, output_settings, transform_path, warped_image, options, segments=None):
  staged = None
  if options.get("propagate"):
    target = options.get("output")
    if target is None or target.GetScene() != slicer.mrmlScene:
      raise ValueError("Select a transformed segmentation output in the current scene.")
    if segments is None:
      raise ValueError("No transformed segmentation was returned.")
    staged = stage_segmentation(logic, segments, options["segments"])
  logic.loadResults(output_settings, transform_path, warped_image)
  if staged is not None:
    target.CopyContent(staged)
    target.SetAndObserveTransformNodeID(None)
    target.CreateDefaultDisplayNodes()


def snapshot_inputs(images, options):
  if options.get("guidance"):
    from UniGradICONHelpers.segmentation import validate_weight
    if validate_weight(options["weight"]) == 0:
      raise ValueError("Segmentation guidance requires a positive Dice loss weight.")
  mode = masking_mode(options)
  if mode != "none":
    for role in ("fixed", "moving"):
      images[role + "_mask"] = binary_mask_on_reference(options.get(role + "Mask"), images[role])
  if options.get("guidance"):
    guidance_images(images, options)
  options["segments"] = []
  if options.get("propagate"):
    source = options.get("propagationSource", options.get("moving"))
    if source is None:
      raise ValueError("Select a moving segmentation to propagate.")
    segmentation = source.GetSegmentation()
    for i in range(segmentation.GetNumberOfSegments()):
      segment_id = segmentation.GetNthSegmentID(i)
      segment = segmentation.GetSegment(segment_id)
      images["propagation_" + str(i)] = snapshot_segment(source, segment_id, allow_empty=True)
      options["segments"].append({"id": segment_id, "name": segment.GetName(), "color": segment.GetColor()})


class SegmentationControls:
  def setupSegmentationControls(self):
    for role in self.logic.SEGMENTATION_REFS | self.logic.MASK_REFS:
      getattr(self.ui, role + "SegmentationComboBox").connect(
        "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    for mode, label in MASKING_MODES.items():
      self.ui.maskingModeComboBox.addItem(label, mode)
    self.ui.maskingModeComboBox.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
    for check_box in (self.ui.diceGuidanceCheckBox, self.ui.propagateSegmentationCheckBox):
      check_box.toggled.connect(self.updateParameterNodeFromGUI)
    self.ui.propagationSegmentationComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.diceWeightSpinBox.valueChanged.connect(self.updateParameterNodeFromGUI)
    self.ui.outputSegmentationComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.onOutputSegmentationChanged)
    self.ui.outputSegmentationComboBox.noneDisplay = "Create automatically"

  def onOutputSegmentationChanged(self, node):
    if slicer.mrmlScene.IsImporting() or self._sceneImportPending:
      return
    if node is not None and not self._updatingGUIFromParameterNode:
      self.ui.propagateSegmentationCheckBox.checked = True
    self.updateParameterNodeFromGUI()

  def saveSegmentationSettings(self):
    node = self._parameterNode
    for role, reference in (self.logic.SEGMENTATION_REFS | self.logic.MASK_REFS).items():
      node.SetNodeReferenceID(reference, getattr(self.ui, role + "SegmentationComboBox").currentNodeID)
    node.SetParameter("SegmentationInputsVersion", "3")
    node.SetNodeReferenceID(self.logic.PROPAGATION_REF, self.ui.propagationSegmentationComboBox.currentNodeID)
    node.SetNodeReferenceID(self.logic.OUTPUT_SEGMENTATION_REF, self.ui.outputSegmentationComboBox.currentNodeID)
    for name, widget in self.segmentationFlags():
      node.SetParameter(name, "true" if widget.checked else "false")
    node.SetParameter("DiceLossWeight", str(self.ui.diceWeightSpinBox.value))
    node.SetParameter("MaskingMode", self.ui.maskingModeComboBox.itemData(self.ui.maskingModeComboBox.currentIndex))

  def segmentationFlags(self):
    return (("UseSegmentationGuidance", self.ui.diceGuidanceCheckBox),
            ("PropagateSegmentation", self.ui.propagateSegmentationCheckBox))

  def restoreSegmentationSettings(self):
    node = self._parameterNode
    settings = self.logic.segmentationSettings(node)
    for role in self.logic.SEGMENTATION_REFS | self.logic.MASK_REFS:
      getattr(self.ui, role + "SegmentationComboBox").setCurrentNode(settings[role])
    self.ui.propagationSegmentationComboBox.setCurrentNode(settings["propagationSource"])
    self.ui.outputSegmentationComboBox.setCurrentNode(node.GetNodeReference(self.logic.OUTPUT_SEGMENTATION_REF))
    for name, widget in self.segmentationFlags():
      widget.checked = node.GetParameter(name) == "true"
    self.ui.diceWeightSpinBox.value = float(node.GetParameter("DiceLossWeight") or DEFAULT_DICE_WEIGHT)
    self.ui.maskingModeComboBox.currentIndex = self.ui.maskingModeComboBox.findData(settings["maskingMode"])

  def validateSegmentationInputs(self):
    guidance, propagation = (widget.checked for _, widget in self.segmentationFlags())
    mode = self.ui.maskingModeComboBox.itemData(self.ui.maskingModeComboBox.currentIndex)
    active_sources = []
    for suffix, enabled, purpose in (("Mask", mode != "none", "masking"), ("", guidance, "IO guidance")):
      if not enabled:
        continue
      for role in ("fixed", "moving"):
        source = getattr(self.ui, role + suffix + "SegmentationComboBox").currentNode()
        if source is None:
          raise ValueError(f"Select the {role} {'mask' if suffix else 'segmentation'} for {purpose}.")
        active_sources.append(source)
    if guidance:
      guidance_classes(self.logic.segmentationSettings(self._parameterNode))
    if propagation:
      source = self.ui.propagationSegmentationComboBox.currentNode()
      if source is None or source.GetSegmentation().GetNumberOfSegments() == 0:
        raise ValueError("Select a moving segmentation with segments to propagate.")
      active_sources.append(source)
    if (guidance or mode == "loss") and self.ui.ioSpinBox.value == 0:
      raise ValueError("Dice label matching and loss-function masking require at least one IO step.")
    output = self.ui.outputSegmentationComboBox.currentNode()
    if propagation and output is not None and output in active_sources:
      raise ValueError("The transformed segmentation must differ from the input segmentations. Create new outputs.")

  def updateSegmentationGUI(self, busy):
    self.ui.segmentationCollapsibleButton.enabled = not busy
    mode = self.ui.maskingModeComboBox.itemData(self.ui.maskingModeComboBox.currentIndex)
    guidance = self.ui.diceGuidanceCheckBox.checked
    propagation = self.ui.propagateSegmentationCheckBox.checked
    self.ui.maskInputsWidget.visible = mode != "none"
    self.ui.guidanceInputsWidget.visible = guidance
    self.ui.propagationInputsWidget.visible = propagation
    io_required = guidance or mode == "loss"
    invalid_io = io_required and self.ui.ioSpinBox.value == 0
    self.ui.ioSpinBox.setStyleSheet("border: 1px solid #b83c2b;" if invalid_io else "")
    self.ui.ioSpinBox.toolTip = "Set IO steps above zero for guidance or loss masking." if invalid_io else "Instance optimization steps; 0 runs inference only."
