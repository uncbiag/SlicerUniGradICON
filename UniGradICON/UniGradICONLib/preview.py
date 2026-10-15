"""Temporary fixed/moving label overlays with reversible viewer changes."""
import numpy as np
import vtk
import SimpleITK as sitk
import slicer

from .segmentation import binary_mask_on_reference, guidance_images, stage_segmentation


class InputPreviewControls:
  def setupInputPreviews(self):
    self._inputPreviewNodes = []
    self._inputPreviewState = None
    self._inputPreviewSignature = None
    self._inputPreviewKind = None
    self._previewObserved = []
    self.ui.previewMasksButton.toggled.connect(lambda checked: self.onInputPreview("masks", checked))
    self.ui.previewGuidanceButton.toggled.connect(lambda checked: self.onInputPreview("guidance", checked))
    slicer.app.connect("aboutToQuit()", lambda: self.clearInputPreview(restore=False, update_gui=False))

  def inputPreviewSignature(self):
    if self._parameterNode is None:
      return None
    options = self.logic.segmentationSettings(self._parameterNode)
    nodes = [self.ui.fixedImageNodeComboBox.currentNode(), self.ui.movingImageNodeComboBox.currentNode()]
    nodes += [options.get(key) for key in ("fixed", "moving", "fixedMask", "movingMask")]
    return (self._parameterNode.GetID(), tuple(node.GetID() if node else None for node in nodes),
            options["maskingMode"], options["guidance"])

  def onInputPreview(self, kind, checked):
    if not checked:
      if self._inputPreviewKind == kind:
        self.clearInputPreview()
      return
    self.clearInputPreview()
    try:
      manager = slicer.app.layoutManager()
      if manager is None:
        raise ValueError("Open the Slicer image viewers to preview the inputs.")
      options = self.logic.segmentationSettings(self._parameterNode)
      volumes = {role: getattr(self.ui, role + "ImageNodeComboBox").currentNode() for role in ("fixed", "moving")}
      images = {role: self.logic.snapshotVolume(node) for role, node in volumes.items()}
      if kind == "masks":
        for role in volumes:
          images[role + "_mask"] = binary_mask_on_reference(options[role + "Mask"], images[role])
      else:
        guidance_images(images, options)
      self.clearCheckerboard()
      state = {"layout": manager.layout, "views": [], "slices": [],
               "backgrounds": {node.GetID() for node in volumes.values()}}
      for slice_node in slicer.util.getNodesByClass("vtkMRMLSliceNode"):
        matrix = vtk.vtkMatrix4x4()
        matrix.DeepCopy(slice_node.GetSliceToRAS())
        state["slices"].append((slice_node, matrix, slice_node.GetFieldOfView()))
      self._inputPreviewState = state
      for composite in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
        state["views"].append((composite, composite.GetBackgroundVolumeID(), composite.GetForegroundVolumeID(),
                               composite.GetLabelVolumeID(), composite.GetForegroundOpacity()))
      manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutSideBySideView)
      palette = [(0.1, 0.8, 0.95), (1.0, 0.5, 0.15), (0.5, 0.9, 0.25), (0.85, 0.3, 0.8)]
      for index, role in enumerate(("fixed", "moving")):
        image = images[role + ("_mask" if kind == "masks" else "_segmentation")]
        values = sitk.GetArrayFromImage(image)
        pairs = [{"label": 1, "name": role.capitalize() + " mask"}] if kind == "masks" else options["labelPairs"]
        masks, metadata = [], []
        for number, pair in enumerate(pairs):
          binary = sitk.GetImageFromArray((values == pair["label"]).astype(np.uint8))
          binary.CopyInformation(image)
          masks.append(binary)
          metadata.append({"id": str(pair["label"]), "name": pair["name"],
                           "color": palette[index if kind == "masks" else number % len(palette)]})
        staged = stage_segmentation(self.logic, masks, metadata)
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "UniGradICON " + role + " input preview")
        node.SetSaveWithScene(False)
        self._inputPreviewNodes.append(node)
        node.CopyContent(staged)
        node.SetHideFromEditors(True)
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        display.SetSaveWithScene(False)
        display.SetOpacity2DFill(0.25)
        display.SetOpacity2DOutline(1)
        slice_widget = manager.sliceWidget("Red" if role == "fixed" else "Yellow")
        display.AddViewNodeID(slice_widget.mrmlSliceNode().GetID())
        slice_widget.mrmlSliceNode().SetOrientationToAxial()
        composite = slice_widget.mrmlSliceCompositeNode()
        composite.SetBackgroundVolumeID(volumes[role].GetID())
        composite.SetForegroundVolumeID(None)
        composite.SetLabelVolumeID(None)
        slice_widget.sliceLogic().FitSliceToAll()
      self._inputPreviewKind = kind
      self.ui.previewMasksButton.text = "Hide mask preview" if kind == "masks" else "Preview masks"
      self.ui.previewGuidanceButton.text = "Hide IO segmentation preview" if kind == "guidance" else "Preview IO segmentations"
      self._inputPreviewSignature = self.inputPreviewSignature()
      observed = list(volumes.values()) + [options[key] for key in
                  (("fixedMask", "movingMask") if kind == "masks" else ("fixed", "moving"))]
      for node in dict.fromkeys(observed):
        tag = node.AddObserver("ModifiedEvent", self.onPreviewInputModified)
        self._previewObserved.append((node, tag))
        tag = node.AddObserver(slicer.vtkMRMLTransformableNode.TransformModifiedEvent, self.onPreviewInputModified)
        self._previewObserved.append((node, tag))
        if node.IsA("vtkMRMLSegmentationNode"):
          segmentation = node.GetSegmentation()
          tag = segmentation.AddObserver(slicer.vtkSegmentation.SegmentModified, self.onPreviewInputModified)
          self._previewObserved.append((segmentation, tag))
        elif node.IsA("vtkMRMLVolumeNode") and node.GetImageData():
          image = node.GetImageData()
          tag = image.AddObserver("ModifiedEvent", self.onPreviewInputModified)
          self._previewObserved.append((image, tag))
      button = self.ui.previewMasksButton if kind == "masks" else self.ui.previewGuidanceButton
      blocked = button.blockSignals(True)
      button.checked = True
      button.blockSignals(blocked)
      self.ui.inputPreviewStatusLabel.text = ""
      self.ui.inputPreviewStatusLabel.visible = False
    except Exception as error:
      self.clearInputPreview()
      self.ui.inputPreviewStatusLabel.text = str(error)
      self.ui.inputPreviewStatusLabel.visible = True

  def onPreviewInputModified(self, caller=None, event=None):
    self.clearInputPreview()
    self.ui.inputPreviewStatusLabel.text = "Input changed. Preview again."
    self.ui.inputPreviewStatusLabel.visible = True

  def clearInputPreview(self, restore=True, update_gui=True):
    for node, tag in getattr(self, "_previewObserved", []):
      node.RemoveObserver(tag)
    self._previewObserved = []
    for node in getattr(self, "_inputPreviewNodes", []):
      if node.GetScene() == slicer.mrmlScene:
        self.logic.removeTemporaryVolume(node)
    self._inputPreviewNodes = []
    state = getattr(self, "_inputPreviewState", None)
    if state and restore:
      backgrounds = state["backgrounds"]
      for composite, background, foreground, label, opacity in state["views"]:
        if (composite.GetScene() == slicer.mrmlScene and composite.GetBackgroundVolumeID() in backgrounds
            and not composite.GetForegroundVolumeID() and not composite.GetLabelVolumeID()):
          composite.SetBackgroundVolumeID(background if slicer.mrmlScene.GetNodeByID(background or "") else None)
          composite.SetForegroundVolumeID(foreground if slicer.mrmlScene.GetNodeByID(foreground or "") else None)
          composite.SetLabelVolumeID(label if slicer.mrmlScene.GetNodeByID(label or "") else None)
          composite.SetForegroundOpacity(opacity)
      manager = slicer.app.layoutManager()
      if manager and manager.layout == slicer.vtkMRMLLayoutNode.SlicerLayoutSideBySideView:
        manager.setLayout(state["layout"])
      for node, matrix, field in state["slices"]:
        if node.GetScene() == slicer.mrmlScene:
          node.GetSliceToRAS().DeepCopy(matrix)
          node.SetFieldOfView(*field)
          node.UpdateMatrices()
    self._inputPreviewState = None
    self._inputPreviewKind = None
    self._inputPreviewSignature = None
    if update_gui and hasattr(self, "ui"):
      for button in (self.ui.previewMasksButton, self.ui.previewGuidanceButton):
        blocked = button.blockSignals(True)
        button.checked = False
        button.blockSignals(blocked)
      self.ui.previewMasksButton.text = "Preview masks"
      self.ui.previewGuidanceButton.text = "Preview IO segmentations"
      self.ui.inputPreviewStatusLabel.text = ""
      self.ui.inputPreviewStatusLabel.visible = False

  def updateInputPreviewGUI(self, busy):
    if not hasattr(self, "_inputPreviewNodes"):
      return
    if self._inputPreviewState and (busy or self._inputPreviewSignature != self.inputPreviewSignature()):
      self.clearInputPreview()
    self.ui.previewMasksButton.enabled = self.ui.previewGuidanceButton.enabled = not busy
