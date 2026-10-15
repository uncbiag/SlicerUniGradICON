"""Overlay and asynchronous checkerboard comparisons, including scene-save cleanup."""
import slicer

from .logic import UniGradICONLogic


class ComparisonControls:
  """Comparison controls mixed into the module widget."""

  def invalidateComparison(self):
    self.clearCheckerboard()
    self._completedResult = None
    self.ui.compareResultButton.checked = False
    self.ui.compareResultButton.text = "Compare result"
    self.setComparisonStatus("Run registration with the selected images to compare a new result.")

  def onSceneStartSave(self, caller, event):
    """Save real image references rather than references to an unsaved preview."""
    self._comparisonSaveRestore = []
    preview = self._checkerboardNode
    result = self._completedResult
    if preview is None or result is None:
      return
    fixed = result["image"]["fixed"]
    foreground = result["outputSettings"]["volume"] if self.ui.compareResultButton.checked else result["image"]["moving"]
    if any(node is None or node.GetScene() != slicer.mrmlScene for node in (preview, fixed, foreground)):
      return
    for composite in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
      if composite.GetBackgroundVolumeID() == preview.GetID():
        self._comparisonSaveRestore.append((composite, preview.GetID(), composite.GetForegroundVolumeID(),
                                            composite.GetForegroundOpacity(), fixed.GetID(), foreground.GetID(),
                                            self._checkerboardGeneration))
        modified = composite.StartModify()
        composite.SetBackgroundVolumeID(fixed.GetID())
        composite.SetForegroundVolumeID(foreground.GetID())
        composite.SetForegroundOpacity(0.5)
        composite.EndModify(modified)

  def onSceneEndSave(self, caller, event):
    restore = self._comparisonSaveRestore
    self._comparisonSaveRestore = []
    for composite, background, foreground, opacity, fixed, moving, generation in restore:
      if (generation == self._checkerboardGeneration and composite.GetScene() == slicer.mrmlScene
          and slicer.mrmlScene.GetNodeByID(background) is not None
          and composite.GetBackgroundVolumeID() == fixed and composite.GetForegroundVolumeID() == moving):
        modified = composite.StartModify()
        composite.SetBackgroundVolumeID(background)
        composite.SetForegroundVolumeID(foreground)
        composite.SetForegroundOpacity(opacity)
        composite.EndModify(modified)

  def onCompareResult(self, checked):
    result = self._completedResult
    if result is None:
      return
    fixed = result["image"]["fixed"]
    foreground = result["outputSettings"]["volume"] if checked else result["image"]["moving"]
    if any(node is None or node.GetScene() != slicer.mrmlScene for node in (fixed, foreground)):
      self.setComparisonStatus("A comparison image was removed. Select the images and run again.", error=True)
      self.ui.compareResultButton.enabled = False
      return
    self.ui.compareResultButton.text = "Show original" if checked else "Compare result"
    if self.ui.comparisonModeComboBox.currentText == "Checkerboard":
      self.queueCheckerboard(fixed, foreground)
    else:
      self.clearCheckerboard(restore=False)
      slicer.util.setSliceViewerLayers(background=fixed, foreground=foreground, foregroundOpacity=0.5)
      if not self._comparisonViewActive:
        slicer.util.resetSliceViews()
      self._comparisonViewActive = True
      self.setComparisonStatus("Overlay: registered result" if checked else "Overlay: original moving image")

  def setComparisonStatus(self, message, error=False):
    self.ui.comparisonStatusLabel.text = message
    self.ui.comparisonStatusLabel.setStyleSheet("color: #b83c2b;" if error else "")

  def onComparisonModeChanged(self, mode):
    self.updateRegistrationGUI()
    if not self.ui.compareResultButton.enabled:
      return
    if mode == "Checkerboard" and not self.ui.compareResultButton.checked:
      self.ui.compareResultButton.checked = True
    else:
      self.onCompareResult(self.ui.compareResultButton.checked)

  def onCheckerboardTilesChanged(self, count):
    if self.ui.compareResultButton.enabled and self.ui.comparisonModeComboBox.currentText == "Checkerboard":
      self.onCompareResult(self.ui.compareResultButton.checked)

  def clearCheckerboard(self, restore=True):
    """Invalidate pending filters; keep their nodes until asynchronous cancellation finishes."""
    self._checkerboardGeneration += 1
    self._checkerboardRefreshTimer.stop()
    self._pendingCheckerboard = None
    for job in list(self._checkerboardJobs.values()):
      cli = job["cli"]
      if cli.IsBusy():
        cli.Cancel()
      else:
        self._onCheckerboardFinished(cli)
    node = self._checkerboardNode
    self._checkerboardNode = None
    if node is not None and node.GetScene() == slicer.mrmlScene:
      if restore and self._completedResult is not None and not slicer.mrmlScene.IsClosing():
        image = self._completedResult["image"]
        if all(image[role].GetScene() == slicer.mrmlScene for role in ("fixed", "moving")):
          slicer.util.setSliceViewerLayers(background=image["fixed"], foreground=image["moving"], foregroundOpacity=0.5)
      self._removeCheckerboardPreview(node)
    self._comparisonViewActive = False

  def cancelCheckerboardOnExit(self):
    # Slicer may already have destroyed the module's Qt widgets and timers.
    # CLI nodes are independent of those widgets and can still be cancelled.
    self._checkerboardGeneration += 1
    for job in list(self._checkerboardJobs.values()):
      if job["cli"].IsBusy():
        job["cli"].Cancel()

  def queueCheckerboard(self, fixed, foreground):
    # Debounce spin-box and comparison changes; stale results must not replace
    # the requested view while the most recent settings wait to start.
    self._checkerboardGeneration += 1
    self._pendingCheckerboard = (fixed, foreground)
    self._checkerboardRefreshTimer.start()
    self.setComparisonStatus("Building checkerboard comparison…")

  def _refreshCheckerboard(self):
    pending = self._pendingCheckerboard
    self._pendingCheckerboard = None
    if pending and self._completedResult is not None and self.ui.comparisonModeComboBox.currentText == "Checkerboard":
      self.startCheckerboard(*pending)

  def _pollCheckerboards(self):
    # Remove CLI nodes only from the Qt event loop, after Slicer has finished
    # handling their status events; deleting them inside an observer can crash.
    for job in list(self._checkerboardJobs.values()):
      if job["cli"].GetScene() != slicer.mrmlScene and not job["cli"].IsBusy():
        self._removeCheckerboardJob(job)
      else:
        self._onCheckerboardFinished(job["cli"])

  def startCheckerboard(self, fixed, foreground):
    module = getattr(slicer.modules, "checkerboardfilter", None)
    if module is None:
      self.setComparisonStatus("Checkerboard Filter is unavailable in this Slicer installation. Choose Overlay instead.", error=True)
      return
    # Superseded runs are cancelled but retain their temporary output until the CLI stops.
    self._checkerboardGeneration += 1
    for job in list(self._checkerboardJobs.values()):
      if job["cli"].IsBusy():
        job["cli"].Cancel()
      else:
        self._onCheckerboardFinished(job["cli"])
    for node in (fixed, foreground):
      if node.GetImageData() is None or node.GetImageData().GetNumberOfPoints() == 0:
        self.setComparisonStatus("A comparison image has no voxel data. Choose the inputs and run again.", error=True)
        return
    output = cli = job = None
    owned = []
    try:
      sourceFixed = self._checkerboardSource(fixed, owned)
      sourceForeground = self._checkerboardSource(foreground, owned)
      output = self.logic.temporaryVolume("UniGradICON checkerboard preview")
      output.CreateDefaultDisplayNodes()
      output.GetDisplayNode().SetSaveWithScene(False)
      fixedDisplay = fixed.GetDisplayNode()
      if fixedDisplay is not None:
        display = output.GetDisplayNode()
        display.SetAndObserveColorNodeID(fixedDisplay.GetColorNodeID())
        display.SetAutoWindowLevel(False)
        display.SetWindowLevel(fixedDisplay.GetWindow(), fixedDisplay.GetLevel())
      cli = slicer.cli.createNode(module)
      cli.SetHideFromEditors(True)
      cli.SetSaveWithScene(False)
      job = {"cli": cli, "output": output, "generation": self._checkerboardGeneration,
             "fixed": fixed, "foreground": foreground, "owned": owned, "id": cli.GetID()}
      self._checkerboardJobs[cli.GetID()] = job
      self._checkerboardTimer.start()
      count = self.ui.checkerboardTilesSpinBox.value
      pattern = [min(count, dimension) for dimension in fixed.GetImageData().GetDimensions()]
      self.setComparisonStatus("Building checkerboard comparison…")
      slicer.cli.run(module, cli, {"inputVolume1": sourceFixed, "inputVolume2": sourceForeground,
                     "outputVolume": output, "checkerPattern": pattern},
                     wait_for_completion=False, update_display=False)
    except Exception as error:
      if job:
        self._removeCheckerboardJob(job)
      else:
        for node in owned + ([output] if output is not None else []):
          self._removeCheckerboardPreview(node)
        if cli is not None and cli.GetScene() == slicer.mrmlScene:
          slicer.mrmlScene.RemoveNode(cli)
      self.setComparisonStatus(f"Checkerboard comparison failed: {error}. Choose Overlay or try again.", error=True)

  def _checkerboardSource(self, node, owned):
    parent = node.GetParentTransformNode()
    if parent is None:
      return node
    if not parent.IsTransformToWorldLinear():
      raise ValueError("Harden nonlinear input transforms before comparing images")
    snapshot = self.logic.temporaryVolume("UniGradICON comparison source")
    owned.append(snapshot)
    snapshot.CopyOrientation(node)
    snapshot.SetAndObserveImageData(node.GetImageData())
    snapshot.SetAndObserveTransformNodeID(node.GetTransformNodeID())
    if not slicer.vtkSlicerTransformLogic().hardenTransform(snapshot):
      raise ValueError("The input transform could not be applied to the comparison snapshot")
    return snapshot

  def _removeCheckerboardJob(self, job, keep_output=False):
    cli = job["cli"]
    self._checkerboardJobs.pop(job["id"], None)
    if not self._checkerboardJobs:
      self._checkerboardTimer.stop()
    if cli.GetScene() == slicer.mrmlScene:
      slicer.mrmlScene.RemoveNode(cli)
    if not keep_output:
      self._removeCheckerboardPreview(job["output"])
    for node in job.get("owned", []):
      self._removeCheckerboardPreview(node)

  @staticmethod
  def _removeCheckerboardPreview(node):
    UniGradICONLogic.removeTemporaryVolume(node)

  def _onCheckerboardFinished(self, cli, event=None):
    job = self._checkerboardJobs.get(cli.GetID())
    if job is None or cli.IsBusy() or cli.GetStatus() == cli.Idle:
      return
    current = (job["generation"] == self._checkerboardGeneration
               and self.ui.comparisonModeComboBox.currentText == "Checkerboard"
               and self._completedResult is not None
               and not (self._startingRegistration or self._registrationTask)
               and all(job[role].GetScene() == slicer.mrmlScene for role in ("fixed", "foreground", "output")))
    success = cli.GetStatus() == cli.Completed
    error = cli.GetErrorText()
    if current and success:
      previous = self._checkerboardNode
      self._checkerboardNode = job["output"]
      if job["output"].GetStorageNode() is not None:
        job["output"].GetStorageNode().SetSaveWithScene(False)
      slicer.util.setSliceViewerLayers(background=job["output"], foreground=None)
      if not self._comparisonViewActive:
        slicer.util.resetSliceViews()
      self._comparisonViewActive = True
      if previous is not None and previous.GetScene() == slicer.mrmlScene:
        self._removeCheckerboardPreview(previous)
      self.setComparisonStatus("Checkerboard: registered result" if self.ui.compareResultButton.checked
                               else "Checkerboard: original moving image")
    self._removeCheckerboardJob(job, keep_output=current and success)
    if current and not success:
      self.setComparisonStatus(f"Checkerboard comparison failed: {error or cli.GetStatusString()}. Choose Overlay or try again.", error=True)
