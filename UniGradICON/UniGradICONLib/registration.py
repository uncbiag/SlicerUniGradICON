"""Interactive registration: output selection, task lifecycle, progress, and cancellation."""
import sys

import qt
import slicer


class RegistrationControls:
  """Registration controls mixed into the module widget."""

  def updateProgress(self, allProgress, currentProgress, elapsedTime=None, remainingTime=None):
    self.ui.progressBar.minimum = 0
    self.ui.progressBar.maximum = max(1, allProgress)
    self.ui.progressBar.value = max(0, min(currentProgress, allProgress))
    self.ui.progressBar.show()

    if elapsedTime is not None:
      self.ui.time.text = f"Optimization elapsed: {elapsedTime:.0f} s"
      if remainingTime is not None and remainingTime > 0:
        self.ui.time.text += f" · Estimated remaining: {remainingTime:.0f} s"
      self.ui.time.show()
    else:
      self.ui.time.hide()

  def onRunRegistrationButton(self):
    if self._registrationTask or self._startingRegistration:
      self.cancelRegistration()
      return
    import time
    try:
      self.validateRunInputs()
      remote = self.ui.remoteProcessingCheckBox.checked
      if remote and not self.ui.serverComboBox.currentText.strip():
        raise ValueError("Enter a registration server address, or clear Remote processing to run locally.")
      if not self.prepareRunOutputs():
        return
      parameters = self.logic.createProcessParameters(self._parameterNode)
      for role, selector in (("fixed", self.ui.fixedImageNodeComboBox), ("moving", self.ui.movingImageNodeComboBox)):
        parameters["image"][role] = selector.currentNode()
      if not self.ui.createVolumeCheckBox.checked:
        parameters["outputSettings"]["volume"] = None
    except Exception as error:
      self.setRegistrationStatus(str(error), error=True)
      return
    self._runParameters = parameters
    self.clearCheckerboard()
    self._registrationStart = time.monotonic()
    self._startingRegistration = True
    self._cancelling = False
    self._completedResult = None
    self.ui.compareResultButton.checked = False
    self.ui.compareResultButton.text = "Compare result"
    self._lastPhase = None
    self._lastRunWarning = None
    self._runSerial += 1
    serial = self._runSerial
    self._reconnectTimer.stop()
    self.setRegistrationStatus("Preparing images…")
    self.ui.progressBar.setRange(0, 0)
    self.ui.progressBar.show()
    self.ui.time.hide()
    self.updateRegistrationGUI()
    # Let Qt paint the immediate feedback before taking the MRML input snapshots.
    qt.QTimer.singleShot(0, lambda: self._beginRegistration(parameters, remote, serial))

  def validateRunInputs(self):
    fixed = self.ui.fixedImageNodeComboBox.currentNode()
    moving = self.ui.movingImageNodeComboBox.currentNode()
    for role, node in (("fixed", fixed), ("moving", moving)):
      if node is None or node.GetImageData() is None or node.GetImageData().GetNumberOfPoints() == 0:
        raise ValueError(f"Select a {role} image that contains voxel data.")
      parent = node.GetParentTransformNode()
      if parent is not None and not parent.IsTransformToWorldLinear():
        raise ValueError(f"Harden the nonlinear parent transform of the {role} image before registration.")
    output = self.ui.outputVolumeComboBox.currentNode()
    if self.ui.createVolumeCheckBox.checked and output in (fixed, moving):
      raise ValueError("The output volume must differ from both input images. Click Create new outputs.")

  @staticmethod
  def outputHasData(node):
    if node is None:
      return False
    if node.IsA("vtkMRMLVolumeNode"):
      return node.GetImageData() is not None and node.GetImageData().GetNumberOfPoints() > 0
    return node.GetTransformToParent() is not None

  def selectedOutputsWithData(self):
    nodes = [self.ui.outputTransformComboBox.currentNode()]
    if self.ui.createVolumeCheckBox.checked:
      nodes.append(self.ui.outputVolumeComboBox.currentNode())
    return [node for node in nodes if self.outputHasData(node)]

  def prepareRunOutputs(self):
    populated = self.selectedOutputsWithData()
    self.ui.outputsCollapsibleButton.text = ("Outputs (transform + volume)" if self.ui.createVolumeCheckBox.checked
                                             else "Outputs (transform only)")
    if populated:
      choice = self.confirmOutputReplacement(populated)
      if choice == "new":
        self.createNewOutputs()
      elif choice != "overwrite":
        return False
    self.createMissingOutputs()
    return True

  def confirmOutputReplacement(self, populated):
    dialog = qt.QMessageBox(slicer.util.mainWindow())
    dialog.setWindowTitle("Preserve existing registration results")
    dialog.setText("These outputs already contain data:\n" + "\n".join(node.GetName() for node in populated))
    newButton = dialog.addButton("Create new outputs", qt.QMessageBox.AcceptRole)
    overwriteButton = dialog.addButton("Overwrite", qt.QMessageBox.DestructiveRole)
    dialog.addButton(qt.QMessageBox.Cancel)
    dialog.setDefaultButton(newButton)
    dialog.exec_()
    return ("new" if dialog.clickedButton() == newButton else
            "overwrite" if dialog.clickedButton() == overwriteButton else "cancel")

  def createMissingOutputs(self, replace=False):
    fixed = self.ui.fixedImageNodeComboBox.currentNode()
    moving = self.ui.movingImageNodeComboBox.currentNode()
    if fixed is None or moving is None:
      raise ValueError("Select the fixed and moving images before creating outputs.")
    base = f"{moving.GetName()} to {fixed.GetName()}"
    for selector, nodeClass, suffix, wanted in (
        (self.ui.outputTransformComboBox, "vtkMRMLTransformNode", "transform", True),
        (self.ui.outputVolumeComboBox, "vtkMRMLScalarVolumeNode", "registered", self.ui.createVolumeCheckBox.checked)):
      if wanted and (replace or selector.currentNode() is None):
        node = slicer.mrmlScene.AddNewNodeByClass(nodeClass, slicer.mrmlScene.GenerateUniqueName(base + " " + suffix))
        selector.setCurrentNode(node)

  def createNewOutputs(self):
    try:
      self.createMissingOutputs(replace=True)
    except ValueError as error:
      self.setRegistrationStatus(str(error), error=True)

  def onOutputVolumeChanged(self, node):
    if not self._updatingGUIFromParameterNode and node is not None:
      self.ui.createVolumeCheckBox.checked = True
    self.updateParameterNodeFromGUI()

  def _beginRegistration(self, parameters, remote, serial):
    if not self._startingRegistration or serial != self._runSerial:
      return
    directory = None
    try:
      import importlib.util
      import tempfile
      from UniGradICONHelpers.task import RegistrationTask
      if not remote and any(importlib.util.find_spec(name) is None for name in ("torch", "itk", "icon_registration")):
        self.setRegistrationStatus("Setting up the local registration backend…")
        self.ensureLocalBackend()
      if not self._startingRegistration or serial != self._runSerial:
        return
      directory = tempfile.mkdtemp(prefix="UniGradICON-task-", dir=slicer.app.temporaryPath)
      images = {role: self.logic.snapshotVolume(parameters["image"][role]) for role in ("fixed", "moving")}
      settings = self.logic.registrationSettings(parameters["image"], parameters["outputSettings"],
                                                  parameters["modelSettings"], parameters["generalSettings"])
      url = self.ui.serverComboBox.currentText.strip() if remote else None
      if remote:
        self.saveServerUrl(url)
        self.logic.setServerToken(url, self.ui.remoteTokenLineEdit.text)
      task = RegistrationTask(images, settings, directory, self.logic.weights_location, sys.executable,
                              url=url, token=self.ui.remoteTokenLineEdit.text if remote else None)
      self._registrationTask = task
      self._startingRegistration = False
      task.start()
      self._taskTimer.start()
      self.updateRegistrationGUI()
    except Exception as error:
      if directory:
        import shutil
        shutil.rmtree(directory, ignore_errors=True)
      self._finishRegistration(error=str(error))

  def cancelRegistration(self):
    if not self._registrationTask and not self._startingRegistration:
      return
    self._cancelling = True
    self.setRegistrationStatus("Cancelling… Waiting for the current transfer to finish."
                               if self.ui.remoteProcessingCheckBox.checked else "Cancelling… Stopping local registration.")
    if self._registrationTask:
      self._registrationTask.cancel()
      self.updateRegistrationGUI()
    else:
      self._runSerial += 1
      self._finishRegistration(cancelled=True)

  def shutdownRegistration(self):
    """Stop ownership of a task without touching Qt widgets during application teardown."""
    self._runSerial += 1
    self._startingRegistration = False
    task = self._registrationTask
    self._registrationTask = None
    if task:
      task.abandon()

  def _pollRegistration(self):
    import queue
    task = self._registrationTask
    if task is None:
      return
    while True:
      try:
        kind, payload = task.events.get_nowait()
      except queue.Empty:
        break
      if kind == "status" and not self._cancelling:
        phase = payload.get("phase", self._lastPhase or "Registering")
        if phase != self._lastPhase:
          self.addLog(phase + "…")
          self._lastPhase = phase
        self.setRegistrationStatus(phase + "…")
        if phase == "Registering" and payload.get("total"):
          self.updateProgress(payload["total"], payload.get("step", 0), payload.get("elapsed"), payload.get("remaining"))
        else:
          self.ui.progressBar.setRange(0, 0)
          self.ui.time.hide()
      elif kind == "warning":
        self._lastRunWarning = payload
        self.addLog(payload)
      elif kind == "result":
        if self._cancelling:
          self._finishRegistration(cancelled=True)
        else:
          try:
            output = self._runParameters["outputSettings"]
            for node in (output.get("transform"), output.get("volume")):
              if node is not None and node.GetScene() != slicer.mrmlScene:
                raise ValueError("An output was removed. Create new outputs and run again.")
            self.logic.loadResults(output, payload[0], payload[1])
            self._completedResult = self._runParameters
            self._finishRegistration()
          except Exception as error:
            self._finishRegistration(error=str(error))
        break
      elif kind in ("error", "cancelled"):
        self._finishRegistration(error=payload if kind == "error" else None,
                                  cancelled=kind == "cancelled" or self._cancelling)
        break

  def _finishRegistration(self, error=None, cancelled=False):
    import time
    task = self._registrationTask
    self._taskTimer.stop()
    self._registrationTask = None
    self._startingRegistration = False
    self._cancelling = False
    self.ui.time.hide()
    self.ui.progressBar.setRange(0, 100)
    self.ui.progressBar.value = 100 if not error and not cancelled else 0
    self.ui.progressBar.visible = not error and not cancelled
    if task:
      task.cleanup()
    if cancelled:
      message = "Registration cancelled. Outputs were preserved. Ready to run again."
    elif error:
      message = f"Registration failed: {error}\nCheck the inputs, selected device and server connection, then run again."
    else:
      elapsed = time.monotonic() - self._registrationStart
      message = f"Completed in {elapsed:.0f} s."
      if self._completedResult["outputSettings"].get("volume") is not None:
        message += " Click Compare result to inspect the registration."
        self.ui.comparisonCollapsibleButton.collapsed = False
        self.setComparisonStatus("Choose a comparison mode, then click Compare result.")
      else:
        message += " Transform ready. Enable Create transformed volume for image comparison."
    if self._lastRunWarning:
      message += "\n" + self._lastRunWarning
    self.setRegistrationStatus(message, error=bool(error) and not cancelled)
    self.addLog(message)
    self.updateProcessingGUI()
    self.updateRegistrationGUI()

  def setRegistrationStatus(self, message, error=False, idle=False):
    self._statusIsIdle = idle
    self.ui.registrationStatusLabel.text = message
    self.ui.registrationStatusLabel.setStyleSheet("color: #b83c2b;" if error else "")

  def updateRegistrationGUI(self):
    busy = self._registrationTask is not None or self._startingRegistration
    for widget in (self.ui.parameterNodeSelector, self.ui.InputsCollapsibleButton,
                   self.ui.generalSettingsCollapsibleButton, self.ui.outputsCollapsibleButton):
      widget.enabled = not busy
    self.ui.runRegistrationButton.text = "Cancelling…" if self._cancelling else "Cancel" if busy else "Run registration"
    inputs = bool(self.ui.fixedImageNodeComboBox.currentNodeID and self.ui.movingImageNodeComboBox.currentNodeID)
    if not busy and self._statusIsIdle:
      self.ui.registrationStatusLabel.text = ("Ready. Click Run registration." if inputs
                                               else "Select the fixed and moving images to begin.")
    self.ui.runRegistrationButton.enabled = not self._cancelling and (busy or inputs)
    self.ui.newOutputsButton.enabled = inputs and not busy
    self.ui.outputTransformComboBox.enabled = not busy
    self.ui.outputVolumeComboBox.enabled = self.ui.createVolumeCheckBox.checked and not busy
    populated = self.selectedOutputsWithData()
    self.ui.outputNoticeLabel.text = ("Existing results require confirmation before replacement." if populated
                                     else "Missing outputs are created automatically.")
    self.ui.outputNoticeLabel.toolTip = "\n".join(node.GetName() for node in populated)
    result = self._completedResult
    if result and not busy:
      if (result["image"]["fixed"] != self.ui.fixedImageNodeComboBox.currentNode()
          or result["image"]["moving"] != self.ui.movingImageNodeComboBox.currentNode()
          or result["outputSettings"].get("volume") != self.ui.outputVolumeComboBox.currentNode()):
        self.invalidateComparison()
        result = None
    self.ui.compareResultButton.enabled = bool(not busy and result and result["outputSettings"].get("volume"))
    self.ui.comparisonControlsFrame.enabled = self.ui.compareResultButton.enabled
    checkerboard = self.ui.comparisonModeComboBox.currentText == "Checkerboard"
    self.ui.checkerboardTilesLabel.visible = checkerboard
    self.ui.checkerboardTilesSpinBox.visible = checkerboard
