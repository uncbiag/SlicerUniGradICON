"""Early local dependency and asynchronous remote capability checks."""
import importlib.util
import queue
import threading

import qt

from UniGradICONHelpers.segmentation import (
  UNIGRADICON_VERSION, backend_packages_current, registration_options, require_server_features,
)


class BackendReadinessControls:
  def setupBackendReadiness(self):
    self._backendResults = queue.Queue()
    self._backendPending = None
    self._backendCheckedKey = None
    self._backendCheckError = ""
    self._backendDebounce = qt.QTimer(self.ui.backendStatusLabel)
    self._backendDebounce.setSingleShot(True)
    self._backendDebounce.timeout.connect(self.checkBackend)
    self._backendPoll = qt.QTimer(self.ui.backendStatusLabel)
    self._backendPoll.setInterval(100)
    self._backendPoll.timeout.connect(self.pollBackendCheck)
    self.ui.checkBackendButton.clicked.connect(self.checkBackend)

  def backendKey(self):
    return (self.ui.serverComboBox.currentText.strip(), self.ui.remoteTokenLineEdit.text)

  def backendReadiness(self):
    if self.ui.remoteProcessingCheckBox.checked:
      if self._serverInfo is None:
        return self._backendCheckError or "Connect to the processing server to check its capabilities.", True, False
      try:
        options = self.logic.segmentationSettings(self._parameterNode) if self._parameterNode else {}
        require_server_features(self._serverInfo, registration_options(options))
        for field, value in (("models", self.ui.modelComboBox.currentText), ("losses", self.ui.lossComboBox.currentText)):
          if field in self._serverInfo and value not in self._serverInfo[field]:
            raise ValueError(f"The server does not support the selected {field[:-1]}: {value}.")
      except ValueError as error:
        return str(error), True, False
      return "Remote backend ready for the selected options.", False, False
    if self.logic._backendRestartRequired:
      return "Backend updated. Restart Slicer before registering.", True, False
    if self._localBackendProblem:
      return self._localBackendProblem, True, False
    if not backend_packages_current() or any(importlib.util.find_spec(name) is None for name in ("torch", "itk")):
      return f"uniGradICON {UNIGRADICON_VERSION} required.", False, True
    return f"Local uniGradICON {UNIGRADICON_VERSION} ready: Dice guidance, multi-label IO and loss masking supported.", False, False

  def updateBackendStatus(self, busy=False):
    if not hasattr(self, "_backendResults"):
      return False
    message, blocked, install = self.backendReadiness()
    if not self.ui.remoteProcessingCheckBox.checked:
      self._backendDebounce.stop()
    if self._backendPending is not None and self.ui.remoteProcessingCheckBox.checked:
      message = "Checking server capabilities…"
    self.ui.backendStatusLabel.text = message if blocked or install or self._backendPending else ""
    self.ui.backendStatusLabel.visible = bool(blocked or install or self._backendPending)
    self.ui.checkBackendButton.toolTip = message
    self.ui.backendStatusLabel.setStyleSheet("color: #b83c2b;" if blocked or install else "")
    self.ui.checkBackendButton.text = "Install or update backend" if install else "Check backend"
    self.ui.checkBackendButton.enabled = not busy and self._backendPending is None
    key = self.backendKey()
    if (self.ui.remoteProcessingCheckBox.checked and key[0] and self._serverInfo is None
        and self._backendPending is None and key != self._backendCheckedKey and not busy):
      self._backendCheckedKey = key
      self._backendDebounce.start(400)
    return blocked

  def validateBackendReadiness(self):
    message, blocked, install = self.backendReadiness()
    if blocked:
      raise ValueError(message)

  def checkBackend(self):
    if not self.ui.remoteProcessingCheckBox.checked:
      try:
        self.ensureLocalBackend()
        self._backendCheckError = ""
      except Exception as error:
        self._backendCheckError = str(error)
      self.updateRegistrationGUI()
      if self._backendCheckError:
        self.ui.backendStatusLabel.text = self._backendCheckError
        self.ui.backendStatusLabel.visible = True
      return
    if self._backendPending is not None:
      return
    key = self.backendKey()
    if not key[0]:
      self.updateRegistrationGUI()
      return
    self._backendPending = key
    self._backendCheckedKey = key
    self._backendCheckError = ""
    results = self._backendResults
    def query():
      from UniGradICONHelpers.remote import RegistrationServerClient
      try:
        results.put((key, RegistrationServerClient(*key).info(timeout=5), None))
      except Exception as error:
        results.put((key, None, str(error)))
    threading.Thread(target=query, name="UniGradICON-backend-check", daemon=True).start()
    self._backendPoll.start()
    self.updateRegistrationGUI()

  def pollBackendCheck(self):
    try:
      key, info, error = self._backendResults.get_nowait()
    except queue.Empty:
      return
    self._backendPoll.stop()
    self._backendPending = None
    if self.ui.remoteProcessingCheckBox.checked and key == self.backendKey():
      self._serverInfo = info
      self._backendCheckError = error or ""
      if info:
        self.logic.setServerToken(key[0], key[1])
        self.updateDeviceChoices()
      self.updateProcessingGUI()
    else:
      self._backendCheckedKey = None
      self.updateRegistrationGUI()

  def stopBackendChecks(self):
    if hasattr(self, "_backendPoll"):
      self._backendPoll.stop()
      self._backendDebounce.stop()
      self._backendPending = None
