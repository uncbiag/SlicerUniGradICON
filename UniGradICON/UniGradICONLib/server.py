"""Local server lifecycle, remote connection settings, and device choices."""
import os
import sys

import numpy as np
import qt
import slicer

from .settings import MODULE_DIRECTORY, REMOTE_PROCESSING_SETTING, SERVER_URL_SETTING


class ServerControls:
  """Server controls mixed into the module widget; all callbacks run on the Qt thread."""

  REMOTE_PROCESSING_SETTING = REMOTE_PROCESSING_SETTING
  SERVER_URL_SETTING = SERVER_URL_SETTING
  SERVER_URL_HISTORY_SETTING = "UniGradICON/ServerUrlHistory"
  SERVER_PORT_SETTING = "UniGradICON/ServerPort"
  SERVER_JOBS_PER_GPU_SETTING = "UniGradICON/ServerJobsPerGpu"
  LOG_CONSOLE_SETTING = "UniGradICON/ServerLogToConsole"
  LOG_GUI_SETTING = "UniGradICON/ServerLogToGui"
  DEFAULT_SERVER_PORT = 8899
  WEIGHTS_FILE_MB = 283
  RECONNECT_TIMEOUT = 5

  def setupServerControls(self):
    """Restore connection preferences and connect the server controls."""
    settings = qt.QSettings()
    self._localBackendProblem = self.detectLocalBackendProblem()
    self.updateServerUrlGUIFromSettings()
    self.ui.serverComboBox.lineEdit().setPlaceholderText("server:8899")
    self.ui.remoteProcessingCheckBox.checked = (
      settings.value(self.REMOTE_PROCESSING_SETTING) == "true" or self._localBackendProblem is not None)
    self.ui.remoteProcessingCheckBox.toggled.connect(self.onRemoteProcessingCheckBoxToggled)
    self.ui.remoteServerButton.toggled.connect(self.onRemoteServerButtonToggled)
    self.ui.remoteTokenLineEdit.text = self.logic.serverToken(self.ui.serverComboBox.currentText)
    self.ui.serverComboBox.currentTextChanged.connect(self.onServerUrlChanged)
    self.ui.remoteTokenLineEdit.textChanged.connect(self.onServerSettingsChanged)

    self.ui.portSpinBox.value = int(settings.value(self.SERVER_PORT_SETTING) or self.DEFAULT_SERVER_PORT)
    self.ui.portSpinBox.valueChanged.connect(lambda port: qt.QSettings().setValue(self.SERVER_PORT_SETTING, str(port)))
    self.ui.jobsPerGpuSpinBox.value = int(settings.value(self.SERVER_JOBS_PER_GPU_SETTING) or 1)
    self.ui.jobsPerGpuSpinBox.valueChanged.connect(
      lambda jobs: qt.QSettings().setValue(self.SERVER_JOBS_PER_GPU_SETTING, str(jobs)))
    for checkBox, key in ((self.ui.logConsoleCheckBox, self.LOG_CONSOLE_SETTING),
                          (self.ui.logGuiCheckBox, self.LOG_GUI_SETTING)):
      if settings.value(key) is not None:
        checkBox.checked = settings.value(key) == "true"
      checkBox.toggled.connect(lambda checked, key=key: qt.QSettings().setValue(key, "true" if checked else "false"))
    self.ui.serverButton.toggled.connect(self.onServerButtonToggled)
    self.ui.copyServerAddressButton.clicked.connect(self.copyServerAddress)
    slicer.app.connect("aboutToQuit()", self.shutdownServer)  # cleanup() does not run on exit

  def detectLocalBackendProblem(self):
    """Why this Slicer cannot run the registration itself, or None.

    A missing PyTorch is no problem: it is installed on the first local registration. One that
    is installed but cannot use this Slicer's NumPy is, and no install fixes it here.
    """
    import importlib.util
    if importlib.util.find_spec("torch") is None:
      return None
    try:
      import torch
      torch.from_numpy(np.zeros(1))
    except Exception as e:
      if "Numpy is not available" in str(e) or "_ARRAY_API" in str(e):
        version = getattr(sys.modules.get("torch"), "__version__", "")
        return (f"PyTorch {version} was built for NumPy 1.x and cannot use NumPy {np.__version__}, "
                "which this Slicer runs on. Use remote processing instead.")
      return f"PyTorch cannot run in this Slicer ({e}). Use remote processing instead."
    return None

  def localReadinessMessage(self):
    """What a registration in this Slicer still needs, found without installing anything."""
    import importlib.util
    from UniGradICONHelpers import registration
    if importlib.util.find_spec("PyTorchUtils") is None:
      return ("To register in this Slicer, install the PyTorch extension from the Extensions Manager "
              "and restart Slicer; or use remote processing.")
    packages = [name for name, module in (("PyTorch", "torch"), ("icon_registration", "icon_registration"))
                if importlib.util.find_spec(module) is None]
    missingWeights = self.logic.modelName not in registration.available_models(self.checkpointFolder)
    steps = []
    if packages:
      steps.append(f"install {' and '.join(packages)}")
    if missingWeights:
      steps.append(f"download the model weights (about {self.WEIGHTS_FILE_MB} MB)")
    if not steps:
      return "Ready to register in this Slicer."
    return f"The first registration in this Slicer will {' and '.join(steps)}."

  def addLog(self, text):
    """Append text to the log window."""
    statusLog = self.ui.statusLog
    if len(statusLog.plainText) > 1024 * 256:
      statusLog.clear()
      statusLog.insertPlainText("Log cleared\n")
    statusLog.moveCursor(qt.QTextCursor.End)
    statusLog.insertPlainText(text + "\n")
    statusLog.ensureCursorVisible()

  def addServerLog(self, *lines):
    for line in lines:
      if self.ui.logConsoleCheckBox.checked:
        print(line)
      if self.ui.logGuiCheckBox.checked:
        self.addLog(line)

  def localDevices(self):
    if self._localBackendProblem is not None:
      return ["CPU"]
    try:
      import torch
      return ["CPU", "GPU"] if torch.cuda.is_available() else ["CPU"]
    except ImportError:
      return ["CPU"]

  def updateDeviceChoices(self):
    """The devices of where the registration runs: this Slicer, or the connected server.

    While the server's devices are not known (not connected yet), the device stored in the parameter node is kept
    as a choice, so showing a node does not change it.
    """
    storedDevice = (self.logic.deviceName(self._parameterNode.GetParameter(self.logic.DEVICE_PARAM))
                    if self._parameterNode is not None and self._parameterNode.GetParameter(self.logic.DEVICE_PARAM)
                    else None)
    if self.ui.remoteProcessingCheckBox.checked:
      if self._serverInfo:
        devices = list(self._serverInfo["devices"])
      else:
        devices = ["CPU"] + ([storedDevice] if storedDevice and storedDevice != "CPU" else [])
    else:
      devices = self.localDevices()
    comboBox = self.ui.deviceComboBox
    current = comboBox.currentText
    wasBlocked = comboBox.blockSignals(True)
    comboBox.clear()
    comboBox.addItems(devices)
    if storedDevice in devices:
      comboBox.currentText = storedDevice
    else:
      comboBox.currentText = current if current in devices else devices[0]
    comboBox.blockSignals(wasBlocked)
    self.updateParameterNodeFromGUI()

  def updateProcessingGUI(self):
    remote = self.ui.remoteProcessingCheckBox.checked
    localProblem = self._localBackendProblem
    # without a working local backend, remote processing is the only way
    self.ui.remoteProcessingCheckBox.enabled = not (localProblem and remote)
    self.ui.remoteProcessingCheckBox.toolTip = localProblem or "Register on a UniGradICON registration server instead of in this Slicer."
    self.ui.serverConnectionFrame.enabled = remote
    self.ui.serverConnectionFrame.visible = remote
    self.ui.remoteTokenLabel.visible = self.ui.remoteTokenLineEdit.visible = remote

    connected = self._serverInfo is not None
    wasBlocked = self.ui.remoteServerButton.blockSignals(True)
    self.ui.remoteServerButton.checked = connected
    self.ui.remoteServerButton.blockSignals(wasBlocked)
    self.ui.remoteServerButton.text = "Connected" if connected else "Connect"

    self.ui.serverCollapsibleButton.enabled = localProblem is None
    self.ui.serverCollapsibleButton.toolTip = localProblem or ""
    serverRunning = self._webServer is not None and self._webServer.isRunning()
    wasBlocked = self.ui.serverButton.blockSignals(True)
    self.ui.serverButton.checked = serverRunning
    self.ui.serverButton.blockSignals(wasBlocked)
    self.ui.serverButton.text = "Stop server" if serverRunning else "Start server"
    self.ui.portSpinBox.enabled = self.ui.serverTokenLineEdit.enabled = not serverRunning
    self.ui.jobsPerGpuSpinBox.enabled = not serverRunning
    for widget in (self.ui.serverAddressTitleLabel, self.ui.serverAddressLabel, self.ui.copyServerAddressButton):
      widget.visible = serverRunning
    self.ui.serverAddressLabel.text = self._webServer.address if serverRunning else ""
    self.updateRegistrationGUI()

  def onRemoteProcessingCheckBoxToggled(self, checked):
    qt.QSettings().setValue(self.REMOTE_PROCESSING_SETTING, "true" if checked else "false")
    self.updateParameterNodeFromGUI()
    self.updateDeviceChoices()
    self.updateProcessingGUI()
    if checked:
      self.scheduleReconnect()

  def onServerUrlChanged(self, url):
    # the token remembered for this server, if any
    wasBlocked = self.ui.remoteTokenLineEdit.blockSignals(True)
    self.ui.remoteTokenLineEdit.text = self.logic.serverToken(url.strip())
    self.ui.remoteTokenLineEdit.blockSignals(wasBlocked)
    self.updateParameterNodeFromGUI()
    self.onServerSettingsChanged()

  def onServerSettingsChanged(self, text=None):
    # another server or token: connect again
    if self._serverInfo is not None:
      self._serverInfo = None
      self.updateDeviceChoices()
    self.updateProcessingGUI()

  def serverUrlHistory(self):
    value = qt.QSettings().value(self.SERVER_URL_HISTORY_SETTING)
    return [url for url in (value or "").split(";") if url]

  def saveServerUrl(self, url):
    settings = qt.QSettings()
    settings.setValue(self.SERVER_URL_SETTING, url)
    history = [url] + [other for other in self.serverUrlHistory() if other != url]
    settings.setValue(self.SERVER_URL_HISTORY_SETTING, ";".join(history[:10]))
    self.updateServerUrlGUIFromSettings()

  def updateServerUrlGUIFromSettings(self):
    comboBox = self.ui.serverComboBox
    wasBlocked = comboBox.blockSignals(True)
    comboBox.clear()
    comboBox.addItems(self.serverUrlHistory())
    comboBox.setCurrentText(qt.QSettings().value(self.SERVER_URL_SETTING) or "")
    comboBox.blockSignals(wasBlocked)

  def connectToServer(self, timeout=None):
    from UniGradICONHelpers import remote
    url = self.ui.serverComboBox.currentText.strip()
    if not url:
      raise remote.ServerError("Enter the address of a registration server.")
    token = self.ui.remoteTokenLineEdit.text
    client = remote.RegistrationServerClient(url, token)
    self._serverInfo = client.info(timeout=timeout) if timeout else client.info()
    self.saveServerUrl(url)
    # remembered on this computer, for registrations run without this panel (never in the scene)
    self.logic.setServerToken(url, token)
    capacity = (f", {self._serverInfo['jobsPerGpu']} job(s) at a time on each of {self._serverInfo['gpus']} GPU(s)"
                if self._serverInfo.get("gpus") else "")
    self.addLog(f"Connected to {client.url} (devices: {', '.join(self._serverInfo['devices'])}{capacity}).")
    self.updateDeviceChoices()
    self.updateProcessingGUI()
    return client

  def scheduleReconnect(self):
    """Check the server again once the GUI is updated (see reconnectToServerQuietly)."""
    self._reconnectTimer.start(0)

  def reconnectToServerQuietly(self):
    """Connect to the server of the shown parameter node again, without dialogs.

    Registrations do not need it, they reach the server by its address, but it shows whether the server answers
    and which devices it offers. Tried once per server and token, until they change or the module is entered again.
    """
    from UniGradICONHelpers import remote
    if (self._parameterNode is None or not self.ui.remoteProcessingCheckBox.checked
        or self._serverInfo is not None or self._registrationTask is not None):
      return
    url = self.ui.serverComboBox.currentText.strip()
    attempt = (url, self.ui.remoteTokenLineEdit.text)
    if not url or attempt == self._lastReconnectAttempt:
      return
    self._lastReconnectAttempt = attempt
    try:
      with slicer.util.WaitCursor():
        self.connectToServer(timeout=self.RECONNECT_TIMEOUT)
    except remote.ServerError as e:
      self._serverInfo = None
      self.addLog(f"Not connected to {url}: {e}")
      self.updateProcessingGUI()

  def onRemoteServerButtonToggled(self, checked):
    from UniGradICONHelpers import remote
    if checked:
      try:
        with slicer.util.WaitCursor():
          self.connectToServer()
      except remote.ServerError as e:
        self._serverInfo = None
        self.addLog(f"Connection failed: {e}")
        slicer.util.warningDisplay(f"{e}\n\nPlease check the address, port, token and connection.")
    else:
      self._serverInfo = None
    self.updateDeviceChoices()
    self.updateProcessingGUI()

  def ensureServerPackages(self):
    import importlib.util
    if all(importlib.util.find_spec(name) for name in ("fastapi", "uvicorn")) and (
        importlib.util.find_spec("python_multipart") or importlib.util.find_spec("multipart")):
      return
    if not slicer.util.confirmOkCancelDisplay(
        "Running a registration server needs the Python packages fastapi, uvicorn and python-multipart. "
        "Install them now?"):
      raise RuntimeError("The server packages are not installed.")
    self.addLog("Installing fastapi, uvicorn and python-multipart...")
    slicer.util.pip_install("fastapi uvicorn python-multipart")

  def onServerButtonToggled(self, checked):
    if checked:
      self.startServer()
    else:
      self.stopServer()

  def startServer(self):
    try:
      self.ensureLocalBackend()
      self.ensureServerPackages()
    except Exception as e:
      self.addLog(f"Failed to start the server: {e}")
      slicer.util.errorDisplay(f"Failed to start the registration server: {e}")
      self.updateProcessingGUI()
      return
    token = self.ui.serverTokenLineEdit.text
    self._webServer = RegistrationServerProcess(
      self.ui.portSpinBox.value, token, self.checkpointFolder, jobsPerGpu=self.ui.jobsPerGpuSpinBox.value,
      logCallback=self.addServerLog, stoppedCallback=self.onServerStopped)
    self._webServer.start()
    if self.ui.logGuiCheckBox.checked:
      self.ui.detailsCollapsibleButton.collapsed = False
    if token:
      self.addLog(f"Server started at {self._webServer.address}; clients need the token.")
    else:
      self.addLog(f"Server started at {self._webServer.address}. It has no token: anyone who can reach "
                  "this computer can use it.")
    self.updateProcessingGUI()

  def stopServer(self):
    if self._webServer is not None:
      self._webServer.stop()  # reports through onServerStopped

  def shutdownServer(self):
    server = self._webServer
    self._webServer = None
    if server is not None:
      server.logCallback = None
      server.stoppedCallback = None
      server.stop()

  def onServerStopped(self, returnCode, stoppedByUser):
    self._webServer = None
    if not hasattr(self, "ui"):
      return
    if stoppedByUser:
      self.addLog("Server was stopped.")
    else:
      self.addLog(f"Server exited with code {returnCode}. Enable 'Log to GUI' and start it again for details.")
    self.updateProcessingGUI()

  def copyServerAddress(self):
    """Put the address of the running server on the clipboard, as a client enters it to connect."""
    address = self.ui.serverAddressLabel.text
    qt.QApplication.clipboard().setText(address)
    slicer.util.showStatusMessage(f"Copied {address} to the clipboard.", 3000)


class RegistrationServerProcess:
  """A registration server (UniGradICONServer/server.py) run with this Slicer's Python.

  Its output is read on a thread, so a full pipe never stalls the server, and handed to
  ``logCallback`` on the main thread; ``stoppedCallback(returnCode, stoppedByUser)`` reports its end.
  """

  OUTPUT_CHECK_INTERVAL_MS = 200

  def __init__(self, port, token=None, weightsDir=None, host="0.0.0.0", jobsPerGpu=1, logCallback=None,
               stoppedCallback=None):
    import socket
    self.port = port
    self.jobsPerGpu = jobsPerGpu
    self.token = token
    self.weightsDir = weightsDir
    self.host = host
    self.logCallback = logCallback
    self.stoppedCallback = stoppedCallback
    self.address = f"{socket.gethostname() if host == '0.0.0.0' else host}:{port}"
    self.process = None
    self._timer = None
    self._stopping = False

  def start(self):
    import queue
    import threading
    script = os.path.join(MODULE_DIRECTORY, "UniGradICONServer", "server.py")
    arguments = ["--host", self.host, "--port", str(self.port), "--jobs-per-gpu", str(self.jobsPerGpu)]
    if self.weightsDir:
      arguments += ["--weights-dir", self.weightsDir]
    # sys.executable is the PythonSlicer launcher, which runs the actual Python as its child:
    # stopping the launcher would leave that running, so the server stops itself once the
    # launcher or this Slicer is gone
    arguments += ["--exit-with-parent", "--parent-pid", str(os.getpid())]
    environment = {"UNIGRADICON_SERVER_TOKEN": self.token} if self.token else None
    self.process = slicer.util.launchConsoleProcess([sys.executable, script] + arguments,
                                                    useStartupEnvironment=False, updateEnvironment=environment)
    self._output = queue.Queue()
    threading.Thread(target=self._readOutput, daemon=True).start()
    self._timer = qt.QTimer()
    self._timer.setInterval(self.OUTPUT_CHECK_INTERVAL_MS)
    self._timer.timeout.connect(self._checkProcess)
    self._timer.start()

  def isRunning(self):
    return self.process is not None and self.process.poll() is None

  def _readOutput(self):
    for line in self.process.stdout:
      self._output.put(line.rstrip())

  def _drainOutput(self):
    import queue
    lines = []
    while True:
      try:
        lines.append(self._output.get_nowait())
      except queue.Empty:
        break
    if lines and self.logCallback:
      self.logCallback(*lines)

  def _checkProcess(self):
    self._drainOutput()
    if not self.isRunning():
      self._finish()

  def _finish(self):
    if self._timer is None:
      return
    self._timer.stop()
    self._timer = None
    self._drainOutput()
    if self.stoppedCallback:
      self.stoppedCallback(self.process.returncode, self._stopping)

  def stop(self):
    self._stopping = True
    if self.isRunning():
      self.process.terminate()
      try:
        self.process.wait(5)
      except Exception:
        self.process.kill()
        self.process.wait()
    self._finish()
