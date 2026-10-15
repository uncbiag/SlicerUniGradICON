"""Slicer entry point and parameter-node bindings for UniGradICON."""
import json

import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
  ScriptedLoadableModule, ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

from UniGradICONLib.preview import InputPreviewControls
from UniGradICONLib.readiness import BackendReadinessControls
from UniGradICONLib.comparison import ComparisonControls
from UniGradICONLib.logic import UniGradICONLogic, RemoteRegistration
from UniGradICONLib.registration import RegistrationControls
from UniGradICONLib.segmentation import SegmentationControls
from UniGradICONLib.server import ServerControls, RegistrationServerProcess
from UniGradICONLib.settings import (
  PresetManager, REMOTE_PROCESSING_SETTING, SERVER_URL_SETTING,
  SERVER_TOKENS_SETTING_GROUP, WEIGHTS_DIRECTORY,
)
from UniGradICONTests.slicer_tests import UniGradICONTest

# Keep the entry module's imports stable for Slicer and existing registration scripts.
__all__ = [
  "UniGradICON", "UniGradICONWidget", "UniGradICONLogic", "UniGradICONTest",
  "RegistrationServerProcess", "RemoteRegistration", "PresetManager",
  "REMOTE_PROCESSING_SETTING", "SERVER_URL_SETTING", "SERVER_TOKENS_SETTING_GROUP",
]


class UniGradICON(ScriptedLoadableModule):
  """Module metadata and Slicer self-test registration."""

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "UniGradICON"
    self.parent.categories = ["Registration"]
    self.parent.dependencies = []
    self.parent.associatedNodeTypes = ["vtkMRMLScriptedModuleNode"]
    self.parent.contributors = ["Basar Demir (University of North Carolina at Chapel Hill), Lin Tian (University of North Carolina at Chapel Hill), Hastings Greer (University of North Carolina at Chapel Hill), Marc Niethammer (University of North Carolina at Chapel Hill)"]
    self.parent.helpText = """
    This module performs medical image registration using the family of foundational GradICON deep registration models. For more information, visit the <a href="https://github.com/uncbiag/uniGradICON">uniGradICON</a> repository.
    """
    self.parent.acknowledgementText = ""


class UniGradICONWidget(ServerControls, RegistrationControls, ComparisonControls, SegmentationControls,
                        InputPreviewControls, BackendReadinessControls,
                        ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Bind the scene's parameter set to the registration, server, and comparison controls."""

  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False
    self._sceneImportPending = False
    self._sceneImportSerial = 0
    self.checkpointFolder = WEIGHTS_DIRECTORY

    # Server and connection state.
    self._localBackendProblem = None
    self._serverInfo = None
    self._webServer = None
    self._lastReconnectAttempt = None

    # Registration task state.
    self._registrationTask = None
    self._runParameters = None
    self._completedResult = None
    self._cancelling = False
    self._lastPhase = None
    self._startingRegistration = False
    self._runSerial = 0
    self._lastRunWarning = None
    self._statusIsIdle = True
    self._registrationStart = None

    # Comparison state and temporary preview ownership.
    self._checkerboardGeneration = 0
    self._checkerboardJobs = {}
    self._checkerboardNode = None
    self._comparisonViewActive = False
    self._pendingCheckerboard = None
    self._comparisonSaveRestore = []
    self._comparisonSegmentationVisibility = []

  def ensureLocalBackend(self):
    """See UniGradICONLogic.ensureLocalBackend."""
    self.logic.ensureLocalBackend()

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    uiWidget = slicer.util.loadUI(self.resourcePath('UI/UniGradICON.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)
    self.ui.parameterNodeSelector.addAttribute("vtkMRMLScriptedModuleNode", "ModuleName", self.moduleName)

    uiWidget.setMRMLScene(slicer.mrmlScene)

    self.logic = UniGradICONLogic(self.checkpointFolder)

    self.ui.modelComboBox.addItems(PresetManager().getPresetNames())
    self.ui.stagesPresetsComboBox = self.ui.modelComboBox  # its name from SlicerANTs, for scripts

    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartSaveEvent, self.onSceneStartSave)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndSaveEvent, self.onSceneEndSave)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onSceneEndImport)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartImportEvent, self.onSceneStartImport)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeAddedEvent, self.onComparisonNodeAdded)

    self.ui.parameterNodeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onParameterNodeSelected)

    # NB: owned by the GUI, so that it cannot fire once Slicer has destroyed the GUI on exit
    self._reconnectTimer = qt.QTimer(uiWidget)
    self._reconnectTimer.setSingleShot(True)
    self._reconnectTimer.timeout.connect(self.reconnectToServerQuietly)
    self._taskTimer = qt.QTimer(uiWidget)
    self._taskTimer.setInterval(100)
    self._taskTimer.timeout.connect(self._pollRegistration)
    self._checkerboardTimer = qt.QTimer(uiWidget)
    self._checkerboardTimer.setInterval(100)
    self._checkerboardTimer.timeout.connect(self._pollCheckerboards)
    self._checkerboardRefreshTimer = qt.QTimer(uiWidget)
    self._checkerboardRefreshTimer.setSingleShot(True)
    self._checkerboardRefreshTimer.setInterval(200)
    self._checkerboardRefreshTimer.timeout.connect(self._refreshCheckerboard)

    # Persist edits in the selected scene parameter node.
    self.ui.lossComboBox.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
    self.ui.outputTransformComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.outputVolumeComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.onOutputVolumeChanged)
    self.ui.createVolumeCheckBox.toggled.connect(self.updateParameterNodeFromGUI)
    self.ui.ioSpinBox.connect("valueChanged(int)", self.updateParameterNodeFromGUI)
    self.ui.deviceComboBox.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)

    self.ui.fixedImageNodeComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.updateImageSettingsFromGUI)
    self.ui.movingImageNodeComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.updateImageSettingsFromGUI)

    self.ui.fixedModalityComboBox.connect("currentIndexChanged(int)", self.updateImageSettingsFromGUI)
    self.ui.movingModalityComboBox.connect("currentIndexChanged(int)", self.updateImageSettingsFromGUI)

    self.ui.modelComboBox.currentTextChanged.connect(self.onPresetSelected)

    self.ui.runRegistrationButton.connect('clicked(bool)', self.onRunRegistrationButton)
    self.ui.newOutputsButton.clicked.connect(self.createNewOutputs)
    self.ui.compareResultButton.toggled.connect(self.onCompareResult)
    self.ui.comparisonModeComboBox.currentTextChanged.connect(self.onComparisonModeChanged)
    self.ui.checkerboardTilesSpinBox.valueChanged.connect(self.onCheckerboardTilesChanged)
    self.ui.comparisonSegmentationsCheckBox.toggled.connect(self.updateComparisonSegmentations)
    self.ui.outputTransformComboBox.noneDisplay = "Create automatically"
    self.ui.outputVolumeComboBox.noneDisplay = "Create automatically"

    self.ui.progressBar.hide()
    self.ui.time.hide()

    self.setupSegmentationControls()
    self.setupServerControls()
    self.setupInputPreviews()
    self.setupBackendReadiness()
    slicer.app.connect("aboutToQuit()", self.shutdownRegistration)
    slicer.app.connect("aboutToQuit()", self.cancelCheckerboardOnExit)

    self.addLog(self._localBackendProblem or self.localReadinessMessage())
    self.updateDeviceChoices()
    self.updateProcessingGUI()

    self.initializeParameterNode()

  def onReload(self):
    """Reload the adapter packages as well as Slicer's entry script."""
    import importlib
    from UniGradICONHelpers import segmentation as segmentation_options
    from UniGradICONLib import settings, logic, server, registration, comparison, segmentation, preview, readiness
    from UniGradICONTests import slicer_tests, segmentation_tests, workflow_tests
    for module in (segmentation_options, settings, logic, server, registration, comparison, segmentation, preview, readiness, segmentation_tests, workflow_tests, slicer_tests):
      importlib.reload(module)
    ScriptedLoadableModuleWidget.onReload(self)

  def nodeEditable(self, node):
    """This module edits its own parameter nodes, also ones created or chosen by other modules."""
    return 0.7 if node and node.GetAttribute("ModuleName") == self.moduleName else 0.0

  def setEditedNode(self, node, role="", context=""):
    if not self.nodeEditable(node):
      return False
    self.setParameterNode(node)
    return True

  def cleanup(self):
    """Release tasks, temporary previews, the local server, and scene observers."""
    self._sceneImportSerial += 1
    self._sceneImportPending = False
    self.shutdownRegistration()
    self.stopBackendChecks()
    self.clearInputPreview()
    self.clearCheckerboard()
    self._taskTimer.stop()
    self.stopServer()
    self.removeObservers()

  def enter(self):
    self.initializeParameterNode()
    self.updateComparisonSegmentations()
    self._lastReconnectAttempt = None  # entering the module checks the server again
    self.scheduleReconnect()

  def exit(self):
    self.restoreComparisonSegmentations()
    # Do not react to parameter node changes (GUI will be updated when the user enters the module)
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  def onSceneStartClose(self, caller, event):
    self._sceneImportSerial += 1
    self._sceneImportPending = False
    self.cancelRegistration()
    self.clearInputPreview()
    self.clearCheckerboard()
    self._completedResult = None
    self._comparisonSaveRestore = []
    self.ui.compareResultButton.enabled = False
    self.ui.progressBar.hide()
    self.setRegistrationStatus("Select the fixed and moving images to begin.", idle=True)
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    if self.parent.isEntered:
      self.initializeParameterNode()

  def onSceneStartImport(self, caller, event):
    self._sceneImportSerial += 1
    self._sceneImportPending = True

  def onSceneEndImport(self, caller, event):
    # Node selectors finish rebuilding after the scene's EndImport observers run.
    serial = self._sceneImportSerial
    qt.QTimer.singleShot(0, lambda: self._finishSceneImport(serial))

  def _finishSceneImport(self, serial):
    if serial != self._sceneImportSerial:
      return
    self._sceneImportPending = False
    self.initializeParameterNode()

  def initializeParameterNode(self):
    """Restore or create the selected scene parameter node, including after module reload."""
    if self._parameterNode is None or not slicer.mrmlScene.IsNodePresent(self._parameterNode):
      self.setParameterNode(self.defaultParameterNode())
    else:
      self.setParameterNode(self._parameterNode)

  def defaultParameterNode(self):
    """The module's own parameter node, else any other in the scene; created only if there is none."""
    node = slicer.mrmlScene.GetSingletonNode(self.moduleName, "vtkMRMLScriptedModuleNode")
    if node is None:
      nodes = [n for n in slicer.util.getNodesByClass("vtkMRMLScriptedModuleNode")
               if n.GetAttribute("ModuleName") == self.moduleName]
      node = nodes[0] if nodes else self.logic.getParameterNode()
    return node

  def onParameterNodeSelected(self, node):
    if slicer.mrmlScene.IsImporting() or self._sceneImportPending:
      return
    if node is None and self.parent.isEntered and not slicer.mrmlScene.IsClosing():
      node = self.defaultParameterNode()  # the last set was removed
    if node is not self._parameterNode:
      self.setParameterNode(node)

  def setParameterNode(self, inputParameterNode):
    """Observe the selected parameter set and display its settings."""

    if inputParameterNode is not self._parameterNode and self._completedResult is not None:
      self.invalidateComparison()
    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)

    # Reflect scene edits from scripts and other modules in the GUI.
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    wasBlocked = self.ui.parameterNodeSelector.blockSignals(True)
    self.ui.parameterNodeSelector.setCurrentNode(self._parameterNode)
    self.ui.parameterNodeSelector.blockSignals(wasBlocked)

    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """Restore scene settings without writing GUI updates back to the parameter node."""

    if self._parameterNode is None or self._updatingGUIFromParameterNode or slicer.mrmlScene.IsImporting():
      return

    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True

    self.updateImageSettingsGUIFromParameter()
    self.restoreSegmentationSettings()

    self.ui.outputTransformComboBox.setCurrentNode(self._parameterNode.GetNodeReference(self.logic.OUTPUT_TRANSFORM_REF))
    self.ui.outputVolumeComboBox.setCurrentNode(self._parameterNode.GetNodeReference(self.logic.OUTPUT_VOLUME_REF))
    self.ui.createVolumeCheckBox.checked = self._parameterNode.GetParameter(self.logic.CREATE_VOLUME_PARAM) == "true"
    self.ui.lossComboBox.currentText = self._parameterNode.GetParameter(self.logic.LOSS_PARAM)
    self.ui.deviceComboBox.currentText = self._parameterNode.GetParameter(self.logic.DEVICE_PARAM)

    self.ui.ioSpinBox.value = int(self._parameterNode.GetParameter(self.logic.IO_PARAM))

    # where to register; a Slicer that cannot run it locally always uses a server
    remoteProcessing = self.logic.isRemoteProcessing(self._parameterNode) or self._localBackendProblem is not None
    processingChanged = remoteProcessing != self.ui.remoteProcessingCheckBox.checked
    wasBlocked = self.ui.remoteProcessingCheckBox.blockSignals(True)
    self.ui.remoteProcessingCheckBox.checked = remoteProcessing
    self.ui.remoteProcessingCheckBox.blockSignals(wasBlocked)
    serverUrl = self._parameterNode.GetParameter(self.logic.SERVER_URL_PARAM)
    if serverUrl != self.ui.serverComboBox.currentText:
      for widget in (self.ui.serverComboBox, self.ui.remoteTokenLineEdit):
        widget.blockSignals(True)
      self.ui.serverComboBox.setCurrentText(serverUrl)
      self.ui.remoteTokenLineEdit.text = self.logic.serverToken(serverUrl)
      for widget in (self.ui.serverComboBox, self.ui.remoteTokenLineEdit):
        widget.blockSignals(False)
      self._serverInfo = None
      processingChanged = True

    self.updateRegistrationGUI()

    self._updatingGUIFromParameterNode = False
    if processingChanged:
      self.updateDeviceChoices()
      self.updateProcessingGUI()
      self.scheduleReconnect()

  def updateImageSettingsGUIFromParameter(self):
    """Restore the inputs and modalities of the selected parameter set."""
    presetParameters = json.loads(self._parameterNode.GetParameter(self.logic.SETTINGS_JSON_PARAM))
    self.ui.fixedImageNodeComboBox.setCurrentNode(self.logic.inputVolumeNode(self._parameterNode, "fixed"))
    self.ui.movingImageNodeComboBox.setCurrentNode(self.logic.inputVolumeNode(self._parameterNode, "moving"))
    # the model of the shown node, without applying its preset again
    wasBlocked = self.ui.modelComboBox.blockSignals(True)
    self.ui.modelComboBox.currentText = presetParameters['modelSettings']['model']
    self.ui.modelComboBox.blockSignals(wasBlocked)
    fixedModality, movingModality = self.logic.modalityNames(
      presetParameters['image']['modality-fixed'], presetParameters['image']['modality-moving'])
    self.ui.fixedModalityComboBox.currentText = fixedModality
    self.ui.movingModalityComboBox.currentText = movingModality
    self.logic.changeModelSettings(presetParameters)

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """Save control values and output references in the selected parameter set."""

    if self._parameterNode is None or self._updatingGUIFromParameterNode or slicer.mrmlScene.IsImporting() or self._sceneImportPending:
      return

    wasModified = self._parameterNode.StartModify()

    self.saveSegmentationSettings()
    self._parameterNode.SetNodeReferenceID(self.logic.OUTPUT_TRANSFORM_REF, self.ui.outputTransformComboBox.currentNodeID)
    self._parameterNode.SetNodeReferenceID(self.logic.OUTPUT_VOLUME_REF, self.ui.outputVolumeComboBox.currentNodeID)
    self._parameterNode.SetParameter(self.logic.LOSS_PARAM, self.ui.lossComboBox.currentText)
    self._parameterNode.SetParameter(self.logic.IO_PARAM, str(self.ui.ioSpinBox.value))
    self._parameterNode.SetParameter(self.logic.DEVICE_PARAM, self.ui.deviceComboBox.currentText)
    self._parameterNode.SetParameter(self.logic.CREATE_VOLUME_PARAM,
                                     "true" if self.ui.createVolumeCheckBox.checked else "false")
    self._parameterNode.SetParameter(self.logic.REMOTE_PROCESSING_PARAM,
                                     "true" if self.ui.remoteProcessingCheckBox.checked else "false")
    self._parameterNode.SetParameter(self.logic.SERVER_URL_PARAM, self.ui.serverComboBox.currentText.strip())

    self._parameterNode.EndModify(wasModified)

  def imageSettingsFromGUI(self):
    return {
      "fixed": self.ui.fixedImageNodeComboBox.currentNodeID,
      "moving": self.ui.movingImageNodeComboBox.currentNodeID,
      "modality-fixed": self.ui.fixedModalityComboBox.currentText,
      "modality-moving": self.ui.movingModalityComboBox.currentText,
    }

  def updateImageSettingsFromGUI(self):
    """Store the selected images and their modalities in the settings (StagesJson)."""
    if self._parameterNode is None or self._updatingGUIFromParameterNode or slicer.mrmlScene.IsImporting() or self._sceneImportPending:
      return
    presetParameters = json.loads(self._parameterNode.GetParameter(self.logic.SETTINGS_JSON_PARAM))
    presetParameters['image'].update(self.imageSettingsFromGUI())

    wasModified = self._parameterNode.StartModify()
    try:
      for role, referenceRole in self.logic.INPUT_VOLUME_REFS.items():
        self._parameterNode.SetNodeReferenceID(referenceRole, presetParameters['image'][role] or "")
      self._parameterNode.SetParameter(self.logic.SETTINGS_JSON_PARAM, json.dumps(presetParameters))
    finally:
      self._parameterNode.EndModify(wasModified)

  def onPresetSelected(self, presetName):
    if presetName == 'Select...' or self._parameterNode is None or self._updatingGUIFromParameterNode:
      return
    wasModified = self._parameterNode.StartModify()
    presetParameters = PresetManager().getPresetParametersByName(presetName)
    presetParameters['image'].update(self.imageSettingsFromGUI())

    self._parameterNode.SetParameter(self.logic.SETTINGS_JSON_PARAM, json.dumps(presetParameters))
    self._parameterNode.EndModify(wasModified)

    self.logic.changeModelSettings(presetParameters)
