"""GUI and scene regression tests; expensive model tests are opt-in."""
import json
import logging
import os
import sys

import numpy as np
import qt
import SimpleITK as sitk
import slicer
import vtk
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest

from UniGradICONLib.logic import UniGradICONLogic, RemoteRegistration
from UniGradICONLib.server import RegistrationServerProcess
from UniGradICONLib.settings import WEIGHTS_DIRECTORY


class UniGradICONTest(ScriptedLoadableModuleTest):
  """Scene and GUI regressions, with optional real model registrations."""

  # the registration tests take minutes and need PyTorch (and, for the remote test, the server
  # packages), so they run only when asked for
  FULL_TESTS_ENVIRONMENT_VARIABLE = "UNIGRADICON_FULL_TESTS"

  @classmethod
  def fullTestsEnabled(cls):
    return os.environ.get(cls.FULL_TESTS_ENVIRONMENT_VARIABLE, "").lower() in ("1", "true", "yes", "on")

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    for test in (self.test_parameterSetInputs, self.test_remoteResultDownloads,
                 self.test_interactiveRegistrationFlow, self.test_checkerboardComparison,
                 self.test_reviewRegressions, self.test_segmentationSupport, self.test_registrationWorkflow, self.test_UniGradICON, self.test_remote,
                 self.test_parameterNodeRegistration):
      self.setUp()
      test()

  def test_registrationWorkflow(self):
    from UniGradICONTests.workflow_tests import run_tests
    run_tests(self)

  def test_segmentationSupport(self):
    from UniGradICONTests.segmentation_tests import run_tests
    run_tests(self)

  def test_interactiveRegistrationFlow(self):
    """Automatic outputs, locking, cancellation, progress, comparison and recovery."""
    import queue
    import shutil
    import time
    from unittest.mock import patch
    from UniGradICONHelpers import task as task_module
    widget = slicer.modules.unigradicon.widgetRepresentation().self()
    widget.initializeParameterNode()
    fixed = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "UX fixed")
    moving = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "UX moving")
    data = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    slicer.util.updateVolumeFromArray(fixed, data)
    slicer.util.updateVolumeFromArray(moving, data)
    widget.ui.fixedImageNodeComboBox.setCurrentNode(fixed)
    widget.ui.movingImageNodeComboBox.setCurrentNode(moving)
    widget.ui.outputTransformComboBox.setCurrentNode(None)
    widget.ui.outputVolumeComboBox.setCurrentNode(None)
    widget.ui.createVolumeCheckBox.checked = True
    widget.ui.remoteProcessingCheckBox.checked = False

    class Task:
      def __init__(self, images, settings, directory, *args, **kwargs):
        self.events = queue.Queue()
        self.directory = directory
        self.settings = settings
      def start(self):
        self.events.put(("status", {"phase": "Loading model"}))
      def cancel(self):
        self.events.put(("cancelled", None))
      def cleanup(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def begin():
      widget.onRunRegistrationButton()
      self.assertEqual(widget.ui.runRegistrationButton.text, "Cancel", widget.ui.registrationStatusLabel.text)
      self.assertFalse(widget.ui.parameterNodeSelector.enabled)
      self.assertFalse(widget.ui.fixedImageNodeComboBox.isEnabled())
      self.assertTrue(widget.ui.runRegistrationButton.enabled)
      deadline = time.monotonic() + 5
      while widget._registrationTask is None:
        self.assertLess(time.monotonic(), deadline)
        slicer.app.processEvents()
      widget._pollRegistration()
      return widget._registrationTask

    with patch.object(task_module, "RegistrationTask", Task), \
         patch.object(slicer.util, "infoDisplay", side_effect=AssertionError("No success popup")):
      current = begin()
      self.assertEqual(widget.ui.progressBar.maximum, 0)
      transform = widget.ui.outputTransformComboBox.currentNode()
      volume = widget.ui.outputVolumeComboBox.currentNode()
      self.assertIn("UX moving to UX fixed", transform.GetName())
      self.assertIsNotNone(volume)
      widget.updateProgress(10, 0)
      self.assertEqual(widget.ui.progressBar.value, 0)
      widget.updateProgress(10, 4, 3, 5)
      self.assertEqual(widget.ui.progressBar.value, 4)
      transformPath = os.path.join(current.directory, "transform.tfm")
      sitk.WriteTransform(sitk.AffineTransform(3), transformPath)
      current.events.put(("result", (transformPath, sitk.GetImageFromArray(data + 1))))
      widget._pollRegistration()
      self.assertIsNone(widget._registrationTask)
      self.assertEqual(widget.ui.runRegistrationButton.text, "Run registration")
      self.assertTrue(widget.ui.parameterNodeSelector.enabled)
      self.assertEqual(widget.ui.progressBar.value, 100)
      self.assertIn("Completed in", widget.ui.registrationStatusLabel.text)
      self.assertTrue(widget.ui.compareResultButton.enabled)
      with patch.object(slicer.util, "setSliceViewerLayers") as layers, patch.object(slicer.util, "resetSliceViews"):
        widget.ui.compareResultButton.checked = True
        layers.assert_called_with(background=fixed, foreground=volume, foregroundOpacity=0.5)
        widget.ui.compareResultButton.checked = False
        layers.assert_called_with(background=fixed, foreground=moving, foregroundOpacity=0.5)
      original = slicer.util.arrayFromVolume(volume).copy()
      for label, expected in (("Create new outputs", "new"), ("Overwrite", "overwrite"), ("Cancel", "cancel")):
        def choose(label=label):
          for popup in qt.QApplication.topLevelWidgets():
            if isinstance(popup, qt.QMessageBox) and popup.isVisible() and popup.windowTitle == "Preserve existing registration results":
              self.assertEqual(popup.defaultButton().text, "Create new outputs")
              for button in popup.buttons():
                if button.text.replace("&", "") == label:
                  button.click()
                  return
        qt.QTimer.singleShot(0, choose)
        self.assertEqual(widget.confirmOutputReplacement([volume]), expected)
      with patch.object(widget, "confirmOutputReplacement", return_value="cancel"):
        self.assertFalse(widget.prepareRunOutputs())
      with patch.object(widget, "confirmOutputReplacement", return_value="new"):
        self.assertTrue(widget.prepareRunOutputs())
      self.assertNotEqual(widget.ui.outputVolumeComboBox.currentNodeID, volume.GetID())
      np.testing.assert_array_equal(slicer.util.arrayFromVolume(volume), original)
      current = begin()
      widget.cancelRegistration()
      self.assertEqual(widget.ui.runRegistrationButton.text, "Cancelling…")
      self.assertFalse(widget.ui.runRegistrationButton.enabled)
      widget._pollRegistration()
      self.assertIsNone(widget._registrationTask)
      self.assertIn("cancelled", widget.ui.registrationStatusLabel.text)
      self.assertFalse(os.path.exists(current.directory))
      current = begin()
      current.events.put(("error", "Server disconnected"))
      widget._pollRegistration()
      self.assertIn("Server disconnected", widget.ui.registrationStatusLabel.text)
      self.assertTrue(widget.ui.runRegistrationButton.enabled)
      widget.ui.createVolumeCheckBox.checked = False
      current = begin()
      self.assertFalse(current.settings["warp"])
      widget.cancelRegistration()
      widget._pollRegistration()
    widget.ui.createVolumeCheckBox.checked = True

  def test_checkerboardComparison(self):
    """Native asynchronous checkerboard: both images, geometry, tile count and cleanup."""
    import time
    from unittest.mock import patch
    slicer.util.selectModule("UniGradICON")
    widget = slicer.modules.unigradicon.widgetRepresentation().self()
    widget.initializeParameterNode()
    nodes = []
    for name, value in (("Checker fixed", 100), ("Checker original", 200), ("Checker result", 300)):
      node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
      slicer.util.updateVolumeFromArray(node, np.full((8, 8, 8), value, dtype=np.float32))
      node.SetSpacing(0.6, 1.2, 1.8)
      node.SetOrigin(11, -9, 5)
      node.CreateDefaultDisplayNodes()
      node.GetDisplayNode().SetAutoWindowLevel(False)
      node.GetDisplayNode().SetWindowLevel(400, 200)
      nodes.append(node)
    fixed, moving, registered = nodes
    widget.ui.fixedImageNodeComboBox.setCurrentNode(fixed)
    widget.ui.movingImageNodeComboBox.setCurrentNode(moving)
    widget.ui.outputVolumeComboBox.setCurrentNode(registered)
    # Non-overlapping original verifies that the CLI resamples in physical space,
    # rather than stitching arrays just because their voxel dimensions match.
    moving.SetOrigin(1000, 1000, 1000)
    widget._completedResult = {"image": {"fixed": fixed, "moving": moving}, "outputSettings": {"volume": registered}}
    widget.ui.checkerboardTilesSpinBox.value = 2
    widget.updateRegistrationGUI()

    def wait():
      deadline = time.monotonic() + 30
      while widget._checkerboardJobs or widget._checkerboardRefreshTimer.isActive():
        self.assertLess(time.monotonic(), deadline, widget.ui.registrationStatusLabel.text)
        slicer.app.processEvents()
        time.sleep(0.01)
      self.assertIsNotNone(widget._checkerboardNode, widget.ui.registrationStatusLabel.text)
      return widget._checkerboardNode

    def verify(node, values, tiles):
      data = slicer.util.arrayFromVolume(node)
      self.assertEqual(set(np.unique(data)), set(values))
      z, y, x = np.indices(data.shape)
      parity = (x // (8 // tiles) + y // (8 // tiles) + z // (8 // tiles)) % 2
      self.assertEqual(len(np.unique(data[parity == 0])), 1)
      self.assertEqual(len(np.unique(data[parity == 1])), 1)
      self.assertNotEqual(data[parity == 0][0], data[parity == 1][0])
      np.testing.assert_allclose(node.GetSpacing(), fixed.GetSpacing(), atol=1e-12, rtol=0)
      np.testing.assert_allclose(node.GetOrigin(), fixed.GetOrigin(), atol=1e-12, rtol=0)
      self.assertTrue(node.GetHideFromEditors())
      self.assertFalse(node.GetSaveWithScene())
      self.assertFalse(node.GetDisplayNode().GetSaveWithScene())

    with patch.object(slicer.util, "setSliceViewerLayers") as layers, patch.object(slicer.util, "resetSliceViews"):
      widget.ui.comparisonModeComboBox.currentText = "Checkerboard"
      output = wait()
      verify(output, (100, 300), 2)
      layers.assert_called_with(background=output, foreground=None)
      self.assertTrue(widget.ui.compareResultButton.checked)
      segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Comparison labels")
      segmentation.CreateDefaultDisplayNodes()
      display = segmentation.GetDisplayNode()
      display.SetVisibility2D(True)
      display.SetVisibility3D(False)
      hidden = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Already hidden labels")
      hidden.CreateDefaultDisplayNodes()
      hidden.GetDisplayNode().SetVisibility2D(False)
      label_view = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSliceCompositeNode", "Label overlay visibility")
      label_view.SetLabelOpacity(0.35)
      widget.ui.comparisonSegmentationsCheckBox.checked = False
      self.assertFalse(display.GetVisibility2D())
      self.assertFalse(display.GetVisibility3D())
      self.assertEqual(label_view.GetLabelOpacity(), 0)
      added = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Labels added during comparison")
      added.CreateDefaultDisplayNodes()
      slicer.app.processEvents()
      self.assertFalse(added.GetDisplayNode().GetVisibility2D())
      widget.updateComparisonSegmentations()
      widget.ui.comparisonSegmentationsCheckBox.checked = True
      self.assertTrue(display.GetVisibility2D())
      self.assertTrue(added.GetDisplayNode().GetVisibility2D())
      self.assertFalse(hidden.GetDisplayNode().GetVisibility2D())
      self.assertAlmostEqual(label_view.GetLabelOpacity(), 0.35)
      widget.ui.comparisonSegmentationsCheckBox.checked = False
      widget.onSceneStartSave(None, None)
      self.assertTrue(display.GetVisibility2D())
      self.assertAlmostEqual(label_view.GetLabelOpacity(), 0.35)
      widget.onSceneEndSave(None, None)
      self.assertFalse(display.GetVisibility2D())
      self.assertEqual(label_view.GetLabelOpacity(), 0)
      slicer.util.selectModule("Data")
      self.assertTrue(display.GetVisibility2D())
      widget.onSceneStartSave(None, None)
      widget.onSceneEndSave(None, None)
      self.assertTrue(display.GetVisibility2D(), "Saving from another module must not hide labels")
      slicer.util.selectModule("UniGradICON")
      self.assertFalse(display.GetVisibility2D())
      # Saving must serialize real image references while leaving the live preview intact.
      import tempfile
      import zipfile
      import xml.etree.ElementTree as ET
      composite = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSliceCompositeNode", "Checkerboard saved view")
      composite.SetBackgroundVolumeID(output.GetID())
      previewID = output.GetID()
      with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "checkerboard-scene.mrb")
        self.assertTrue(slicer.util.saveScene(path))
        with zipfile.ZipFile(path) as archive:
          mrml = next(name for name in archive.namelist() if name.endswith(".mrml"))
          saved = ET.fromstring(archive.read(mrml))
        self.assertFalse(any(previewID == value for element in saved.iter() for value in element.attrib.values()))
      self.assertEqual(composite.GetBackgroundVolumeID(), previewID)
      widget.ui.compareResultButton.checked = False
      output = wait()
      verify(output, (0, 100), 2)
      widget.ui.compareResultButton.checked = True
      widget.ui.checkerboardTilesSpinBox.value = 3
      widget.ui.checkerboardTilesSpinBox.value = 4
      output = wait()
      verify(output, (100, 300), 4)
      widget.ui.comparisonModeComboBox.currentText = "Overlay"
      self.assertTrue(display.GetVisibility2D())
      self.assertFalse(hidden.GetDisplayNode().GetVisibility2D())
      self.assertAlmostEqual(label_view.GetLabelOpacity(), 0.35)
      self.assertTrue(widget.ui.comparisonSegmentationsCheckBox.isHidden())
      self.assertIsNone(widget._checkerboardNode)
      self.assertTrue(display.GetVisibility2D())
      self.assertFalse(widget._comparisonSegmentationVisibility)
      self.assertIsNone(output.GetScene())
      layers.assert_called_with(background=fixed, foreground=registered, foregroundOpacity=0.5)
      widget.ui.comparisonModeComboBox.currentText = "Checkerboard"
      widget._checkerboardRefreshTimer.stop()
      widget._pendingCheckerboard = None
      widget.startCheckerboard(fixed, registered)
      widget.clearCheckerboard()
      deadline = time.monotonic() + 30
      while widget._checkerboardJobs:
        self.assertLess(time.monotonic(), deadline)
        slicer.app.processEvents()
        time.sleep(0.01)
      self.assertIsNone(widget._checkerboardNode)
      self.assertEqual(len(slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")), 3)
      self.assertEqual(len(slicer.util.getNodesByClass("vtkMRMLScalarVolumeDisplayNode")), 3)
    widget._completedResult = None
    widget.ui.comparisonSegmentationsCheckBox.checked = True
    widget.ui.comparisonModeComboBox.currentText = "Overlay"
    widget.updateRegistrationGUI()

  def test_reviewRegressions(self):
    """World geometry, transactional result loading, setup cancellation and control layout."""
    import tempfile
    import time
    import importlib.util
    import sitkUtils
    from unittest.mock import patch
    from UniGradICONHelpers import task as task_module
    widget = slicer.modules.unigradicon.widgetRepresentation().self()
    widget.initializeParameterNode()
    logic = widget.logic
    data = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    fixed = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Review fixed")
    moving = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Review moving")
    for node in (fixed, moving):
      slicer.util.updateVolumeFromArray(node, data)
      node.SetSpacing(0.6, 1.2, 2.4)
      node.SetOrigin(11, -7, 3)
    parent = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "Review parent")
    affine = vtk.vtkTransform()
    affine.Translate(10, -20, 30)
    affine.RotateZ(37)
    parent.SetMatrixTransformToParent(affine.GetMatrix())
    moving.SetAndObserveTransformNodeID(parent.GetID())
    snapshot = logic.snapshotVolume(moving)
    index = (1, 2, 3)
    ijk = vtk.vtkMatrix4x4()
    moving.GetIJKToRASMatrix(ijk)
    ras = affine.TransformPoint(ijk.MultiplyPoint((*index, 1))[:3])
    expected = np.array(ras) * (-1, -1, 1)
    np.testing.assert_allclose(snapshot.TransformIndexToPhysicalPoint(index), expected, atol=1e-12, rtol=0)
    self.assertEqual(moving.GetTransformNodeID(), parent.GetID())
    np.testing.assert_array_equal(slicer.util.arrayFromVolume(moving), data)
    owned = []
    source = widget._checkerboardSource(moving, owned)
    sourceImage = logic.snapshotVolume(source)
    np.testing.assert_allclose(sourceImage.TransformIndexToPhysicalPoint(index), expected, atol=1e-12, rtol=0)
    self.assertIsNone(source.GetParentTransformNode())
    self.assertEqual(moving.GetTransformNodeID(), parent.GetID())
    for node in owned:
      logic.removeTemporaryVolume(node)
    outputTransform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode")
    outputTransform.SetAndObserveTransformToParent(affine)
    outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    slicer.util.updateVolumeFromArray(outputVolume, data + 777)
    outputVolume.CreateDefaultDisplayNodes()
    for node in (outputVolume, outputTransform):
      node.SetAndObserveTransformNodeID(parent.GetID())
    settings = {"transform": outputTransform, "volume": outputVolume}
    original = outputTransform.GetTransformToParent().TransformPoint((0, 0, 0))
    with tempfile.TemporaryDirectory() as directory:
      path = os.path.join(directory, "result.h5")
      sitk.WriteTransform(sitk.AffineTransform(3), path)
      count = slicer.mrmlScene.GetNumberOfNodes()
      with patch.object(sitkUtils, "PushVolumeToSlicer", side_effect=RuntimeError("image load failed")):
        with self.assertRaisesRegex(RuntimeError, "image load failed"):
          logic.loadResults(settings, path, sitk.GetImageFromArray(data))
      np.testing.assert_array_equal(slicer.util.arrayFromVolume(outputVolume), data + 777)
      self.assertEqual(outputTransform.GetTransformToParent().TransformPoint((0, 0, 0)), original)
      self.assertEqual(outputVolume.GetTransformNodeID(), parent.GetID())
      self.assertEqual(slicer.mrmlScene.GetNumberOfNodes(), count)
      logic.loadResults(settings, path, sitk.GetImageFromArray(data))
      np.testing.assert_array_equal(slicer.util.arrayFromVolume(outputVolume), data)
      self.assertIsNone(outputVolume.GetTransformNodeID())
      self.assertIsNone(outputTransform.GetTransformNodeID())
      self.assertEqual(slicer.mrmlScene.GetNumberOfNodes(), count)
    widget.ui.fixedImageNodeComboBox.setCurrentNode(fixed)
    widget.ui.movingImageNodeComboBox.setCurrentNode(moving)
    widget.ui.outputVolumeComboBox.setCurrentNode(None)
    widget.ui.outputTransformComboBox.setCurrentNode(None)
    widget.ui.remoteProcessingCheckBox.checked = False
    original_find = importlib.util.find_spec
    with patch.object(importlib.util, "find_spec", side_effect=lambda name: None if name == "torch" else original_find(name)), \
         patch.object(widget, "ensureLocalBackend", side_effect=widget.cancelRegistration), \
         patch.object(task_module, "RegistrationTask") as task:
      widget.onRunRegistrationButton()
      deadline = time.monotonic() + 5
      while widget._startingRegistration:
        self.assertLess(time.monotonic(), deadline)
        slicer.app.processEvents()
      self.assertIsNone(widget._registrationTask)
      task.assert_not_called()
      self.assertIn("cancelled", widget.ui.registrationStatusLabel.text)
    from UniGradICONLib import logic as logic_module
    upgrade_logic = logic_module.UniGradICONLogic()
    with patch.object(logic_module, "backend_packages_current", return_value=False), \
         patch.object(slicer.util, "confirmOkCancelDisplay", return_value=True), \
         patch.object(slicer.util, "pip_install") as install, \
         patch.dict(sys.modules, {"unigradicon": __import__("types").SimpleNamespace()}):
      with self.assertRaisesRegex(RuntimeError, "Restart Slicer"):
        upgrade_logic.ensureLocalBackend()
      install.assert_called_once_with("unigradicon==1.0.6")
    with patch.object(logic_module, "backend_packages_current", return_value=True), \
         patch.object(slicer.util, "pip_install") as install:
      with self.assertRaisesRegex(RuntimeError, "Restart Slicer"):
        upgrade_logic.ensureLocalBackend()
      install.assert_not_called()
    self.assertFalse(upgrade_logic._localBackendReady)
    self.assertEqual(widget.ui.runRegistrationButton.parent(), widget.ui.registrationRunFrame)
    self.assertEqual(widget.ui.progressBar.parent(), widget.ui.registrationRunFrame)
    self.assertEqual(widget.ui.time.parent(), widget.ui.registrationRunFrame)
    self.assertEqual(widget.ui.outputVolumeComboBox.parent(), widget.ui.outputsCollapsibleButton)
    widget._completedResult = {"image": {"fixed": fixed, "moving": moving}, "outputSettings": {"volume": outputVolume}}
    widget.ui.outputVolumeComboBox.setCurrentNode(outputVolume)
    widget.updateRegistrationGUI()
    self.assertTrue(widget.ui.compareResultButton.enabled)
    widget.ui.fixedImageNodeComboBox.setCurrentNode(moving)
    self.assertIsNone(widget._completedResult)
    self.assertFalse(widget.ui.compareResultButton.enabled)

  def test_remoteResultDownloads(self):
    """Download the server's ITK image only when requested, before deleting the job."""
    import shutil
    import sitkUtils
    from unittest.mock import patch
    from UniGradICONHelpers import remote

    fixed = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Fixed download test")
    moving = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Moving download test")
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    slicer.util.updateVolumeFromArray(fixed, data)
    slicer.util.updateVolumeFromArray(moving, data)
    image = sitk.GetImageFromArray(data)
    image.SetOrigin((4, 5, 6))
    image.SetSpacing((1.5, 2, 2.5))

    class Client:
      def __init__(self):
        self.events = []
        self.params = None

      def submit(self, fixedPath, movingPath, params, maskPath):
        self.params = params
        return {"id": "test-job"}

      def status(self, jobID):
        return {"state": "done"}

      def download(self, jobID, result, path):
        self.events.append(result)
        if result == "transform":
          sitk.WriteTransform(sitk.AffineTransform(3), path)
        else:
          sitk.WriteImage(image, path, True)
        return path

      def cancel(self, jobID):
        self.events.append("delete")

    for withVolume in (False, True):
      outputTransform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode")
      outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode") if withVolume else None
      parameters = {
        "image": {"fixed": fixed, "moving": moving, "modality-fixed": "MRI", "modality-moving": "MRI"},
        "modelSettings": {"model": "unigradicon"},
        "outputSettings": {"transform": outputTransform, "volume": outputVolume, "loss": "LNCC"},
        "generalSettings": {"device": "CPU", "io_steps": 0},
      }
      client = Client()
      registration = RemoteRegistration(UniGradICONLogic(), client, parameters)
      try:
        registration.start()
        registration.poll()
        self.assertTrue(registration.finished)
        self.assertIsNone(registration.error)
        self.assertEqual(client.params["warp"], withVolume)
        self.assertEqual(client.events, ["transform", "warped", "delete"] if withVolume else ["transform", "delete"])
        self.assertIsNotNone(outputTransform.GetTransformToParent())
        self.assertFalse(os.path.exists(registration.workDir))
        if withVolume:
          np.testing.assert_array_equal(slicer.util.arrayFromVolume(outputVolume), data)
          loaded = sitkUtils.PullVolumeFromSlicer(outputVolume)
          self.assertEqual(loaded.GetOrigin(), image.GetOrigin())
          self.assertEqual(loaded.GetSpacing(), image.GetSpacing())
      finally:
        registration.timer.stop()
        if registration.workDir:
          shutil.rmtree(registration.workDir, ignore_errors=True)
    failed = RemoteRegistration(UniGradICONLogic(), Client(), parameters)
    with patch.object(failed.client, "submit", side_effect=remote.ServerError("upload failed")):
      with self.assertRaisesRegex(remote.ServerError, "upload failed"):
        failed.start()
    self.assertTrue(failed.finished)
    self.assertFalse(os.path.exists(failed.workDir))
    callbacks = []
    completed = RemoteRegistration(UniGradICONLogic(), Client(), parameters,
                                    finishedCallback=lambda error, cancelled: callbacks.append((error, cancelled)))
    completed.start()
    with patch.object(completed.client, "cancel", side_effect=remote.ServerError("cleanup unavailable")):
      completed.poll()
      completed.poll()
    self.assertIsNone(completed.error)
    self.assertEqual(callbacks, [(None, False)])
    self.assertFalse(os.path.exists(completed.workDir))

  def test_parameterSetInputs(self):
    """Selections follow the parameter set, including legacy JSON and imported scenes."""
    import tempfile

    widget = slicer.modules.unigradicon.widgetRepresentation().self()
    logic = widget.logic
    sets = []
    for name in ("A", "B"):
      fixed = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Fixed " + name)
      moving = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Moving " + name)
      parameterNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScriptedModuleNode", "Set " + name)
      parameterNode.SetAttribute("ModuleName", "UniGradICON")
      parameterNode.SetAttribute("RegressionSet", name)
      logic.setDefaultParameters(parameterNode)
      settings = json.loads(parameterNode.GetParameter(logic.SETTINGS_JSON_PARAM))
      settings["image"].update(fixed=fixed.GetID(), moving=moving.GetID())
      parameterNode.SetParameter(logic.SETTINGS_JSON_PARAM, json.dumps(settings))
      sets.append((parameterNode, fixed, moving))

    for parameterNode, fixed, moving in (sets[0], sets[1], sets[0]):
      widget.setParameterNode(parameterNode)
      self.assertEqual(widget.ui.fixedImageNodeComboBox.currentNodeID, fixed.GetID())
      self.assertEqual(widget.ui.movingImageNodeComboBox.currentNodeID, moving.GetID())
      self.assertEqual(parameterNode.GetNodeReference(logic.INPUT_VOLUME_REFS["fixed"]), fixed)
      self.assertEqual(parameterNode.GetNodeReference(logic.INPUT_VOLUME_REFS["moving"]), moving)

    widget.setParameterNode(sets[1][0])
    widget.ui.movingImageNodeComboBox.setCurrentNode(None)
    self.assertIsNone(logic.inputVolumeNode(sets[1][0], "moving"))
    self.assertIsNone(logic.createProcessParameters(sets[1][0])["image"]["moving"])
    self.assertFalse(widget.ui.runRegistrationButton.enabled)
    widget.ui.movingImageNodeComboBox.setCurrentNode(sets[1][2])

    # Import into the same scene to force ID collisions and MRML reference remapping.
    originalIDs = {n.GetID() for n in slicer.util.getNodesByClass("vtkMRMLNode")}
    oldURL = slicer.mrmlScene.GetURL()
    try:
      with tempfile.TemporaryDirectory() as directory:
        slicer.mrmlScene.SetURL(os.path.join(directory, "parameter-sets.mrml"))
        self.assertTrue(slicer.mrmlScene.Commit())
        self.assertTrue(slicer.mrmlScene.Import())
    finally:
      slicer.mrmlScene.SetURL(oldURL or "")
    imported = [n for n in slicer.util.getNodesByClass("vtkMRMLScriptedModuleNode")
                if n.GetID() not in originalIDs and n.GetAttribute("RegressionSet") == "B"]
    self.assertEqual(len(imported), 1)
    widget.setParameterNode(imported[0])
    parameters = logic.createProcessParameters(imported[0])
    for role, selector in (("fixed", widget.ui.fixedImageNodeComboBox),
                           ("moving", widget.ui.movingImageNodeComboBox)):
      node = logic.inputVolumeNode(imported[0], role)
      self.assertIsNotNone(node)
      self.assertNotIn(node.GetID(), originalIDs)
      self.assertEqual(selector.currentNodeID, node.GetID())
      self.assertEqual(parameters["image"][role], node.GetID())

  def test_parameterNodeRegistration(self):
    """Register with a parameter node and a fixed mask, in this Slicer and on a server, as plugins do."""
    if self.fullTestsEnabled():
      self.runParameterNodeRegistrationTest()
    else:
      logging.warning('Parameter node registration test was skipped. Enable it by starting Slicer with '
                      f'{self.FULL_TESTS_ENVIRONMENT_VARIABLE}=1.')

  def runParameterNodeRegistrationTest(self, port=8897):
    import time
    import vtk
    import SampleData
    from UniGradICONHelpers import remote

    self.delayDisplay("Registering with a parameter node and a fixed mask")
    sampleDataLogic = SampleData.SampleDataLogic()
    fixed = sampleDataLogic.downloadMRBrainTumor1()
    moving = sampleDataLogic.downloadMRBrainTumor2()
    mask = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', 'HalfMask')
    maskArray = np.zeros(slicer.util.arrayFromVolume(fixed).shape, dtype=np.uint8)
    maskArray[:, :, : maskArray.shape[2] // 2] = 1
    slicer.util.updateVolumeFromArray(mask, maskArray)
    ijkToRas = vtk.vtkMatrix4x4()
    fixed.GetIJKToRASMatrix(ijkToRas)
    mask.SetIJKToRASMatrix(ijkToRas)

    logic = UniGradICONLogic()
    logic.isSingletonParameterNode = False
    token = "test-token"
    url = f"127.0.0.1:{port}"
    weightsFolder = WEIGHTS_DIRECTORY
    server = RegistrationServerProcess(port, token=token, weightsDir=weightsFolder, host="127.0.0.1")
    server.start()
    try:
      results = {}
      for location in ("local", "remote"):
        parameterNode = logic.createParameterNode()
        slicer.mrmlScene.AddNode(parameterNode)
        logic.setDefaultParameters(parameterNode)
        stages = json.loads(parameterNode.GetParameter(logic.SETTINGS_JSON_PARAM))
        stages["image"]["modality-fixed"] = stages["image"]["modality-moving"] = "MRI"
        parameterNode.SetParameter(logic.SETTINGS_JSON_PARAM, json.dumps(stages))
        parameterNode.SetParameter(logic.REMOTE_PROCESSING_PARAM, "true" if location == "remote" else "false")
        parameterNode.SetParameter(logic.SERVER_URL_PARAM, url)
        if location == "remote":
          UniGradICONLogic.setServerToken(url, token)
          deadline = time.time() + 120
          while True:
            try:
              remote.RegistrationServerClient(url, token).info()
              break
            except remote.ServerError:
              self.assertLess(time.time(), deadline, "the registration server did not start")
              slicer.app.processEvents()
              time.sleep(0.5)
        outputTransform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode', f"{location}Transform")
        logic.registerVolumesUsingParameterNode(parameterNode, fixed, moving, outputTransform, fixedMaskNode=mask)
        self.assertIsNotNone(outputTransform.GetTransformToParent())
        results[location] = outputTransform
        # the token stays in this computer's settings, out of the scene
        self.assertFalse(any(token in parameterNode.GetParameter(name) for name in parameterNode.GetParameterNames()))

      points = [[10.0, -20.0, 5.0], [-30.0, 15.0, 20.0]]
      for point in points:
        local = results["local"].GetTransformToParent().TransformPoint(point)
        remoteResult = results["remote"].GetTransformToParent().TransformPoint(point)
        self.assertLess(max(abs(a - b) for a, b in zip(local, remoteResult)), 1e-3)
      self.delayDisplay('Parameter node registration test passed!')
    finally:
      UniGradICONLogic.setServerToken(url, "")
      server.stop()

  def test_remote(self):
    """Register through a registration server started on this computer."""
    # Needs PyTorch and the server packages (fastapi, uvicorn, python-multipart), and takes
    # a few minutes on a CPU: disabled by default, like the logic tests below.
    testRemote = self.fullTestsEnabled()
    if testRemote:
      self.runRemoteRegistrationTest()
    else:
      logging.warning('Remote registration test was skipped. Enable it by starting Slicer with '
                      f'{self.FULL_TESTS_ENVIRONMENT_VARIABLE}=1.')

  def runRemoteRegistrationTest(self, port=8898):
    import time
    import SampleData
    from UniGradICONHelpers import remote

    self.delayDisplay("Registering on a registration server on this computer")
    sampleDataLogic = SampleData.SampleDataLogic()
    fixed = sampleDataLogic.downloadMRBrainTumor1()
    moving = sampleDataLogic.downloadMRBrainTumor2()
    outputVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
    outputTransform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode')
    weightsFolder = WEIGHTS_DIRECTORY

    def logServer(*lines):
      for line in lines:
        logging.info("UniGradICON server: " + line)
    server = RegistrationServerProcess(port, weightsDir=weightsFolder, host="127.0.0.1", logCallback=logServer)
    server.start()
    process = server.process
    try:
      client = remote.RegistrationServerClient(f"http://127.0.0.1:{port}")
      deadline = time.time() + 120
      while True:
        try:
          client.info()
          break
        except remote.ServerError:
          if time.time() > deadline or process.poll() is not None:
            raise
          time.sleep(0.5)

      parameters = {
        "image": {"fixed": fixed, "moving": moving, "modality-fixed": "MRI", "modality-moving": "MRI"},
        "modelSettings": {"model": "unigradicon"},
        "outputSettings": {"transform": outputTransform, "volume": outputVolume, "loss": "LNCC"},
        "generalSettings": {"io_steps": 0, "device": "CPU"},
      }
      results = []
      registration = RemoteRegistration(UniGradICONLogic(weightsFolder), client, parameters,
                                        finishedCallback=lambda error, cancelled: results.append((error, cancelled)))
      registration.start()
      deadline = time.time() + 900
      while not results:
        self.assertLess(time.time(), deadline, "the remote registration did not finish")
        slicer.app.processEvents()
        time.sleep(0.1)

      self.assertEqual(results[0], (None, False))
      self.assertEqual(slicer.util.arrayFromVolume(outputVolume).shape, slicer.util.arrayFromVolume(fixed).shape)
      self.assertIsNotNone(outputTransform.GetTransformToParent())
      self.delayDisplay('Remote registration test passed!')
    finally:
      server.stop()


  def registrationParameters(self, loss="LNCC", ioSteps=0, device="CPU"):
    return {
      "image": {"fixed": self.fixed, "moving": self.moving,
                "modality-fixed": "MRI", "modality-moving": "MRI"},
      "outputSettings": {"transform": self.outputTransform, "volume": self.outputVolume, "loss": loss},
      "generalSettings": {"io_steps": ioSteps, "device": device},
    }

  # Preserve the parameter-builder names used by older test scripts.
  def test_wo_IO(self):
    return self.registrationParameters()

  def test_w_LNCC(self):
    return self.registrationParameters(ioSteps=1)

  def test_w_SquaredLNCC(self):
    return self.registrationParameters(loss="Squared LNCC", ioSteps=1)

  def test_w_MINDSSC(self):
    return self.registrationParameters(loss="MIND-SSC", ioSteps=1)

  def test_GPU(self):
    return self.registrationParameters(device="GPU")

  def test_UniGradICON(self):
    """Exercise the model with each supported loss and the available devices."""
    if not self.fullTestsEnabled():
      logging.warning("Model tests skipped. Set %s=1 to enable them.", self.FULL_TESTS_ENVIRONMENT_VARIABLE)
      return

    import SampleData
    import torch
    samples = SampleData.SampleDataLogic()
    self.fixed = samples.downloadMRBrainTumor1()
    self.moving = samples.downloadMRBrainTumor2()
    self.outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    self.outputTransform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode")
    logic = UniGradICONLogic()
    cases = [self.test_wo_IO, self.test_w_LNCC, self.test_w_SquaredLNCC, self.test_w_MINDSSC]
    if torch.cuda.is_available():
      cases.append(self.test_GPU)
    for parameters in cases:
      logic.process(**parameters())
      self.assertIsNotNone(self.outputTransform.GetTransformToParent())
      self.assertEqual(slicer.util.arrayFromVolume(self.outputVolume).shape,
                       slicer.util.arrayFromVolume(self.fixed).shape)
    self.delayDisplay("Model registration tests passed")
