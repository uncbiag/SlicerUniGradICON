"""Multi-label input matching, preview ownership, IO controls and backend preflight."""
from pathlib import Path
import threading
import time
from unittest.mock import patch
import qt
import tempfile

import numpy as np
import SimpleITK as sitk
import slicer
import vtk

from UniGradICONLib.segmentation import guidance_images, snapshot_inputs


def run_tests(test):
  widget = slicer.modules.unigradicon.widgetRepresentation().self()
  widget.initializeParameterNode()
  widget.ui.segmentationCollapsibleButton.collapsed = False
  widget.ui.remoteProcessingCheckBox.checked = False
  logic = widget.logic
  shape = (12, 14, 16)
  labels = np.zeros(shape, np.uint16)
  labels[2:5, 2:6, 2:6] = 2
  labels[7:10, 8:12, 9:13] = 7
  volumes, segmentations = {}, {}
  for role in ("fixed", "moving"):
    volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", role + " multi-label image")
    slicer.util.updateVolumeFromArray(volume, np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
    volume.SetSpacing(1.2, 1.3, 1.4)
    volumes[role] = volume
    segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", role + " multi-label segmentation")
    segmentation.SetReferenceImageGeometryParameterFromVolumeNode(volume)
    # Segment order and opaque IDs differ; semantic names establish correspondence.
    for value, name in ([(2, "Kidney"), (7, "Liver")] if role == "fixed" else [(7, "Liver"), (2, "Kidney")]):
      sid = segmentation.GetSegmentation().AddEmptySegment(role + str(value), name)
      slicer.util.updateSegmentBinaryLabelmapFromArray((labels == value).astype(np.uint8), segmentation, sid, volume)
    segmentation.GetSegmentation().CollapseBinaryLabelmaps()
    segmentations[role] = segmentation
    getattr(widget.ui, role + "ImageNodeComboBox").setCurrentNode(volume)
    getattr(widget.ui, role + "SegmentationComboBox").setCurrentNode(segmentation)
    getattr(widget.ui, role + "MaskSegmentationComboBox").setCurrentNode(segmentation)
  widget.ui.diceGuidanceCheckBox.checked = True
  widget.ui.maskingModeComboBox.currentText = "Loss-function masking"
  widget.ui.ioSpinBox.value = 0
  test.assertFalse(widget.ui.runRegistrationButton.enabled)
  test.assertFalse(hasattr(widget.ui, "guidanceIOSpinBox"))
  test.assertEqual(widget.ui.checkBackendButton.parent(), widget.ui.backendControlsFrame)
  widget.ui.ioSpinBox.value = 3
  test.assertEqual(widget._parameterNode.GetParameter(logic.IO_PARAM), "3")
  options = logic.segmentationSettings(widget._parameterNode)
  images = {role: logic.snapshotVolume(volume) for role, volume in volumes.items()}
  snapshot_inputs(images, options)
  expected = np.zeros(shape, np.uint32)
  expected[labels == 2] = 1
  expected[labels == 7] = 2
  for role in volumes:
    np.testing.assert_array_equal(sitk.GetArrayFromImage(images[role + "_segmentation"]), expected)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(images[role + "_mask"]), labels != 0)
  test.assertEqual([pair["name"] for pair in options["labelPairs"]], ["Kidney", "Liver"])

  source = segmentations["fixed"]
  sid = "fixed7"
  slicer.util.updateSegmentBinaryLabelmapFromArray((labels != 0).astype(np.uint8), source, sid, volumes["fixed"])
  with test.assertRaisesRegex(ValueError, "overlap"):
    guidance_images(images, options)
  slicer.util.updateSegmentBinaryLabelmapFromArray((labels == 7).astype(np.uint8), source, sid, volumes["fixed"])

  manager = slicer.app.layoutManager()
  if manager is not None:
    manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
    source.CreateDefaultDisplayNodes()
    source.GetDisplayNode().SetVisibility(False)
    widget.ui.previewMasksButton.checked = True
    test.assertEqual(widget._inputPreviewKind, "masks", widget.ui.inputPreviewStatusLabel.text)
    test.assertEqual(len(widget._inputPreviewNodes), 2)
    test.assertEqual(manager.layout, slicer.vtkMRMLLayoutNode.SlicerLayoutSideBySideView)
    test.assertFalse(source.GetDisplayNode().GetVisibility())
    for node in widget._inputPreviewNodes:
      test.assertFalse(node.GetSaveWithScene())
      test.assertTrue(node.GetHideFromEditors())
      test.assertEqual(node.GetSegmentation().GetNumberOfSegments(), 1)
    widget.ui.previewGuidanceButton.checked = True
    test.assertEqual(widget._inputPreviewKind, "guidance", widget.ui.inputPreviewStatusLabel.text)
    test.assertEqual(len(widget._inputPreviewNodes), 2)
    for node in widget._inputPreviewNodes:
      test.assertEqual(node.GetSegmentation().GetNumberOfSegments(), 2)
    with tempfile.TemporaryDirectory() as directory:
      test.assertTrue(slicer.util.saveScene(directory))
      text = next(Path(directory).rglob("*.mrml")).read_text()
      test.assertNotIn("input preview", text)
    widget.ui.previewGuidanceButton.checked = False
    test.assertEqual(widget._inputPreviewNodes, [])
    test.assertEqual(manager.layout, slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
    test.assertFalse(source.GetDisplayNode().GetVisibility())
    widget.ui.previewMasksButton.checked = True
    slicer.util.updateSegmentBinaryLabelmapFromArray((labels == 2).astype(np.uint8), source, "fixed2", volumes["fixed"])
    test.assertEqual(widget._inputPreviewNodes, [])
    for transformed in (source, volumes["fixed"]):
      parent = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "Preview input placement")
      transformed.SetAndObserveTransformNodeID(parent.GetID())
      widget.ui.previewMasksButton.checked = True
      test.assertEqual(widget._inputPreviewKind, "masks")
      matrix = vtk.vtkMatrix4x4()
      matrix.SetElement(0, 3, 1)
      parent.SetMatrixTransformToParent(matrix)
      test.assertEqual(widget._inputPreviewNodes, [], "Changing input placement must invalidate its preview")
      transformed.SetAndObserveTransformNodeID(None)
      slicer.mrmlScene.RemoveNode(parent)

  # A multi-label volume retains its numeric classes, rather than becoming foreground 1.
  labelmaps = {}
  for role in volumes:
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", role + " label values")
    node.CopyOrientation(volumes[role])
    slicer.util.updateVolumeFromArray(node, labels)
    labelmaps[role] = node
    getattr(widget.ui, role + "SegmentationComboBox").setCurrentNode(node)
  options = logic.segmentationSettings(widget._parameterNode)
  guidance_images(images, options)
  for role in volumes:
    np.testing.assert_array_equal(sitk.GetArrayFromImage(images[role + "_segmentation"]), labels)
  original = widget._parameterNode
  other = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScriptedModuleNode", "Alternate IO set")
  other.SetAttribute("ModuleName", "UniGradICON")
  widget.setParameterNode(other)
  widget.setParameterNode(original)
  test.assertEqual(widget.ui.fixedSegmentationComboBox.currentNode(), labelmaps["fixed"])

  widget.ui.remoteProcessingCheckBox.checked = True
  widget.ui.serverComboBox.currentText = "http://example.invalid:8899"
  widget._backendDebounce.stop()
  widget._serverInfo = {"devices": ["CPU"], "features": ["segmentationGuidance", "lossFunctionMasking"]}
  widget.updateRegistrationGUI()
  test.assertFalse(widget.ui.runRegistrationButton.enabled)
  test.assertIn("preserve multiple", widget.ui.backendStatusLabel.text)
  count = slicer.mrmlScene.GetNumberOfNodes()
  widget.onRunRegistrationButton()
  test.assertEqual(slicer.mrmlScene.GetNumberOfNodes(), count)
  test.assertIn("preserve multiple", widget.ui.registrationStatusLabel.text)
  widget._serverInfo["features"].append("multiLabelSegmentation")
  widget.updateRegistrationGUI()
  test.assertTrue(widget.ui.runRegistrationButton.enabled)
  old_key = ("old-server:8899", "old-token")
  widget._backendPending = old_key
  widget._backendResults.put((old_key, {"devices": ["CPU"], "features": []}, None))
  current_info = widget._serverInfo
  widget.pollBackendCheck()
  test.assertIs(widget._serverInfo, current_info)
  # A slow capability request must not block the image viewers or Qt timers.
  release, finished = threading.Event(), threading.Event()
  ticks = []
  def delayed_info(client, timeout=5):
    release.wait(3)
    finished.set()
    return {"devices": ["CPU"], "features": ["segmentationGuidance", "multiLabelSegmentation", "lossFunctionMasking"]}
  timer = qt.QTimer()
  timer.setInterval(10)
  timer.timeout.connect(lambda: ticks.append(1))
  with patch("UniGradICONHelpers.remote.RegistrationServerClient.info", delayed_info):
    widget._serverInfo = None
    widget.checkBackend()
    test.assertFalse(finished.is_set())
    timer.start()
    deadline = time.monotonic() + 5
    while len(ticks) < 3:
      test.assertLess(time.monotonic(), deadline)
      slicer.app.processEvents()
      time.sleep(0.01)
    release.set()
    while widget._backendPending:
      test.assertLess(time.monotonic(), deadline)
      slicer.app.processEvents()
      time.sleep(0.01)
    test.assertIsNotNone(widget._serverInfo)
  timer.stop()
  widget.ui.remoteProcessingCheckBox.checked = False
  widget._backendDebounce.stop()
