"""Segmentation scene persistence, physical geometry, and interactive result handling."""
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk
import slicer
import vtk

from UniGradICONHelpers import task
from UniGradICONLib.segmentation import snapshot_segment, segment_on_reference, snapshot_inputs, load_results, binary_mask_on_reference


def run_tests(test):
  widget = slicer.modules.unigradicon.widgetRepresentation().self()
  widget.initializeParameterNode()
  widget.ui.segmentationCollapsibleButton.collapsed = False
  logic = widget.logic
  test.assertEqual(widget.ui.diceWeightSpinBox.value, 10)
  test.assertEqual(widget._parameterNode.GetParameter("MaskingMode"), "none")
  test.assertTrue(widget.ui.maskInputsWidget.isHidden())
  test.assertTrue(widget.ui.guidanceInputsWidget.isHidden())
  test.assertTrue(widget.ui.propagationInputsWidget.isHidden())
  for role in ("fixed", "moving", "fixedMask", "movingMask"):
    test.assertIsNone(getattr(widget.ui, role + "SegmentationComboBox").currentNode())
  unused = {"guidance": False, "maskingMode": "none", "propagate": False}
  images = {}
  snapshot_inputs(images, unused)
  test.assertEqual(images, {})
  fixed = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Fixed segmentation test")
  moving = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Moving segmentation test")
  data = np.arange(8 * 9 * 10, dtype=np.float32).reshape(8, 9, 10)
  for volume in (fixed, moving):
    volume.SetOrigin(12, -8, 30)
    volume.SetSpacing(1.5, 2, 3)
    slicer.util.updateVolumeFromArray(volume, data)
  labels = np.zeros(data.shape, np.uint8)
  labels[2:6, 3:7, 4:8] = 1
  source = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Moving labels")
  source.SetReferenceImageGeometryParameterFromVolumeNode(moving)
  labels_two = np.zeros_like(labels)
  labels_two[1:3, 0:2, 0:2] = 1
  for segment_id, name, color in (("one", "Organ", (0.2, 0.6, 0.9)), ("two", "Overlapping organ", (0.9, 0.2, 0.1))):
    source.GetSegmentation().AddEmptySegment(segment_id, name, color)
    slicer.util.updateSegmentBinaryLabelmapFromArray(labels if segment_id == "one" else labels_two, source, segment_id, moving)
  source.GetSegmentation().AddEmptySegment("empty", "Empty segment", (0.5, 0.5, 0.5))
  test.assertFalse(sitk.GetArrayFromImage(snapshot_segment(source, "empty", allow_empty=True)).any())
  reference = logic.snapshotVolume(fixed)
  np.testing.assert_array_equal(sitk.GetArrayFromImage(segment_on_reference(source, "two", reference)), labels_two)
  parent = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
  affine = vtk.vtkTransform()
  affine.Translate(7, -5, 2)
  parent.SetAndObserveTransformToParent(affine)
  untransformed = snapshot_segment(source, "one")
  source.SetAndObserveTransformNodeID(parent.GetID())
  transformed = snapshot_segment(source, "one")
  np.testing.assert_allclose(np.array(transformed.GetOrigin()) - untransformed.GetOrigin(), (-7, 5, 2))
  source.SetAndObserveTransformNodeID(None)

  widget.ui.fixedImageNodeComboBox.setCurrentNode(fixed)
  widget.ui.movingImageNodeComboBox.setCurrentNode(moving)
  masks = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Independent ROI masks")
  masks.SetReferenceImageGeometryParameterFromVolumeNode(moving)
  masks.GetSegmentation().AddEmptySegment("roi", "Broad ROI")
  mask_labels = np.zeros_like(labels)
  mask_labels[1:7, 2:8, 3:9] = 1
  slicer.util.updateSegmentBinaryLabelmapFromArray(mask_labels, masks, "roi", moving)
  masks.GetSegmentation().AddEmptySegment("extra", "Additional foreground")
  extra = np.zeros_like(labels)
  extra[0, 0, 0] = 1
  slicer.util.updateSegmentBinaryLabelmapFromArray(extra, masks, "extra", moving)
  masks.GetSegmentation().AddEmptySegment("empty", "Empty mask segment")
  mask_labels[0, 0, 0] = 1
  np.testing.assert_array_equal(sitk.GetArrayFromImage(binary_mask_on_reference(masks, reference)), mask_labels)
  labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "Nonzero ROI labelmap")
  labelmap.CopyOrientation(moving)
  slicer.util.updateVolumeFromArray(labelmap, mask_labels.astype(np.uint16) * 7)
  np.testing.assert_array_equal(sitk.GetArrayFromImage(binary_mask_on_reference(labelmap, reference)), mask_labels)
  empty_mask = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Empty mask")
  with test.assertRaisesRegex(ValueError, "empty or does not overlap"):
    binary_mask_on_reference(empty_mask, reference)
  widget.ui.fixedMaskSegmentationComboBox.setCurrentNode(masks)
  widget.ui.movingMaskSegmentationComboBox.setCurrentNode(labelmap)
  widget.ui.propagationSegmentationComboBox.setCurrentNode(source)
  widget.ui.fixedSegmentationComboBox.setCurrentNode(source)
  widget.ui.movingSegmentationComboBox.setCurrentNode(source)
  widget.ui.maskingModeComboBox.currentText = "Input masking"
  widget.ui.diceGuidanceCheckBox.checked = True
  widget.ui.propagateSegmentationCheckBox.checked = True
  widget.ui.diceWeightSpinBox.value = 0.25
  widget.ui.ioSpinBox.value = 0
  with test.assertRaisesRegex(ValueError, "at least one IO step"):
    widget.validateRunInputs()
  widget.ui.ioSpinBox.value = 1
  test.assertFalse(widget.ui.maskInputsWidget.isHidden())
  test.assertFalse(widget.ui.guidanceInputsWidget.isHidden())
  test.assertFalse(widget.ui.propagationInputsWidget.isHidden())
  selected = logic.segmentationSettings(widget._parameterNode)
  independent = {"fixed": reference, "moving": logic.snapshotVolume(moving)}
  snapshot_inputs(independent, selected)
  for role in ("fixed", "moving"):
    np.testing.assert_array_equal(sitk.GetArrayFromImage(independent[role + "_mask"]), mask_labels)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(independent[role + "_segmentation"]), labels.astype(np.uint32)*2 + labels_two.astype(np.uint32)*3)
  original = widget._parameterNode
  original.SetSingletonTag(None)
  other = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScriptedModuleNode", "Other set")
  other.SetAttribute("ModuleName", "UniGradICON")
  widget.setParameterNode(other)
  test.assertEqual(widget.ui.maskingModeComboBox.currentText, "None")
  test.assertTrue(widget.ui.maskInputsWidget.isHidden())
  test.assertTrue(widget.ui.guidanceInputsWidget.isHidden())
  widget.setParameterNode(original)
  test.assertEqual(widget.ui.fixedSegmentationComboBox.currentNodeID, widget.ui.movingSegmentationComboBox.currentNodeID)
  test.assertTrue(widget.ui.diceGuidanceCheckBox.checked)
  test.assertEqual(widget.ui.diceWeightSpinBox.value, 0.25)
  test.assertEqual(widget.ui.maskingModeComboBox.currentText, "Input masking")
  test.assertEqual(widget.ui.fixedMaskSegmentationComboBox.currentNode(), masks)
  test.assertEqual(widget.ui.propagationSegmentationComboBox.currentNode(), source)
  old_url = slicer.mrmlScene.GetURL()
  before_ids = {node.GetID() for node in slicer.util.getNodesByClass("vtkMRMLNode")}
  original.SetAttribute("SegmentationRegression", "Original")
  try:
    with tempfile.TemporaryDirectory() as directory:
      test.assertTrue(slicer.util.saveScene(directory))
      slicer.mrmlScene.SetURL(str(next(Path(directory).rglob("*.mrml"))))
      test.assertTrue(slicer.mrmlScene.Import())
      slicer.app.processEvents()
  finally:
    slicer.mrmlScene.SetURL(old_url or "")
  imported = next(node for node in slicer.util.getNodesByClass("vtkMRMLScriptedModuleNode")
                  if node.GetID() not in before_ids and node.GetAttribute("SegmentationRegression") == "Original")
  widget.setParameterNode(imported)
  test.assertNotEqual(widget.ui.movingSegmentationComboBox.currentNodeID, source.GetID())
  test.assertEqual(widget.ui.fixedSegmentationComboBox.currentNodeID, widget.ui.movingSegmentationComboBox.currentNodeID)
  test.assertEqual(widget.ui.maskingModeComboBox.currentText, "Input masking")
  imported_settings = logic.segmentationSettings(imported)
  test.assertNotEqual(imported_settings["fixed"], imported_settings["fixedMask"])
  test.assertTrue(imported_settings["fixedMask"].IsA("vtkMRMLSegmentationNode"))
  test.assertTrue(imported_settings["movingMask"].IsA("vtkMRMLLabelMapVolumeNode"))
  test.assertIsNotNone(imported_settings["propagationSource"])
  widget.setParameterNode(original)

  widget.ui.diceGuidanceCheckBox.checked = False
  widget.ui.maskingModeComboBox.currentText = "None"
  widget.ui.createVolumeCheckBox.checked = False
  widget.ui.remoteProcessingCheckBox.checked = False
  widget.ui.outputTransformComboBox.setCurrentNode(None)
  widget.ui.outputSegmentationComboBox.setCurrentNode(None)
  widget.ui.deviceComboBox.currentText = "CPU"
  widget.ui.ioSpinBox.value = 0
  captured = []
  requested_modes = ("None", "Input masking", "Loss-function masking")
  def fake_local(job, paths):
    captured.append(paths)
    mode = requested_modes[len(captured) - 1]
    if mode != "None":
      test.assertIn("fixed_mask", paths)
      test.assertIn("moving_mask", paths)
      test.assertEqual(job.settings["input_masking"], mode == "Input masking")
      test.assertEqual(job.settings["loss_function_masking"], mode == "Loss-function masking")
      for role in ("fixed_mask", "moving_mask"):
        np.testing.assert_array_equal(sitk.GetArrayFromImage(sitk.ReadImage(paths[role])), mask_labels)
    else:
      test.assertNotIn("fixed_mask", paths)
      test.assertNotIn("moving_mask", paths)
    test.assertNotIn("fixed_segmentation", paths)
    test.assertNotIn("moving_segmentation", paths)
    transform = os.path.join(job.directory, "result.h5")
    sitk.WriteTransform(sitk.TranslationTransform(3, [1.5, 0, 0]), transform)
    return transform, None
  with patch.object(task.RegistrationTask, "_local", fake_local):
    for mode in requested_modes:
      widget.ui.maskingModeComboBox.currentText = mode
      widget.ui.ioSpinBox.value = 0 if mode != "Loss-function masking" else 1
      widget.ui.outputTransformComboBox.setCurrentNode(None)
      widget.ui.outputSegmentationComboBox.setCurrentNode(None)
      if mode == "Loss-function masking":
        widget.ui.ioSpinBox.value = 0
        with test.assertRaisesRegex(ValueError, "at least one IO step"):
          widget.validateRunInputs()
        widget.ui.ioSpinBox.value = 1
      widget.onRunRegistrationButton()
      deadline = time.monotonic() + 20
      while widget._startingRegistration or widget._registrationTask:
        test.assertLess(time.monotonic(), deadline)
        slicer.app.processEvents()
        time.sleep(0.01)
      test.assertTrue(widget.ui.registrationStatusLabel.text.startswith("Completed in"), widget.ui.registrationStatusLabel.text)
  test.assertTrue(widget.ui.registrationStatusLabel.text.startswith("Completed in"), widget.ui.registrationStatusLabel.text)
  output = widget.ui.outputSegmentationComboBox.currentNode()
  test.assertEqual(output.GetSegmentation().GetNumberOfSegments(), 3)
  test.assertFalse(sitk.GetArrayFromImage(snapshot_segment(output, "empty", allow_empty=True)).any())
  test.assertIn("propagation_2", captured[0])
  expected = np.zeros_like(labels)
  expected[:, :, 1:] = labels[:, :, :-1]
  expected_two = np.zeros_like(labels_two)
  expected_two[:, :, 1:] = labels_two[:, :, :-1]
  for segment_id in ("one", "two"):
    np.testing.assert_array_equal(sitk.GetArrayFromImage(segment_on_reference(output, segment_id, reference)), expected if segment_id == "one" else expected_two)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(segment_on_reference(source, segment_id, reference)), labels if segment_id == "one" else labels_two)
    test.assertEqual(output.GetSegmentation().GetSegment(segment_id).GetName(), source.GetSegmentation().GetSegment(segment_id).GetName())
    np.testing.assert_allclose(output.GetSegmentation().GetSegment(segment_id).GetColor(), source.GetSegmentation().GetSegment(segment_id).GetColor())
  test.assertTrue(widget.outputHasData(output))
  test.assertIn(output, widget.selectedOutputsWithData())
  widget.ui.outputSegmentationComboBox.setCurrentNode(source)
  with test.assertRaisesRegex(ValueError, "differ from the input"):
    widget.validateRunInputs()
  widget.ui.outputSegmentationComboBox.setCurrentNode(output)

  def wait_for_cancel(job, paths):
    while True:
      job.check_cancelled()
      time.sleep(0.01)
  with patch.object(task.RegistrationTask, "_local", wait_for_cancel), \
       patch.object(widget, "confirmOutputReplacement", return_value="overwrite"):
    widget.onRunRegistrationButton()
    deadline = time.monotonic() + 10
    while widget._startingRegistration:
      test.assertLess(time.monotonic(), deadline)
      slicer.app.processEvents()
    widget.cancelRegistration()
    while widget._registrationTask:
      test.assertLess(time.monotonic(), deadline)
      slicer.app.processEvents()
      time.sleep(0.01)
  test.assertIn("cancelled", widget.ui.registrationStatusLabel.text)
  np.testing.assert_array_equal(sitk.GetArrayFromImage(segment_on_reference(output, "one", reference)), expected)

  # An incomplete segmentation must fail before replacing an existing transform or labels.
  parameters = logic.createProcessParameters(original)
  snapshots = {"fixed": reference, "moving": logic.snapshotVolume(moving)}
  snapshot_inputs(snapshots, parameters["segmentationSettings"])
  transform = parameters["outputSettings"]["transform"]
  before = transform.GetTransformToParent().TransformPoint((0, 0, 0))
  with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, "other.h5")
    sitk.WriteTransform(sitk.TranslationTransform(3, [100, 0, 0]), path)
    with test.assertRaisesRegex(ValueError, "incomplete"):
      load_results(logic, parameters["outputSettings"], path, None, parameters["segmentationSettings"], [])
    test.assertEqual(transform.GetTransformToParent().TransformPoint((0, 0, 0)), before)
  widget.ui.propagateSegmentationCheckBox.checked = False

  # Remaining labels are used automatically; missing input nodes require a new selection.
  source.GetSegmentation().RemoveSegment("two")
  slicer.app.processEvents()
  widget.ui.maskingModeComboBox.currentText = "None"
  widget.ui.diceGuidanceCheckBox.checked = True
  widget.validateRunInputs()
  widget.ui.fixedSegmentationComboBox.setCurrentNode(None)
  with test.assertRaisesRegex(ValueError, "Select the fixed"):
    widget.validateRunInputs()
  widget.ui.diceGuidanceCheckBox.checked = False
  widget.ui.maskingModeComboBox.currentText = "None"
  widget.validateRunInputs()
  widget.ui.fixedSegmentationComboBox.setCurrentNode(None)
  widget.ui.movingSegmentationComboBox.setCurrentNode(None)
  widget.validateRunInputs()
  settings = logic.segmentationSettings(original)
  test.assertIsNone(settings["fixed"])
  test.assertIsNone(settings["moving"])

  # An initial transform must not silently move the image without its moving mask.
  initial = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "Initial placement")
  transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "API registration result")
  for mode in ("Input masking", "Loss-function masking"):
    widget.ui.maskingModeComboBox.currentText = mode
    with test.assertRaisesRegex(ValueError, "harden the initial transform"):
      logic.registerVolumesUsingParameterNode(original, fixed, moving, transform, initialTransformNode=initial)

  # Explicit segmentation settings must select the same fixed mask in both APIs.
  import itk
  import shutil
  from unittest.mock import Mock
  from UniGradICONHelpers import registration
  from UniGradICONLib.logic import RemoteRegistration
  widget.ui.maskingModeComboBox.currentText = "Input masking"
  widget.ui.propagateSegmentationCheckBox.checked = False
  widget.ui.fixedMaskSegmentationComboBox.setCurrentNode(masks)
  widget.ui.movingMaskSegmentationComboBox.setCurrentNode(labelmap)
  parameters = logic.createProcessParameters(original)
  parameters["image"].update(fixed=fixed, moving=moving)
  parameters["outputSettings"].update(transform=transform, volume=None)
  parameters["generalSettings"].update(device="CPU", io_steps=1)
  legacy_mask = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "Legacy fixed-only mask")
  legacy_mask.CopyOrientation(fixed)
  slicer.util.updateVolumeFromArray(legacy_mask, np.ones(data.shape, np.uint8))
  captured = []
  def register(*args, **kwargs):
    captured.append(itk.array_from_image(kwargs["fixed_mask"]).copy())
    return itk.TranslationTransform[itk.D, 3].New(), None
  with patch.object(registration, "load_model", return_value=None), \
       patch.object(registration, "register_images", side_effect=register), \
       patch.object(logic, "loadResults"):
    logic.process(**parameters, fixedMask=legacy_mask)
  np.testing.assert_array_equal(captured[0], mask_labels)
  client = Mock()
  client.info.return_value = {"features": ["inputMasking"]}
  uploaded = []
  def submit(fixed_path, moving_path, settings, fixed_mask_path, **kwargs):
    uploaded.append(sitk.GetArrayFromImage(sitk.ReadImage(fixed_mask_path)))
    return {"id": "mask-consistency"}
  client.submit.side_effect = submit
  remote = RemoteRegistration(logic, client, parameters, fixedMask=legacy_mask)
  try:
    remote._uploadInputs()
    np.testing.assert_array_equal(uploaded[0], captured[0])
  finally:
    remote.timer.stop()
    shutil.rmtree(remote.workDir, ignore_errors=True)
