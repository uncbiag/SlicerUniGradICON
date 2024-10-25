import os
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import json
import glob
import os
import numpy as np

input_shape = [1, 1, 175, 175, 175]

class UniGradICON(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "UniGradICON"
    self.parent.categories = ["Registration"]
    self.parent.dependencies = []
    self.parent.contributors = ["Basar Demir"]
    self.parent.helpText = """
    This module performs registration of images using uniGradICON or multiGradICON. See more information in the <a href="https://github.com/uncbiag/uniGradICON">uniGradICON</a> repository.
    """
    self.parent.acknowledgementText = ""

#
# unigradiconWidget
#

class UniGradICONWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False
    self.checkpointFolder = self.resourcePath("UI") + "/../../../model_checkpoints/"
    
    global sitkUtils
    global sitk
    global np
    global itk
    global icon
    global torch
    global network_wrappers
    global networks
    global compute_warped_image_multiNC
    global itk_wrapper
    global SampleData
    global icon_helper
    
    import numpy as np
    import sitkUtils
    import SampleData
    import shutil
    

    if not os.path.exists(self.checkpointFolder):
      os.makedirs(self.checkpointFolder)
        
    if not os.path.exists(self.checkpointFolder + "unigradicon_weights.pth"):
      slicer.progressWindow = slicer.util.createProgressDialog()
      self.sampleDataLogic = SampleData.SampleDataLogic()
      self.sampleDataLogic.logMessage = self.reportProgress
              
      weights_location = self.sampleDataLogic.downloadFileIntoCache("https://github.com/uncbiag/uniGradICON/releases/download/unigradicon_weights/Step_2_final.trch", "unigradicon_weights.trch")
      if self.sampleDataLogic.downloadPercent and self.sampleDataLogic.downloadPercent == 100:
        shutil.copyfile(weights_location, self.checkpointFolder + "unigradicon_weights.pth")
        slicer.progressWindow.close()
    
    if not os.path.exists(self.checkpointFolder + "multigradicon_weights.pth"):
      slicer.progressWindow = slicer.util.createProgressDialog()
      self.sampleDataLogic = SampleData.SampleDataLogic()
      self.sampleDataLogic.logMessage = self.reportProgress
      weights_location = self.sampleDataLogic.downloadFileIntoCache("https://github.com/uncbiag/uniGradICON/releases/download/multigradicon_weights/Step_2_final.trch", "Step_2_final.trch")
      if self.sampleDataLogic.downloadPercent and self.sampleDataLogic.downloadPercent == 100:
        shutil.copyfile(weights_location, self.checkpointFolder + "multigradicon_weights.pth")
        slicer.progressWindow.close()
    
    try:
      import icon_registration
    except ModuleNotFoundError:
      if slicer.util.confirmOkCancelDisplay(
      "'icon_registration' is missing. Click OK to install it."
      ): 
        slicer.util.pip_install("icon_registration")
    try:
      import icon_registration as icon
      import icon_registration.network_wrappers as network_wrappers
      import icon_registration.networks as networks
      from icon_registration.mermaidlite import compute_warped_image_multiNC
      import icon_registration.itk_wrapper as itk_wrapper
      import itk
      import torch
      import icon_helper
    except ModuleNotFoundError:
      raise RuntimeError("There is a problem about the installation of 'hydra' package. Please try again to install!")
    
    try:
      import SimpleITK
    except ModuleNotFoundError:
      if slicer.util.confirmOkCancelDisplay(
      "'SimpleITK' is missing. Click OK to install it."
      ): 
        slicer.util.pip_install("SimpleITK")
    try:
      import SimpleITK as sitk
    except ModuleNotFoundError:
      raise RuntimeError("There is a problem about the installation of 'SimpleITK' package. Please try again to install!")

  def reportProgress(self, msg, level=None):
    if slicer.progressWindow.wasCanceled:
        raise Exception("Download aborted")
    slicer.progressWindow.show()
    slicer.progressWindow.activateWindow()
    slicer.progressWindow.setValue(int(self.sampleDataLogic.downloadPercent))
    slicer.progressWindow.setLabelText("Downloading checkpoints...")
    slicer.app.processEvents()
            
  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/UniGradICON.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)
    
    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = UniGradICONLogic(self.checkpointFolder)

    self.ui.stagesPresetsComboBox.addItems(PresetManager().getPresetNames())
    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).
    self.ui.lossComboBox.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
    self.ui.outputTransformComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.outputVolumeComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.ioSpinBox.connect("valueChanged(int)", self.updateParameterNodeFromGUI)
    self.ui.deviceComboBox.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)

    self.ui.fixedImageNodeComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.updateStagesFromFixedMovingNodes)
    self.ui.movingImageNodeComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.updateStagesFromFixedMovingNodes)

    self.ui.fixedModalityComboBox.connect("currentIndexChanged(int)", self.updateStagesFromFixedMovingNodes)
    self.ui.movingModalityComboBox.connect("currentIndexChanged(int)", self.updateStagesFromFixedMovingNodes)

    # Preset Stages
    self.ui.stagesPresetsComboBox.currentTextChanged.connect(self.onPresetSelected)

    # Buttons
    self.ui.runRegistrationButton.connect('clicked(bool)', self.onRunRegistrationButton)
    
    # Add GPU option if available
    if torch.cuda.is_available():
      self.ui.deviceComboBox.addItem("GPU")
      
    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()
  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

  def enter(self):
    """
    Called each time the user opens this module.
    """
    # Make sure parameter node exists and observed
    self.initializeParameterNode()

  def exit(self):
    """
    Called each time the user opens a different module.
    """
    # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  def onSceneStartClose(self, caller, event):
    """
    Called just before the scene is closed.
    """
    # Parameter node will be reset, do not use it anymore
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    """
    Called just after the scene is closed.
    """
    # If this module is shown while the scene is closed then recreate a new parameter node immediately
    if self.parent.isEntered:
      self.initializeParameterNode()

  def initializeParameterNode(self):
    """
    Ensure parameter node exists and observed.
    """
    # Parameter node stores all user choices in parameter values, node selections, etc.
    # so that when the scene is saved and reloaded, these settings are restored.
    self.setParameterNode(self.logic.getParameterNode() if not self._parameterNode else self._parameterNode)

  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)

    # Unobserve previously selected parameter node and add an observer to the newly selected.
    # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
    # those are reflected immediately in the GUI.
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True

    self.updateStagesGUIFromParameter()

    self.ui.outputTransformComboBox.setCurrentNode(self._parameterNode.GetNodeReference(self.logic.OUTPUT_TRANSFORM_REF))
    self.ui.outputVolumeComboBox.setCurrentNode(self._parameterNode.GetNodeReference(self.logic.OUTPUT_VOLUME_REF))
    self.ui.lossComboBox.currentText = self._parameterNode.GetParameter(self.logic.LOSS_PARAM)
    self.ui.deviceComboBox.currentText = self._parameterNode.GetParameter(self.logic.DEVICE_PARAM)
    
    self.ui.ioSpinBox.value = int(self._parameterNode.GetParameter(self.logic.IO_PARAM))

    self.ui.runRegistrationButton.enabled = self.ui.fixedImageNodeComboBox.currentNodeID and self.ui.movingImageNodeComboBox.currentNodeID and\
                                            (self.ui.outputTransformComboBox.currentNodeID and self.ui.outputVolumeComboBox.currentNodeID)

    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False

  def updateStagesGUIFromParameter(self):
    presetParameters = json.loads(self._parameterNode.GetParameter(self.logic.STAGES_JSON_PARAM))
    self.ui.fixedModalityComboBox.currentText = presetParameters['image']['modality-fixed']
    self.ui.movingModalityComboBox.currentText = presetParameters['image']['modality-moving']
    self.logic.changeModelSettings(presetParameters)

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch
    
    self._parameterNode.SetNodeReferenceID(self.logic.OUTPUT_TRANSFORM_REF, self.ui.outputTransformComboBox.currentNodeID)
    self._parameterNode.SetNodeReferenceID(self.logic.OUTPUT_VOLUME_REF, self.ui.outputVolumeComboBox.currentNodeID)
    self._parameterNode.SetParameter(self.logic.LOSS_PARAM, self.ui.lossComboBox.currentText)
    self._parameterNode.SetParameter(self.logic.IO_PARAM, str(self.ui.ioSpinBox.value))
    self._parameterNode.SetParameter(self.logic.DEVICE_PARAM, self.ui.deviceComboBox.currentText)

    self._parameterNode.EndModify(wasModified)


  def updateStagesFromFixedMovingNodes(self):
    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return
    presetParameters = json.loads(self._parameterNode.GetParameter(self.logic.STAGES_JSON_PARAM))
    presetParameters['image']['fixed'] = self.ui.fixedImageNodeComboBox.currentNodeID
    presetParameters['image']['moving'] = self.ui.movingImageNodeComboBox.currentNodeID
    
    presetParameters['image']['modality-fixed'] = self.ui.fixedModalityComboBox.currentText
    presetParameters['image']['modality-moving'] = self.ui.movingModalityComboBox.currentText
  
    self._parameterNode.SetParameter(self.logic.STAGES_JSON_PARAM, json.dumps(presetParameters))

  def onPresetSelected(self, presetName):
    if presetName == 'Select...' or self._parameterNode is None or self._updatingGUIFromParameterNode:
      return
    wasModified = self._parameterNode.StartModify()  # Modify in a single batch
    presetParameters = PresetManager().getPresetParametersByName(presetName)
    presetParameters['image']['fixed'] = self.ui.fixedImageNodeComboBox.currentNodeID
    presetParameters['image']['moving'] = self.ui.movingImageNodeComboBox.currentNodeID
    presetParameters['image']['modality-fixed'] = self.ui.fixedModalityComboBox.currentText
    presetParameters['image']['modality-moving'] = self.ui.movingModalityComboBox.currentText
      
    self._parameterNode.SetParameter(self.logic.STAGES_JSON_PARAM, json.dumps(presetParameters))
    self._parameterNode.EndModify(wasModified)
    
    self.logic.changeModelSettings(presetParameters)

  def onRunRegistrationButton(self):
    if self.ui.runRegistrationButton.text == 'Cancel':
      return

    parameters = self.logic.createProcessParameters(self._parameterNode)
    self.logic.process(**parameters)

#
# unigradiconLogic
#

class UniGradICONLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  OUTPUT_TRANSFORM_REF = "OutputTransform"
  OUTPUT_VOLUME_REF = "OutputVolume"
  LOSS_PARAM = "LNCC"
  STAGES_JSON_PARAM = "StagesJson"
  IO_PARAM = "0"
  DEVICE_PARAM = "CPU"

  def __init__(self, weights_location):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)
    self.weights_location = weights_location
    self.device = torch.device("cpu")
    print(self.device)
    self.model = icon_helper.make_network(input_shape, include_last_step=True, loss_fn=icon.LNCC(sigma=5), device=self.device)
    self.model.regis_net.load_state_dict(torch.load(self.weights_location + "unigradicon_weights.pth", map_location=self.device))
    self.model.to(self.device)
    

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """
    presetParameters = PresetManager().getPresetParametersByName()
    if not parameterNode.GetParameter(self.STAGES_JSON_PARAM):
      parameterNode.SetParameter(self.STAGES_JSON_PARAM, json.dumps(presetParameters))
    if not parameterNode.GetNodeReference(self.OUTPUT_TRANSFORM_REF):
      parameterNode.SetNodeReferenceID(self.OUTPUT_TRANSFORM_REF, "")
    if not parameterNode.GetNodeReference(self.OUTPUT_VOLUME_REF):
      parameterNode.SetNodeReferenceID(self.OUTPUT_VOLUME_REF, "")
    if not parameterNode.GetParameter(self.LOSS_PARAM):
      parameterNode.SetParameter(self.LOSS_PARAM, str(presetParameters["modelSettings"]["loss"]))
    if not parameterNode.GetParameter(self.IO_PARAM):
      parameterNode.SetParameter(self.IO_PARAM, str(presetParameters["modelSettings"]["io_steps"]))
    if not parameterNode.GetParameter(self.DEVICE_PARAM):
      parameterNode.SetParameter(self.DEVICE_PARAM, str(presetParameters["modelSettings"]["device"]))

  def createProcessParameters(self, paramNode):
    parameters = json.loads(paramNode.GetParameter(self.STAGES_JSON_PARAM))

    parameters['outputSettings'] = {}
    parameters['outputSettings']['transform'] = paramNode.GetNodeReference(self.OUTPUT_TRANSFORM_REF)
    parameters['outputSettings']['volume'] = paramNode.GetNodeReference(self.OUTPUT_VOLUME_REF)
    parameters['outputSettings']['loss'] = paramNode.GetParameter(self.LOSS_PARAM)

    parameters['generalSettings'] = {}
    parameters['generalSettings']['io_steps'] = int(paramNode.GetParameter(self.IO_PARAM))
    parameters['generalSettings']['device'] = str(paramNode.GetParameter(self.DEVICE_PARAM))
    
    return parameters

  def itk2sitk(self, itk_image):
    sitkImage = sitk.GetImageFromArray(itk.GetArrayFromImage(itk_image))

    sitkImage.SetOrigin(tuple(itk_image.GetOrigin()))
    sitkImage.SetSpacing(tuple(itk_image.GetSpacing()))
    sitkImage.SetDirection(itk.GetArrayFromMatrix(itk_image.GetDirection()).flatten()) 
    
    return sitkImage
  
  def sitk2itk(self, sitk_image):
    itk_image = itk.GetImageFromArray(sitk.GetArrayFromImage(sitk_image))
    image_dimension = 3
    
    itk_image.SetOrigin(sitk_image.GetOrigin())
    itk_image.SetSpacing(sitk_image.GetSpacing())
    itk_image.SetDirection(itk.GetMatrixFromArray(np.reshape(np.array(sitk_image.GetDirection()), [image_dimension]*2)))
    
    return itk_image
  
  def changeModelSettings(self, presetParameters):
    if presetParameters['modelSettings']['model'] == 'unigradicon':
      self.model.regis_net.load_state_dict(torch.load(self.weights_location + "unigradicon_weights.pth", map_location=self.device))
    if presetParameters['modelSettings']['model'] == 'multigradicon':
      self.model.regis_net.load_state_dict(torch.load(self.weights_location + "multigradicon_weights.pth", map_location=self.device))
    
    self.model.similarity = icon_helper.make_sim(presetParameters['modelSettings']['loss'])
    self.model.eval()

  def process(self, image, outputSettings, modelSettings=None, generalSettings=None, wait_for_completion=False):
    """
    :param stages: list defining registration stages
    :param outputSettings: dictionary defining output settings
    :param initialTransformSettings: dictionary defining initial moving transform
    :param generalSettings: dictionary defining general registration settings
    :param wait_for_completion: flag to enable waiting for completion
    See presets examples to see how these are specified
    """
    
    fixed_image = sitkUtils.PullVolumeFromSlicer(image['fixed'])
    moving_image = sitkUtils.PullVolumeFromSlicer(image['moving'])
    
    fixed_image_modality = image['modality-fixed']
    moving_image_modality = image['modality-moving']
    
    #convert to itk
    fixed = self.sitk2itk(fixed_image)
    moving = self.sitk2itk(moving_image)
    
    if generalSettings['device'] == 'GPU':
      self.model.cuda()
    else:
      self.model.cpu()
    
    #convert to itk by preserving metadata
    phi_AB, _ = icon_helper.register_pair(
        self.model,
        icon_helper.preprocess(moving, moving_image_modality), 
        icon_helper.preprocess(fixed, fixed_image_modality), 
        finetune_steps= None if generalSettings["io_steps"] == 0 else generalSettings["io_steps"],
    )
   
    itk.transformwrite([phi_AB], f'{slicer.app.temporaryPath}/transform.tfm')
    node_reference = outputSettings['transform']
    transform_node = slicer.util.loadTransform(f'{slicer.app.temporaryPath}/transform.tfm')
    
    #get name of node_reference
    nodeName = node_reference.GetName()
    slicer.mrmlScene.RemoveNode(node_reference)
    transform_node.SetName(nodeName)

    moving = itk.CastImageFilter[type(moving), itk.Image[itk.F, 3]].New()(moving)
    interpolator = itk.LinearInterpolateImageFunction.New(moving)
    warped_moving_image = itk.resample_image_filter(
            moving,
            transform=phi_AB,
            interpolator=interpolator,
            use_reference_image=True,
            reference_image=fixed
            )

    outputVolumeNode = outputSettings['volume']
    outputVolumeNode.CreateDefaultDisplayNodes()
    sitkUtils.PushVolumeToSlicer(self.itk2sitk(warped_moving_image), outputVolumeNode)
    
#
# Preset Manager
#

class PresetManager:
  def __init__(self):
      self.presetPath = os.path.join(os.path.dirname(__file__), 'Resources', 'Presets')

  def saveStagesAsPreset(self, stages):
    from PythonQt import BoolResult
    ok = BoolResult()
    presetName = \
      qt.QInputDialog().getText(qt.QWidget(), 'Save Preset', 'Preset name: ', qt.QLineEdit.Normal, 'my_preset', ok)
    if not ok:
      return
    if presetName in self.getPresetNames():
      slicer.util.warningDisplay(f'{presetName} already exists. Set another name.')
      return self.saveStagesAsPreset(stages)
    outFilePath = os.path.join(self.presetPath, f'{presetName}.json')
    saveSettings = self.getPresetParametersByName()
    saveSettings['stages'] = stages
    try:
      with open(outFilePath, 'w') as outfile:
        json.dump(saveSettings, outfile)
    except:
      slicer.util.warningDisplay(f'Unable to write into {outFilePath}')
      return
    slicer.util.infoDisplay(f'Saved preset to {outFilePath}.')
    return presetName

  def getPresetParametersByName(self, name='unigradicon'):
    presetFilePath = os.path.join(self.presetPath, name + '.json')
    with open(presetFilePath) as presetFile:
      return json.load(presetFile)

  def getPresetNames(self):
    G = glob.glob(os.path.join(self.presetPath, '*.json'))
    return [os.path.splitext(os.path.basename(g))[0] for g in G]


#
# unigradiconTest
#

class UniGradICONTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_UniGradICON()

  def test_UniGradICON(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")

    # Get/create input data

    import SampleData
    sampleDataLogic = SampleData.SampleDataLogic()
    fixed = sampleDataLogic.downloadMRBrainTumor1()
    moving = sampleDataLogic.downloadMRBrainTumor2()

    outputVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')

    logic = UniGradICONLogic()
    presetParameters = PresetManager().getPresetParametersByName('QuickSyN')

    presetParameters['outputSettings']['volume'] = outputVolume
    presetParameters['outputSettings']['transform'] = None
    presetParameters['outputSettings']['log'] = None
    logic.process(**presetParameters)

    logic.cliNode.AddObserver('ModifiedEvent', self.onProcessingStatusUpdate)

  def onProcessingStatusUpdate(self, caller, event):
    if caller.GetStatus() & caller.Completed:
      self.delayDisplay('Test passed!')
