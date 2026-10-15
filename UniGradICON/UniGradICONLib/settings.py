"""Shared settings keys and paths for source and installed modules."""
import glob
import json
import os

MODULE_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIRECTORY = os.path.join(os.path.dirname(MODULE_DIRECTORY), "model_checkpoints")

REMOTE_PROCESSING_SETTING = "UniGradICON/RemoteProcessing"
SERVER_URL_SETTING = "UniGradICON/ServerUrl"
SERVER_TOKENS_SETTING_GROUP = "UniGradICON/ServerTokens"


class PresetManager:
  """The model choices: one JSON file per model in Resources/Presets, with its default settings."""

  def __init__(self):
    self.presetPath = os.path.join(MODULE_DIRECTORY, 'Resources', 'Presets')

  def getPresetParametersByName(self, name='unigradicon'):
    presetFilePath = os.path.join(self.presetPath, name + '.json')
    with open(presetFilePath) as presetFile:
      return json.load(presetFile)

  def getPresetNames(self):
    presetFiles = glob.glob(os.path.join(self.presetPath, '*.json'))
    return sorted(os.path.splitext(os.path.basename(path))[0] for path in presetFiles)
