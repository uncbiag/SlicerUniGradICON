#-----------------------------------------------------------------------------
set(MODULE_NAME UniGradICON)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  )

set(EXTENSION_HOMEPAGE "https://github.com/uncbiag/SlicerUniGradICON")
set(EXTENSION_CATEGORY "Registration")
set(EXTENSION_CONTRIBUTORS "Basar Demir (University of North Carolina at Chapel Hill), Lin Tian (University of North Carolina at Chapel Hill), Hastings Greer (University of North Carolina at Chapel Hill), Marc Niethammer (University of North Carolina at Chapel Hill)")
set(EXTENSION_DESCRIPTION "This module performs medical image registration using the family of foundational GradICON deep registration models.")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/uncbiag/SlicerUniGradICON/main/screenshots/interface.png https://raw.githubusercontent.com/uncbiag/SlicerUniGradICON/main/screenshots/brain_example.png https://raw.githubusercontent.com/uncbiag/SlicerUniGradICON/main/screenshots/knee_example.png")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/uncbiag/SlicerUniGradICON/main/UniGradICON/Resources/Icons/UniGradICON.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
