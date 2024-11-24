cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(UniGradICON)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/uncbiag/SlicerUniGradICON")
set(EXTENSION_CATEGORY "Registration")
set(EXTENSION_CONTRIBUTORS "Basar Demir (University of North Carolina at Chapel Hill), Lin Tian (University of North Carolina at Chapel Hill), Hastings Greer (University of North Carolina at Chapel Hill), Marc Niethammer (University of North Carolina at Chapel Hill)")
set(EXTENSION_DESCRIPTION "This module performs medical image registration using the family of foundational GradICON deep registration models.")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/uncbiag/SlicerUniGradICON/main/screenshots/interface.png https://raw.githubusercontent.com/uncbiag/SlicerUniGradICON/main/screenshots/brain_example.png https://raw.githubusercontent.com/uncbiag/SlicerUniGradICON/main/screenshots/knee_example.png")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/uncbiag/SlicerUniGradICON/main/UniGradICON/Resources/Icons/UniGradICON.png")
set(EXTENSION_DEPENDS "PyTorch") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(UniGradICON)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})