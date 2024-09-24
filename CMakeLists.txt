cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(UniGradICON)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/uncbiag/SlicerUniGradICON")
set(EXTENSION_CATEGORY "Registration")
set(EXTENSION_CONTRIBUTORS "Basar Demir (UNC)")
set(EXTENSION_DESCRIPTION "This module performs registration of medical images using uniGradICON or multiGradICON.")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

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
