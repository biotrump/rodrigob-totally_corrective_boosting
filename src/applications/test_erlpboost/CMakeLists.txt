# This is a CMake build file, for more information consult:
# http://en.wikipedia.org/wiki/CMake
# and
# http://www.cmake.org/Wiki/CMake
# http://www.cmake.org/cmake/help/syntax.html
# http://www.cmake.org/Wiki/CMake_Useful_Variables
# http://www.cmake.org/cmake/help/cmake-2-8-docs.html

# to compile the local code you can use: cmake ./ && make -j2

# ----------------------------------------------------------------------
# Base CMake setup
cmake_minimum_required(VERSION 2.6)
set(project_root_folder "../../..")
set(src_folder "../..")

# ----------------------------------------------------------------------
# Setup the project
include(FindPkgConfig)
project (TotallyCorrectiveEntropyRegularizedBoosting)

option(USE_TAO "Should the TAO optimizer be used ?" ON)
option(USE_CLP "Should the COIN linear programming optimizer be used ?" ON)

# ----------------------------------------------------------------------
# Setup link and include directories

include_directories(
  "${project_root_folder}/libs"
  "${project_root_folder}/src"
)

# ----------------------------------------------------------------------
# Collect source files

file(GLOB SrcCpp 
  "./*.cpp"
  "${src_folder}/*.cpp"

  "${project_root_folder}/libs/lbfgsb/*.cpp"
  "${project_root_folder}/libs/lbfgsb/ap/*.cpp"
)

# Files that will not compile on all setups
file(GLOB BlackListCpp
  "${src_folder}/LpBoost.cpp"
  "${src_folder}/TaoOptimizer.cpp"
)

list(REMOVE_ITEM SrcCpp ${BlackListCpp})

if(USE_TAO)
file(GLOB TaoCpp
  "${src_folder}/TaoOptimizer.cpp"
)

list(APPEND SrcCpp ${TaoCpp})
add_definitions(-DUSE_TAO)
endif(USE_TAO)

if(USE_CLP)

file(GLOB ClpCpp
  "${src_folder}/LpBoost.cpp"
)

list(APPEND SrcCpp ${ClpCpp})
add_definitions(-DUSE_CLP)
endif(USE_CLP)


# ----------------------------------------------------------------------
# Create the executable

add_executable(test_erlpboost
  ${SrcCpp}
  ${HelpersCpp}
)

target_link_libraries(test_erlpboost
   boost_program_options-mt boost_filesystem-mt boost_system-mt boost_thread-mt 
   #protobuf
   gomp 
   
   #csparse sparse spblas mv
   #lapack blas atlas
)   

if(USE_CLP)
target_link_libraries(test_erlpboost
  Clp
)
endif(USE_CLP)

