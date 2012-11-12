# This is a CMake build file, for more information consult:
# http://en.wikipedia.org/wiki/CMake
# and
# http://www.cmake.org/Wiki/CMake
# http://www.cmake.org/cmake/help/syntax.html
# http://www.cmake.org/Wiki/CMake_Useful_Variables
# http://www.cmake.org/cmake/help/cmake-2-8-docs.html

# to compile the local code you can use: cmake ./ and make -j2

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
  "${src_folder}/*.cpp"
  "${src_folder}/weak_learners/*.cpp"
  "${src_folder}/oracles/*.cpp"
  "${src_folder}/optimizers/*.cpp"
  "${src_folder}/boosters/*.cpp"
  "${src_folder}/math/*.cpp"



  "${project_root_folder}/libs/lbfgsb/*.cpp"
  "${project_root_folder}/libs/lbfgsb/ap/*.cpp"
)

# Files that will not compile on all setups
file(GLOB BlackListCpp
  "${src_folder}/boosters/LpBoost.cpp"
  "${src_folder}/optimizers/TaoOptimizer.cpp"
)

list(REMOVE_ITEM SrcCpp ${BlackListCpp})

if(USE_TAO)
file(GLOB TaoCpp
  "${src_folder}/optimizers/TaoOptimizer.cpp"
)

list(APPEND SrcCpp ${TaoCpp})
add_definitions(-DUSE_TAO)
endif(USE_TAO)

if(USE_CLP)

file(GLOB ClpCpp
  "${src_folder}/boosters/LpBoost.cpp"
)

list(APPEND SrcCpp ${ClpCpp})
add_definitions(-DUSE_CLP)
endif(USE_CLP)

# ----------------------------------------------------------------------
# set default compilation flags and default build
set(OPT_CXX_FLAGS "-fopenmp -ffast-math -funroll-loops -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -DNDEBUG -DBOOST_DISABLE_ASSERTS ${OPT_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")


# ----------------------------------------------------------------------
# Create the library
add_library( totally_corrective_boosting SHARED
   ${SrcCpp}
   ${HelpersCpp}
)

target_link_libraries( totally_corrective_boosting

   boost_program_options-mt boost_filesystem-mt boost_system-mt boost_thread-mt
   #protobuf
   gomp

   #csparse sparse spblas mv
   #lapack blas atlas
)


if(USE_CLP)
target_link_libraries(totally_corrective_boosting
  Clp
)
endif(USE_CLP)

# ----------------------------------------------------------------------
# Create the executable

add_executable(test_erlpboost "test_erlpboost.cpp")

target_link_libraries(test_erlpboost

   totally_corrective_boosting

   boost_program_options-mt
   boost_filesystem-mt
   boost_system-mt
   boost_thread-mt
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

