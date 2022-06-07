CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(EASENA)#_DEBUG VERSION "2.20.0")
set(PROJECT_NAME "EASENA")
set(PROJECT_VER  "RE")
set(PROJECT_VER_MAJOR 2)
set(PROJECT_VER_MINOR 20)
set(PROJECT_VER_PATCH 0)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/libeasea/include)

message("-- PROJECT NAME: ${PROJECT_NAME}")
message("-- PROJECT VERSION: ${PROJECT_VER}")
message("-- PROJECT VERSION MAJOR: ${PROJECT_VER_MAJOR}")
message("-- PROJECT VERSION MINOR: ${PROJECT_VER_MINOR}")

SET(CMAKE_STATIC_LIBRARY_PREFIX="")
INCLUDE_DIRECTORIES(${PROJECT_BINARY_BIN})
INCLUDE_DIRECTORIES(libeasea/include)
INCLUDE_DIRECTORIES(libeasna/include)
INCLUDE_DIRECTORIES(cxxopts)

option(USE_OPENMP "If available, use OpenMP for parallelization." ON)

if (USE_OPENMP)
  find_package(OpenMP)
endif ()
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else ()
  # Disable warnings for all the unknown OpenMP pragmas.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
endif ()

SET(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)

# Creation of a static library for EASEA that will be linked to the main program easena
FILE(GLOB libeasea_src libeasea/*.cpp )
if(WIN32)
        FILE(GLOB libeasea_src_win  "libeasea/win32/*.cpp" )
endif(WIN32)
ADD_LIBRARY( libeasea STATIC ${libeasea_src} ${libeasea_src_win})
SET_TARGET_PROPERTIES(libeasea PROPERTIES OUTPUT_NAME "libeasea" PREFIX "" COMPILE_FLAGS "-std=c++14 -O3")
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
endif()
# -fopenmp")

# Creation of a static library for EASNA that will be linked to the main program easena 
# Program based on the one that can be found in https://git.unistra.fr/rorhand/perceptron/
FILE(GLOB_RECURSE libeasna_src libeasna/*.cpp )
ADD_LIBRARY(libeasna STATIC ${libeasna_src})
SET_TARGET_PROPERTIES(libeasna PROPERTIES OUTPUT_NAME "libeasna" PREFIX "" COMPILE_FLAGS "-pedantic -Wall -Wextra -Woverloaded-virtual -Wfloat-equal -Wwrite-strings -Wpointer-arith -Wcast-qual -Wcast-align -Wshadow -Wredundant-decls -Wdouble-promotion -Winit-self -Wswitch-default -Wlogical-op -std=c++11 -O3 -march=native -mtune=native -s")
#if(CMAKE_VERSION VERSION_GREATER 3.9.0)
#        TARGET_LINK_LIBRARIES(libeasna PUBLIC OpenMP::OpenMP_CXX)
#else ()
#        TARGET_LINK_LIBRARIES(libeasna PUBLIC "${OpenMP_CXX_FLAGS}")
#endif ()

# FLEX and BISON management to create the easea language and soon the easna/neural language
FIND_PACKAGE(FLEX)
if(APPLE)
        SET(BISON_EXECUTABLE "/usr/local/opt/bison/bin/bison" CACHE PATH "Bison executable")
endif(APPLE)
FIND_PACKAGE(BISON)
FLEX_TARGET( EaseaLexer 
        compiler/EaseaLex.l 
	${CMAKE_SOURCE_DIR}/compiler/EaseaLex.cpp
)
BISON_TARGET( EaseaParser
        compiler/EaseaParse.y 
	${CMAKE_SOURCE_DIR}/compiler/EaseaParse.cpp 
        COMPILE_FLAGS -d
        DEFINES_FILE compiler/EaseaParse.hpp
)
FLEX_TARGET( NeuralLexer 
        compiler/NeuralLex.l 
        ${CMAKE_SOURCE_DIR}/compiler/NeuralLex.cpp
        DEFINES_FILE compiler/NeuralLex.h
)
BISON_TARGET( NeuralParser
        compiler/NeuralParse.y 
        ${CMAKE_SOURCE_DIR}/compiler/NeuralParse.cpp 
        COMPILE_FLAGS -d
        DEFINES_FILE compiler/NeuralParse.hpp
)

#Easena compiler
SET(easena_src 
        compiler/EaseaLex.cpp
        compiler/EaseaParse.cpp
        compiler/EaseaSym.cpp
        compiler/EaseaYTools.cpp
        compiler/easena.cpp
        compiler/NeuralLex.cpp
        compiler/NeuralParse.cpp
)

ADD_EXECUTABLE(easena ${easena_src})
SET_TARGET_PROPERTIES(easena PROPERTIES COMPILE_FLAGS "-std=c++14 -w -Wno-deprecated -Wno-write-strings -fmessage-length=0")
TARGET_LINK_LIBRARIES(easena libeasea libeasna)
ADD_CUSTOM_COMMAND(TARGET easena POST_BUILD 
	COMMAND $(CMAKE_COMMAND) -E make_directory libeasna
	COMMAND $(CMAKE_COMMAND) -E make_directory libeasea
        COMMAND $(CMAKE_COMMAND) -E copy libeasea.a libeasea/
        COMMAND $(CMAKE_COMMAND) -E copy libeasna.a libeasna/
)

INSTALL(TARGETS easena DESTINATION /usr/local/easena/bin )
INSTALL(TARGETS libeasea DESTINATION /usr/local/easena/libeasea )
INSTALL(TARGETS libeasna DESTINATION /usr/local/easena/libeasna )
INSTALL(DIRECTORY tpl DESTINATION /usr/local/easena )
INSTALL(DIRECTORY "libeasea/include" DESTINATION "/usr/local/easena/libeasea")
INSTALL(DIRECTORY "libeasna/include" DESTINATION "/usr/local/easena/libeasna")
INSTALL(DIRECTORY "libeasna/src" DESTINATION "/usr/local/easena/libeasna")
INSTALL(DIRECTORY examples DESTINATION /usr/local/easena/examples)
INSTALL(FILES easeagrapher/EaseaGrapher.jar DESTINATION /usr/local/easena/easeagrapher )

SET(MAJOR_VERSION 3)
SET(MINOR_VERSION 0)
SET(PATCH_VERSION 0)

ADD_SUBDIRECTORY(Test/xor_mnist)

CONFIGURE_FILE(config.h.in ${CMAKE_SOURCE_DIR}/libeasea/include/config.h)

## OUTDATED !
## IF(EXISTS "${CMAKE_ROOT}/Modules/CPack.cmake")
## INCLUDE(InstallRequiredSystemLibraries)
##  
## SET(CPACK_SET_DESTDIR "on")
## SET(CPACK_PACKAGING_INSTALL_PREFIX "/tmp")
## SET(CPACK_GENERATOR "DEB")
##  
## SET(CPACK_PACKAGE_DESCRIPTION "EASEA framework")
## SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "EASEA: A distributed and massively parallel evolutionary engine")
## SET(CPACK_PACKAGE_VENDOR "University of Strasbourg")
## SET(CPACK_PACKAGE_CONTACT "Pallamidessi Joseph")
## SET(CPACK_PACKAGE_VERSION_MAJOR "${MAJOR_VERSION}")
## SET(CPACK_PACKAGE_VERSION_MINOR "${MINOR_VERSION}")
## SET(CPACK_PACKAGE_VERSION_PATCH "${PATCH_VERSION}")
## SET(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}_${MAJOR_VERSION}.${MINOR_VERSION}.${CPACK_PACKAGE_VERSION_PATCH}")
## SET(CPACK_SOURCE_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}_${MAJOR_VERSION}.${MINOR_VERSION}.${CPACK_PACKAGE_VERSION_PATCH}")
##  
## SET(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.1.3), libgcc1 (>= 1:4.1.1), libstdc++6 (>= 4.2.1), build-essential, openjdk-8-jre")
## SET(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
## SET(CPACK_DEBIAN_PACKAGE_SECTION "devel")
## SET(CPACK_DEBIAN_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
##  
## SET(CPACK_COMPONENTS_ALL Libraries ApplicationData)
## INCLUDE(CPack)
##  
## ENDIF(EXISTS "${CMAKE_ROOT}/Modules/CPack.cmake")
