CMAKE_MINIMUM_REQUIRED(VERSION 3.9) # 3.9: OpenMP improved support

### Metadata ###

# Debug ONLY
set(CMAKE_VERBOSE_MAKEFILE TRUE)

PROJECT(EASENA VERSION 3.0.0)
#set(PROJECT_NAME "EASENA")

message("-- PROJECT NAME: ${PROJECT_NAME}")
message("-- PROJECT VERSION: ${PROJECT_VERSION}")
message("-- PROJECT VERSION MAJOR: ${PROJECT_VERSION_MAJOR}")
message("-- PROJECT VERSION MINOR: ${PROJECT_VERSION_MINOR}")
message("-- PROJECT VERSION PATCH: ${PROJECT_VERSION_PATCH}")

SET(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)

### Configuration file

CONFIGURE_FILE(config.h.in ${CMAKE_SOURCE_DIR}/libeasea/include/config.h)

### Target declarations ###
FILE(GLOB libeasea_src libeasea/*.cpp )
if(WIN32)
        FILE(GLOB libeasea_src_win  "libeasea/win32/*.cpp" )
endif(WIN32)

ADD_LIBRARY(libeasea STATIC ${libeasea_src} ${libeasea_src_win})
FILE(GLOB_RECURSE libeasna_src libeasna/*.cpp )
# Creation of a static library for EASNA that will be linked to the main program easena 
ADD_LIBRARY(libeasna STATIC ${libeasna_src}) # Program based on the one that can be found in https://git.unistra.fr/rorhand/perceptron/
#Easena compiler
SET(easena_src 
        ${CMAKE_SOURCE_DIR}/compiler/EaseaLex.cpp
        ${CMAKE_SOURCE_DIR}/compiler/EaseaParse.cpp
        ${CMAKE_SOURCE_DIR}/compiler/EaseaSym.cpp
        ${CMAKE_SOURCE_DIR}/compiler/EaseaYTools.cpp
        ${CMAKE_SOURCE_DIR}/compiler/easena.cpp
        ${CMAKE_SOURCE_DIR}/compiler/NeuralLex.cpp
        ${CMAKE_SOURCE_DIR}/compiler/NeuralParse.cpp
)
ADD_EXECUTABLE(easena ${easena_src})
SET_TARGET_PROPERTIES(libeasea PROPERTIES OUTPUT_NAME "libeasea" PREFIX "")
SET_TARGET_PROPERTIES(libeasna PROPERTIES OUTPUT_NAME "libeasna" PREFIX "")

### Base flags
SET(CMAKE_STATIC_LIBRARY_PREFIX="")
target_compile_features(libeasea PUBLIC cxx_std_14)
target_compile_features(libeasna PUBLIC cxx_std_14)
target_compile_options(libeasea PUBLIC
	$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -march=native -mtune=native>
	$<$<CXX_COMPILER_ID:MSVC>:/O2>
	)
target_compile_options(libeasna PUBLIC
	$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -march=native -mtune=native>
	$<$<CXX_COMPILER_ID:MSVC>:/O2>
	$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-pedantic -Wall -Wextra -Woverloaded-virtual -Wfloat-equal -Wwrite-strings -Wpointer-arith -Wcast-qual -Wcast-align -Wshadow -Wredundant-decls -Wdouble-promotion -Winit-self -Wswitch-default -Wlogical-op>
	$<$<CXX_COMPILER_ID:MSVC>:/W4>
	)
target_compile_options(easena PUBLIC
	$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w -Wno-deprecated -Wno-write-strings -fmessage-length=0>
	)
TARGET_LINK_LIBRARIES(easena libeasea libeasna)

### Includes ###

INCLUDE_DIRECTORIES(${PROJECT_BINARY_BIN})
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/libeasea/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/libeasna/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/cxxopts)

### Dependencies ###

## OpenMP
option(USE_OPENMP "If available, use OpenMP for parallelization." ON)

if (USE_OPENMP)
	find_package(OpenMP REQUIRED)
endif ()
if (OpenMP_CXX_FOUND)
	target_link_libraries(libeasea PUBLIC OpenMP::OpenMP_CXX)
	target_link_libraries(libeasna PUBLIC OpenMP::OpenMP_CXX)
else ()
  	# Disable warnings for all the unknown OpenMP pragmas.
  	target_compile_options(libeasna $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-unknown-pragmas>)
endif ()

## Boost (asio)
find_package(Boost REQUIRED)
target_include_directories(libeasea PUBLIC ${Boost_INCLUDE_DIRS})

## Bison & Flex: FLEX and BISON management to create the easea language and soon the easna/neural language

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
        DEFINES_FILE ${CMAKE_SOURCE_DIR}/compiler/EaseaParse.hpp
)
FLEX_TARGET( NeuralLexer 
        compiler/NeuralLex.l 
        ${CMAKE_SOURCE_DIR}/compiler/NeuralLex.cpp
        DEFINES_FILE ${CMAKE_SOURCE_DIR}/compiler/NeuralLex.h
)
BISON_TARGET( NeuralParser
        compiler/NeuralParse.y 
        ${CMAKE_SOURCE_DIR}/compiler/NeuralParse.cpp 
        COMPILE_FLAGS -d
        DEFINES_FILE ${CMAKE_SOURCE_DIR}/compiler/NeuralParse.hpp
)

### Installation ###
set(OUTPUT_DIR string(TOLOWER ${PROJECT_NAME}))
## Binaries
INSTALL(TARGETS easena libeasea libeasna)
INSTALL(TARGETS libeasna DESTINATION ${OUTPUT_DIR}/libeasea)
INSTALL(TARGETS libeasea DESTINATION ${OUTPUT_DIR}/libeasna)

## Text
FILE(GLOB templates ${CMAKE_SOURCE_DIR}/tpl/*)
INSTALL(FILES ${templates} DESTINATION ${OUTPUT_DIR}/tpl)
INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/libeasea/include DESTINATION ${OUTPUT_DIR}/libeasea)
INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/libeasna/include DESTINATION ${OUTPUT_DIR}/libeasna)
INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/libeasna/src DESTINATION ${OUTPUT_DIR}/libeasna)
INSTALL(DIRECTORY examples DESTINATION ${OUTPUT_DIR})
INSTALL(FILES "easeagrapher/EaseaGrapher.jar" DESTINATION ${OUTPUT_DIR}/easeagrapher)

### Tests ###

ADD_SUBDIRECTORY(Test/xor_mnist)


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
