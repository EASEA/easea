CMAKE_MINIMUM_REQUIRED(VERSION 3.9) # 3.9: OpenMP improved support

### Metadata ###

# Debug ONLY
set(CMAKE_VERBOSE_MAKEFILE TRUE)

PROJECT(EASENA VERSION 3.2.0)

message("-- PROJECT NAME: ${PROJECT_NAME}")
message("-- PROJECT VERSION: ${PROJECT_VERSION}")
message("-- PROJECT VERSION MAJOR: ${PROJECT_VERSION_MAJOR}")
message("-- PROJECT VERSION MINOR: ${PROJECT_VERSION_MINOR}")
message("-- PROJECT VERSION PATCH: ${PROJECT_VERSION_PATCH}")

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release")
endif()

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
target_compile_features(libeasea PUBLIC cxx_std_17)
target_compile_features(libeasna PUBLIC cxx_std_17)
target_compile_options(libeasea PUBLIC
	$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -g>
	$<$<AND:$<NOT:$<CONFIG:Debug>>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-O3 -march=native -mtune=native>
	$<$<AND:$<CONFIG:Debug>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-O0 -g -fsanitize=address -fsanitize=undefined -fno-sanitize=vptr>

	$<$<CXX_COMPILER_ID:MSVC>:/W3>
	$<$<AND:$<NOT:$<CONFIG:Debug>>,$<CXX_COMPILER_ID:MSVC>>:/O2>
	$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/DEBUG:FULL /W4 /fsanitize=address>
	)
target_compile_options(libeasna PUBLIC
	$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-g -pedantic -Wall -Wextra -Woverloaded-virtual -Wfloat-equal -Wwrite-strings -Wpointer-arith -Wcast-qual -Wcast-align -Wshadow -Wredundant-decls -Wdouble-promotion -Winit-self -Wswitch-default>
	$<$<AND:$<NOT:$<CONFIG:Debug>>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-O3 -march=native -mtune=native>
	$<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g -fsanitize=address -fsanitize=undefined -fno-sanitize=vptr>

	$<$<CXX_COMPILER_ID:MSVC>:/W3>
	$<$<AND:$<NOT:$<CONFIG:Debug>>,$<CXX_COMPILER_ID:MSVC>>:/O2>
	$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/DEBUG:FULL /fsanitize=address>
	)
target_compile_options(easena PUBLIC
	$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-g -Wall -Wextra -w -Wno-deprecated -Wno-write-strings -fmessage-length=0>
	$<$<AND:$<NOT:$<CONFIG:Debug>>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-O3 -march=native -mtune=native>
	$<$<AND:$<CONFIG:Debug>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-g -fsanitize=address -fsanitize=undefined -fno-sanitize=vptr>

	$<$<CXX_COMPILER_ID:MSVC>:/W3>
	$<$<AND:$<NOT:$<CONFIG:Debug>>,$<CXX_COMPILER_ID:MSVC>>:/O2>
	$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/DEBUG:FULL /fsanitize=address>
	)
TARGET_LINK_LIBRARIES(easena libeasea libeasna)
target_link_options(libeasea PUBLIC
	$<$<AND:$<CONFIG:Debug>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=address -fsanitize=undefined -fno-sanitize=vptr>
	$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/fsanitize=address>
	)
target_link_options(libeasna PUBLIC $<$<AND:$<CONFIG:Debug>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=address -fsanitize=undefined -fno-sanitize=vptr>
	$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/fsanitize=address>
	)
target_link_options(easena PUBLIC $<$<AND:$<CONFIG:Debug>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=address -fsanitize=undefined -fno-sanitize=vptr>
	$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/fsanitize=address>
	)
target_compile_definitions(easena PUBLIC $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>) # :(

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
	target_compile_options(libeasna PUBLIC $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-unknown-pragmas>)
endif ()

## Boost (asio)
find_package(Boost REQUIRED COMPONENTS iostreams serialization)
target_include_directories(libeasea PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(libeasea PUBLIC ${Boost_LIBRARIES})

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
if (WIN32)
	set(OUTPUT_DIR ".")
else()
	string(TOLOWER ${PROJECT_NAME} OUTPUT_DIR)
endif()
## Binaries
INSTALL(TARGETS easena libeasea libeasna)
INSTALL(TARGETS easena DESTINATION ${OUTPUT_DIR}/bin)
INSTALL(TARGETS libeasea DESTINATION ${OUTPUT_DIR}/libeasea)
INSTALL(TARGETS libeasna DESTINATION ${OUTPUT_DIR}/libeasna)

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

### Configuration file

CONFIGURE_FILE(config.h.in ${CMAKE_SOURCE_DIR}/libeasea/include/config.h)
