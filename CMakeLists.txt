CMAKE_MINIMUM_REQUIRED(VERSION 3.9) # 3.9: OpenMP improved support

### Metadata ###

# Debug ONLY
set(CMAKE_VERBOSE_MAKEFILE TRUE)

PROJECT(EASEA VERSION 3.2.0)

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

#Easea compiler
SET(easea_src 
        ${CMAKE_SOURCE_DIR}/compiler/EaseaLex.cpp
        ${CMAKE_SOURCE_DIR}/compiler/EaseaParse.cpp
        ${CMAKE_SOURCE_DIR}/compiler/EaseaSym.cpp
        ${CMAKE_SOURCE_DIR}/compiler/EaseaYTools.cpp
        ${CMAKE_SOURCE_DIR}/compiler/easea.cpp
        ${CMAKE_SOURCE_DIR}/compiler/errors.cpp
)
ADD_EXECUTABLE(easea ${easea_src})
SET_TARGET_PROPERTIES(libeasea PROPERTIES OUTPUT_NAME "libeasea" PREFIX "")

### Base flags
SET(CMAKE_STATIC_LIBRARY_PREFIX="")
target_compile_features(libeasea PUBLIC cxx_std_17)
target_compile_options(libeasea PUBLIC
	$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -g>
	$<$<AND:$<NOT:$<CONFIG:Debug>>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-O3 -march=native -mtune=native>
	$<$<AND:$<CONFIG:Debug>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-O0 -g -fsanitize=address -fsanitize=undefined -fno-sanitize=vptr>

	$<$<CXX_COMPILER_ID:MSVC>:/W3>
	$<$<AND:$<NOT:$<CONFIG:Debug>>,$<CXX_COMPILER_ID:MSVC>>:/O2>
	$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/DEBUG:FULL /W4 /fsanitize=address>
	)
target_compile_options(easea PUBLIC
	$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-g -Wall -Wextra -w -Wno-deprecated -Wno-write-strings -fmessage-length=0>
	$<$<AND:$<NOT:$<CONFIG:Debug>>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-O3 -march=native -mtune=native>
	$<$<AND:$<CONFIG:Debug>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-g -fsanitize=address -fsanitize=undefined -fno-sanitize=vptr>

	$<$<CXX_COMPILER_ID:MSVC>:/W3>
	$<$<AND:$<NOT:$<CONFIG:Debug>>,$<CXX_COMPILER_ID:MSVC>>:/O2>
	$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/DEBUG:FULL /W4 /fsanitize=address>
	)
TARGET_LINK_LIBRARIES(easea libeasea)
target_link_options(libeasea PUBLIC
	$<$<AND:$<CONFIG:Debug>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=address -fsanitize=undefined -fno-sanitize=vptr>
	$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/fsanitize=address>
	)
target_link_options(easea PUBLIC $<$<AND:$<CONFIG:Debug>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=address -fsanitize=undefined -fno-sanitize=vptr>
	$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/fsanitize=address>
	)
target_compile_definitions(easea PUBLIC $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>) # :(

### Includes ###

INCLUDE_DIRECTORIES(${PROJECT_BINARY_BIN})
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/libeasea/include)

### Dependencies ###

## OpenMP
option(USE_OPENMP "If available, use OpenMP for parallelization." ON)

if (USE_OPENMP)
	find_package(OpenMP REQUIRED)
endif ()
if (OpenMP_CXX_FOUND)
	target_link_libraries(libeasea PUBLIC OpenMP::OpenMP_CXX)
else ()
  	# Disable warnings for all the unknown OpenMP pragmas.
	target_compile_options(libeasea PUBLIC $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-unknown-pragmas>)
endif ()

## Boost
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
	add_definitions(-DBOOST_ALL_NO_LIB)
	set(Boost_USE_STATIC_LIBS ON)
	set(Boost_USE_MULTITHREADED ON)
	set(Boost_USE_STATIC_RUNTIME OFF)
endif()
find_package(Boost REQUIRED COMPONENTS iostreams serialization program_options)
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

### Installation ###
if (WIN32)
	set(OUTPUT_DIR ".")
else()
	string(TOLOWER ${PROJECT_NAME} OUTPUT_DIR)
endif()
## Binaries
INSTALL(TARGETS easea libeasea)
INSTALL(TARGETS easea DESTINATION ${OUTPUT_DIR}/bin)
INSTALL(TARGETS libeasea DESTINATION ${OUTPUT_DIR}/libeasea)

## Text
FILE(GLOB templates ${CMAKE_SOURCE_DIR}/tpl/*)
INSTALL(FILES ${templates} DESTINATION ${OUTPUT_DIR}/tpl)
INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/libeasea/include DESTINATION ${OUTPUT_DIR}/libeasea)
INSTALL(DIRECTORY examples DESTINATION ${OUTPUT_DIR})
INSTALL(FILES "easeagrapher/EaseaGrapher.jar" DESTINATION ${OUTPUT_DIR}/easeagrapher)

### Tests ###

### Configuration file

CONFIGURE_FILE(config.h.in ${CMAKE_SOURCE_DIR}/libeasea/include/config.h)
