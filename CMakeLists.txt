CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

project(EASEA)
SET(CMAKE_STATIC_LIBRARY_PREFIX="")
INCLUDE_DIRECTORIES(libeasea/include boost)

ADD_DEFINITIONS(-g -w -Wno-deprecated -Wno-write-strings -fmessage-length=0 )


if(WIN32)
  FILE(GLOB libeasea_src_win  "libeasea/win32/*.cpp" )
endif(WIN32)

#Libeasea static library
FILE(GLOB libeasea_src libeasea/*.cpp )
ADD_LIBRARY( libeasea STATIC ${libeasea_src} ${libeasea_src_win})
SET_TARGET_PROPERTIES(libeasea PROPERTIES OUTPUT_NAME "libeasea" PREFIX "" COMPILE_FLAGS "-O3 -fopenmp")


#Boost program_options static library
FILE(GLOB boost_src boost/*.cpp)
ADD_LIBRARY( program_options STATIC ${boost_src} )
SET_TARGET_PROPERTIES(program_options PROPERTIES OUTPUT_NAME "program_options" PREFIX "" COMPILE_FLAGS "-O2")


find_package(BISON)
find_package(FLEX)
FLEX_TARGET(Lexer compiler/EaseaLex.l compiler/EaseaLex.cpp)
BISON_TARGET(Parser compiler/EaseaParse.y compiler/EaseaParse.cpp COMPILE_FLAGS -d)

#Easea compiler
SET(easea_src 
        compiler/EaseaLex.cpp
        compiler/EaseaParse.cpp
        compiler/EaseaSym.cpp
        compiler/EaseaYTools.cpp)


ADD_EXECUTABLE(easea ${easea_src})
ADD_DEPENDENCIES(easea Lexer Parser)
SET_TARGET_PROPERTIES(easea PROPERTIES COMPILE_FLAGS "")
TARGET_LINK_LIBRARIES(easea libeasea program_options)
ADD_CUSTOM_COMMAND(TARGET easea POST_BUILD 
                    COMMAND $(CMAKE_COMMAND) -E copy libeasea.a libeasea/ 
                    COMMAND $(CMAKE_COMMAND) -E copy program_options.a boost/
                   )

INSTALL(TARGETS easea DESTINATION /usr/local/easea/bin )
INSTALL(TARGETS libeasea DESTINATION /usr/local/easea/libeasea )
INSTALL(TARGETS program_options DESTINATION /usr/local/easea/boost )
INSTALL(DIRECTORY tpl DESTINATION /usr/local/easea )
INSTALL(DIRECTORY "libeasea/include" DESTINATION "/usr/local/easea/libeasea")
INSTALL(DIRECTORY boost/boost DESTINATION /usr/local/easea/boost)
INSTALL(DIRECTORY examples DESTINATION /usr/local/easea/examples)
INSTALL(FILES easeagrapher/EaseaGrapher.jar DESTINATION /usr/local/easea/easeagrapher )

SET(MAJOR_VERSION 0)
SET(MINOR_VERSION 1)
SET(PATCH_VERSION 0)
 
IF(EXISTS "${CMAKE_ROOT}/Modules/CPack.cmake")
INCLUDE(InstallRequiredSystemLibraries)
 
SET(CPACK_SET_DESTDIR "on")
SET(CPACK_PACKAGING_INSTALL_PREFIX "/tmp")
SET(CPACK_GENERATOR "DEB")
 
SET(CPACK_PACKAGE_DESCRIPTION "EASEA framework")
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "EASEA: A distributed and massively parallel evolutionary engine")
SET(CPACK_PACKAGE_VENDOR "University of Strasbourg")
SET(CPACK_PACKAGE_CONTACT "Pallamidessi Joseph")
SET(CPACK_PACKAGE_VERSION_MAJOR "${MAJOR_VERSION}")
SET(CPACK_PACKAGE_VERSION_MINOR "${MINOR_VERSION}")
SET(CPACK_PACKAGE_VERSION_PATCH "${PATCH_VERSION}")
SET(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}_${MAJOR_VERSION}.${MINOR_VERSION}.${CPACK_PACKAGE_VERSION_PATCH}")
SET(CPACK_SOURCE_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}_${MAJOR_VERSION}.${MINOR_VERSION}.${CPACK_PACKAGE_VERSION_PATCH}")
 
SET(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.1.3), libgcc1 (>= 1:4.1.1), libstdc++6 (>= 4.2.1), build-essential, openjdk-8-jre")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "postinst;prerm;")
SET(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
SET(CPACK_DEBIAN_PACKAGE_SECTION "devel")
SET(CPACK_DEBIAN_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
 
SET(CPACK_COMPONENTS_ALL Libraries ApplicationData)
INCLUDE(CPack)
 
ENDIF(EXISTS "${CMAKE_ROOT}/Modules/CPack.cmake")
