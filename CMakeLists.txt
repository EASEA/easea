CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

project(EASEA)
SET(CMAKE_STATIC_LIBRARY_PREFIX="")
INCLUDE_DIRECTORIES(libeasea/include alexyacc/include boost)

ADD_DEFINITIONS(-g -w -Wno-deprecated -Wno-write-strings -fmessage-length=0 )


if(WIN32)
  FILE(GLOB libeasea_src_win  "libeasea/win32/*.cpp" )
endif(WIN32)


#Alexyacc static library
FILE(GLOB libalex_src alexyacc/*.cpp)
ADD_LIBRARY( libalex STATIC ${libalex_src} )


#Libeasea static library
FILE(GLOB libeasea_src libeasea/*.cpp )
ADD_LIBRARY( libeasea STATIC ${libeasea_src} ${libeasea_src_win})
SET_TARGET_PROPERTIES(libeasea PROPERTIES OUTPUT_NAME "libeasea" PREFIX "" COMPILE_FLAGS "-O3")


#Boost program_options static library
FILE(GLOB boost_src boost/*.cpp)
ADD_LIBRARY( program_options STATIC ${boost_src} )
SET_TARGET_PROPERTIES(program_options PROPERTIES OUTPUT_NAME "program_options" PREFIX "" COMPILE_FLAGS "-O2")

#Easea compiler
SET(easea_src 
        compiler/EaseaLex.cpp
        compiler/EaseaParse.cpp
        compiler/EaseaSym.cpp
        compiler/EaseaYTools.cpp)


ADD_EXECUTABLE(easea ${easea_src})
SET_TARGET_PROPERTIES(easea PROPERTIES COMPILE_FLAGS "")
TARGET_LINK_LIBRARIES(easea libeasea libalex program_options)
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
INSTALL(FILES easeagrapher/EaseaGrapher.jar DESTINATION /usr/local/easea/easeagrapher )
