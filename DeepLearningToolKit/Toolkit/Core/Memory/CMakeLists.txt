### Lib file
#add_library(cSanMemoryManager STATIC cSanMemoryManager.h cSanMemoryManager.cpp)
#set_target_properties(cSanMemoryManager PROPERTIES LINKER_LANGUAGE CXX)

#add_library(cSanMemoryManagerAlgorithms STATIC cSanMemoryManagerAlgorithms.h cSanMemoryManagerAlgorithms.cpp)
#set_target_properties(cSanMemoryManagerAlgorithms PROPERTIES LINKER_LANGUAGE CXX)
#
#add_library(cSanStackManager STATIC cSanStackManager.h cSanStackManager.cpp)
#set_target_properties(cSanStackManager PROPERTIES LINKER_LANGUAGE CXX)

add_library(cSanMemoryManagerDef STATIC cSanMemoryManagerDef.h)
set_target_properties(cSanMemoryManagerDef PROPERTIES LINKER_LANGUAGE CXX)

