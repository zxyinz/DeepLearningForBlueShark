### Core/Container CMakeLists

### Lib file
add_library(SanContainerDef STATIC SanContainerDef.h)
set_target_properties(SanContainerDef PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(SanContainerDef cSanMemoryManagerDef)

add_library(SanStreamT STATIC SanStreamIterator.h SanStreamT.h)
set_target_properties(SanStreamT PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(SanStreamT cSanMemoryManagerDef SanContainerDef)

add_library(SanStream STATIC SanStream.h SanStream.cpp)
set_target_properties(SanStream PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(SanStream cSanMemoryManagerDef SanContainerDef SanStreamT)

add_library(SanTree STATIC SanTree)
set_target_properties(SanTree PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(SanTree cSanMemoryManagerDef SanContainerDef)