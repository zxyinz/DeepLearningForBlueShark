### Core/Mathematics CMakeLists

### Lib file
add_library(SanMathematicsVector STATIC SanMathematicsVector.h SanMathematicsVector.cpp)
set_target_properties(SanMathematicsVector PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(SanMathematicsVector SanMemory SanMathematics SanContainer)

add_library(SanMathematicsMatrix STATIC SanMathematicsMatrix.h SanMathematicsMatrix.cpp)
set_target_properties(SanMathematicsMatrix PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(SanMathematicsMatrix SanMemory SanMathematics SanMathematicsVector SanContainer)

add_library(SanMathematicsVectorT STATIC SanMathematicsVectorT.h)
set_target_properties(SanMathematicsVectorT PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(SanMathematicsVectorT SanMemory SanMathematics SanMathematicsVector SanContainer)


add_library(Vector STATIC Vector.h Vector.cpp)
set_target_properties(Vector PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(Vector SanMemory SanStreamT SanContainer)

add_library(Matrix STATIC Matrix.h Matrix.cpp)
set_target_properties(Matrix PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(Matrix SanMemory SanStreamT SanContainer Vector)

add_library(Tensor STATIC Tensor.h Tensor.cpp)
set_target_properties(Tensor PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(Tensor SanMemory SanStreamT SanContainer Vector Matrix)