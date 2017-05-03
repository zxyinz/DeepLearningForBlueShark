### ML/NeuralNetwork/Node CMakeLists

### Lib file
add_library(cNeuralNode STATIC cNeuralNode.h cNeuralNode.cpp)
set_target_properties(cNeuralNode PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(cNeuralNode SanMathematics SanContainer MachineLearningDataTypeDef)

add_library(cSigmoidNode STATIC cSigmoidNode.h cSigmoidNode.cpp)
set_target_properties(cSigmoidNode PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(cSigmoidNode cNeuralNode)

add_library(cLogisticNode STATIC cLogisticNode.h cLogisticNode.cpp)
set_target_properties(cLogisticNode PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(cLogisticNode cSigmoidNode)

add_library(cConvNode STATIC cConvNode.h cConvNode.cpp)
set_target_properties(cConvNode PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(cConvNode cNeuralNode)