#include"cLogisticNode.h"
using namespace std;
using namespace San;
using namespace San::Mathematics;
using namespace San::MachineLearning;
sfloat cLogisticFunc::iForward(const sfloat &Val)
{
	return 1.0 / (1.0 + exp(-Val));
}
sfloat cLogisticFunc::iBackward(const sfloat &PredictVal, const sfloat &Error)
{
	return PredictVal * (1.0 - PredictVal) * Error;
}
cLogisticNode::cLogisticNode(const uint32 Size)
	:cSigmoidNode(Size)
{
}
cLogisticNode::cLogisticNode(const cSigmoidNode &Node)
	:cSigmoidNode(Node.size())
{
}
cLogisticNode::~cLogisticNode()
{
	this->_ReleaseNode();
}
sfloat cLogisticNode::iCalcFuncValue(const sfloat &Net) const
{
	return this->iForward(Net);
}
sfloat cLogisticNode::iCalcDeltaValue(const _tensor_ref<sfloat> &Error) const
{
	return this->iBackward(this->m_FunctionOutput, Error.iSum());
}
cNeuralNode* cLogisticNode::iCopy() const
{
	cLogisticNode* pNode = new cLogisticNode(*this);
	return pNode;
}