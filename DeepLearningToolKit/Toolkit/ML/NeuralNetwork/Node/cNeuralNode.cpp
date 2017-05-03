#include"cNeuralNode.h"
using namespace std;
using namespace San;
using namespace San::Mathematics;
using namespace San::MachineLearning;
cNeuralNode::cNeuralNode(const uint32 Size)
	:m_WeightArray(Size - 1, 0.0), m_BiasWeight(0.0), m_ValueArray(Size - 1, 0.0)
{
}
cNeuralNode::cNeuralNode(const cNeuralNode &Node)
	:m_WeightArray(Node.m_WeightArray), m_BiasWeight(Node.m_BiasWeight), m_ValueArray(Node.m_ValueArray)
{
}
cNeuralNode::~cNeuralNode()
{
	this->_ReleaseNode();
}
void cNeuralNode::_CreateNode(const cNeuralNode* pNode)
{
}
void cNeuralNode::_ReleaseNode()
{
}
sfloat cNeuralNode::_GetRandomValue(_rengine &RandomEngine, const sfloat &LowerBound, const sfloat &UpperBound) const
{
	const uint32 MaxTimes = 32;

	uniform_real_distribution<> Dist(LowerBound, UpperBound);

	sfloat Val = Dist(RandomEngine);
	uint32 CurrentTimes = 1;

	while (::gloIsFloatEqual(Val, 0.0) && (CurrentTimes < MaxTimes)) { Val = Dist(RandomEngine); CurrentTimes = CurrentTimes + 1; }

	return Val;
}
const sfloat& cNeuralNode::operator[](const uint32 Index) const
{
	return Index == 0 ? this->m_BiasWeight : this->m_WeightArray[Index - 1];
}
sfloat& cNeuralNode::operator[](const uint32 Index)
{
	return Index == 0 ? this->m_BiasWeight : this->m_WeightArray[Index - 1];
}
sfloat cNeuralNode::iPredict(const _tensor_ref<sfloat> &Input, _tensor_ref<sfloat>* pOutput)
{
	return 0.0;
}
sfloat cNeuralNode::iPredict(const _tensor_ref<sfloat> &Input, _tensor_ref<sfloat>* pOutput) const
{
	return 0.0;
}
sfloat cNeuralNode::iFeedback(const sfloat &Delta, const sfloat LearningRate, const sfloat Momentum)
{
	return 0.0;
}
sfloat cNeuralNode::iCalcFuncValue(const sfloat &Val) const
{
	return Val;
}
sfloat cNeuralNode::iCalcDeltaValue(const _tensor_ref<sfloat> &Error) const
{
	return Error.iSum();
}
uint32 cNeuralNode::iResize(const uint32 Size, _rengine &RandomEngine, const sfloat &LowerBound, const sfloat &UpperBound)
{
	this->m_WeightArray.iReSizeStream(Size - 1);
	this->iReset(RandomEngine, LowerBound, UpperBound);

	this->m_ValueArray.iReSizeStream(Size - 1);
	this->m_ValueArray.iSetStream(0.0);

	return this->m_WeightArray.size();
}
uint32 cNeuralNode::iReset(_rengine &RandomEngine, const sfloat &LowerBound, const sfloat &UpperBound)
{
	for (auto &Val : this->m_WeightArray){ Val = this->_GetRandomValue(RandomEngine, LowerBound, UpperBound); }
	this->m_BiasWeight = this->_GetRandomValue(RandomEngine, LowerBound, UpperBound);

	return this->size();
}
cNeuralNode* cNeuralNode::iCopy() const
{
	cNeuralNode* pNode = new cNeuralNode(*this);
	return pNode;
}
//cNeuralNode::const_iterator cNeuralNode::cbegin() const
//{
//	return this->m_WeightArray.cbegin();
//}
//cNeuralNode::const_iterator cNeuralNode::cend() const
//{
//	return this->m_WeightArray.cend();
//}
//cNeuralNode::const_iterator cNeuralNode::begin() const
//{
//	return this->m_WeightArray.begin();
//}
//cNeuralNode::iterator cNeuralNode::begin()
//{
//	return this->m_WeightArray.begin();
//}
//cNeuralNode::const_iterator cNeuralNode::end() const
//{
//	return this->m_WeightArray.end();
//}
//cNeuralNode::iterator cNeuralNode::end()
//{
//	return this->m_WeightArray.end();
//}
uint32 cNeuralNode::size() const
{
	return this->m_WeightArray.size() + 1;
}
bool cNeuralNode::empty() const
{
	return false;
}