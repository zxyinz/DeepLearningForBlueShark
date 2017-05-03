#include"cSigmoidNode.h"
using namespace std;
using namespace San;
using namespace San::Mathematics;
using namespace San::MachineLearning;
cSigmoidNode::cSigmoidNode(const uint32 Size)
	:cNeuralNode(Size)
{
	this->_CreateNode(nullptr);
}
cSigmoidNode::cSigmoidNode(const cSigmoidNode &Node)
	:cNeuralNode(Node.size())
{
	this->_CreateNode(&Node);
}
cSigmoidNode::~cSigmoidNode()
{
	this->_ReleaseNode();
}
void cSigmoidNode::_CreateNode(const cNeuralNode* pNode)
{
	if (pNode == nullptr)
	{
		this->m_Delta = 0.0;
		this->m_PrevDelta = 0.0;
		this->m_LinearOutput = 0.0;
		this->m_FunctionOutput = 0.0;
		this->m_PredictError = 0.0;
	}
	else
	{
		cSigmoidNode* pCurrentNode = (cSigmoidNode*) pNode;

		this->m_Delta = pCurrentNode->m_Delta;
		this->m_PrevDelta = pCurrentNode->m_PrevDelta;
		this->m_LinearOutput = pCurrentNode->m_LinearOutput;
		this->m_FunctionOutput = pCurrentNode->m_FunctionOutput;
		this->m_PredictError = pCurrentNode->m_PredictError;
	}
}
void cSigmoidNode::_ReleaseNode()
{
}
sfloat cSigmoidNode::iPredict(const _tensor_ref<sfloat> &Input, _tensor_ref<sfloat>* pOutput)
{
	//Size mismatch
	if (Input.iGetSize() != this->m_WeightArray.size()){ return 0.0; }

	const _vector_ref<sfloat> InputVec(this->m_WeightArray);
	_vector_ref<sfloat> ValVec(this->m_ValueArray);

	ValVec.iSet(_vector_ref<sfloat>(this->size() - 1, this->m_WeightArray));
	ValVec.iMul(InputVec);

	this->m_LinearOutput = 1.0 * this->m_BiasWeight + ValVec.iSum();
	this->m_FunctionOutput = this->iCalcFuncValue(this->m_LinearOutput);

	if ((pOutput != nullptr) && (pOutput->iGetSize() > 0))
	{
		(*pOutput)[0][0][0] = this->m_FunctionOutput;
	}
	return this->m_FunctionOutput;
}
sfloat cSigmoidNode::iPredict(const _tensor_ref<sfloat> &Input, _tensor_ref<sfloat>* pOutput) const
{
	//Size mismatch
	if (Input.iGetSize() != this->m_WeightArray.size()){ return 0.0; }

	const _vector_ref<sfloat> InputVec(this->m_WeightArray);
	const _vector_ref<sfloat> WeightVec(this->m_ValueArray);

	sfloat LinearOutput = 1.0 * this->m_BiasWeight;

	const _vector_ref<sfloat> &Vec = Input[0][0];

	auto CalcVal = [&LinearOutput, &Vec](const sfloat &Weight, const _vector_ref<sfloat>::_dimension Pos)
	{
		LinearOutput = LinearOutput + Weight * Vec[Pos];
	};
	WeightVec.iOperationC(CalcVal);

	sfloat FunctionOutput = this->iCalcFuncValue(LinearOutput);

	if ((pOutput != nullptr) && (pOutput->iGetSize() > 0))
	{
		(*pOutput)[0][0][0] = this->m_FunctionOutput;
	}

	return FunctionOutput;
}
sfloat cSigmoidNode::iFeedback(const sfloat &Delta, const sfloat LearningRate, const sfloat Momentum)
{
	sfloat Changes = 0.0;

	this->m_PrevDelta = this->m_Delta;
	this->m_Delta = Delta;

	// Update each weight
	// sfloat WDelta = LearningRate * this->m_Delta * W.x + Momentum * this->m_PrevDelta;
	// W.w = W.w + WDelta;
	// Changes = Changes + abs(WDelta);

	_vector_ref<sfloat> WeightSet(this->m_WeightArray);
	_vector_ref<sfloat> ValueSet(this->m_ValueArray);

	const sfloat UpdateRate = LearningRate * this->m_Delta;
	const sfloat MomentumRate = Momentum * this->m_PrevDelta;

	ValueSet.iMul(UpdateRate);
	ValueSet.iAdd(MomentumRate);

	WeightSet.iAdd(ValueSet);
	this->m_BiasWeight = this->m_BiasWeight + UpdateRate /** 1.0*/ + MomentumRate;

	auto SumChanges = [&Changes](sfloat Value, _vector_ref<sfloat>::_dimension &Dimension)
	{
		Changes = Changes + abs(Value);
	};

	ValueSet.iOperationC(SumChanges);

	Changes = Changes + UpdateRate + MomentumRate; //Bias changes

	return Changes;
}
cNeuralNode* cSigmoidNode::iCopy() const
{
	cSigmoidNode* pNode = new cSigmoidNode(*this);
	return pNode;
}