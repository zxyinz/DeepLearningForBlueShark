#include"cConvNode.h"
using namespace std;
using namespace San;
using namespace San::Mathematics;
using namespace San::MachineLearning;
cConvNode::cConvNode(const cConvNode::_dimension Dimension)
	:m_Dimension(Dimension), cNeuralNode(Dimension.x * Dimension.y * Dimension.z)
{
	this->_CreateNode(nullptr);
}
cConvNode::cConvNode(const cConvNode &Node)
	:m_Dimension(Node.m_Dimension), cNeuralNode(Node.size())
{
	this->_CreateNode(&Node);
}
cConvNode::~cConvNode()
{
	this->_ReleaseNode();
}
void cConvNode::_CreateNode(const cNeuralNode* pNode)
{
	m_WeightTensorRef.iSetTensor(this->m_Dimension, this->m_WeightArray);
	m_ValueTensorRef.iSetTensor(this->m_Dimension, this->m_ValueArray);

	if (pNode == nullptr)
	{
		cConvNode::_dimension OutputDim = this->m_Dimension - (this->m_Dimension * 0.5);

		this->m_Delta = 0.0;
		this->m_PrevDelta = 0.0;

		this->m_LinearOutput.iSetDimension(OutputDim);
		this->m_FunctionOutput.iSetDimension(OutputDim);
		this->m_PredictError.iSetDimension(OutputDim);
	}
	else
	{
		cConvNode* pCurrentNode = (cConvNode*) pNode;

		this->m_Delta = pCurrentNode->m_Delta;
		this->m_PrevDelta = pCurrentNode->m_PrevDelta;

		this->m_LinearOutput = pCurrentNode->m_LinearOutput;
		this->m_FunctionOutput = pCurrentNode->m_FunctionOutput;
		this->m_PredictError = pCurrentNode->m_PredictError;
	}
}
void cConvNode::_ReleaseNode()
{
}
sfloat cConvNode::iPredict(const _tensor_ref<sfloat> &Input, _tensor_ref<sfloat>* pOutput)
{
	cConvNode::_dimension OutputDim = this->m_Dimension - (this->m_Dimension * 0.5);

	this->m_LinearOutput.iSet(0.0);
	this->m_FunctionOutput.iSet(0.0);
	this->m_PredictError.iSet(0.0);

	const auto &Weight = this->m_WeightTensorRef;
	const auto &Bias = this->m_BiasWeight;
	auto &ValSum = this->m_ValueTensorRef;
	auto &CurrentRes = this->m_PredictError;

	auto ConvFunc = [&Weight, &Bias, &CurrentRes, &ValSum](const _tensor_ref<sfloat> &DataBlock, const cConvNode::_dimension &Pos)
	{
		CurrentRes.iSet(DataBlock);
		CurrentRes.iMul(Weight);

		ValSum.iAdd(CurrentRes);

		return CurrentRes.iSum() + Bias;
	};
	Input.iConvolution(this->m_LinearOutput, this->m_Dimension, ConvFunc);

	this->m_FunctionOutput.iSet(this->m_LinearOutput);
	
	auto &Node = *this;
	auto CalcFuncVal = [&Node](const sfloat &Val, const _tensor_ref<sfloat>::_dimension Pos)
	{
		return Node.iCalcFuncValue(Val);
	};
	this->m_FunctionOutput.iOperation(CalcFuncVal);

	if ((pOutput != nullptr) && (pOutput->iGetDimension() == OutputDim))
	{
		pOutput->iSet(this->m_FunctionOutput);
	}

	return 1.0;
}
sfloat cConvNode::iPredict(const _tensor_ref<sfloat> &Input, _tensor_ref<sfloat>* pOutput) const
{
	cConvNode::_dimension OutputDim = this->m_Dimension - (this->m_Dimension * 0.5);

	if ((pOutput != nullptr) && (pOutput->iGetDimension() == OutputDim))
	{
		return 0.0;
	}

	Input.iConv(*pOutput, this->m_Dimension, this->m_WeightTensorRef);

	const auto &Bias = this->m_BiasWeight;
	auto &Node = *this;

	auto CalcFuncVal = [&Bias, &Node](const sfloat &Val, const _tensor_ref<sfloat>::_dimension Pos)
	{
		return Node.iCalcFuncValue(Val + Bias);
	};
	pOutput->iOperation(CalcFuncVal);

	return 1.0;
}
sfloat cConvNode::iFeedback(const sfloat &Delta, const sfloat LearningRate, const sfloat Momentum)
{
	sfloat Changes = 0.0;

	this->m_PrevDelta = this->m_Delta;
	this->m_Delta = Delta;

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
cNeuralNode* cConvNode::iCopy() const
{
	cConvNode* pNode = new cConvNode(*this);
	return pNode;
}