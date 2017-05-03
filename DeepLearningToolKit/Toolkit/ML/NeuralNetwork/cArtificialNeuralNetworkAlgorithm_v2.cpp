#include"cArtificialNeuralNetworkAlgorithm_v2.h"
#include"time.h"
using namespace std;
using namespace San;
using namespace San::Mathematics;
using namespace San::MachineLearning;
stNEURALNODE::stNEURALNODE()
	:LinearOutput(0.0), FunctionOutput(0.0), PredictError(0.0), Delta(0.0), PrevDelta(0.0)
{
	this->_CreateNode();
}
stNEURALNODE::stNEURALNODE(const stNEURALNODE &Node)
	:LinearOutput(Node.LinearOutput), FunctionOutput(0.0), PredictError(Node.PredictError), Delta(Node.Delta), PrevDelta(Node.PrevDelta)
{
	this->_CreateNode();
}
stNEURALNODE::~stNEURALNODE()
{
	this->_ReleaseNode();
}
void stNEURALNODE::_CreateNode()
{
}
void stNEURALNODE::_ReleaseNode()
{
}
sfloat stNEURALNODE::_CalcFuncValue(const sfloat &Val) const
{
	return Val;
}
const ANNWEIGHTPAIR& stNEURALNODE::operator[](const uint32 Index) const
{
	return this->WeightArray[Index];
}
ANNWEIGHTPAIR& stNEURALNODE::operator[](const uint32 Index)
{
	return this->WeightArray[Index];
}
sfloat stNEURALNODE::iPredict(const vector<sfloat> &InputData)
{
	const uint32 Size = this->WeightArray.size();

	this->LinearOutput = this->WeightArray[0].w;
	this->WeightArray[0].x = 1.0;

	for (int32 seek = 1; seek < Size; seek = seek + 1)
	{
		ANNWEIGHTPAIR &W = this->WeightArray[seek];

		W.x = InputData[seek - 1];
		this->LinearOutput = this->LinearOutput + W.w * W.x;
	}

	this->FunctionOutput = this->_CalcFuncValue(this->LinearOutput);
	return this->FunctionOutput;
}
sfloat stNEURALNODE::iPredict(const vector<sfloat> &InputData) const
{
	const uint32 Size = this->WeightArray.size();

	sfloat Val = this->WeightArray[0].w;
	for (int32 seek = 1; seek < Size; seek = seek + 1)
	{
		Val = Val + this->WeightArray[seek].w * InputData[seek - 1];
	}

	return this->_CalcFuncValue(Val);
}
sfloat stNEURALNODE::iFeedback(const sfloat &Delta, const sfloat LearningRate, const sfloat Momentum)
{
	sfloat Changes = 0.0;

	this->PrevDelta = this->Delta;
	this->Delta = Delta;

	const sfloat UpdateRate = LearningRate * this->Delta;
	const sfloat MomentumRate = Momentum * this->PrevDelta;

	for (auto & W : this->WeightArray)
	{
		sfloat WDelta = UpdateRate * W.x + MomentumRate;
		W.w = W.w + WDelta;
		Changes = Changes + abs(WDelta);
	}

	return Changes;
}
uint32 stNEURALNODE::iResize(const uint32 WeightArraySize,const uint32 RandomSeed)
{
	this->WeightArray.resize(WeightArraySize);
	this->iReset(RandomSeed);

	return this->WeightArray.size();
}
uint32 stNEURALNODE::iReset(const uint32 RandomSeed)
{
	this->LinearOutput = 0.0;
	this->FunctionOutput = 0.0;
	this->Delta = 0.0;
	this->PrevDelta = 0.0;

	return 0;
}
stNEURALNODE* stNEURALNODE::iCopy() const
{
	stNEURALNODE* pNode = new stNEURALNODE();

	pNode->WeightArray = this->WeightArray;
	pNode->Delta = this->Delta;
	pNode->PrevDelta = this->PrevDelta;

	return pNode;
}
stNEURALNODE::const_iterator stNEURALNODE::cbegin() const
{
	return this->WeightArray.cbegin();
}
stNEURALNODE::const_iterator stNEURALNODE::cend() const
{
	return this->WeightArray.cend();
}
stNEURALNODE::const_iterator stNEURALNODE::begin() const
{
	return this->WeightArray.begin();
}
stNEURALNODE::iterator stNEURALNODE::begin()
{
	return this->WeightArray.begin();
}
stNEURALNODE::const_iterator stNEURALNODE::end() const
{
	return this->WeightArray.end();
}
stNEURALNODE::iterator stNEURALNODE::end()
{
	return this->WeightArray.end();
}
uint32 stNEURALNODE::size() const
{
	return this->WeightArray.size();
}
bool stNEURALNODE::empty() const
{
	return this->WeightArray.empty();
}
stNEURALLAYER::stNEURALLAYER(const stNEURALLAYER &Layer)
	: NodePtrArray(Layer.NodePtrArray)
{
	this->pIdentityNode = Layer.pIdentityNode->iCopy();
	for (auto &ptr : this->NodePtrArray){ ptr = this->pIdentityNode->iCopy(); }
}
stNEURALLAYER::~stNEURALLAYER()
{
	for (auto &ptr : this->NodePtrArray)
	{
		delete ptr;
		ptr = nullptr;
	}
	delete this->pIdentityNode;
	this->pIdentityNode = nullptr;
}
stNEURALLAYER& stNEURALLAYER::operator=(const stNEURALLAYER &Layer)
{
	NEURALNODE* pNode = Layer.pIdentityNode->iCopy();
	delete this->pIdentityNode;

	this->pIdentityNode = pNode;

	uint32 Size = Layer.size();

	vector<NEURALNODE*> PtrArray(Size, nullptr);

	for (uint32 seek = 0; seek < Size; seek = seek + 1)
	{
		PtrArray[seek] = Layer[seek]->iCopy();
	}

	Size = this->NodePtrArray.size();
	for (uint32 seek = 0; seek < Size; seek = seek + 1)
	{
		delete this->NodePtrArray[seek];
		this->NodePtrArray[seek] = nullptr;
	}

	this->NodePtrArray = PtrArray;

	return *this;
}
const NEURALNODE* stNEURALLAYER::operator[](const uint32 Index) const
{
	return this->NodePtrArray[Index];
}
NEURALNODE* stNEURALLAYER::operator[](const uint32 Index)
{
	return this->NodePtrArray[Index];
}
uint32 stNEURALLAYER::iResize(const uint32 NodeNumber)
{
	const uint32 CurrentSize = this->NodePtrArray.size();
	if (CurrentSize < NodeNumber)
	{
		this->NodePtrArray.resize(NodeNumber);
		for (int32 seek = CurrentSize; seek < NodeNumber; seek = seek + 1)
		{
			this->NodePtrArray[seek] = this->pIdentityNode->iCopy();
		}
	}
	else
	{
		for (int32 seek = NodeNumber; seek < CurrentSize; seek = seek + 1)
		{
			delete this->NodePtrArray[seek];
			this->NodePtrArray[seek] = nullptr;
		}
		this->NodePtrArray.resize(NodeNumber);
	}

	return this->NodePtrArray.size();
}
stNEURALLAYER::const_iterator stNEURALLAYER::cbegin() const
{
	return this->NodePtrArray.cbegin();
}
stNEURALLAYER::const_iterator stNEURALLAYER::cend() const
{
	return this->NodePtrArray.cend();
}
stNEURALLAYER::const_iterator stNEURALLAYER::begin() const
{
	return this->NodePtrArray.begin();
}
stNEURALLAYER::iterator stNEURALLAYER::begin()
{
	return this->NodePtrArray.begin();
}
stNEURALLAYER::const_iterator stNEURALLAYER::end() const
{
	return this->NodePtrArray.end();
}
stNEURALLAYER::iterator stNEURALLAYER::end()
{
	return this->NodePtrArray.end();
}
uint32 stNEURALLAYER::size() const
{
	return this->NodePtrArray.size();
}
bool stNEURALLAYER::empty() const
{
	return this->NodePtrArray.empty();
}
cArtificialNeuralNetworkAlgorithm_v2::cArtificialNeuralNetworkAlgorithm_v2(const sfloat LearningRate, const sfloat MomentumValue, const sfloat RandomMinBoundary, const sfloat RandomMaxBoundary)
	:m_InputLayerSize(1), m_LearningRate(LearningRate), m_MomentumValue(MomentumValue)
{
	m_RandomBoundary[0] = RandomMinBoundary;
	m_RandomBoundary[1] = RandomMaxBoundary;
	this->m_CurrentNetwork.clear();
	this->m_OutputArray.resize(1);
	this->m_OutputArray.front().resize(this->m_InputLayerSize, 0.0);
}
cArtificialNeuralNetworkAlgorithm_v2::~cArtificialNeuralNetworkAlgorithm_v2()
{
	this->m_CurrentNetwork.clear();
	this->m_OutputArray.clear();
	this->m_LearningRate = 0.0;
	this->m_MomentumValue = 0.0;
	this->m_RandomBoundary[0] = 0.0;
	this->m_RandomBoundary[1] = 0.0;
}
void cArtificialNeuralNetworkAlgorithm_v2::_ResetNeuralNetwork(const uint32 RandomSeed)
{
	const uint32 HiddenLayerSize = this->m_CurrentNetwork.size();
	this->m_OutputArray.resize(HiddenLayerSize + 1);
	this->m_OutputArray.front().resize(this->m_InputLayerSize, 0.0);

	for (uint32 seek = 0; seek < HiddenLayerSize; seek = seek + 1)
	{
		const uint32 LayerSize = this->m_CurrentNetwork[seek].size();
		this->m_OutputArray[seek + 1].resize(LayerSize, 0.0);

		for (auto &pNode : this->m_CurrentNetwork[seek])
		{
			pNode->iResize(this->m_OutputArray[seek].size() + 1, RandomSeed);

			for (auto &Weight : *pNode)
			{
				Weight.w = this->_GenerateRandomValue();
				Weight.x = 0.0;
			}
		}
	}
}
sfloat cArtificialNeuralNetworkAlgorithm_v2::_GenerateRandomValue() const
{
	const sfloat Base = 1000.0;
	sfloat RandValue = 0.0;
	while (true)
	{
		RandValue = ::rand() % (uint32) 1000;
		RandValue = RandValue / Base;
		RandValue = this->m_RandomBoundary[0] + (this->m_RandomBoundary[1] - this->m_RandomBoundary[0])*RandValue;
		if (!::gloIsFloatEqual(RandValue, 0.0, 0.0001))
		{
			break;
		}
	}
	return RandValue;
}
cArtificialNeuralNetworkAlgorithm_v2::_layer cArtificialNeuralNetworkAlgorithm_v2::_GetHiddenLayer(const uint32 LayerID) const
{
	if (LayerID == 0){ return this->m_CurrentNetwork.back(); }
	return this->m_CurrentNetwork[LayerID];
}
bool cArtificialNeuralNetworkAlgorithm_v2::iResizeLayer(const uint32 LayerID, const uint32 NodeNumber)
{
	if (NodeNumber == 0) { return false; }
	if (this->m_CurrentNetwork.size() <= LayerID) { return false; }

	this->m_CurrentNetwork[LayerID].iResize(NodeNumber);
	this->_ResetNeuralNetwork();

	return true;
}
bool cArtificialNeuralNetworkAlgorithm_v2::iInitializeLayer(const uint32 LayerID, const vector<vector<sfloat>> InitValue)
{
	if (LayerID == 0) { return false; }
	if (this->m_CurrentNetwork.size() < LayerID) { return false; }
	if (this->m_CurrentNetwork[LayerID - 1].size() != InitValue.size()) { return false; }

	const uint32 NodeSize = this->m_CurrentNetwork[LayerID - 1].size();

	if (NodeSize == 0){ return false; }

	const uint32 WeightVectorSize = this->m_CurrentNetwork[LayerID - 1][0]->size();

	for (uint32 seek = 0; seek < NodeSize; seek = seek + 1)
	{
		if (InitValue[seek].size() != WeightVectorSize)
		{
			return false;
		}
	}

	for (uint32 seek = 0; seek < NodeSize; seek = seek + 1)
	{
		for (uint32 seek_w = 0; seek_w < WeightVectorSize; seek_w = seek_w + 1)
		{
			(*this->m_CurrentNetwork[LayerID - 1][seek])[seek_w].w = InitValue[seek][seek_w];
			(*this->m_CurrentNetwork[LayerID - 1][seek])[seek_w].x = 0.0;
		}
	}
	return true;
}
void cArtificialNeuralNetworkAlgorithm_v2::iReleaseLayer(const uint32 LayerID)
{
	if (LayerID == 0) { return; }
	if (this->m_CurrentNetwork.size() <= LayerID) { return; }

	vector<_layer>::iterator pLayer = this->m_CurrentNetwork.begin();
	uint32 Count = 0;
	while (Count < LayerID) { pLayer++; }

	this->m_CurrentNetwork.erase(pLayer);
}
bool cArtificialNeuralNetworkAlgorithm_v2::iUpdateLearningRate(const sfloat Rate)
{
	if (::gloIsFloatEqual(Rate, 0.0))
	{
		return false;
	}
	this->m_LearningRate = Rate;
}
bool cArtificialNeuralNetworkAlgorithm_v2::iUpdateMomentumValue(const sfloat Momentum)
{
	if (::gloIsFloatEqual(Momentum, 0.0))
	{
		return false;
	}
	this->m_MomentumValue = Momentum;
}
bool cArtificialNeuralNetworkAlgorithm_v2::iUpdateRandomBoundary(const sfloat RandomMinBoundary, const sfloat RandomMaxBoundary)
{
	if (RandomMinBoundary >= RandomMaxBoundary)
	{
		return false;
	}
	if (::gloIsFloatEqual(RandomMinBoundary, 0.0) && ::gloIsFloatEqual(RandomMaxBoundary, 0.0))
	{
		return false;
	}
	this->m_RandomBoundary[0] = RandomMinBoundary;
	this->m_RandomBoundary[1] = RandomMaxBoundary;
	return true;
}
bool cArtificialNeuralNetworkAlgorithm_v2::iUpdateInputLayerSize(const uint32 FeatureSize)
{
	this->m_InputLayerSize = FeatureSize;
	this->_ResetNeuralNetwork();

	return this->m_InputLayerSize;
}
sfloat cArtificialNeuralNetworkAlgorithm_v2::iGetLearningRate() const
{
	return this->m_LearningRate;
}
sfloat cArtificialNeuralNetworkAlgorithm_v2::iGetMomentumValue() const
{
	return this->m_MomentumValue;
}
vector<sfloat> cArtificialNeuralNetworkAlgorithm_v2::iGetRandomBoundary() const
{
	vector<sfloat> Bound(2);
	Bound[0] = this->m_RandomBoundary[0];
	Bound[1] = this->m_RandomBoundary[1];
	return Bound;
}
uint32 cArtificialNeuralNetworkAlgorithm_v2::iUpdateInputLayerSize() const
{
	return this->m_InputLayerSize;
}
void cArtificialNeuralNetworkAlgorithm_v2::iTraining(const ANNTRAININGSET &TrainingSet, const uint32 TrainingTimes, const sfloat MinWeightChanges, const sfloat MinErrorChanges, const bool bGenerateHistory)
{
	auto DataGenerator = [](const ANNTRAININGSET::_data_type &Data){ return Data; };
	this->iTraining(TrainingSet, DataGenerator, TrainingTimes, MinWeightChanges, MinErrorChanges, bGenerateHistory);
}
void cArtificialNeuralNetworkAlgorithm_v2::iTrainingWithValidation(const ANNTRAININGSET &TrainingSet, const ANNTRAININGSET &ValidationSet, const uint32 TrainingTimes, const sfloat MinWeightChanges, const sfloat MinErrorChanges)
{
	auto DataGenerator = [](const ANNTRAININGSET::_data_type &Data){ return Data; };
	this->iTrainingWithValidation(TrainingSet, ValidationSet, DataGenerator, TrainingTimes, MinWeightChanges, MinErrorChanges);
}
void cArtificialNeuralNetworkAlgorithm_v2::iPredict(const vector<sfloat> &InputArray, const int32 Index)
{
	if (this->m_CurrentNetwork.empty()){ return; }
	if (InputArray.size() != this->m_InputLayerSize){ return; }

//	_network &CurrentNetwork = Index == -1 ? this->m_CurrentNetwork : (Index > this->m_HistoryList.size() ? this->m_CurrentNetwork : this->m_HistoryList[Index]);
//
//#pragma omp parallel for //private(InputArray)
//	for (int32 seek = 0; seek < this->m_InputLayerSize; seek = seek + 1)
//	{
//		this->m_OutputArray[0][seek] = InputArray[seek];
//	}
//
//	const uint32 LayerSize = CurrentNetwork.size();
//
//	for (uint32 seek = 0; seek < LayerSize; seek = seek + 1)
//	{
//		_layer* pLayer = &CurrentNetwork[seek];
//		int32 NodeSize = pLayer->size();
//
//#pragma omp parallel for firstprivate(pLayer)
//		for (int32 seek_node = 0; seek_node < NodeSize; seek_node = seek_node + 1)
//		{
//			sfloat Output = (*pLayer)[seek_node]->iPredict(this->m_OutputArray[seek]);
//			this->m_OutputArray[seek + 1][seek_node] = Output;
//		}
//	}

	_network* pCurrentNetwork = Index == -1 ? &this->m_CurrentNetwork : (Index > this->m_HistoryList.size() ? &this->m_CurrentNetwork : &this->m_HistoryList[Index]);
	vector<vector<sfloat>>* pOutputArray = &this->m_OutputArray;

	const uint32 LayerSize = pCurrentNetwork->size();

#pragma omp parallel firstprivate(pCurrentNetwork, pOutputArray, LayerSize)
	{
#pragma omp for //private(InputArray)
		for (int32 seek = 0; seek < this->m_InputLayerSize; seek = seek + 1)
		{
			(*pOutputArray)[0][seek] = InputArray[seek];
		}

		for (uint32 seek = 0; seek < LayerSize; seek = seek + 1)
		{
			_layer* pLayer = &(*pCurrentNetwork)[seek];
			const int32 NodeSize = pLayer->size();

			vector<sfloat>* pInput = &(*pOutputArray)[seek];
			vector<sfloat>* pOutput = &(*pOutputArray)[seek + 1];

#pragma omp for //firstprivate(pLayer)
			for (int32 seek_node = 0; seek_node < NodeSize; seek_node = seek_node + 1)
			{
				sfloat Output = (*pLayer)[seek_node]->iPredict(*pInput);
				(*pOutput)[seek_node] = Output;
			}
		}
	}
}
vector<sfloat> cArtificialNeuralNetworkAlgorithm_v2::iPredict(const vector<sfloat> &InputArray, const int32 Index) const
{
	if (this->m_CurrentNetwork.empty()){ return vector<sfloat>(); }
	if (InputArray.size() != this->m_InputLayerSize){ return vector<sfloat>(); }

//	const _network &CurrentNetwork = Index == -1 ? this->m_CurrentNetwork : (Index > this->m_HistoryList.size() ? this->m_CurrentNetwork : this->m_HistoryList[Index]);
//
//#pragma omp parallel for //private(InputArray)
//	for (int32 seek = 0; seek < this->m_InputLayerSize; seek = seek + 1)
//	{
//		this->m_OutputArray[0][seek] = InputArray[seek];
//	}
//
//	const uint32 LayerSize = CurrentNetwork.size();
//
//	for (uint32 seek = 0; seek < LayerSize; seek = seek + 1)
//	{
//		const _layer* pLayer = &CurrentNetwork[seek];
//		int32 NodeSize = pLayer->size();
//
//#pragma omp parallel for firstprivate(pLayer)
//		for (int32 seek_node = 0; seek_node < NodeSize; seek_node = seek_node + 1)
//		{
//			sfloat Output = (*pLayer)[seek_node]->iPredict(this->m_OutputArray[seek]);
//			this->m_OutputArray[seek + 1][seek_node] = Output;
//		}
//	}
//
//	return this->m_OutputArray.back();

	const _network* pCurrentNetwork = Index == -1 ? &this->m_CurrentNetwork : (Index > this->m_HistoryList.size() ? &this->m_CurrentNetwork : &this->m_HistoryList[Index]);

	const uint32 LayerSize = pCurrentNetwork->size();

#pragma omp parallel firstprivate(pCurrentNetwork, LayerSize)
	{
#pragma omp for nowait //private(InputArray)
		for (int32 seek = 0; seek < this->m_InputLayerSize; seek = seek + 1)
		{
			this->m_OutputArray[0][seek] = InputArray[seek];
		}

		for (uint32 seek = 0; seek < LayerSize; seek = seek + 1)
		{
			const _layer* pLayer = &(*pCurrentNetwork)[seek];
			int32 NodeSize = pLayer->size();

#pragma omp for nowait//firstprivate(pLayer)
			for (int32 seek_node = 0; seek_node < NodeSize; seek_node = seek_node + 1)
			{
				sfloat Output = (*pLayer)[seek_node]->iPredict(this->m_OutputArray[seek]);
				this->m_OutputArray[seek + 1][seek_node] = Output;
			}
		}
	}

	return this->m_OutputArray.back();
}
pair<sfloat, sfloat> cArtificialNeuralNetworkAlgorithm_v2::iFeedback(const vector<sfloat> &TrainingOutputArray)
{
	const int32 OutputSize = this->m_OutputArray.back().size();
	const uint32 LayerSize = this->m_CurrentNetwork.size();

	if (OutputSize > TrainingOutputArray.size())
	{
		return make_pair(0.0, 0.0);
	}

	sfloat WeightChanges = 0.0;
	sfloat PredictError = 0.0;

//	//Calc Sum of Error, store in m_OutputArray
//	vector<sfloat> &PredictVector = this->m_OutputArray.back();
//
//#pragma omp parallel for
//	for (int32 seek = 0; seek < OutputSize; seek = seek + 1)
//	{
//		sfloat OutputVal = PredictVector[seek];
//		sfloat Error = OutputVal * (1.0 - OutputVal) * (TrainingOutputArray[seek] - OutputVal);
//
//		PredictVector[seek] = Error;
//		PredictError = PredictError + abs(Error);
//	}
//
//	for (int32 seek = LayerSize - 2; seek >= 0; seek = seek - 1)
//	{
//		const _layer* pOLayer = &this->m_CurrentNetwork[seek + 1]; //next layer
//
//		vector<sfloat> &HiddenVector = this->m_OutputArray[seek + 1];
//		const vector<sfloat>* pOutputDeltaArray = &this->m_OutputArray[seek + 2]; //Output Delta
//
//		const int32 HiddenNodeArraySize = HiddenVector.size();
//		const int32 OutputNodeArraySize = pOLayer->size();
//
//#pragma omp parallel for firstprivate(pOLayer, pOutputDeltaArray)
//		for (int32 seek_node = 0; seek_node < HiddenNodeArraySize; seek_node = seek_node + 1)
//		{
//			/*Calculate the delta value*/
//			sfloat OutputVal = HiddenVector[seek_node];
//			sfloat ErrorSum = 0.0;
//
//			for (uint32 seek_k = 0; seek_k < OutputNodeArraySize; seek_k = seek_k + 1)
//			{
//				ErrorSum = ErrorSum + (*(*pOLayer)[seek_k])[seek_node + 1].w * (*pOutputDeltaArray)[seek_k];
//			}
//
//			HiddenVector[seek_node] = OutputVal * (1.0 - OutputVal) * ErrorSum;
//		}
//	}
//
//	for (uint32 seek = 0; seek < LayerSize; seek = seek + 1)
//	{
//		_layer &Layer = this->m_CurrentNetwork[seek];
//		vector<sfloat> &ErrorArray = this->m_OutputArray[seek + 1];
//
//		const int32 NodeSize = Layer.size();
//
//#pragma omp parallel for //private(ErrorArray)
//		for (int32 seek_node = 0; seek_node < NodeSize; seek_node = seek_node + 1)
//		{
//			sfloat ErrorSum = ErrorArray[seek_node];
//			WeightChanges = WeightChanges + Layer[seek_node]->iFeedback(ErrorSum, this->m_LearningRate, this->m_MomentumValue);
//		}
//	}

	//Calc Sum of Error, store in m_OutputArray
	vector<sfloat>* pPredictVector = &this->m_OutputArray.back();

#pragma omp parallel
	{
#pragma omp for reduction(+:PredictError)
		for (int32 seek = 0; seek < OutputSize; seek = seek + 1)
		{
			sfloat &OutputVal = (*pPredictVector)[seek];
			sfloat Error = OutputVal * (1.0 - OutputVal) * (TrainingOutputArray[seek] - OutputVal);

			OutputVal = Error;
			PredictError = PredictError + abs(Error);
		}

		for (int32 seek = LayerSize - 2; seek >= 0; seek = seek - 1)
		{
			const _layer* pOLayer = &this->m_CurrentNetwork[seek + 1]; //next layer

			vector<sfloat>* pHiddenVector = &this->m_OutputArray[seek + 1];
			const vector<sfloat>* pOutputDeltaArray = &this->m_OutputArray[seek + 2]; //Output Delta

			const int32 HiddenNodeArraySize = pHiddenVector->size();
			const int32 OutputNodeArraySize = pOLayer->size();

#pragma omp for
			for (int32 seek_node = 0; seek_node < HiddenNodeArraySize; seek_node = seek_node + 1)
			{
				/*Calculate the delta value*/
				sfloat &OutputVal = (*pHiddenVector)[seek_node];
				sfloat ErrorSum = 0.0;

				for (uint32 seek_k = 0; seek_k < OutputNodeArraySize; seek_k = seek_k + 1)
				{
					ErrorSum = ErrorSum + (*(*pOLayer)[seek_k])[seek_node + 1].w * (*pOutputDeltaArray)[seek_k];
				}

				OutputVal = OutputVal * (1.0 - OutputVal) * ErrorSum;
			}
		}

		for (uint32 seek = 0; seek < LayerSize; seek = seek + 1)
		{
			_layer* pLayer = &this->m_CurrentNetwork[seek];
			vector<sfloat>* pErrorArray = &this->m_OutputArray[seek + 1];

			const int32 NodeSize = pLayer->size();

#pragma omp for nowait reduction(+ : WeightChanges)
			for (int32 seek_node = 0; seek_node < NodeSize; seek_node = seek_node + 1)
			{
				WeightChanges = WeightChanges + (*pLayer)[seek_node]->iFeedback((*pErrorArray)[seek_node], this->m_LearningRate, this->m_MomentumValue);
			}
		}
	}

	return make_pair(WeightChanges, PredictError);
}
void cArtificialNeuralNetworkAlgorithm_v2::iPruning(const ANNTRAININGSET &ValidationSet)
{
	if (this->m_HistoryList.empty()){ return; }

	const uint32 HistoryArraySize = this->m_HistoryList.size();
	const uint32 ValidationSetSize = ValidationSet.size();
	const int32 OutputArraySize = this->m_OutputArray.back().size();

	sfloat MinError = FLT_MAX;
	uint32 PIndex = 0;

	for (uint32 seek_network = 0; seek_network < HistoryArraySize; seek_network = seek_network + 1)
	{
		sfloat Error = 0.0;

		for (uint32 seek = 0; seek < ValidationSetSize; seek = seek + 1)
		{
			this->iPredict(ValidationSet[seek].Data, seek_network);
			for (int32 seek_node = 0; seek_node < OutputArraySize; seek_node = seek_node + 1)
			{
				sfloat SubError = ValidationSet[seek].Label[seek_node] - this->m_OutputArray.back()[seek_node];
				Error = Error + SubError * SubError;
			}
		}

		if (Error <= MinError)
		{
			MinError = Error;
			PIndex = seek_network;
		}
	}

	::cout << "Index:\t\t" << PIndex << "\n";
	this->m_CurrentNetwork = this->m_HistoryList[PIndex];
	this->m_HistoryList.clear();
}
void cArtificialNeuralNetworkAlgorithm_v2::iReset()
{
	this->_ResetNeuralNetwork();
}
void cArtificialNeuralNetworkAlgorithm_v2::iClearHistory()
{
	this->m_HistoryList.clear();
}
vector<sfloat> cArtificialNeuralNetworkAlgorithm_v2::iGetOutputArray(int32 LayerID) const
{
	LayerID = LayerID <= 0 ? (this->m_OutputArray.size() - LayerID - 1): LayerID;
	LayerID = LayerID < 0 ? 0 : LayerID;
	LayerID = LayerID >= this->m_OutputArray.size() ? (this->m_OutputArray.size() - 1) : LayerID;

	return this->m_OutputArray[LayerID];
}
SString cArtificialNeuralNetworkAlgorithm_v2::iPrintNeuralNetwork() const
{
	const SString strLine = _SSTR("--------------------\r\n");

	SString strOutput = _SSTR("INPUT LAYER:\r\n\r\n");

	uint32 LayerSize = this->m_CurrentNetwork.size();

	for (uint32 seek = 0; seek < this->m_OutputArray[0].size(); seek = seek + 1)
	{
		strOutput = strOutput + ::gloFToS(this->m_OutputArray[0][seek], _SSTR("6.3")) + _SSTR("\t");
	}

	strOutput = strOutput + _SSTR("\r\n");
	strOutput = strOutput + _SSTR("\r\n");

	for (uint32 seek = 0; seek < LayerSize; seek = seek + 1)
	{
		if (seek != (LayerSize - 1))
		{
			strOutput = strOutput + _SSTR("HIDDEN LAYER # ") + ::gloIToS(seek) + _SSTR(":\r\n\r\n");
		}
		else
		{
			strOutput = strOutput + _SSTR("OUTPUT LAYER\r\n\r\n");
		}
		for (uint32 seek_node = 0; seek_node < this->m_CurrentNetwork[seek].size(); seek_node = seek_node + 1)
		{
			strOutput = strOutput + ::gloFToS(this->m_CurrentNetwork[seek][seek_node]->FunctionOutput, _SSTR("6.3")) + _SSTR(" = ");
			for (uint32 seek_w = 0; seek_w < this->m_CurrentNetwork[seek][seek_node]->size(); seek_w = seek_w + 1)
			{
				strOutput = strOutput + ::gloFToS((*this->m_CurrentNetwork[seek][seek_node])[seek_w].w, _SSTR("6.3")) + _SSTR(" ");
			}
			strOutput = strOutput + _SSTR("\r\n");
		}
		strOutput = strOutput + _SSTR("\r\n");
	}
	return strOutput;
}