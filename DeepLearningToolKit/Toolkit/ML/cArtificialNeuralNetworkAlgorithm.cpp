#include"time.h"
#include"cArtificialNeuralNetworkAlgorithm.h"
using namespace std;
using namespace San;
using namespace San::Mathematics;
using namespace San::MachineLearning;
cArtificialNeuralNetworkAlgorithm::cArtificialNeuralNetworkAlgorithm(const sfloat LearningRate, const sfloat MomentumValue,const sfloat RandomMinBoundary, const sfloat RandomMaxBoundary)
	:m_LearningRate(LearningRate),
	m_MomentumValue(MomentumValue)
{
	//::srand(time(0));
	m_RandomBoundary[0] = RandomMinBoundary;
	m_RandomBoundary[1] = RandomMaxBoundary;
	this->m_LayerArray.clear();
	this->m_FeatureArray.clear();
	ANNLAYER InputLayer;
	InputLayer.LayerID = 0;
	InputLayer.NodeArray.clear();
	this->m_LayerArray.push_back(InputLayer);
}
cArtificialNeuralNetworkAlgorithm::~cArtificialNeuralNetworkAlgorithm()
{
	this->m_LayerArray.clear();
	this->m_FeatureArray.clear();
	this->m_LearningRate = 0.0;
	this->m_MomentumValue = 0.0;
	this->m_RandomBoundary[0] = 0.0;
	this->m_RandomBoundary[1] = 0.0;
}
void cArtificialNeuralNetworkAlgorithm::_ResetNeuralNetwork()
{
	uint32 FeatureSize = this->m_FeatureArray.size();
	this->m_LayerArray[0].NodeArray.resize(FeatureSize);
	for (uint32 seek = 0; seek < FeatureSize; seek = seek + 1)
	{
		this->m_LayerArray[0][seek].SigmoidOutput = 0.0;
	}
	uint32 LayerSize = this->m_LayerArray.size();
	for (uint32 seek = 1; seek < LayerSize; seek = seek + 1)
	{
		uint32 WeightSize = this->m_LayerArray[seek - 1].NodeArray.size() + 1;
		uint32 NodeSize = this->m_LayerArray[seek].NodeArray.size();
		for (uint32 seek_node = 0; seek_node < NodeSize; seek_node = seek_node + 1)
		{
			this->m_LayerArray[seek][seek_node].NodeID = seek_node;
			this->m_LayerArray[seek][seek_node].PrevDelta = 0.0;
			this->m_LayerArray[seek][seek_node].CurrentDelta = 0.0;
			this->m_LayerArray[seek][seek_node].WeightVector.resize(WeightSize);
			this->m_LayerArray[seek][seek_node].SigmoidOutput = 0.0;
			for (uint32 seek_w = 0; seek_w < WeightSize; seek_w = seek_w + 1)
			{
				this->m_LayerArray[seek][seek_node][seek_w].w = this->_GenerateRandomValue();
				this->m_LayerArray[seek][seek_node][seek_w].x = 0.0;
			}
		}
	}
}
sfloat cArtificialNeuralNetworkAlgorithm::_CalcSigmoidValue(const sfloat Net) const
{ 
	return 1.0 / (1.0 + ::exp(-Net));
}
sfloat cArtificialNeuralNetworkAlgorithm::_GenerateRandomValue() const
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
bool cArtificialNeuralNetworkAlgorithm::iCreateFeatureNode(const SString &strFeatureName, const SANSTREAM &UserParam, const ANNFEATUREUPDATEFUNC FeatureUpdateFunc)
{
	if (strFeatureName.empty())
	{
		return false;
	}
	for (uint32 seek = 0; seek < this->m_FeatureArray.size(); seek = seek + 1)
	{
		if (this->m_FeatureArray[seek].strFeatrueName == strFeatureName)
		{
			return false;
		}
	}
	ANNFEATURE Feature;
	Feature.strFeatrueName = strFeatureName;
	Feature.UserParam = UserParam;
	Feature.UpdateFunc = FeatureUpdateFunc;
	this->m_FeatureArray.push_back(Feature);
	this->_ResetNeuralNetwork();
	return true;
}
bool cArtificialNeuralNetworkAlgorithm::iUpdateFeatureNode(const SString &strFeatureName, const SANSTREAM &UserParam, const ANNFEATUREUPDATEFUNC FeatureUpdateFunc)
{
	if (strFeatureName.empty())
	{
		return false;
	}
	for (uint32 seek = 0; seek < this->m_FeatureArray.size(); seek = seek + 1)
	{
		if (this->m_FeatureArray[seek].strFeatrueName == strFeatureName)
		{
			this->m_FeatureArray[seek].UserParam = UserParam;
			this->m_FeatureArray[seek].UpdateFunc = FeatureUpdateFunc;
			return true;
		}
	}
	return false;
}
void cArtificialNeuralNetworkAlgorithm::iReleaseFeatureNode(const SString &strFeatureName)
{
	vector<ANNFEATURE>::iterator pFeature = this->m_FeatureArray.begin();
	while (pFeature!=this->m_FeatureArray.end())
	{
		if (pFeature->strFeatrueName == strFeatureName)
		{
			this->m_FeatureArray.erase(pFeature);
			this->_ResetNeuralNetwork();
			return;
		}
	}
}
uint32 cArtificialNeuralNetworkAlgorithm::iCreateLayer(const uint32 NodeNumber)
{
	if (NodeNumber == 0)
	{
		return false;
	}
	ANNLAYER Layer;
	Layer.LayerID = this->m_LayerArray.size();
	Layer.NodeArray.resize(NodeNumber);
	this->m_LayerArray.push_back(Layer);
	this->_ResetNeuralNetwork();
	return Layer.LayerID;
}
bool cArtificialNeuralNetworkAlgorithm::iResizeLayer(const uint32 LayerID, const uint32 NodeNumber)
{
	if (LayerID == 0)
	{
		return false;
	}
	if (NodeNumber == 0)
	{
		return false;
	}
	if (this->m_LayerArray.size() <= LayerID)
	{
		return false;
	}
	this->m_LayerArray[LayerID].NodeArray.resize(NodeNumber);
	this->_ResetNeuralNetwork();
	return true;
}
bool cArtificialNeuralNetworkAlgorithm::iInitializeLayer(const uint32 LayerID, const vector<vector<sfloat>> InitValue)
{
	if (LayerID == 0)
	{
		return false;
	}
	if (this->m_LayerArray.size() <= LayerID)
	{
		return false;
	}
	if (this->m_LayerArray[LayerID].NodeArray.size() != InitValue.size())
	{
		return false;
	}
	uint32 NodeSize = this->m_LayerArray[LayerID].NodeArray.size();
	uint32 WeightVectorSize = this->m_LayerArray[LayerID][0].WeightVector.size();
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
			this->m_LayerArray[LayerID][seek][seek_w].w = InitValue[seek][seek_w];
			this->m_LayerArray[LayerID][seek][seek_w].x = 0.0;
		}
	}
	return true;
}
void cArtificialNeuralNetworkAlgorithm::iReleaseLayer(const uint32 LayerID)
{
	if (LayerID == 0)
	{
		return;
	}
	if (this->m_LayerArray.size() <= LayerID)
	{
		return;
	}
	vector<ANNLAYER>::iterator pLayer = this->m_LayerArray.begin();
	uint32 Count = 1;
	while (pLayer != this->m_LayerArray.end())
	{
		if (pLayer->LayerID == LayerID)
		{
			pLayer = this->m_LayerArray.erase(pLayer);
		}
		else
		{
			pLayer->LayerID = Count;
			Count = Count + 1;
			pLayer++;
		}
	}
}
bool cArtificialNeuralNetworkAlgorithm::iUpdateLearningRate(const sfloat Rate)
{
	if (::gloIsFloatEqual(Rate, 0.0))
	{
		return false;
	}
	this->m_LearningRate = Rate;
}
bool cArtificialNeuralNetworkAlgorithm::iUpdateMomentumValue(const sfloat Momentum)
{
	if (::gloIsFloatEqual(Momentum, 0.0))
	{
		return false;
	}
	this->m_MomentumValue = Momentum;
}
bool cArtificialNeuralNetworkAlgorithm::iUpdateRandomBoundary(const sfloat RandomMinBoundary, const sfloat RandomMaxBoundary)
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
sfloat cArtificialNeuralNetworkAlgorithm::iGetLearningRate() const
{
	return this->m_LearningRate;
}
sfloat cArtificialNeuralNetworkAlgorithm::iGetMomentumValue() const
{
	return this->m_MomentumValue;
}
vector<sfloat> cArtificialNeuralNetworkAlgorithm::iGetRandomBoundary() const
{
	vector<sfloat> Bound(2);
	Bound[0] = this->m_RandomBoundary[0];
	Bound[1] = this->m_RandomBoundary[1];
	return Bound;
}
void cArtificialNeuralNetworkAlgorithm::iTraining(const ANNTRAININGSET &TrainingSet, const uint32 TrainingTimes,const bool bGenerateHistory)
{
	uint32 TrainingSetSize = TrainingSet.size();
	if (bGenerateHistory)
	{
		this->m_HistoryList.resize(TrainingTimes);
	}
	for (uint32 seek = 0; seek < TrainingTimes; seek = seek + 1)
	{
		//::cout << "Training Iteration: " << seek << "\n";
		if (bGenerateHistory)
		{
			this->m_HistoryList[seek] = this->m_LayerArray;
		}
		for (uint32 seek_set = 0; seek_set < TrainingSetSize; seek_set = seek_set + 1)
		{
			this->iPredict(TrainingSet[seek_set].first);
			this->iFeedback(TrainingSet[seek_set].second);
		}
	}
}
void cArtificialNeuralNetworkAlgorithm::iTrainingWithValidation(const ANNTRAININGSET &TrainingSet, const ANNTRAININGSET &ValidationSet, const uint32 TrainingTimes)
{
	uint32 TrainingSetSize = TrainingSet.size();
	uint32 ValidationSetSize = ValidationSet.size();
	uint32 OutputArraySize = this->m_LayerArray.back().NodeArray.size();
	sfloat MinErrorValue = FLT_MAX;
	vector<ANNLAYER> MinErrorNetwork = this->m_LayerArray;
	sfloat CurrentError = 0.0;
	sfloat Error = 0.0;
	/*Calculate the initialize error*/
	for (uint32 seek = 0; seek < ValidationSetSize; seek = seek + 1)
	{
		this->iPredict(ValidationSet[seek].first);
		for (uint32 seek_node = 0; seek_node < OutputArraySize; seek_node = seek_node + 1)
		{
			Error = ValidationSet[seek].second[seek_node] - this->m_LayerArray.back()[seek_node].SigmoidOutput;
			MinErrorValue = MinErrorValue + Error*Error;
		}
	}
	uint32 Index = 0;
	for (uint32 seek = 0; seek < TrainingTimes; seek = seek + 1)
	{
		//::cout << "Training Iteration: " << seek << "\n";
		/*Training*/
		for (uint32 seek_set = 0; seek_set < TrainingSetSize; seek_set = seek_set + 1)
		{
			this->iPredict(TrainingSet[seek_set].first);
			this->iFeedback(TrainingSet[seek_set].second);
		}
		/*Pruning*/
		CurrentError = 0.0;
		for (uint32 seek_set = 0; seek_set < ValidationSetSize; seek_set = seek_set + 1)
		{
			this->iPredict(ValidationSet[seek_set].first);
			for (uint32 seek_node = 0; seek_node < OutputArraySize; seek_node = seek_node + 1)
			{
				Error = ValidationSet[seek_set].second[seek_node] - this->m_LayerArray.back()[seek_node].SigmoidOutput;
				CurrentError = CurrentError + Error*Error;
			}
		}
		if (CurrentError < MinErrorValue)
		{
			MinErrorNetwork = this->m_LayerArray;
			MinErrorValue = CurrentError;
			Index = seek;
		}
	}
	//::cout << Index << "\n";
	this->m_LayerArray = MinErrorNetwork;
}
void cArtificialNeuralNetworkAlgorithm::iPredict(const vector<sfloat> &InputArray,const int32 Index)
{
	uint32 InputLayerSize = this->m_FeatureArray.size();
	if (InputArray.size() != InputLayerSize)
	{
		return;
	}
	vector<ANNLAYER>* pCurrentNetwork = &(this->m_LayerArray);
	if (Index != -1)
	{
		if (Index < this->m_HistoryList.size())
		{
			pCurrentNetwork = &(this->m_HistoryList[Index]);
		}
	}
	(*pCurrentNetwork)[0].NodeArray.resize(InputLayerSize);
	for (uint32 seek = 0; seek < InputLayerSize; seek = seek + 1)
	{
		(*pCurrentNetwork)[0][seek].SigmoidOutput = InputArray[seek];
	}
	uint32 LayerSize = pCurrentNetwork->size();
	for (uint32 seek = 1; seek < LayerSize; seek = seek + 1)
	{
		uint32 NodeSize = (*pCurrentNetwork)[seek].NodeArray.size();
//#pragma omp parallel for
		for (int32 seek_node = 0; seek_node < NodeSize; seek_node = seek_node + 1)
		{
			sfloat Net = (*pCurrentNetwork)[seek][seek_node][0].w;
			(*pCurrentNetwork)[seek][seek_node][0].x = 1.0;
			uint32 WeightVectorSize = (*pCurrentNetwork)[seek][seek_node].WeightVector.size();
			for (uint32 seek_weight = 1; seek_weight < WeightVectorSize; seek_weight = seek_weight + 1)
			{
				(*pCurrentNetwork)[seek][seek_node][seek_weight].x = (*pCurrentNetwork)[seek - 1][seek_weight - 1].SigmoidOutput;
				Net = Net + (*pCurrentNetwork)[seek][seek_node][seek_weight].w*(*pCurrentNetwork)[seek][seek_node][seek_weight].x;
			}
			(*pCurrentNetwork)[seek][seek_node].SigmoidOutput = this->_CalcSigmoidValue(Net);
		}
	}
}
void cArtificialNeuralNetworkAlgorithm::iPredict(const int32 Index)
{
	uint32 InputLayerSize = this->m_FeatureArray.size();
	vector<sfloat> InputArray(InputLayerSize);
	for (uint32 seek = 0; seek < InputLayerSize; seek = seek + 1)
	{
		if (this->m_FeatureArray[seek].UpdateFunc == nullptr)
		{
			return;
		}
		InputArray[seek] = this->m_FeatureArray[seek].UpdateFunc(this->m_FeatureArray[seek].UserParam);
	}
	this->iPredict(InputArray, Index);
}
void cArtificialNeuralNetworkAlgorithm::iFeedback(const vector<sfloat> &TrainingOutputArray)
{
	if (this->m_LayerArray.back().NodeArray.size() != TrainingOutputArray.size())
	{
		return;
	}
	uint32 OutputSize = this->m_LayerArray.back().NodeArray.size();
	sfloat OutputVal = 0.0;
	uint32 WeightVectorSize = 0;
	ANNNODE* pNode = nullptr;
	for (uint32 seek = 0; seek < OutputSize; seek = seek + 1)
	{
		pNode = &(this->m_LayerArray.back()[seek]);

		/*Calculate the delta value*/
		OutputVal = pNode->SigmoidOutput;
		pNode->PrevDelta = pNode->CurrentDelta;
		pNode->CurrentDelta = OutputVal*(1.0 - OutputVal)*(TrainingOutputArray[seek] - OutputVal);
	}
	uint32 NodeArraySize = 0;
	for (uint32 seek = this->m_LayerArray.size() - 2; seek > 0; seek = seek - 1)
	{
		NodeArraySize = this->m_LayerArray[seek].NodeArray.size();
//#pragma omp parallel for
		for (int32 seek_node = 0; seek_node < NodeArraySize; seek_node = seek_node + 1)
		{
			uint32 OutputLayerNodeSize = this->m_LayerArray[seek + 1].NodeArray.size();;
			sfloat ErrorSum = 0.0;

			pNode = &(this->m_LayerArray[seek][seek_node]);

			/*Calculate the delta value*/
			OutputVal = pNode->SigmoidOutput;
			pNode->PrevDelta = pNode->CurrentDelta;
			for (uint32 seek_error = 0; seek_error < OutputLayerNodeSize; seek_error = seek_error + 1)
			{
				ErrorSum = ErrorSum + this->m_LayerArray[seek + 1][seek_error][seek_node + 1].w*this->m_LayerArray[seek + 1][seek_error].CurrentDelta;
			}
			pNode->CurrentDelta = OutputVal*(1.0 - OutputVal)*ErrorSum;
		}
	}
	uint32 LayerArraySize = this->m_LayerArray.size();
	for (uint32 seek = 1; seek < LayerArraySize; seek = seek + 1)
	{
		NodeArraySize = this->m_LayerArray[seek].NodeArray.size();
		for (uint32 seek_node = 0; seek_node < NodeArraySize; seek_node = seek_node + 1)
		{
			WeightVectorSize = this->m_LayerArray[seek][seek_node].WeightVector.size();
			for (uint32 seek_weight = 0; seek_weight < WeightVectorSize; seek_weight = seek_weight + 1)
			{
				/*W_ji = W_ji + LearningRate * Delta_W * X_ji + Momentum * PrevDelta*/
				sfloat DeltaW = this->m_LearningRate*this->m_LayerArray[seek][seek_node].CurrentDelta*this->m_LayerArray[seek][seek_node][seek_weight].x;
				sfloat Momentum = this->m_MomentumValue*this->m_LayerArray[seek][seek_node].PrevDelta;
				this->m_LayerArray[seek][seek_node][seek_weight].w = this->m_LayerArray[seek][seek_node][seek_weight].w + DeltaW + Momentum;
			}
		}
	}
}
void cArtificialNeuralNetworkAlgorithm::iPruning(const ANNTRAININGSET &ValidationSet)
{
	if (this->m_HistoryList.empty())
	{
		return;
	}
	uint32 HistoryArraySize = this->m_HistoryList.size();
	uint32 ValidationSetSize = ValidationSet.size();
	uint32 OutputArraySize = this->m_LayerArray.back().NodeArray.size();
	sfloat MinError = FLT_MAX;
	uint32 PIndex = 0;
	for (uint32 seek_network = 0; seek_network < HistoryArraySize;seek_network=seek_network+1)
	{
		sfloat Error = 0.0;
		for (uint32 seek = 0; seek < ValidationSetSize; seek = seek + 1)
		{
			this->iPredict(ValidationSet[seek].first, seek_network);
			for (uint32 seek_node = 0; seek_node < OutputArraySize; seek_node = seek_node + 1)
			{
				sfloat SubError = ValidationSet[seek].second[seek_node] - this->m_HistoryList[seek_network].back()[seek_node].SigmoidOutput;
				Error = Error + SubError*SubError;
			}
		}
		if (Error <= MinError)
		{
			MinError = Error;
			PIndex = seek_network;
		}
	}
	::cout << "Index:\t\t" << PIndex << "\n";
	this->m_LayerArray = this->m_HistoryList[PIndex];
	this->m_HistoryList.clear();
}
void cArtificialNeuralNetworkAlgorithm::iReset()
{
	this->_ResetNeuralNetwork();
}
void cArtificialNeuralNetworkAlgorithm::iClearHistory()
{
	this->m_HistoryList.clear();
}
vector<sfloat> cArtificialNeuralNetworkAlgorithm::iGetOutputArray() const
{
	uint32 OutputArraySize = this->m_LayerArray.back().NodeArray.size();
	vector<sfloat> OutputArray(OutputArraySize);
	for (uint32 seek = 0; seek < OutputArraySize; seek = seek + 1)
	{
		OutputArray[seek] = this->m_LayerArray.back()[seek].SigmoidOutput;
	}
	return OutputArray;
}
SString cArtificialNeuralNetworkAlgorithm::iPrintNeuralNetwork() const
{
	const SString strLine = _SSTR("--------------------\r\n");

	SString strOutput = _SSTR("INPUT LAYER:\r\n\r\n");

	uint32 LayerSize = this->m_LayerArray.size();

	for (uint32 seek = 0; seek < this->m_FeatureArray.size(); seek = seek + 1)
	{
		strOutput = strOutput + _SSTR("  ") + this->m_FeatureArray[seek].strFeatrueName + _SSTR("\r\n");
	}

	strOutput = strOutput + _SSTR("\r\n\r\n");

	for (uint32 seek = 0; seek < this->m_LayerArray[0].NodeArray.size(); seek = seek + 1)
	{
		strOutput = strOutput + ::gloFToS(this->m_LayerArray[0][seek].SigmoidOutput, _SSTR("6.3")) + _SSTR("\t");
	}

	strOutput = strOutput + _SSTR("\r\n");
	strOutput = strOutput + _SSTR("\r\n");

	for (uint32 seek = 1; seek < LayerSize; seek = seek + 1)
	{
		if (seek != (LayerSize - 1))
		{
			strOutput = strOutput + _SSTR("HIDDEN LAYER # ") + ::gloIToS(seek) + _SSTR(":\r\n\r\n");
		}
		else
		{
			strOutput = strOutput + _SSTR("OUTPUT LAYER\r\n\r\n");
		}
		for (uint32 seek_node = 0; seek_node < this->m_LayerArray[seek].NodeArray.size(); seek_node = seek_node + 1)
		{
			strOutput = strOutput + ::gloFToS(this->m_LayerArray[seek][seek_node].SigmoidOutput, _SSTR("6.3")) + _SSTR(" = ");
			for (uint32 seek_w = 0; seek_w < this->m_LayerArray[seek][seek_node].WeightVector.size(); seek_w = seek_w + 1)
			{
				strOutput = strOutput + ::gloFToS(this->m_LayerArray[seek][seek_node][seek_w].w, _SSTR("6.3")) + _SSTR(" ");
			}
			strOutput = strOutput + _SSTR("\r\n");
		}
		strOutput = strOutput + _SSTR("\r\n");
	}
	return strOutput;
}