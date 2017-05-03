#include"cDeepNeuralNetworkAlgorithm.h"
using namespace std;
using namespace San;
using namespace San::MachineLearning;
cDeepNeuralNetworkAlgorithm::cDeepNeuralNetworkAlgorithm(const sfloat LearningRate, const sfloat MomentumValue, const sfloat RandomMinBoundary, const sfloat RandomMaxBoundary)
	:cArtificialNeuralNetworkAlgorithm_v2(LearningRate,MomentumValue,RandomMinBoundary,RandomMaxBoundary)
{
}
cDeepNeuralNetworkAlgorithm::~cDeepNeuralNetworkAlgorithm()
{

}
void cDeepNeuralNetworkAlgorithm::iPreProcessing(const _rbm_dataset &TrainingSet, const sfloat RBMLearningRate, const uint32 TrainingTimes, _rengine &REngine, const sfloat MinWeightChanges, const sfloat MinErrorChanges)
{
	const int32 TrainingSetSize = TrainingSet.size();
	int32 RBMVisibleLayerSize = this->m_InputLayerSize + 1;
	int32 RBMHiddenLayerSize = this->m_CurrentNetwork[0].NodePtrArray.size();

	cRestrictedBoltzmannMachineAlgorithm RBM(RBMVisibleLayerSize, RBMHiddenLayerSize, this->m_RandomBoundary[0], this->m_RandomBoundary[1]);

	_rbm_dataset RBMInputSet = TrainingSet;
	_rbm_dataset RBMOutputSet;

	_rbm_dataset::_instance OutputInstance(_sstream<sfloat>(RBMHiddenLayerSize), 0);

	for (int32 seek = 0; seek < TrainingSetSize; seek = seek + 1)
	{
		RBMOutputSet.iPush(OutputInstance);
	}

	RBM.iTrainingRBM(RBMInputSet, RBMOutputSet, TrainingTimes, 1, RBMLearningRate, REngine, MinWeightChanges, MinErrorChanges);

	/*Update Weight*/
	const auto &WeightMatrix = RBM.iGetWeightMatrix();
	for (int32 seek_hidden = 0; seek_hidden < RBMHiddenLayerSize; seek_hidden = seek_hidden + 1)
	{
		const auto &WeightArray = WeightMatrix[seek_hidden];
		auto &Node = (*this->m_CurrentNetwork[0][seek_hidden]);

		for (int32 seek_visible = 0; seek_visible < RBMVisibleLayerSize; seek_visible = seek_visible + 1)
		{
			Node[seek_visible].w = WeightArray[seek_visible];
		}
	}

	int32 HiddenLayerSize = this->m_CurrentNetwork.size() - 1;

	for (int32 seek = 1; seek < HiddenLayerSize; seek = seek + 1)
	{
		RBMVisibleLayerSize = this->m_CurrentNetwork[seek - 1].NodePtrArray.size() + 1;
		RBMHiddenLayerSize = this->m_CurrentNetwork[seek].NodePtrArray.size();

		for (int32 seek_instance = 0; seek_instance < TrainingSetSize; seek_instance = seek_instance + 1)
		{
			auto &InputDataRef = RBMInputSet[seek_instance].Data;
			auto &OutputDataRef = RBMOutputSet[seek_instance].Data;

			InputDataRef.resize(RBMVisibleLayerSize);

			InputDataRef[0] = 1.0; //W0

			for (int32 seek_attrib = 1; seek_attrib <= RBMVisibleLayerSize; seek_attrib = seek_attrib + 1)
			{
				InputDataRef[seek_attrib] = OutputDataRef[seek_attrib - 1];
			}

			OutputDataRef.resize(RBMHiddenLayerSize);
		}

		RBM.iUpdateDBMState(RBMVisibleLayerSize, RBMHiddenLayerSize, this->m_RandomBoundary[0], this->m_RandomBoundary[1]);
		RBM.iTrainingRBM(RBMInputSet, RBMOutputSet, TrainingTimes, 1, RBMLearningRate, REngine, MinWeightChanges, MinErrorChanges);

		/*Update Weight*/
		const auto &WeightMatrix = RBM.iGetWeightMatrix();
		for (int32 seek_hidden = 0; seek_hidden < RBMHiddenLayerSize; seek_hidden = seek_hidden + 1)
		{
			const auto &WeightArray = WeightMatrix[seek_hidden];
			auto &Node = (*this->m_CurrentNetwork[seek][seek_hidden]);

			for (int32 seek_visible = 0; seek_visible < RBMVisibleLayerSize; seek_visible = seek_visible + 1)
			{
				Node[seek_visible].w = WeightArray[seek_visible];
			}
		}
	}
}
void cDeepNeuralNetworkAlgorithm::iGenerateTrainingSet(const ANNTRAININGSET &TrainingSet, _rbm_dataset &DestDataSet) const
{
	const int32 TrainingSetSize = TrainingSet.size();
	int32 RBMVisibleLayerSize = this->m_InputLayerSize + 1;
	int32 RBMHiddenLayerSize = this->m_CurrentNetwork[0].NodePtrArray.size();

	_rbm_dataset::_instance InputInstance(_sstream<sfloat>(RBMVisibleLayerSize), 0);

	auto &Data = InputInstance.Data;

	for (int32 seek = 0; seek < TrainingSetSize; seek = seek + 1)
	{
		Data.resize(RBMVisibleLayerSize);

		Data[0] = 1.0;
		for (int32 seek_attrib = 1; seek_attrib < RBMVisibleLayerSize; seek_attrib = seek_attrib + 1)
		{
			Data[seek_attrib] = TrainingSet[seek].Data[seek_attrib - 1];
		}

		DestDataSet.iPush(InputInstance);
	}
}
void cDeepNeuralNetworkAlgorithm::iGenerateTrainingSet(const ANNTRAININGSET_IT &TrainingSet, _rbm_dataset &DestDataSet) const
{
	const int32 TrainingSetSize = TrainingSet.size();
	int32 RBMVisibleLayerSize = this->m_InputLayerSize + 1;
	int32 RBMHiddenLayerSize = this->m_CurrentNetwork[1].NodePtrArray.size();

	_rbm_dataset::_instance InputInstance(_sstream<sfloat>(RBMVisibleLayerSize), 0);

	auto &Data = InputInstance.Data;

	for (int32 seek = 0; seek < TrainingSetSize; seek = seek + 1)
	{
		Data.resize(RBMVisibleLayerSize);

		Data[0] = 1.0;
		for (int32 seek_attrib = 1; seek_attrib < RBMVisibleLayerSize; seek_attrib = seek_attrib + 1)
		{
			Data[seek_attrib] = TrainingSet[seek].Data->Data[seek_attrib - 1];
		}

		DestDataSet.iPush(InputInstance);
	}
}