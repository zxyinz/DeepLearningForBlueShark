#include"cArtificialNeuralNetworkAlgorithm.h"
#include"cArtificialNeuralNetworkAlgorithm_v2.h"
#include"cRestrictedBoltzmannMachineAlgorithm.h"
using namespace std;
#pragma once
namespace San
{
	namespace MachineLearning
	{
#ifndef __CDEEPNEURALNETWORKALGORITHM_H__
#define __CDEEPNEURALNETWORKALGORITHM_H__
		class cDeepNeuralNetworkAlgorithm : public cArtificialNeuralNetworkAlgorithm_v2//cArtificialNeuralNetworkAlgorithm
		{
		public:
			typedef cRestrictedBoltzmannMachineAlgorithm::_rbm_dataset _rbm_dataset;
			typedef cRestrictedBoltzmannMachineAlgorithm::_rengine _rengine;
		private:
		public:
			cDeepNeuralNetworkAlgorithm(const sfloat LearningRate = 0.1, const sfloat MomentumValue = 0.0, const sfloat RandomMinBoundary = -0.05, const sfloat RandomMaxBoundary = 0.05);
			~cDeepNeuralNetworkAlgorithm();

			void iPreProcessing(const _rbm_dataset &TrainingSet, const sfloat RBMLearningRate, const uint32 TrainingTimes, _rengine &REngine, const sfloat MinWeightChanges = 0.0, const sfloat MinErrorChanges = 0.0);

			void iGenerateTrainingSet(const ANNTRAININGSET &TrainingSet, _rbm_dataset &DestDataSet) const;
			void iGenerateTrainingSet(const ANNTRAININGSET_IT &TrainingSet, _rbm_dataset &DestDataSet) const;
		};
#endif
	}
}