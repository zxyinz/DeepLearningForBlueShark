#include"../../Core/SanContainer.h"
#include"../../Core/SanMathematics.h"
#include"../cDataSet.h"
using namespace std;
using namespace San::Mathematics;
#pragma once
namespace San
{
	namespace MachineLearning
	{
#ifndef __CRESTRICTEDBOLTZMANNMACHINEALGORITHM_H__
#define __CRESTRICTEDBOLTZMANNMACHINEALGORITHM_H__
		class cRestrictedBoltzmannMachineAlgorithm
		{
		public:
			typedef _dataset<_sstream<sfloat>, uint32> _rbm_dataset;
			typedef default_random_engine _rengine;
		private:
			_sstream<sfloat> m_HiddenLayerBias;
			_sstream<sfloat> m_VisibleLayerBias;
			_sstream<_rengine> m_HiddenLayerREngine;
			_sstream<_rengine> m_VisibleLayerREngine;
			vector<vector<sfloat>> m_WeightMatrix; // Size = HiddenLayerSize * VisibleLayerSize, 2D Matrix
			sfloat m_RandomBoundary[2];
		protected:
			bool _InitializeRBM(const _rbm_dataset &TrainingSet, _rengine &REngine);

			void _SampleHV(const _sstream<sfloat> &Sample, _sstream<sfloat> &HiddenNodeProbability, _sstream<sfloat> &HiddenNodeSample, const int32 VisibleLayerSize, _rengine &REngine);
			void _SampleVH(const _sstream<sfloat> &Sample, _sstream<sfloat> &VisibleNodeProbability, _sstream<sfloat> &VisibleNodeSample, const int32 HiddenLayerSize, _rengine &REngine);

			void _SampleHV(const _sstream<sfloat> &Sample, _sstream<sfloat> &HiddenNodeProbability, _sstream<sfloat> &HiddenNodeSample, const int32 VisibleLayerSize); // Parallel
			void _SampleVH(const _sstream<sfloat> &Sample, _sstream<sfloat> &VisibleNodeProbability, _sstream<sfloat> &VisibleNodeSample, const int32 HiddenLayerSize); // Parallel

			sfloat _CalcProbabilityUp(const uint32 HiddenIndex, const _sstream<sfloat> &Sample, const int32 VisibleLayerSize) const;
			sfloat _CalcProbabilityDown(const uint32 VisibleIndex, const _sstream<sfloat> &Sample, const int32 HiddenLayerSize) const;

			sfloat _CalcBinomialValue(const sfloat N, const sfloat Probability, _rengine &REngine) const;
			sfloat _CalcSigmoidValue(const sfloat Net) const;
			sfloat _CalcUniformDistribution(const sfloat MinBoundary, const sfloat MaxBoundary, _rengine &REngine) const;
		public:
			cRestrictedBoltzmannMachineAlgorithm(const uint32 VisibleLayerSize = 1, const uint32 HiddenLayerSize = 1, const sfloat MinBoundary = -0.05, const sfloat MaxBoundary = 0.05);
			~cRestrictedBoltzmannMachineAlgorithm();

			bool iUpdateDBMState(const uint32 VisibleLayerSize, const uint32 HiddenLayerSize, const sfloat MinBoundary, const sfloat MaxBoundary);

			void iTrainingRBM(const _rbm_dataset &TrainingSet, _rbm_dataset &OutputSet, const uint32 TrainingTimes, const uint32 KTimes, const sfloat LearningRate, _rengine &REngine, const sfloat MinWeightChanges = 0.0, const sfloat MinErrorChanges = 0.0);

			void iReconstruct(_sstream<sfloat> &VisibleSample, _sstream<sfloat> &ReconstructResult);

			const vector<vector<sfloat>>& iGetWeightMatrix() const;
			vector<vector<sfloat>>& iGetWeightMatrix();

			void iPrintRBM();
		};
#endif
	}
}