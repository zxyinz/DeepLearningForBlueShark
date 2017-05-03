#include"..\Core\SanContainer.h"
#include"cDecisionTreeAlgorithm.h"
using namespace std;
#pragma once;
namespace San
{
	namespace MachineLearning
	{
#ifndef __CGENETICALGORITHMHEPOTHESIS_H__
#define __CGENETICALGORITHMHEPOTHESIS_H__
		/*All the hypothesis type which supported by cGeneticAlgorithm class*/
		enum eGENETICALGORITHMHYPOTHESISTYPE
		{
			GAHT_RULESET,
			GAHT_RULESET_CONTINUE,
		};
		typedef eGENETICALGORITHMHYPOTHESISTYPE eGAHYPOTHESISTYPE;

		/*Instance Definition*/
		template<class InputType, class OutputType>
		using GAINSTANCE = pair<vector<InputType>, vector<OutputType>>;

		/*Instance Set Definition*/
		template<class InputType, class OutputType>
		using GAINSTANCESET = vector<GAINSTANCE<InputType, OutputType>>;

		/*Genetic algotrithm hypothesis class*/
		template<class I, class O, class UserDataType>
		class cGeneticAlgorithmHypothesis
		{
		private:
			eGAHYPOTHESISTYPE m_Type;
		public:
			cGeneticAlgorithmHypothesis(const eGAHYPOTHESISTYPE Type = GAHT_RULTSET)
				:m_Type(Type)
			{
			};
			~cGeneticAlgorithmHypothesis(){};
			eGAHYPOTHESISTYPE iGetType() const
			{
				return this->m_Type;
			};
			virtual bool iInitialize(const uint32 Index, const UserDataType &UserData, const GAINSTANCESET<I, O> &InstanceSpace) = 0;
			virtual void iRelease() = 0;
			virtual bool iPredict(const GAINSTANCESET<I, O> &InstanceSpace, vector<O> &Result) = 0;
			virtual bool iTraining(const GAINSTANCESET<I, O> &TrainingSet) = 0;
			virtual bool iCrossoverSP(const cGeneticAlgorithmHypothesis<I, O, UserDataType>* pHypothesis, vector<cGeneticAlgorithmHypothesis<I, O, UserDataType>*> &OffspringSet) = 0;
			virtual bool iCrossoverTP(const cGeneticAlgorithmHypothesis<I, O, UserDataType>* pHypothesis, vector<cGeneticAlgorithmHypothesis<I, O, UserDataType>*> &OffspringSet) = 0;
			virtual bool iCrossoverUNI(const cGeneticAlgorithmHypothesis<I, O, UserDataType>* pHypothesis, vector<cGeneticAlgorithmHypothesis<I, O, UserDataType>*> &OffspringSet) = 0;
			virtual bool iMutation() = 0;
			virtual sfloat iEvaluate(const GAINSTANCESET<I, O> &InstanceSpace) = 0;
			virtual sfloat iAccuracy(const GAINSTANCESET<I, O> &InstanceSpace) = 0;
			virtual SString iPrintHypothesisSpace() const = 0;
			virtual SString iPrintDebugInformtion() const = 0;
			virtual sfloat iGetHypothesisSize() const = 0;
		};
#endif
	}
}