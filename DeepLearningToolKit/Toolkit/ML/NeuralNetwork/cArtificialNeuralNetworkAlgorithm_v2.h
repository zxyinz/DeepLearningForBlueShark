#include"../../Core/SanTypes.h"
#include"../../Core/SanMathematics.h"
#include"../cDataSet.h"
#include"cArtificialNeuralNetworkAlgorithm.h"
using namespace std;
#pragma once
namespace San
{
	namespace MachineLearning
	{
#ifndef __CARTIFICIALNEURALNETWORKALGORITHMV2_H__
#define __CARTIFICIALNEURALNETWORKALGORITHMV2_H__
		struct stNEURALNODE
		{
		public:
			typedef vector<ANNWEIGHTPAIR>::iterator iterator;
			typedef vector<ANNWEIGHTPAIR>::const_iterator const_iterator;
		protected:
			vector<ANNWEIGHTPAIR> WeightArray;
			sfloat Delta;
			sfloat PrevDelta;
		public:
			sfloat LinearOutput;
			sfloat FunctionOutput;
			sfloat PredictError;
		protected:
			virtual void _CreateNode();
			virtual void _ReleaseNode();
			virtual sfloat _CalcFuncValue(const sfloat &Net) const;
		public:
			stNEURALNODE();
			stNEURALNODE(const stNEURALNODE &Node);
			~stNEURALNODE();

			const ANNWEIGHTPAIR& operator[](const uint32 Index) const;
			ANNWEIGHTPAIR& operator[](const uint32 Index);

			/*No size check, may cause error*/
			virtual sfloat iPredict(const vector<sfloat> &InputData);
			virtual sfloat iPredict(const vector<sfloat> &InputData) const;

			virtual sfloat iFeedback(const sfloat &Delta, const sfloat LearningRate, const sfloat Momentum = 0.0);

			virtual uint32 iResize(const uint32 WeightArraySize, const uint32 RandomSeed);
			virtual uint32 iReset(const uint32 RandomSeed);
			virtual stNEURALNODE* iCopy() const;

			/*STL function*/
			const_iterator cbegin() const;
			const_iterator cend() const;

			const_iterator begin() const;
			iterator begin();
			const_iterator end() const;
			iterator end();

			uint32 size() const;
			bool empty() const;
		};
		typedef stNEURALNODE	NEURALNODE;
		typedef stNEURALNODE*	lpNEURALNODE;

		template<class _SigmoidFunction>
		struct _SIGMOIDNODE : public NEURALNODE
		{
		private:
			_SigmoidFunction SigmoidFunc;
		protected:
			sfloat _CalcFuncValue(const sfloat &Net) const override final 
			{ 
				return SigmoidFunc(Net); 
			};
		public:
			_SIGMOIDNODE(const _SigmoidFunction &Func)
				:SigmoidFunc(Func)
			{
			};
			_SIGMOIDNODE(const _SIGMOIDNODE<_SigmoidFunction> &Node)
				:NEURALNODE(NEURALNODE(Node)), SigmoidFunc(Node.SigmoidFunc)
			{
			};
			~_SIGMOIDNODE(){};

			stNEURALNODE* iCopy() const override final
			{
				stNEURALNODE* pNode = new _SIGMOIDNODE<_SigmoidFunction>(*this);
				return pNode;
			};
		};

		template<class _SigmoidFunction>
		using _sigmoidnode = _SIGMOIDNODE<_SigmoidFunction>;

		struct stNEURALLAYER
		{
		public:
			typedef vector<NEURALNODE*>::iterator iterator;
			typedef vector<NEURALNODE*>::const_iterator const_iterator;
		public:
			NEURALNODE* pIdentityNode;
			vector<NEURALNODE*> NodePtrArray;
		public:
			template<class _NodeType>
			stNEURALLAYER(const _NodeType &Node, const uint32 NodeNumber)
				: NodePtrArray(NodeNumber, nullptr)
			{
				this->pIdentityNode = Node.iCopy();
				for (auto &ptr : this->NodePtrArray){ ptr = this->pIdentityNode->iCopy(); }
			};
			stNEURALLAYER(const stNEURALLAYER &Layer);
			~stNEURALLAYER();

			stNEURALLAYER& operator=(const stNEURALLAYER &Layer);

			const NEURALNODE* operator[](const uint32 Index) const;
			NEURALNODE* operator[](const uint32 Index);

			uint32 iResize(const uint32 NodeNumber);

			const_iterator cbegin() const;
			const_iterator cend() const;

			const_iterator begin() const;
			iterator begin();
			const_iterator end() const;
			iterator end();

			uint32 size() const;
			bool empty() const;
		};
		typedef stNEURALLAYER	NEURALLAYER;
		typedef stNEURALLAYER*	lpNEURALLAYER;

		class cArtificialNeuralNetworkAlgorithm_v2
		{
		public:
			typedef NEURALNODE _node;
			typedef NEURALLAYER _layer;
			typedef vector<_layer> _network;
		protected:
			_network m_CurrentNetwork;				// ID = size() - 1, Output Layer
			vector<_network> m_HistoryList;			// Layer history, for validation use, contain all the history from 0-state ~ (current - 1)-state
			sfloat m_LearningRate;
			sfloat m_MomentumValue;
			sfloat m_RandomBoundary[2];				// The boundary of random initialize value, default -0.05 ~ 0.05
			int32 m_InputLayerSize;
			mutable vector<vector<sfloat>> m_OutputArray;
		protected:
			void _ResetNeuralNetwork(const uint32 RandomSeed = 0);
			sfloat _GenerateRandomValue() const;
			_layer _GetHiddenLayer(const uint32 LayerID) const; //For incremental learning
		public:
			cArtificialNeuralNetworkAlgorithm_v2(const sfloat LearningRate = 0.1, const sfloat MomentumValue = 0.0, const sfloat RandomMinBoundary = -0.05, const sfloat RandomMaxBoundary = 0.05);
			~cArtificialNeuralNetworkAlgorithm_v2();

			/* Layer operation function, the last layer user created is deemed to output layer */
			template<class _NodeType>
			int32 iCreateLayer(const _NodeType &Node, const uint32 NodeNumber)
			{
				if (NodeNumber == 0) { return -1; }

				this->m_CurrentNetwork.push_back(_layer(Node, NodeNumber));
				this->_ResetNeuralNetwork();
				return this->m_CurrentNetwork.size() - 1;
			};
			bool iResizeLayer(const uint32 LayerID, const uint32 NodeNumber);
			bool iInitializeLayer(const uint32 LayerID, const vector<vector<sfloat>> InitValue = vector<vector<sfloat>>(0));
			void iReleaseLayer(const uint32 LayerID);

			/* Learning param operation functions */
			bool iUpdateLearningRate(const sfloat Rate);
			bool iUpdateMomentumValue(const sfloat Momentum);
			bool iUpdateRandomBoundary(const sfloat RandomMinBoundary, const sfloat RandomMaxBoundary);
			bool iUpdateInputLayerSize(const uint32 FeatureSize);

			sfloat iGetLearningRate() const;
			sfloat iGetMomentumValue() const;
			vector<sfloat> iGetRandomBoundary() const;
			uint32 iUpdateInputLayerSize() const;

			/* Training the artificial neural network, this function will generate the layer history			*/
			/* and the layer history will continue exit unitl the function iPruning or iClearHistory be called	*/
			/* TrainingTimes: iteration times																	*/
			void iTraining(const ANNTRAININGSET &TrainingSet, const uint32 TrainingTimes = 1, const sfloat MinWeightChanges = 0.0, const sfloat MinErrorChanges = 0.0, const bool bGenerateHistory = true);

			template<class _ann_dataset, class _RawDataGenerator>
			void iTraining(const _ann_dataset &TrainingSet, _RawDataGenerator &RawDataGenerator, const uint32 TrainingTimes = 1, const sfloat MinWeightChanges = 0.0, const sfloat MinErrorChanges = 0.0, const bool bGenerateHistory = true)
			{
				uint32 TrainingSetSize = TrainingSet.size();

				if (bGenerateHistory)
				{
					this->m_HistoryList.resize(TrainingTimes);
				}

				sfloat PrevErrorSum = 0.0;
				for (uint32 seek = 0; seek < TrainingTimes; seek = seek + 1)
				{
					sfloat CurrentWeightChanges = 0.0;
					sfloat CurrentErrorSum = 0.0;

					if (bGenerateHistory)
					{
						this->m_HistoryList[seek] = this->m_CurrentNetwork;
					}

					for (uint32 seek_set = 0; seek_set < TrainingSetSize; seek_set = seek_set + 1)
					{
						const ANNTRAININGSET::_data_type &Data = RawDataGenerator(TrainingSet[seek_set].Data);
						this->iPredict(Data);

						pair<sfloat, sfloat> Res = this->iFeedback(TrainingSet[seek_set].Label);

						CurrentWeightChanges = CurrentWeightChanges + Res.first;
						CurrentErrorSum = CurrentErrorSum + Res.second;
					}

					sfloat CurrentErrorChanges = abs(CurrentErrorSum - PrevErrorSum);
					PrevErrorSum = CurrentErrorSum;

					_SDEBUG(::cout << "Iteration: " << seek << "\tW: " << CurrentWeightChanges << "\tE: " << CurrentErrorSum << "\tEC: " << CurrentErrorChanges << "\n");

					if (CurrentWeightChanges < MinWeightChanges){ break; }
					if (CurrentErrorChanges < MinErrorChanges){ break; }
				}
			};

			/* Training and pruning at the same time, this function will not generate any layer histiry			*/
			/* should be faster than iTraining function															*/
			void iTrainingWithValidation(const ANNTRAININGSET &TrainingSet, const ANNTRAININGSET &ValidationSet, const uint32 TrainingTimes = 1, const sfloat MinWeightChanges = 0.0, const sfloat MinErrorChanges = 0.0);

			template<class _ann_dataset, class _RawDataGenerator>
			void iTrainingWithValidation(const _ann_dataset &TrainingSet, const _ann_dataset &ValidationSet, _RawDataGenerator &RawDataGenerator, const uint32 TrainingTimes = 1, const sfloat MinWeightChanges = 0.0, const sfloat MinErrorChanges = 0.0)
			{
				const uint32 TrainingSetSize = TrainingSet.size();
				const uint32 ValidationSetSize = ValidationSet.size();
				const uint32 OutputArraySize = this->m_CurrentNetwork.back().size();

				sfloat MinErrorValue = FLT_MAX;
				_network MinErrorNetwork;

				sfloat CurrentError = 0.0;
				sfloat PredictError = 0.0;

				/*Calculate the initialize error*/
				for (uint32 seek = 0; seek < ValidationSetSize; seek = seek + 1)
				{
					const ANNTRAININGSET::_data_type &Data = RawDataGenerator(ValidationSet[seek].Data);
					this->iPredict(Data);

					for (uint32 seek_node = 0; seek_node < OutputArraySize; seek_node = seek_node + 1)
					{
						PredictError = ValidationSet[seek].Label[seek_node] - this->m_OutputArray.back()[seek_node];
						MinErrorValue = MinErrorValue + PredictError * PredictError;
					}
				}

				sfloat PrevErrorSum = 0.0;

				for (uint32 seek = 0; seek < TrainingTimes; seek = seek + 1)
				{
					//::cout << "Training Iteration: " << seek << "\n";
					/*Training*/
					sfloat CurrentWeightChanges = 0.0;
					sfloat CurrentErrorSum = 0.0;

					for (uint32 seek_set = 0; seek_set < TrainingSetSize; seek_set = seek_set + 1)
					{
						const ANNTRAININGSET::_data_type &Data = RawDataGenerator(TrainingSet[seek_set].Data);
						this->iPredict(Data);

						pair<sfloat, sfloat> Res = this->iFeedback(TrainingSet[seek_set].Label);

						CurrentWeightChanges = CurrentWeightChanges + Res.first;
						CurrentErrorSum = CurrentErrorSum + Res.second;
					}

					/*Pruning*/
					CurrentError = 0.0;
					for (uint32 seek_set = 0; seek_set < ValidationSetSize; seek_set = seek_set + 1)
					{
						const ANNTRAININGSET::_data_type &Data = RawDataGenerator(ValidationSet[seek_set].Data);
						this->iPredict(Data);

						for (uint32 seek_node = 0; seek_node < OutputArraySize; seek_node = seek_node + 1)
						{
							PredictError = ValidationSet[seek_set].Label[seek_node] - this->m_OutputArray.back()[seek_node];
							CurrentError = CurrentError + PredictError * PredictError;
						}
					}
					if (CurrentError < MinErrorValue)
					{
						MinErrorNetwork = this->m_CurrentNetwork;
						MinErrorValue = CurrentError;
					}

					sfloat CurrentErrorChanges = CurrentErrorSum - PrevErrorSum;
					PrevErrorSum = CurrentErrorSum;

					if (CurrentWeightChanges < MinWeightChanges){ break; }
					if (CurrentErrorChanges < MinErrorChanges){ break; }
				}
				//::cout << Index << "\n";
				this->m_CurrentNetwork = MinErrorNetwork;
			};

			/* Predict the output with given input array, the input array size should same as the input layer unit size	*/
			void iPredict(const vector<sfloat> &InputArray, const int32 Index = -1);

			/*Predict, not change the network value, slower than iPredict*/
			vector<sfloat> iPredict(const vector<sfloat> &InputArray, const int32 Index = -1) const;

			/* Update the hidden unit and output unit weight by training output array	*/
			/* the training output array size should equal with output layer size		*/
			pair<sfloat, sfloat> iFeedback(const vector<sfloat> &TrainingOutputArray);

			/* Pruning the artificial neural network with the given training set and validation set	*/
			void iPruning(const ANNTRAININGSET &ValidationSet);

			/* Reset all the hiddenlayer and output layer weight by random values(re-initialize)	*/
			void iReset();

			/* Clear the layer history	*/
			void iClearHistory();

			vector<sfloat> iGetOutputArray(int32 LayerID = -1) const;
			SString iPrintNeuralNetwork() const;
		};
		//typedef cArtificialNeuralNetworkAlgorithm cANNAlgorithm;
#endif
	}
}