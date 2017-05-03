#include"../../Core/SanMathematics.h"
#include"Node/cNeuralNode.h"
#include"cArtificialNeuralNetworkAlgorithm_v2.h"
using namespace std;
using namespace San::Mathematics;
#pragma once
namespace San
{
	namespace MachineLearning
	{
#ifndef __CCONVOLUTIONALNEURALNETWORKALGORITHM_H__
#define __CCONVOLUTIONALNEURALNETWORKALGORITHM_H__
		struct stCNNNEURALLAYER
		{
		public:
			typedef vector<NEURALNODE*>::iterator iterator;
			typedef vector<NEURALNODE*>::const_iterator const_iterator;
		public:
			NEURALNODE* pIdentityNode;
			vector<NEURALNODE*> NodePtrArray;
		public:
			template<class _NodeType>
			stCNNNEURALLAYER(const _NodeType &Node, const uint32 NodeNumber)
				: NodePtrArray(NodeNumber, nullptr)
			{
				this->pIdentityNode = Node.iCopy();
				for (auto &ptr : this->NodePtrArray){ ptr = this->pIdentityNode->iCopy(); }
			};
			stCNNNEURALLAYER(const stCNNNEURALLAYER &Layer);
			~stCNNNEURALLAYER();

			stCNNNEURALLAYER& operator=(const stCNNNEURALLAYER &Layer);

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
		typedef stCNNNEURALLAYER	CNNNEURALLAYER;
		typedef stCNNNEURALLAYER*	lpCNNNEURALLAYER;

		class cForwardNeuralNetworkAlgorithm
		{
		public:
			typedef cNeuralNode _node;
			typedef _tensor<sfloat> _node_data;
			typedef NEURALLAYER _layer;
			typedef vector<_layer> _network;
		protected:
			_network m_CurrentNetwork;				// ID = size() - 1, Output Layer
			vector<_network> m_HistoryList;			// Layer history, for validation use, contain all the history from 0-state ~ (current - 1)-state
			sfloat m_LearningRate;
			sfloat m_MomentumValue;
			sfloat m_RandomBoundary[2];				// The boundary of random initialize value, default -0.05 ~ 0.05
			int32 m_InputLayerSize;
			mutable vector<vector<_node_data>> m_OutputArray;
		protected:
			void _ResetNeuralNetwork(const uint32 RandomSeed = 0);
			sfloat _GenerateRandomValue() const;
			_layer _GetHiddenLayer(const uint32 LayerID) const; //For incremental learning
		public:
			cForwardNeuralNetworkAlgorithm(const sfloat LearningRate = 0.1, const sfloat MomentumValue = 0.0, const sfloat RandomMinBoundary = -0.05, const sfloat RandomMaxBoundary = 0.05);
			~cForwardNeuralNetworkAlgorithm();

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

			/* Training and pruning at the same time, this function will not generate any layer histiry			*/
			/* should be faster than iTraining function															*/
			void iTrainingWithValidation(const ANNTRAININGSET &TrainingSet, const ANNTRAININGSET &ValidationSet, const uint32 TrainingTimes = 1, const sfloat MinWeightChanges = 0.0, const sfloat MinErrorChanges = 0.0);

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