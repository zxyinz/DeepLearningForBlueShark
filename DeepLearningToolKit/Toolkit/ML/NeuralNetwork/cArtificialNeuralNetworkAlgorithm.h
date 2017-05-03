#include"../../Core/SanTypes.h"
#include"../../Core/SanMathematics.h"
#include"../cDataSet.h"
using namespace std;
#pragma once
namespace San
{
	namespace MachineLearning
	{
#ifndef __CARTIFICIALNEURALNETWORKALGORITHM_H__
#define __CARTIFICIALNEURALNETWORKALGORITHM_H__

		typedef sfloat(*ANNFEATUREUPDATEFUNC)(SANSTREAM&);

		struct ANNFEATURE
		{
			SString strFeatureName;
			SANSTREAM UserParam;
			ANNFEATUREUPDATEFUNC UpdateFunc;
		};

		struct ANNWEIGHTPAIR
		{
		public:
			sfloat x;
			sfloat w;
		public:
			ANNWEIGHTPAIR(const sfloat x = 0.0, const sfloat w = 0.0) :x(x), w(w){};
			ANNWEIGHTPAIR(const ANNWEIGHTPAIR &WeightPair) : x(WeightPair.x), w(WeightPair.w){};
			~ANNWEIGHTPAIR(){};
		};

		struct ANNNODE
		{
		public:
			uint32 NodeID;
			vector<ANNWEIGHTPAIR> WeightVector;
			sfloat PrevDelta;
			sfloat CurrentDelta;
			sfloat SigmoidOutput;
			//uint32 BinomialOutput;
			sfloat VisibleBias;		// For deep learning, RBM
			sfloat HiddenBias;		// For deep learning, RBM
		public:
			ANNNODE(){};
			~ANNNODE(){};

			ANNWEIGHTPAIR& operator[](const uint32 Index)
			{
				return this->WeightVector[Index];
			};
			const ANNWEIGHTPAIR& operator[](const uint32 Index) const
			{
				return this->WeightVector[Index];
			}
		};

		struct ANNLAYER
		{
		public:
			uint32 LayerID;
			vector<ANNNODE> NodeArray;
		public:
			ANNLAYER(){};
			~ANNLAYER(){};

			ANNNODE& operator[](const uint32 Index)
			{
				return NodeArray[Index];
			};
			const ANNNODE& operator[](const uint32 Index) const
			{
				return NodeArray[Index];
			};
		};

		typedef _dataset<vector<sfloat>,vector<sfloat>> ANNTRAININGSET;
		typedef _dataset<ANNTRAININGSET::const_iterator, vector<sfloat>> ANNTRAININGSET_IT;

		class cArtificialNeuralNetworkAlgorithm
		{
		protected:
			vector<ANNFEATURE> m_FeatureArray;		// Feature update function array, update the input layer values
			vector<ANNLAYER> m_LayerArray;			// ID = 0, Input Layer, contain all the input values which updated by Feature update function
			// ID = size() - 1, Output Layer
			vector<vector<ANNLAYER>> m_HistoryList;	// Layer history, for validation use, contain all the history from 0-state ~ (current - 1)-state
			sfloat m_LearningRate;
			sfloat m_MomentumValue;
			sfloat m_RandomBoundary[2];			// The boundary of random initialize value, default -0.05 ~ 0.05
		protected:
			void _ResetNeuralNetwork();
			sfloat _CalcSigmoidValue(const sfloat Net) const;
			sfloat _GenerateRandomValue() const;
			ANNLAYER _GetHiddenLayer(const uint32 LayerID) const; //For incremental learning
		public:
			cArtificialNeuralNetworkAlgorithm(const sfloat LearningRate = 0.1, const sfloat MomentumValue = 0.0, const sfloat RandomMinBoundary = -0.05, const sfloat RandomMaxBoundary = 0.05);
			~cArtificialNeuralNetworkAlgorithm();

			/* Feature operation functions, the FeatureUnpdateFunc is for ANN class update the feature value automatically	- using in iPredict() function */
			bool iCreateFeatureNode(const SString &strFeatureName, const SANSTREAM &UserParam, const ANNFEATUREUPDATEFUNC FeatureUpdateFunc);
			bool iUpdateFeatureNode(const SString &strFeatureName, const SANSTREAM &UserParam, const ANNFEATUREUPDATEFUNC FeatureUpdateFunc);
			void iReleaseFeatureNode(const SString &strFeatureName);

			/* Layer operation function, the last layer user created is deemed to output layer */
			uint32 iCreateLayer(const uint32 NodeNumber);
			bool iResizeLayer(const uint32 LayerID, const uint32 NodeNumber);
			bool iInitializeLayer(const uint32 LayerID, const vector<vector<sfloat>> InitValue = vector<vector<sfloat>>(0));
			void iReleaseLayer(const uint32 LayerID);

			/* Learning param operation functions */
			bool iUpdateLearningRate(const sfloat Rate);
			bool iUpdateMomentumValue(const sfloat Momentum);
			bool iUpdateRandomBoundary(const sfloat RandomMinBoundary, const sfloat RandomMaxBoundary);

			sfloat iGetLearningRate() const;
			sfloat iGetMomentumValue() const;
			vector<sfloat> iGetRandomBoundary() const;

			/* Training the artificial neural network, this function will generate the layer history			*/
			/* and the layer history will continue exit unitl the function iPruning or iClearHistory be called	*/
			/* TrainingTimes: iteration times																	*/
			void iTraining(const ANNTRAININGSET &TrainingSet, const uint32 TrainingTimes = 1, const sfloat MinWeightChanges = 0.001, const sfloat MinErrorRate = 0.001, const bool bGenerateHistory = true);

			/* Training and pruning at the same time, this function will not generate any layer histiry			*/
			/* should be faster than iTraining function															*/
			void iTrainingWithValidation(const ANNTRAININGSET &TrainingSet, const ANNTRAININGSET &ValidationSet, const uint32 TrainingTimes = 1, const sfloat MinWeightChanges = 0.001, const sfloat MinErrorRate = 0.001);

			/* Predict the output with given input array, the input array size should same as the input layer unit size	*/
			void iPredict(const vector<sfloat> &InputArray, const int32 Index = -1);

			/*Predict, not change the network value, slower than iPredict*/
			vector<sfloat> iPredictC(const vector<sfloat> &InputArray, const int32 Index = -1) const;

			/* Predict the output with automatically generate the input array, the FeatureUpdateFunc should not be an nullptr	*/
			void iPredict(const int32 Index = -1);

			/* Update the hidden unit and output unit weight by training output array	*/
			/* the training output array size should equal with output layer size		*/
			pair<sfloat,sfloat> iFeedback(const vector<sfloat> &TrainingOutputArray);

			/* Pruning the artificial neural network with the given training set and validation set	*/
			void iPruning(const ANNTRAININGSET &ValidationSet);

			/* Reset all the hiddenlayer and output layer weight by random values(re-initialize)	*/
			void iReset();

			/* Clear the layer history	*/
			void iClearHistory();

			vector<sfloat> iGetOutputArray() const;
			SString iPrintNeuralNetwork() const;
		};
		//typedef cArtificialNeuralNetworkAlgorithm cANNAlgorithm;
#endif
	}
}