#include"../../../Core/SanTypes.h"
#include"../../../Core/SanMathematics.h"
#include"../../cDataSet.h"
#pragma once
using namespace std;
using namespace San::Mathematics;
namespace San
{
	namespace MachineLearning
	{
#ifndef __CNEURALNODE_H__
#define __CNEURALNODE_H__
		class cNeuralNode : public _smemobj<>
		{
		public:
			typedef vector<sfloat>::iterator iterator;
			typedef vector<sfloat>::const_iterator const_iterator;

			typedef default_random_engine _rengine;
		protected:
			_sstream<sfloat> m_WeightArray;
			_sstream<sfloat> m_ValueArray;
			sfloat m_BiasWeight;
		protected:
			virtual void _CreateNode(const cNeuralNode* pNode);
			virtual void _ReleaseNode();
			//virtual void _Copy(cNeuralNode* pNode) const;

			virtual sfloat _GetRandomValue(_rengine &RandomEngine, const sfloat &LowerBound, const sfloat &UpperBound) const;
		public:
			cNeuralNode(const uint32 Size = 0);
			cNeuralNode(const cNeuralNode &Node);
			~cNeuralNode();

			const sfloat& operator[](const uint32 Index) const;
			sfloat& operator[](const uint32 Index);

			/*No size check, may cause error*/
			virtual sfloat iPredict(const _tensor_ref<sfloat> &Input, _tensor_ref<sfloat>* pOutput);
			virtual sfloat iPredict(const _tensor_ref<sfloat> &Input, _tensor_ref<sfloat>* pOutput) const;

			virtual sfloat iFeedback(const sfloat &Delta, const sfloat LearningRate, const sfloat Momentum = 0.0);

			virtual sfloat iCalcFuncValue(const sfloat &Net) const;
			virtual sfloat iCalcDeltaValue(const _tensor_ref<sfloat> &Error) const;

			virtual uint32 iResize(const uint32 Size, _rengine &RandomEngine, const sfloat &LowerBound = -0.05, const sfloat &UpperBound = 0.05);
			virtual uint32 iReset(_rengine &RandomEngine, const sfloat &LowerBound = -0.05, const sfloat &UpperBound = 0.05);

			virtual cNeuralNode* iCopy() const;

			/*STL function*/
			/*const_iterator cbegin() const;
			const_iterator cend() const;

			const_iterator begin() const;
			iterator begin();
			const_iterator end() const;
			iterator end();*/

			uint32 size() const;
			bool empty() const;
		};
#endif
	}
}