#include"cNeuralNode.h"
#pragma once
using namespace std;
using namespace San::Mathematics;
namespace San
{
	namespace MachineLearning
	{
#ifndef __CCONVNODE_H__
#define __CCONVNODE_H__
		class cConvNode : public cNeuralNode
		{
		public:
			typedef _tensor_ref<sfloat>::_dimension _dimension;
		protected:
			_tensor_ref<sfloat> m_WeightTensorRef;
			_tensor_ref<sfloat> m_ValueTensorRef;
			_dimension m_Dimension;
			sfloat m_Delta;
			sfloat m_PrevDelta;
		public:
			_tensor<sfloat> m_LinearOutput;
			_tensor<sfloat> m_FunctionOutput;
			_tensor<sfloat> m_PredictError;
		protected:
			void _CreateNode(const cNeuralNode* pNode) override;
			void _ReleaseNode() override;
		public:
			cConvNode(const _dimension Dimension);
			cConvNode(const cConvNode &Node);
			~cConvNode();

			/*No size check, may cause error*/
			sfloat iPredict(const _tensor_ref<sfloat> &Input, _tensor_ref<sfloat>* pOutput) override;
			sfloat iPredict(const _tensor_ref<sfloat> &Input, _tensor_ref<sfloat>* pOutput) const override;

			sfloat iFeedback(const sfloat &Delta, const sfloat LearningRate, const sfloat Momentum = 0.0) override;

			cNeuralNode* iCopy() const override;
		};
#endif
	}
}