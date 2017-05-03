#include"cNeuralNode.h"
#pragma once
using namespace std;
using namespace San::Mathematics;
namespace San
{
	namespace MachineLearning
	{
#ifndef __CSIGMOIDNODE_H__
#define __CSIGMOIDNODE_H__
		class cSigmoidNode : public cNeuralNode, public _smemobj<>
		{
		protected:
			sfloat m_Delta;
			sfloat m_PrevDelta;
		public:
			sfloat m_LinearOutput;
			sfloat m_FunctionOutput;
			sfloat m_PredictError;
		protected:
			void _CreateNode(const cNeuralNode* pNode) override;
			void _ReleaseNode() override;
		public:
			cSigmoidNode(const uint32 Size = 0);
			cSigmoidNode(const cSigmoidNode &Node);
			~cSigmoidNode();

			/*No size check, may cause error*/
			sfloat iPredict(const _tensor_ref<sfloat> &Input, _tensor_ref<sfloat>* pOutput) override;
			sfloat iPredict(const _tensor_ref<sfloat> &Input, _tensor_ref<sfloat>* pOutput) const override;

			sfloat iFeedback(const sfloat &Delta, const sfloat LearningRate, const sfloat Momentum = 0.0) override;

			cNeuralNode* iCopy() const override;
		};
#endif
	}
}