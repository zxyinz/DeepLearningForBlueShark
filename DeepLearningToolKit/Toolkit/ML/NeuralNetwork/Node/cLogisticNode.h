#include"cSigmoidNode.h"
#pragma once
using namespace std;
using namespace San::Mathematics;
namespace San
{
	namespace MachineLearning
	{
#ifndef __CLOGISTICNODE_H__
#define __CLOGISTICNODE_H__
		class cLogisticFunc
		{
		public:
			static sfloat iForward(const sfloat &Val);
			static sfloat iBackward(const sfloat &PredictVal, const sfloat &Error);
		};

		class cLogisticNode : public cSigmoidNode, protected cLogisticFunc, public _smemobj<>
		{
		protected:
		public:
			cLogisticNode(const uint32 Size = 0);
			cLogisticNode(const cSigmoidNode &Node);
			~cLogisticNode();

			sfloat iCalcFuncValue(const sfloat &Net) const override;
			sfloat iCalcDeltaValue(const _tensor_ref<sfloat> &Error) const override;

			cNeuralNode* iCopy() const override;
		};
#endif
	}
}