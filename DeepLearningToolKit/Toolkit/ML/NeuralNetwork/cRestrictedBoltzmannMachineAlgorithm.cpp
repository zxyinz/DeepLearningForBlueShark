#include"cRestrictedBoltzmannMachineAlgorithm.h"
using namespace std;
using namespace San;
using namespace San::MachineLearning;
cRestrictedBoltzmannMachineAlgorithm::cRestrictedBoltzmannMachineAlgorithm(const uint32 VisibleLayerSize, const uint32 HiddenLayerSize, const sfloat MinBoundary, const sfloat MaxBoundary)
{
	this->iUpdateDBMState(VisibleLayerSize, HiddenLayerSize, MinBoundary, MaxBoundary);
}
cRestrictedBoltzmannMachineAlgorithm::~cRestrictedBoltzmannMachineAlgorithm()
{
}
bool cRestrictedBoltzmannMachineAlgorithm::_InitializeRBM(const _rbm_dataset &TrainingSet, _rengine &REngine)
{
	if (TrainingSet.empty())
	{
		return false;
	}

	int32 TrainingSetSize = TrainingSet.size();
	int32 VisibleLayerSize = this->m_VisibleLayerBias.size();
	int32 HiddenLayerSize = this->m_HiddenLayerBias.size();

	///*Calculate attribute probability for whole set*/
	//_sstream<sfloat> ProbabilityArray(InputArraySize);
	//for(int32 seek = 0; seek < InputArraySize; seek = seek + 1)
	//{
	//	ProbabilityArray[seek] = 0.0;
	//}
	//for(int32 seek = 0; seek < TrainingSetSize; seek = seek + 1)
	//{
	//	for(int32 seek_input = 0; seek_input < InputArraySize; seek_input = seek_input + 1)
	//	{
	//		ProbabilityArray[seek_input] = ProbabilityArray[seek_input] + this->_CalcSigmoidValue(TrainingSet[seek][seek_input]);
	//	}
	//}
	//for(int32 seek = 0; seek < InputArraySize; seek = seek + 1)
	//{
	//	ProbabilityArray[seek] = ProbabilityArray[seek] / (sfloat) TrainingSetSize;
	//}

	///*For Visible Bias, Initialize with log[Pi / (1 - Pi)]*/
	//uint32 VisibleLayerSize = InputArraySize;
	//for(int32 seek = 0; seek < VisibleLayerSize; seek = seek + 1)
	//{
	//	sfloat Base = (1.0 - ProbabilityArray[seek]);
	//	if (Base == 0)
	//	{
	//		Base = 0.0001;
	//	}
	//	pVisibleLayer->NodeArray[seek].VisibleBias = ::log(ProbabilityArray[seek] / Base);
	//}

	uniform_int_distribution<> Dist;

	/*For Visible Bias, Initialize with 0*/
	for(auto &Val : this->m_VisibleLayerBias){ Val = 0.0; }

	/*For Hidden Bias, Initialize with 0*/
	for(auto &Val : this->m_HiddenLayerBias) { Val = 0.0; }

	for (auto &Engine : this->m_VisibleLayerREngine){ Engine.seed(Dist(REngine)); }
	for (auto &Engine : this->m_HiddenLayerREngine) { Engine.seed(Dist(REngine)); }

	sfloat Base = 1.0 / (sfloat)this->m_VisibleLayerBias.size();

	/*Random Initialize Weight Array*/
	for(int32 seek = 0; seek < HiddenLayerSize; seek = seek + 1)
	{
		auto &WeightArray = this->m_WeightMatrix[seek];

		for(int32 seek_vis = 0; seek_vis < VisibleLayerSize; seek_vis = seek_vis + 1)
		{
			WeightArray[seek_vis] = this->_CalcUniformDistribution(-Base, Base, REngine);
		}
	}
	return true;
}
void cRestrictedBoltzmannMachineAlgorithm::iPrintRBM()
{
	::cout << "W\n";
	for (int32 i = 0; i < this->m_HiddenLayerBias.size(); i++)
	{
		for (int32 j = 0; j < this->m_VisibleLayerBias.size(); j++)
		{
			::cout << this->m_WeightMatrix[i][j] << " ";
		}
		::cout << "\n";
	}

	::cout << "H\n";
	for (int32 i = 0; i < this->m_HiddenLayerBias.size(); i++)
	{
		::cout << this->m_HiddenLayerBias[i] << " ";
	}
	::cout << "\n";

	::cout << "V\n";
	for (int32 j = 0; j < this->m_VisibleLayerBias.size(); j++)
	{
		::cout << this->m_VisibleLayerBias[j] << " ";
	}
	::cout << "\n";
}
void cRestrictedBoltzmannMachineAlgorithm::iTrainingRBM(const _rbm_dataset &TrainingSet, _rbm_dataset &OutputSet, const uint32 TrainingTimes, const uint32 KTimes, const sfloat LearningRate, _rengine &REngine, const sfloat MinWeightChanges, const sfloat MinErrorChanges)
{
	int32 TrainingSetSize = TrainingSet.size();
	int32 VisibleLayerSize = this->m_VisibleLayerBias.size();
	int32 HiddenLayerSize = this->m_HiddenLayerBias.size();

	this->_InitializeRBM(TrainingSet, REngine);

	_sstream<sfloat> HiddenProbability(HiddenLayerSize, 0.0);
	_sstream<sfloat> HiddenSample(HiddenLayerSize, 0.0);

	_sstream<sfloat> NewVisibleProbability(VisibleLayerSize, 0.0);
	_sstream<sfloat> NewVisibleSample(VisibleLayerSize, 0.0);

	_sstream<sfloat> NewHiddenProbability(HiddenLayerSize, 0.0);
	_sstream<sfloat> NewHiddenSample(HiddenLayerSize, 0.0);

	sfloat PrevError = MAX_FLOAT32_VALUE;

	for(int32 seek_itaration = 0; seek_itaration < TrainingTimes; seek_itaration = seek_itaration + 1)
	{
		/*Update Weight*/
		sfloat WeightChanges = 0.0;
		sfloat Error = 0.0;

		for(int32 seek_instance = 0; seek_instance < TrainingSetSize; seek_instance = seek_instance + 1)
		{
			const auto &Instance = TrainingSet[seek_instance];
			/*Contrastive Divergence*/

			this->_SampleHV(Instance.Data, HiddenProbability, HiddenSample, VisibleLayerSize);

			/*Gibbs Sample*/
			if (KTimes > 0) // seek == 0
			{
				this->_SampleVH(HiddenSample, NewVisibleProbability, NewVisibleSample, HiddenLayerSize);
				this->_SampleHV(NewVisibleSample, NewHiddenProbability, NewHiddenSample, VisibleLayerSize);
			}

			for(int32 seek = 1; seek < KTimes; seek = seek + 1)
			{
				this->_SampleVH(NewHiddenSample, NewVisibleProbability, NewVisibleSample, HiddenLayerSize);
				this->_SampleHV(NewVisibleSample, NewHiddenProbability, NewHiddenSample, VisibleLayerSize);
			}

			for(int32 seek = 0; seek < HiddenLayerSize; seek = seek + 1)
			{
				auto &WeightArray = this->m_WeightMatrix[seek];

				const sfloat HiddenProb = HiddenProbability[seek];
				const sfloat NewHiddenProb = NewHiddenProbability[seek];

				for(int32 seek_visible = 0; seek_visible < VisibleLayerSize; seek_visible = seek_visible + 1)
				{
					const sfloat DeltaWeight = (HiddenProb * Instance.Data[seek_visible] - NewHiddenProb * NewVisibleSample[seek_visible]) / (sfloat) TrainingSetSize;
					WeightArray[seek_visible] = WeightArray[seek_visible] + DeltaWeight * LearningRate;
				}

				const sfloat CurrentError = (HiddenSample[seek] - NewHiddenProbability[seek]) / (sfloat) TrainingSetSize;
				const sfloat Delta = LearningRate * CurrentError;

				this->m_HiddenLayerBias[seek] = this->m_HiddenLayerBias[seek] + Delta;

				WeightChanges = WeightChanges + abs(Delta);
				Error = Error + abs(CurrentError);
			}

			for(int32 seek = 0; seek < VisibleLayerSize; seek = seek + 1)
			{
				const sfloat CurrentError = (Instance.Data[seek] - NewVisibleSample[seek]) / (sfloat) TrainingSetSize;
				const sfloat Delta = LearningRate * CurrentError;

				this->m_VisibleLayerBias[seek] = this->m_VisibleLayerBias[seek] + Delta;

				Error = Error + abs(CurrentError);
			}
		}

		//::cout << "Iteration: " << seek_itaration << "\t Weight Changes: " << WeightChanges << "\t Error Changes: " << abs(PrevError - Error) << "\r\n";

		//Stop threshold

		if (WeightChanges <= MinWeightChanges){ break; }
		if (abs(PrevError - Error) < MinErrorChanges) { break; }

		PrevError = Error;
	}

	//this->iPrintRBM();

	/*Encode Output Set For Next RBM Training*/
	uniform_int_distribution<> Dist; //For parallel
	for (int32 seek_instance = 0; seek_instance < TrainingSetSize; seek_instance = seek_instance + 1)
	{
		OutputSet[seek_instance].Label = Dist(REngine);
	}

	const _rbm_dataset* pTrainingSet = &TrainingSet;
	_rbm_dataset* pOutputSet = &OutputSet;
	_sstream<sfloat>* pHiddenSample = &HiddenSample;

#pragma omp parallel firstprivate(pTrainingSet, pOutputSet, pHiddenSample, VisibleLayerSize)
	{
#pragma omp for nowait
		for (int32 seek_instance = 0; seek_instance < TrainingSetSize; seek_instance = seek_instance + 1)
		{
			_rengine CurrentEngine((*pOutputSet)[seek_instance].Label);
			this->_SampleHV((*pTrainingSet)[seek_instance].Data, (*pOutputSet)[seek_instance].Data, *pHiddenSample, VisibleLayerSize, CurrentEngine);
		}
	}
}
void cRestrictedBoltzmannMachineAlgorithm::_SampleHV(const _sstream<sfloat> &Sample, _sstream<sfloat> &HiddenNodeProbability, _sstream<sfloat> &HiddenNodeSample, const int32 VisibleLayerSize, _rengine &REngine)
{
	const int32 HiddenLayerSize = this->m_HiddenLayerBias.size();

	for(int32 seek = 0; seek < HiddenLayerSize; seek = seek + 1)
	{
		HiddenNodeProbability[seek] = this->_CalcProbabilityUp(seek, Sample, VisibleLayerSize);
		HiddenNodeSample[seek] = this->_CalcBinomialValue(1, HiddenNodeProbability[seek], REngine);
	}
}
void cRestrictedBoltzmannMachineAlgorithm::_SampleVH(const _sstream<sfloat> &Sample, _sstream<sfloat> &VisibleNodeProbability, _sstream<sfloat> &VisibleNodeSample, const int32 HiddenLayerSize, _rengine &REngine)
{
	const int32 VisibleLayerSize = this->m_VisibleLayerBias.size();

	for(int32 seek = 0; seek < VisibleLayerSize; seek = seek + 1)
	{
		VisibleNodeProbability[seek] = this->_CalcProbabilityDown(seek, Sample, HiddenLayerSize);
		VisibleNodeSample[seek] = this->_CalcBinomialValue(1, VisibleNodeProbability[seek], REngine);
	}
}
void cRestrictedBoltzmannMachineAlgorithm::_SampleHV(const _sstream<sfloat> &Sample, _sstream<sfloat> &HiddenNodeProbability, _sstream<sfloat> &HiddenNodeSample, const int32 VisibleLayerSize)
{
	const int32 HiddenLayerSize = this->m_HiddenLayerBias.size();

	const _sstream<sfloat>* pSample = &Sample;
	sfloat* pHiddenProb = HiddenNodeProbability.iGetPtr();
	sfloat* pHiddenSample = HiddenNodeSample.iGetPtr();
	_rengine* pREngine = this->m_HiddenLayerREngine.iGetPtr();

#pragma omp parallel firstprivate(pSample, pHiddenProb, pHiddenSample, pREngine)
	{
#pragma omp for nowait
		for (int32 seek = 0; seek < HiddenLayerSize; seek = seek + 1)
		{
			pHiddenProb[seek] = this->_CalcProbabilityUp(seek, *pSample, VisibleLayerSize);
			pHiddenSample[seek] = this->_CalcBinomialValue(1, pHiddenProb[seek], pREngine[seek]);
		}
	}
}
void cRestrictedBoltzmannMachineAlgorithm::_SampleVH(const _sstream<sfloat> &Sample, _sstream<sfloat> &VisibleNodeProbability, _sstream<sfloat> &VisibleNodeSample, const int32 HiddenLayerSiz)
{
	const int32 VisibleLayerSize = this->m_VisibleLayerBias.size();

	const _sstream<sfloat>* pSample = &Sample;
	sfloat* pVisibleProb = VisibleNodeProbability.iGetPtr();
	sfloat* pVisibleSample = VisibleNodeSample.iGetPtr();
	_rengine* pREngine = this->m_VisibleLayerREngine.iGetPtr();

#pragma omp parallel firstprivate(pSample, pVisibleProb, pVisibleSample, pREngine)
	{
#pragma omp for nowait
		for (int32 seek = 0; seek < VisibleLayerSize; seek = seek + 1)
		{
			pVisibleProb[seek] = this->_CalcProbabilityDown(seek, (*pSample), HiddenLayerSiz);
			pVisibleSample[seek] = this->_CalcBinomialValue(1, pVisibleProb[seek], pREngine[seek]);
		}
	}
}
sfloat cRestrictedBoltzmannMachineAlgorithm::_CalcProbabilityUp(const uint32 HiddenIndex, const _sstream<sfloat> &Sample, const int32 VisibleLayerSize) const
{
	/*P(Hi | v£© = sig(sum(Vj * Wij) + Bi)*/
	const auto &WeightArray = this->m_WeightMatrix[HiddenIndex];

	sfloat Probability = 0.0;

	for (int32 seek = 0; seek < VisibleLayerSize; seek = seek + 1)
	{
		Probability = Probability + Sample[seek] * WeightArray[seek];
	}

	Probability = Probability + this->m_HiddenLayerBias[HiddenIndex];
	return this->_CalcSigmoidValue(Probability);
}
sfloat cRestrictedBoltzmannMachineAlgorithm::_CalcProbabilityDown(const uint32 VisibleIndex, const _sstream<sfloat> &Sample, const int32 HiddenLayerSize) const
{
	/*P(Vi | h) = sig(sum(Hj*Wij) + Ai)*/
	sfloat Probability = 0.0;

	for (int32 seek = 0; seek < HiddenLayerSize; seek = seek + 1)
	{
		Probability = Probability + Sample[seek] * this->m_WeightMatrix[seek][VisibleIndex];
	}

	Probability = Probability + this->m_VisibleLayerBias[VisibleIndex];
	return this->_CalcSigmoidValue(Probability);
}
sfloat cRestrictedBoltzmannMachineAlgorithm::_CalcBinomialValue(const sfloat N, const sfloat Probability, _rengine &REngine) const
{
	if ((Probability<0.0) || (Probability>1.0))
	{
		return 0.0;
	}

	::uniform_real_distribution<> Dist(0.0, 1.0);
	sfloat Res = 0;

	for(int32 seek = 0; seek < N; seek = seek + 1)
	{
		const sfloat RandVal = Dist(REngine);

		if (RandVal < Probability)
		{
			Res = Res + 1.0;
		}
	}
	return Res;
}
sfloat cRestrictedBoltzmannMachineAlgorithm::_CalcSigmoidValue(const sfloat Net) const
{
	return 1.0 / (1.0 + ::exp(-Net));
}
sfloat cRestrictedBoltzmannMachineAlgorithm::_CalcUniformDistribution(const sfloat MinBoundary, const sfloat MaxBoundary, _rengine &REngine) const
{
	::uniform_real_distribution<> Dist(MinBoundary, MaxBoundary);

	return Dist(REngine);

	for (int32 seek = 0; seek < 32; seek = seek + 1)
	{
		const sfloat RandVal = Dist(REngine);
		if (!::gloIsFloatEqual(RandVal, 0.0, 0.0001))
		{
			return RandVal;
		}
	}

	_SDEBUG(::cout << "Error: can not generate target random value, file: cRBM.cpp, func: _CalcUniformDistribution\r\n";)

	return 0.0;
}
bool cRestrictedBoltzmannMachineAlgorithm::iUpdateDBMState(const uint32 VisibleLayerSize, const uint32 HiddenLayerSize, const sfloat MinBoundary, const sfloat MaxBoundary)
{
	if ((VisibleLayerSize==0)||(HiddenLayerSize==0))
	{
		return false;
	}
	if (MinBoundary >= MaxBoundary)
	{
		return false;
	}

	this->m_VisibleLayerBias.resize(VisibleLayerSize);
	this->m_HiddenLayerBias.resize(HiddenLayerSize);
	this->m_VisibleLayerREngine.resize(VisibleLayerSize);
	this->m_HiddenLayerREngine.resize(HiddenLayerSize);
	//this->m_WeightMatrix.iSetDimension(_matrix<sfloat>::_dimension(VisibleLayerSize, HiddenLayerSize));
	this->m_WeightMatrix.resize(HiddenLayerSize);
	for (int32 seek = 0; seek < HiddenLayerSize; seek = seek + 1)
	{
		this->m_WeightMatrix[seek].resize(VisibleLayerSize);
	}

	this->m_RandomBoundary[0] = MinBoundary;
	this->m_RandomBoundary[1] = MaxBoundary;

	return true;
}
const vector<vector<sfloat>>& cRestrictedBoltzmannMachineAlgorithm::iGetWeightMatrix() const
{
	return this->m_WeightMatrix;
}
vector<vector<sfloat>>& cRestrictedBoltzmannMachineAlgorithm::iGetWeightMatrix()
{
	return this->m_WeightMatrix;
}
void cRestrictedBoltzmannMachineAlgorithm::iReconstruct(_sstream<sfloat> &VisibleSample, _sstream<sfloat> &ReconstructResult)
{
	int32 VisibleLayerSize = this->m_VisibleLayerBias.size();
	int32 HiddenLayerSize = this->m_HiddenLayerBias.size();

	if (ReconstructResult.size() != VisibleLayerSize)
	{
		ReconstructResult.resize(VisibleLayerSize);
	}

	_sstream<sfloat> HiddenProbability(HiddenLayerSize);

	/*Calculate Hidden Layer Probability*/
	for(int32 seek = 0; seek < HiddenLayerSize; seek = seek + 1)
	{
		HiddenProbability[seek] = this->_CalcProbabilityUp(seek, VisibleSample, VisibleLayerSize);
	}

	for(int32 seek = 0; seek < VisibleLayerSize; seek = seek + 1)
	{
		sfloat SigVal = 0.0;
		for(int32 seek_hidden = 0; seek_hidden < HiddenLayerSize; seek_hidden = seek_hidden + 1)
		{
			SigVal = SigVal + this->m_WeightMatrix[seek_hidden][seek] * HiddenProbability[seek_hidden];
		}

		SigVal = SigVal + this->m_VisibleLayerBias[seek];
		ReconstructResult[seek] = this->_CalcSigmoidValue(SigVal);
	}
}