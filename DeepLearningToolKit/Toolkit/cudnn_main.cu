/*
* This code is released into the public domain.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
* OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*/
#include"map"
#include"readubyte.h"
#include"LeNet.cuh"

using namespace std;

// Application parameters
const int GPUID = 0;//The GPU ID to use
const int TrainingIteration = 5;//Number of TrainingIteration for training
const int RandomSeed = 0;//Override random seed (default uses random_device)
const int TestImageSize = -1;//Number of images to TestImageSize to compute error rate (default uses entire test set)

// Batch parameters
const int batch_size = 32;//Batch size for training

// Filenames
#define _MNIST
//#define _CIFAR10

#ifdef _MNIST
const string strTrainDataPath = "../Dataset/timg.bin";//Training images filename
const string strTrainLabelPath = "../Dataset/tlabel.bin";//Training labels filename
const string strTestDataPath = "../Dataset/test_img.bin";//Test images filename
const string strTestLabelPath = "../Dataset/test_label.bin";//Test labels filename
#endif

#ifdef _CIFAR10
const vector<string> TrainDataPath = { "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin" };//Training images filename
const vector<string> TestDataPath = { "test_batch.bin" };//Test images filename
#endif

// Solver parameters
const double LearningRate = 0.01;//Base learning rate
const double Gamma = 0.0001;//Learning rate policy gamma
const double Power = 0.75;//Learning rate policy power

_SAN_PAIR_DEF(RESULT_PAIR, float, Error, 0.0, float, Base, 0.0, , );

RESULT_PAIR Evaluate(TrainingContext &context, cDataLayer &DataLayer, const vector<float> &DataSet, const vector<float> &LabelSet, const size_t DataSetSize, const cDimension &DataSetShape, cFullyConnectedLayer &FC)
{
	RESULT_PAIR Result;

	const int BatchSize = DataSetShape.batches;
	const int BatchNumber = DataSetSize / DataSetShape.batches;
	const int PerBatchSize = DataSetShape.size;

	for (int seek_batch = 0; seek_batch < BatchNumber; seek_batch++)
	{
		int Offset = seek_batch * PerBatchSize;

		// Prepare current batch on device
		DataLayer.iOutput().iWrite(&DataSet[Offset], PerBatchSize, 0);

		// Forward propagate test image
		context.Forward(DataLayer);

		FC.iOutput().iSynchronize();

		for (int seek = 0; seek < BatchSize; seek = seek + 1)
		{
			// Determine classification according to maximal response
			auto pOutput = FC.iOutput().iGetPtr(HOST_MEM) + seek * 10;

			vector<float> Vec(10, 0.0);
			for (int ID = 0; ID < 10; ID = ID + 1) { Vec[ID] = pOutput[ID]; }

			int MaxID = 0;
			for (int ID = 1; ID < 10; ++ID)
			{
				MaxID = pOutput[MaxID] < pOutput[ID] ? ID : MaxID;
			}

			Result.Error = Result.Error + (MaxID == LabelSet[Result.Base + seek] ? 0 : 1);
		}

		Result.Base = Result.Base + BatchSize;
	}

	return Result;
}
int main(int argc, char **argv)
{
	cDimension Block(1, 1, 1);

	// Open input data
	std::printf("Reading input data\n");

	// Read dataset sizes
#ifdef _MNIST
	size_t train_size = MNISTDataSetLoader(strTrainDataPath, strTrainLabelPath, vector<float>(), vector<float>(), Block);
	size_t test_size = MNISTDataSetLoader(strTestDataPath, strTestLabelPath, vector<float>(), vector<float>(), Block);
#endif

#ifdef _CIFAR10
	size_t train_size = CIFAR10DataSetLoader(TrainDataPath, vector<float>(), vector<float>(), Block);
	size_t test_size = CIFAR10DataSetLoader(TestDataPath, vector<float>(), vector<float>(), Block);
#endif

	if (train_size == 0) { return 1; }

	Block.iUpdate();

	vector<float> TrainingImageSet(train_size * Block.size);
	vector<float> TrainingLabelSet(train_size);
	vector<float> TestImageSet(test_size * Block.size);
	vector<float> TestLabelSet(test_size);

	// Read data from datasets
#ifdef _MNIST
	if (MNISTDataSetLoader(strTrainDataPath, strTrainLabelPath, TrainingImageSet, TrainingLabelSet, Block) != train_size) { return 2; }
	if (MNISTDataSetLoader(strTestDataPath, strTestLabelPath, TestImageSet, TestLabelSet, Block) != test_size) { return 3; }
#endif

#ifdef _CIFAR10
	if (CIFAR10DataSetLoader(TrainDataPath, TrainingImageSet, TrainingLabelSet, Block) != train_size) { return 2; }
	if (CIFAR10DataSetLoader(TestDataPath, TestImageSet, TestLabelSet, Block) != test_size) { return 3; }
#endif

	std::printf("Done. Training dataset size: %d, Test dataset size: %d\n", (int) train_size, (int) test_size);
	std::printf("Batch size: %lld, TrainingIteration: %d\n", batch_size, TrainingIteration);


	// Choose GPU
	int num_GPUIDs;
	checkCudaErrors(cudaGetDeviceCount(&num_GPUIDs), LOCATION_STRING);

	if (GPUID < 0 || GPUID >= num_GPUIDs)
	{
		printf("ERROR: Invalid GPU ID %d (There are %d GPUs on this machine)\n", GPUID, num_GPUIDs);
		return 4;
	}

	Block.batches = batch_size;
	Block.iUpdate();

	// Create the LeNet network architecture
	cDataLayer DataLayer(Block);
	cConvLayer conv1(cDimension(5, 5, 1, 1), 20);
	cMaxPoolLayer pool1(cDimension(2, 2, 1));
	cConvLayer conv2(cDimension(5, 5, 1, 1), 50);
	cMaxPoolLayer pool2(cDimension(2, 2, 1));
	cFullyConnectedLayer fc1(500, cFullyConnectedLayer::FT_RELU);
	cFullyConnectedLayer fc2(10, cFullyConnectedLayer::FT_SOFTMAX);
	cLabelLayer LabelLayer(fc2);

	// Initialize CUDNN/CUBLAS training context
	TrainingContext context(GPUID);

	context.iPushLayer(&DataLayer);
	context.iPushLayer(&conv1);
	context.iPushLayer(&pool1);
	context.iPushLayer(&conv2);
	context.iPushLayer(&pool2);
	context.iPushLayer(&fc1);
	context.iPushLayer(&fc2);
	context.iPushLayer(&LabelLayer);

	// Create random network
	random_device RandomDevice;
	default_random_engine Engine(RandomSeed < 0 ? RandomDevice() : static_cast<unsigned int>(RandomSeed));

	context.iInitNetwork(Engine);

	// Forward propagation data
	cMemObj<float> LabelSet;
	LabelSet.iResize(cDimension(1, 1, 1, Block.batches));

	std::printf("Training...\n");

	// Use SGD to train the network
	checkCudaErrors(cudaDeviceSynchronize(), LOCATION_STRING);

	const int PerBatchSize = Block.size;
	const int BatchNumber = train_size / Block.batches;

	::cout << "Epoch\tPer Epoch TC\tPer Batch TC\tTraining Error\tTest Error\r\n";

	for (int seek_epoch = 0; seek_epoch < TrainingIteration; ++seek_epoch)
	{
		::cout << "#" << seek_epoch + 1 << "\t";

		const float CurrentLearningRate = static_cast<float>(LearningRate * pow((1.0 + Gamma * seek_epoch), (-Power)));

		auto StartTime = chrono::high_resolution_clock::now();

		for (int seek_batch = 0; seek_batch < BatchNumber; seek_batch = seek_batch + 1)
		{
			// Prepare current batch on device
			checkCudaErrors(cudaMemcpyAsync(DataLayer.iOutput().iGetPtr(), &TrainingImageSet[seek_batch * PerBatchSize], sizeof(float) * PerBatchSize, cudaMemcpyHostToDevice), LOCATION_STRING);
			checkCudaErrors(cudaMemcpyAsync(LabelSet.iGetPtr(), &TrainingLabelSet[seek_batch * Block.batches], sizeof(float) * Block.batches, cudaMemcpyHostToDevice), LOCATION_STRING);

			context.Forward(DataLayer);
			context.Backward(DataLayer, LabelLayer, LabelSet, CurrentLearningRate);
		}

		auto Cost = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - StartTime).count();

		RESULT_PAIR TrainRes = Evaluate(context, DataLayer, TrainingImageSet, TrainingLabelSet, train_size, Block, fc2);
		RESULT_PAIR TestRes = Evaluate(context, DataLayer, TestImageSet, TestLabelSet, test_size, Block, fc2);

		std::printf("%6.2f s\t%6.2f ms\t%.4f%%\t\t%.4f%%\n", Cost / 1000000.0f, (Cost / 1000.0f) / BatchNumber, (TrainRes.Error / TrainRes.Base) * 100.0f, (TestRes.Error / TestRes.Base) * 100.0f);
	}
	checkCudaErrors(cudaDeviceSynchronize(), LOCATION_STRING);

	const int TestSize = TestImageSize < 0 ? (int) test_size : TestImageSize;

	// Test the resulting neural network's classification
	if (TestSize > 0)
	{
		RESULT_PAIR Result = Evaluate(context, DataLayer, TestImageSet, TestLabelSet, TestSize, Block, fc2);
		std::printf("Test result: %.2f%% error [%d/%d]\r\n", (Result.Error / Result.Base) * 100.0f, static_cast<int>(Result.Error), static_cast<int>(Result.Base));
	}

	// Free data structures
	checkCudaErrors(cudaSetDevice(GPUID), LOCATION_STRING);

	::system("pause");

	return 0;
}
