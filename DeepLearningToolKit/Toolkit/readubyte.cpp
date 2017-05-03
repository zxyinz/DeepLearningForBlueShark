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
#include"readubyte.h"
using namespace std;

const unsigned int UBYTE_IMAGE_MAGIC = 2051;
const unsigned int  UBYTE_LABEL_MAGIC = 2049;

#ifdef _MSC_VER
    #define bswap(x) _byteswap_ulong(x)
#else
    #define bswap(x) __builtin_bswap32(x)
#endif

#pragma pack(push, 1)
struct UByteImageDataset 
{
    /// Magic number (UBYTE_IMAGE_MAGIC).
    uint32_t magic;

    /// Number of images in dataset.
    uint32_t length;

    /// The height of each image.
    uint32_t height;

    /// The width of each image.
    uint32_t width;

    void Swap()
    {
        magic = bswap(magic);
        length = bswap(length);
        height = bswap(height);
        width = bswap(width);
    }
};

struct UByteLabelDataset
{
    /// Magic number (UBYTE_LABEL_MAGIC).
    uint32_t magic;

    /// Number of labels in dataset.
    uint32_t length;

    void Swap()
    {
        magic = bswap(magic);
        length = bswap(length);
    }
};
#pragma pack(pop)

size_t MNISTDataSetLoader(const string &strImgFilePath, const string &strLabelFilePath, vector<float> &DataSet, vector<float> &LabelSet, cDimension &cDimension)
{
    FILE *imfp = fopen(strImgFilePath.c_str(), "rb");
    if (!imfp)
    {
        printf("ERROR: Cannot open image dataset %s\n", strImgFilePath);
        return 0;
    }
    FILE *lbfp = fopen(strLabelFilePath.c_str(), "rb");
    if (!lbfp)
    {
        fclose(imfp);
        printf("ERROR: Cannot open label dataset %s\n", strLabelFilePath);
        return 0;
    }

    UByteImageDataset image_header;
    UByteLabelDataset label_header;
    
    // Read and verify file headers
    if (fread(&image_header, sizeof(UByteImageDataset), 1, imfp) != 1)
    {
        printf("ERROR: Invalid dataset file (image file header)\n");
        fclose(imfp);
        fclose(lbfp);
        return 0;
    }
    if (fread(&label_header, sizeof(UByteLabelDataset), 1, lbfp) != 1)
    {
        printf("ERROR: Invalid dataset file (label file header)\n");
        fclose(imfp);
        fclose(lbfp);
        return 0;
    }

    // Byte-swap data structure values (change endianness)
    image_header.Swap();
    label_header.Swap();

    // Verify datasets
    if (image_header.magic != UBYTE_IMAGE_MAGIC)
    {
        printf("ERROR: Invalid dataset file (image file magic number)\n");
        fclose(imfp);
        fclose(lbfp);
        return 0;
    }
    if (label_header.magic != UBYTE_LABEL_MAGIC)
    {
        printf("ERROR: Invalid dataset file (label file magic number)\n");
        fclose(imfp);
        fclose(lbfp);
        return 0;
    }
    if (image_header.length != label_header.length)
    {
        printf("ERROR: Dataset file mismatch (number of images does not match the number of labels)\n");
        fclose(imfp);
        fclose(lbfp);
        return 0;
    }
    
    // Output dimensions
    cDimension.width = image_header.width;
	cDimension.height = image_header.height;
	cDimension.channels = 1;

	cDimension.iUpdate();

    // Read images and labels (if requested)
	if (!DataSet.empty())
    {
		vector<unsigned char> Buffer(DataSet.size());

		if (fread(&Buffer[0], sizeof(uint8_t), Buffer.size(), imfp) != Buffer.size())
        {
            printf("ERROR: Invalid dataset file (partial image dataset)\n");
            fclose(imfp);
            fclose(lbfp);
            return 0;
        }

		for (int seek = 0; seek < Buffer.size(); seek = seek + 1)
		{
			DataSet[seek] = (float) Buffer[seek] / 255.0;
		}
    }
    if (!LabelSet.empty())
    {
		vector<unsigned char> Buffer(LabelSet.size());

		if (fread(&Buffer[0], sizeof(uint8_t), label_header.length, lbfp) != label_header.length)
        {
            printf("ERROR: Invalid dataset file (partial label dataset)\n");
            fclose(imfp);
            fclose(lbfp);
            return 0;
        }

		for (int seek = 0; seek < Buffer.size(); seek = seek + 1)
		{
			LabelSet[seek] = Buffer[seek];
		}
    }
    
    fclose(imfp);
    fclose(lbfp);

    return image_header.length;
}
size_t CIFAR10DataSetLoader(const vector<string> &FilePathSet, vector<float> &DataSet, vector<float> &LabelSet, cDimension &cDimension)
{
	const size_t PerInstanceSize = 32 * 32 * 3 + 1;

	size_t InstanceCount = 0;
	size_t ImageOffset = 0;
	size_t LabelOffset = 0;

	// Output dimensions
	cDimension.width = 32;
	cDimension.height = 32;
	cDimension.channels = 1;

	cDimension.iUpdate();

	for (const auto strPath : FilePathSet)
	{
		fstream* pFile = new fstream(strPath, ios::binary | ios::in, 0x40);

		if (pFile == nullptr)
		{ 
			printf("ERROR: Cannot open image dataset %s\n", strPath);
			
			pFile->close();
			delete pFile;
			pFile = nullptr;

			continue;
		}

		pFile->seekg(0, ios::end);

		const size_t FileSize = pFile->tellg();
		pFile->clear();
		pFile->seekg(0, ios::beg);

		if ((FileSize <= 0) || (((unsigned int) FileSize % PerInstanceSize) != 0))
		{ 
			printf("ERROR: Invalid file size %s\n", strPath);

			pFile->close();
			delete pFile;
			pFile = nullptr;

			continue;
		}

		const size_t CurrentInstanceSize = (unsigned int) FileSize / PerInstanceSize;
		InstanceCount = InstanceCount + CurrentInstanceSize;

		// Read images and labels (if requested)
		if ((!DataSet.empty())&&(!LabelSet.empty()))
		{
			vector<char> Buffer(FileSize);
			pFile->read(&Buffer[0], FileSize);

			for (int seek_instance = 0; seek_instance < CurrentInstanceSize; seek_instance = seek_instance + 1)
			{
				size_t Index = seek_instance * (cDimension.size * 3 + 1);
				float* pData = &DataSet[ImageOffset];

				for (int seek = 0; seek < cDimension.resolution; seek = seek + 1) { pData[seek] = 0.0; }

				LabelSet[LabelOffset] = Buffer[Index];
				LabelOffset = LabelOffset + 1;

				Index = Index + 1;

				//R
				for (int seek = 0; seek < cDimension.resolution; seek = seek + 1)
				{
					const float Channel = (float) Buffer[Index + seek] / 255.0;
					pData[seek] = pData[seek] + Channel * 0.21;
				}
				Index = Index + cDimension.resolution;

				//G
				for (int seek = 0; seek < cDimension.resolution; seek = seek + 1)
				{
					const float Channel = (float) Buffer[Index + seek] / 255.0;
					pData[seek] = pData[seek] + Channel * 0.71;
				}
				Index = Index + cDimension.resolution;

				//B
				for (int seek = 0; seek < cDimension.resolution; seek = seek + 1)
				{
					const float Channel = (float) Buffer[Index + seek] / 255.0;
					pData[seek] = pData[seek] + Channel * 0.08;
				}
				Index = Index + cDimension.resolution;

				ImageOffset = ImageOffset + cDimension.size;
			}
		}

		pFile->close();
		delete pFile;
		pFile = nullptr;
	}

	return InstanceCount;
}