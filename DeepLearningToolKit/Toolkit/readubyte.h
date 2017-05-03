#include"cDimension.h"
#pragma once
using namespace std;

#ifndef __CUDNN_TRAINING_READUBYTE_H
#define __CUDNN_TRAINING_READUBYTE_H

size_t MNISTDataSetLoader(const string &strImgFilePath, const string &strLabelFilePath, vector<float> &DataSet, vector<float> &LabelSet, cDimension &cDimension);
size_t CIFAR10DataSetLoader(const vector<string> &FilePathSet, vector<float> &DataSet, vector<float> &LabelSet, cDimension &cDimension);

#endif  // __CUDNN_TRAINING_READUBYTE_H
