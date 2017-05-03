#include"iostream"
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include"cublas_v2.h"
#include"cudnn.h"
#pragma comment(lib,"cudnn.lib")
#pragma comment(lib,"cublas.lib")
#pragma once
using namespace std;
#ifndef __DEBUGDEF_H__
#define __DEBUGDEF_H__

#define LOCATION_STRING __FILE__, __LINE__

inline string ToString(const cudaError &status)
{
	switch (status)
	{
	case cudaSuccess: return "cudaSuccess"; break;
	case cudaErrorMissingConfiguration: return "cudaErrorMissingConfiguration"; break;
	case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation"; break;
	case cudaErrorInitializationError: return "cudaErrorInitializationError"; break;
	case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure"; break;
	case cudaErrorPriorLaunchFailure: return "cudaErrorPriorLaunchFailure"; break;
	case cudaErrorLaunchTimeout: return "cudaErrorLaunchTimeout"; break;
	case cudaErrorLaunchOutOfResources: return "cudaErrorLaunchOutOfResources"; break;
	case cudaErrorInvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction"; break;
	case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration"; break;
	case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice"; break;
	case cudaErrorInvalidValue: return "cudaErrorInvalidValue"; break;
	case cudaErrorInvalidPitchValue: return "cudaErrorInvalidPitchValue"; break;
	case cudaErrorInvalidSymbol: return "cudaErrorInvalidSymbol"; break;
	case cudaErrorMapBufferObjectFailed: return "cudaErrorMapBufferObjectFailed"; break;
	case cudaErrorUnmapBufferObjectFailed: return "cudaErrorUnmapBufferObjectFailed"; break;
	case cudaErrorInvalidHostPointer: return "cudaErrorInvalidHostPointer"; break;
	case cudaErrorInvalidDevicePointer: return "cudaErrorInvalidDevicePointer"; break;
	case cudaErrorInvalidTexture: return "cudaErrorInvalidTexture"; break;
	case cudaErrorInvalidTextureBinding: return "cudaErrorInvalidTextureBinding"; break;
	case cudaErrorInvalidChannelDescriptor: return "cudaErrorInvalidChannelDescriptor"; break;
	case cudaErrorInvalidMemcpyDirection: return "cudaErrorInvalidMemcpyDirection"; break;
	case cudaErrorAddressOfConstant: return "cudaErrorAddressOfConstant"; break;
	case cudaErrorTextureFetchFailed: return "cudaErrorTextureFetchFailed"; break;
	case cudaErrorTextureNotBound: return "cudaErrorTextureNotBound"; break;
	case cudaErrorSynchronizationError: return "cudaErrorSynchronizationError"; break;
	case cudaErrorInvalidFilterSetting: return "cudaErrorInvalidFilterSetting"; break;
	case cudaErrorInvalidNormSetting: return "cudaErrorInvalidNormSetting"; break;
	case cudaErrorMixedDeviceExecution: return "cudaErrorMixedDeviceExecution"; break;
	case cudaErrorCudartUnloading: return "cudaErrorCudartUnloading"; break;
	case cudaErrorUnknown: return "cudaErrorUnknown"; break;
	case cudaErrorNotYetImplemented: return "cudaErrorNotYetImplemented"; break;
	case cudaErrorMemoryValueTooLarge: return "cudaErrorMemoryValueTooLarge"; break;
	case cudaErrorInvalidResourceHandle: return "cudaErrorInvalidResourceHandle"; break;
	case cudaErrorNotReady: return "cudaErrorNotReady"; break;
	case cudaErrorInsufficientDriver: return "cudaErrorInsufficientDriver"; break;
	case cudaErrorSetOnActiveProcess: return "cudaErrorSetOnActiveProcess"; break;
	case cudaErrorInvalidSurface: return "cudaErrorInvalidSurface"; break;
	case cudaErrorNoDevice: return "cudaErrorNoDevice"; break;
	case cudaErrorECCUncorrectable: return "cudaErrorECCUncorrectable"; break;
	case cudaErrorSharedObjectSymbolNotFound: return "cudaErrorSharedObjectSymbolNotFound"; break;
	case cudaErrorSharedObjectInitFailed: return "cudaErrorSharedObjectInitFailed"; break;
	case cudaErrorUnsupportedLimit: return "cudaErrorUnsupportedLimit"; break;
	case cudaErrorDuplicateVariableName: return "cudaErrorDuplicateVariableName"; break;
	case cudaErrorDuplicateTextureName: return "cudaErrorDuplicateTextureName"; break;
	case cudaErrorDuplicateSurfaceName: return "cudaErrorDuplicateSurfaceName"; break;
	case cudaErrorDevicesUnavailable: return "cudaErrorDevicesUnavailable"; break;
	case cudaErrorInvalidKernelImage: return "cudaErrorInvalidKernelImage"; break;
	case cudaErrorNoKernelImageForDevice: return "cudaErrorNoKernelImageForDevice"; break;
	case cudaErrorIncompatibleDriverContext: return "cudaErrorIncompatibleDriverContext"; break;
	case cudaErrorPeerAccessAlreadyEnabled: return "cudaErrorPeerAccessAlreadyEnabled"; break;
	case cudaErrorPeerAccessNotEnabled: return "cudaErrorPeerAccessNotEnabled"; break;
	case cudaErrorDeviceAlreadyInUse: return "cudaErrorDeviceAlreadyInUse"; break;
	case cudaErrorProfilerDisabled: return "cudaErrorProfilerDisabled"; break;
	case cudaErrorProfilerNotInitialized: return "cudaErrorProfilerNotInitialized"; break;
	case cudaErrorProfilerAlreadyStarted: return "cudaErrorProfilerAlreadyStarted"; break;
	case cudaErrorProfilerAlreadyStopped: return "cudaErrorProfilerAlreadyStopped"; break;
	case cudaErrorAssert: return "cudaErrorAssert"; break;
	case cudaErrorTooManyPeers: return "cudaErrorTooManyPeers"; break;
	case cudaErrorHostMemoryAlreadyRegistered: return "cudaErrorHostMemoryAlreadyRegistered"; break;
	case cudaErrorHostMemoryNotRegistered: return "cudaErrorHostMemoryNotRegistered"; break;
	case cudaErrorOperatingSystem: return "cudaErrorOperatingSystem"; break;
	case cudaErrorPeerAccessUnsupported: return "cudaErrorPeerAccessUnsupported"; break;
	case cudaErrorLaunchMaxDepthExceeded: return "cudaErrorLaunchMaxDepthExceeded"; break;
	case cudaErrorLaunchFileScopedTex: return "cudaErrorLaunchFileScopedTex"; break;
	case cudaErrorLaunchFileScopedSurf: return "cudaErrorLaunchFileScopedSurf"; break;
	case cudaErrorSyncDepthExceeded: return "cudaErrorSyncDepthExceeded"; break;
	case cudaErrorLaunchPendingCountExceeded: return "cudaErrorLaunchPendingCountExceeded"; break;
	case cudaErrorNotPermitted: return "cudaErrorNotPermitted"; break;
	case cudaErrorNotSupported: return "cudaErrorNotSupported"; break;
	case cudaErrorHardwareStackError: return "cudaErrorHardwareStackError"; break;
	case cudaErrorIllegalInstruction: return "cudaErrorIllegalInstruction"; break;
	case cudaErrorMisalignedAddress: return "cudaErrorMisalignedAddress"; break;
	case cudaErrorInvalidAddressSpace: return "cudaErrorInvalidAddressSpace"; break;
	case cudaErrorInvalidPc: return "cudaErrorInvalidPc"; break;
	case cudaErrorIllegalAddress: return "cudaErrorIllegalAddress"; break;
	case cudaErrorInvalidPtx: return "cudaErrorInvalidPtx"; break;
	case cudaErrorInvalidGraphicsContext: return "cudaErrorInvalidGraphicsContext"; break;
	case cudaErrorStartupFailure: return "cudaErrorStartupFailure"; break;
	case cudaErrorApiFailureBase: return "cudaErrorApiFailureBase"; break;
	default: break;
	}

	return string();
};

inline string ToString(const cudnnStatus_t &status)
{
	switch (status)
	{
	case CUDNN_STATUS_SUCCESS: return "CUDNN_STATUS_SUCCESS"; break;
	case CUDNN_STATUS_NOT_INITIALIZED: return "CUDNN_STATUS_NOT_INITIALIZED"; break;
	case CUDNN_STATUS_ALLOC_FAILED: return "CUDNN_STATUS_ALLOC_FAILED"; break;
	case CUDNN_STATUS_BAD_PARAM: return "CUDNN_STATUS_BAD_PARAM"; break;
	case CUDNN_STATUS_INTERNAL_ERROR: return "CUDNN_STATUS_INTERNAL_ERROR"; break;
	case CUDNN_STATUS_INVALID_VALUE: return "CUDNN_STATUS_INVALID_VALUE"; break;
	case CUDNN_STATUS_ARCH_MISMATCH: return "CUDNN_STATUS_ARCH_MISMATCH"; break;
	case CUDNN_STATUS_MAPPING_ERROR: return "CUDNN_STATUS_MAPPING_ERROR"; break;
	case CUDNN_STATUS_EXECUTION_FAILED: return "CUDNN_STATUS_EXECUTION_FAILED"; break;
	case CUDNN_STATUS_NOT_SUPPORTED: return "CUDNN_STATUS_NOT_SUPPORTED"; break;
	case CUDNN_STATUS_LICENSE_ERROR: return "CUDNN_STATUS_LICENSE_ERROR"; break;
	default: break;
	}

	return string();
};
inline string ToString(const cublasStatus_t &status)
{
	switch (status)
	{
	case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS"; break;
	case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED"; break;
	case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED"; break;
	case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; break;
	case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; break;
	case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR"; break;
	case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; break;
	case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; break;
	case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED"; break;
	case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR"; break;
	default: break;
	}

	return string();
};
inline void FatalError(const string &strMsg, const string &strFile, const int &Line)
{
	::cerr << strMsg.c_str() << "\r\n\t" << strFile.c_str() << ':' << Line << "\r\nAborting...\n";  cudaDeviceReset();
	::system("pause");
};
template<class _type>
inline void checkCUDNN(const _type &status, const string &strFile, const int &Line)
{
	const int Success = 0;

	if (status != Success)
	{
		stringstream error;
		error << "Error: " << ToString(status) << " " << cudnnGetErrorString(status);
		FatalError(error.str(), strFile, Line);
	}
};
template<class _type>
inline void checkCudaErrors(const _type &status, const string &strFile, const int &Line)
{
	if (status != 0)
	{
		stringstream error;
		error << "Cuda failure: " << status << " " << ToString(status);
		FatalError(error.str(), strFile, Line);
	}
};
#endif