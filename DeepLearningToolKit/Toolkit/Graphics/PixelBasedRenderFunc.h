#include"..\Core\SanTypes.h"
#include"SanGraphics.h"
namespace San
{
	namespace Graphics
	{
#ifndef __SANPIXELBASEDRENDERFUNC_H__
#define __SANPIXELBASEDRENDERFUNC_H__
		struct PIXCAMERA
		{
			SVECTOR3 Coord;
			SVECTOR3 LookAt;
			SVECTOR3 Dir;
			SVECTOR3 UpDir;
			sfloat Cutoff;
			sfloat Near;
			sfloat Far;
			SVECTOR3 ScreenVec[4];
		};

		void ClearScreenRGBA(sfloat* p_buffer, const uint32 width, const uint32 height, const SANCOLORF &Color);
		void DrawImageRGBA(sfloat* p_buffer, const uint32 buf_width, const uint32 buf_height, sfloat* p_img, const uint32 img_width, const uint32 img_height, const SVECTOR3 &pos, bool bUseAlpha = false);
		void DrawCrossRGBA(sfloat* p_buffer, const uint32 width, uint32 height, const SVECTOR3 pos, const SANCOLORF color, const uint32 size,const uint32 thickness);
		void DrawDotRGBA(sfloat* p_buffer, const uint32 width, uint32 height, const SVECTOR3 pos, const SANCOLORF color, sfloat radius);
		void DrawQuadrRGBA(sfloat* p_buffer, const uint32 buf_width, const uint32 buf_height, const uint32 quadr_width, const uint32 quadr_height, const SANCOLORF color, const SVECTOR3 &pos, const uint32 size);
		void DrawLineRGBA(sfloat* p_buffer, const uint32 buf_width, const uint32 buf_height, const SVECTOR3 pos_begin, const SVECTOR3 pos_end, const SANCOLORF color, const uint32 size);
		void UpdateCamera(PIXCAMERA &Camera, uint32 ScreenWidth, uint32 ScreenHeight);
		SVECTOR3 GetScreenCoord(const SVECTOR3 Coord, const PIXCAMERA Camera, uint32 ScreenWidth, uint32 ScreenHeight);
		sfloat GetHitPoint(const SVECTOR3* pRayCoord, const SVECTOR3* pRayDir, const SVECTOR3* pVec1, const SVECTOR3* pVec2, const SVECTOR3* pVec3, const SVECTOR3* pNormal, SVECTOR3* pHitPoint);
		SVECTOR3 Interpolation(const SVECTOR3* pVecSrc, const SVECTOR3* pVec1, const SVECTOR3* pVec2, const SVECTOR3* pVec3);
		void DrawDot3DRGBA(sfloat* p_buffer, const uint32 width, uint32 height, const SVECTOR3 pos, const SANCOLORF color, sfloat radius, const PIXCAMERA camera, const SVECTOR3 offset = SVECTOR3(0.0, 0.0, 0.0));
		void DrawLine3DRGBA(sfloat* p_buffer, const uint32 buf_width, const uint32 buf_height, const SVECTOR3 coord_begin, const SVECTOR3 coord_end, const SANCOLORF color, const uint32 size, const PIXCAMERA camera, const SVECTOR3 offset = SVECTOR3(0.0, 0.0, 0.0));
		void DrawLineGraph(sfloat* p_buffer, const uint32 buf_width, const uint32 buf_height, const uint32 dest_width, const uint32 dest_height, sfloat* p_dataset, uint32 data_size, const SVECTOR3 pos, const SANCOLORF color, const uint32 size);

		template<class DType, class SType> void BufferCopy(DType* pDest, SType* pSrc, uint32 BufferSize, sfloat min = 0.0f, sfloat max = 255.0f)
		{
#pragma omp parallel for
			for (int32 seek = 0; seek < BufferSize; seek = seek + 1)
			{
				pDest[seek] = pSrc[seek]<min ? min : (pSrc[seek]>max ? max : pSrc[seek]);
			}
		}
#endif
	}
}