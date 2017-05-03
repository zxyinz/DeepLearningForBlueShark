#include"SanGraphics.h"
using namespace std;
using namespace San;
using namespace San::Mathematics;
using namespace San::Graphics;
stSANCOLOR& stSANCOLOR::operator+(const stSANCOLOR& Color)
{
	int16 value=this->r+Color.r;
	this->r=value>255?255:value;
	value=this->g+Color.g;
	this->g=value>255?255:value;
	value=this->b+Color.b;
	this->b=value>255?255:value;
	value=this->a+Color.a;
	this->a=value>255?255:value;
	return *this;
}
stSANCOLOR& stSANCOLOR::operator-(const stSANCOLOR& Color)
{
	int16 value=this->r-Color.r;
	this->r=value<0?0:value;
	value=this->g-Color.g;
	this->g=value<0?0:value;
	value=this->b-Color.b;
	this->b=value<0?0:value;
	value=this->a-Color.a;
	this->a=value<0?0:value;
	return *this;
}
stSANCOLOR& stSANCOLOR::operator=(const stSANCOLORF& Color)
{
	this->r=Color.r*255;
	this->g=Color.g*255;
	this->b=Color.b*255;
	this->a=Color.a*255;
	return *this;
}
/*Color Float*/
stSANCOLORF& stSANCOLORF::operator+(const stSANCOLORF& Color)
{
	sfloat value=this->r+Color.r;
	//this->r=value>1.0f?1.0f:value;
	value=this->g+Color.g;
	//this->g=value>1.0f?1.0f:value;
	value=this->b+Color.b;
	//this->b=value>1.0f?1.0f:value;
	value=this->a+Color.a;
	//this->a=value>1.0f?1.0f:value;
	return *this;
}
stSANCOLORF& stSANCOLORF::operator-(const stSANCOLORF& Color)
{
	sfloat value=this->r-Color.r;
	//this->r=value<0.0f?0.0f:value;
	value=this->g-Color.g;
	//this->g=value<0.0f?0.0f:value;
	value=this->b-Color.b;
	//this->b=value<0.0f?0.0f:value;
	value=this->a-Color.a;
	//this->a=value<0.0f?0.0f:value;
	return *this;
}
stSANCOLORF& stSANCOLORF::operator=(const stSANCOLOR& Color)
{
	this->r = Color.r;// / 255;
	this->g = Color.g;// / 255;
	this->b = Color.b;// / 255;
	this->a = Color.a;// / 255;
	return *this;
}
/*Graphics Vector*/
stSANGRAPHICSPOINT::stSANGRAPHICSPOINT()
	:p_vt(nullptr),
	vt_size(0)
{}
stSANGRAPHICSPOINT::stSANGRAPHICSPOINT(const stSANGRAPHICSPOINT& P)
{
	v=P.v;
	vn=P.vn;
	vc=P.vc;
	if(P.vt_size!=0)
	{
		vt_size=P.vt_size;
		p_vt=new SVECTOR3[vt_size];
		::memcpy(p_vt,P.p_vt,sizeof(SVECTOR3)*vt_size);
	}
}
stSANGRAPHICSPOINT::~stSANGRAPHICSPOINT()
{
	if(p_vt!=nullptr)
	{
		delete[] p_vt;
		p_vt=nullptr;
		vt_size=0;
	}
}
stSANGRAPHICSPOINT& stSANGRAPHICSPOINT::operator=(const stSANGRAPHICSPOINT& P)
{
	lpSVECTOR3 buffer=nullptr;
	this->v=P.v;
	this->vn=P.vn;
	this->vc=P.vc;
	if(P.vt_size!=0)
	{
		buffer=new SVECTOR3[this->vt_size];
		::memcpy(buffer,P.p_vt,vt_size*sizeof(SVECTOR3));
	}
	if(this->vt_size!=0)
	{
		delete[] this->p_vt;// may cause problem when vt_size equal 1
	}
	this->vt_size=P.vt_size;
	this->p_vt=buffer;
	buffer=nullptr;
	return *this;
}