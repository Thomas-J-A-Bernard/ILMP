//---------------------------------------------------------------------------
// RDAAM (Flowers et al. 2009) implementation provided by R. Ketcham and
// modified by C. Glotzbach to handle:
// 1. U,Th,Sm profiles as input e.g. derived by LA-ICP-MS measurements
// 2. Use all important stopping distances making up >99% of all alpha ejections
// 3. Calculates the He profile for a short axis of an infinite cylinder

// Please cite the following references:
// Flowers, R. M., Ketcham, R. A., Shuster, D. L., & Farley, K. A. (2009). Apatite (U–Th)/He thermochronometry using a radiation damage accumulation and annealing model. Geochimica et Cosmochimica Acta, 73(8), 2347–2365. https://doi.org/10.1016/j.gca.2009.01.015
// Ketcham, R. A. (2005). Forward and Inverse Modeling of Low-Temperature Thermochronometry Data. Reviews in Mineralogy and Geochemistry, 58(1), 275–314. https://doi.org/10.2138/rmg.2005.58.11
// Glotzbach, C., & Ehlers, T. A. (2024). Interpreting cooling dates and histories from laser ablation in-situ (U-Th-Sm)/He thermochronometry.

#pragma hdrstop

#include "RDAAM_profile.h"
#include <iostream>
#include <numeric>

//---------------------------------------------------------------------------

#pragma package(smart_init)

// Arrays to hold radial information for calculation
double *aDepl238, *aDepl235, *aDepl232, *aDepl147; // Alpha depletion including zoning (% retained)
double *aEjOnly238, *aEjOnly235, *aEjOnly232, *aEjOnly147;  // Alpha ejection only
double *nmpg238, *nmpg235, *nmpg232, *nmpg147;   // Compositional profiles (nanomoles per gram)
double *nmpg238z, *nmpg235z, *nmpg232z, *nmpg147z;


// FD diffusion solvers
double *diag, *b, *u, *gam, *prodHe;

double *ppmU, *ppmTh, *ppmSm;
double total238, total235, total232, total147;
double max238, max235, max232, max147;
double totalHe;

float alphaCorrFactor;
float ft238, ft235, ft232, ft147;

double radius;				// Grain radius, in um

// Alpha damage tracking for trapping model
double *alphaDamage;  // Total alpha damage (as nm/g)
double *betaTrap;     // Represents radially varying diffusivity in trapping model (cm^2/s)

int     precision;    // PREC_GOOD, PREC_BETTER, PREC_BEST
int     rdimLog2;     // Number of nodes = 2^rdimLog2 + 1
double  maxTempStep;  // Maximum temperature change for any time step (deg C)
double  gridSpacing;  // Space between nodes (cm)

int     rdim;         // Number of nodes DO NOT SET DIRECTLY
int     rombInit, rombLimit;  // Romberg integration parameters
double	ageConv;			// Desired age precision, in years.

double  E;            // Activation energy (kcal/mol)
double  dInf;         // Diffusivity at infinite temperature (cm^2/s)
double  Et;           // Trapping activation energy (kcal/mol)
double  psi;          // Trapping model extra parameter
double  polyA;				// Polynomial term for psi
double  trapRmr0;     // FT rmr0 parameter for trap annealing
int			psiUnits;	 		// Units for psi

alphaStoppingDistRec	asDist;
double maxSDU238,maxSDU235,maxSDTh232,maxSDSm147;


int     endNode;      // Earliest t-T node to calculate on (when damage starts being retained)

bool		nonzeroProduction;	// Whether there is any He production at all
bool 		paramsOK;

double	heModelAge;       // Calculated uncorrected model age
double	heCorrModelAge;		// Calculated corrected model age

vector<float> heProfile;
vector<float> He;
vector<double> sdU238,frU238,sdU235,frU235,sdTh232,frTh232,sdSm147,frSm147;


vector<TTPathPoint> tTPath;  // Interpolated time-temperature path
vector<TTPathPoint> *tTDef;  // Pointer to input path
//vector<StoppingDist> SD;    // all significant stopping distances


void GeneralInit(int precision, double grainRadius, double *ppm_U, double *ppm_Th, double *ppm_Sm)
{
	ageConv = 100.0;  // Ages calculated to within 100 years

// Initialize pointers
	diag = NULL;
	b = NULL;
	gam = NULL;
	u = NULL;
	prodHe = NULL;
	aDepl238 = NULL;
	aDepl235 = NULL;
	aDepl232 = NULL;
	aDepl147 = NULL;
	aEjOnly238 = NULL;
	aEjOnly235 = NULL;
	aEjOnly232 = NULL;
	aEjOnly147 = NULL;
	nmpg238 = NULL;
	nmpg235 = NULL;
	nmpg232 = NULL;
	nmpg147 = NULL;
    nmpg238z = NULL;
    nmpg235z = NULL;
    nmpg232z = NULL;
    nmpg147z = NULL;
	alphaDamage = NULL;
	betaTrap = NULL;
    ppmU = NULL;
    ppmTh = NULL;
    ppmSm = NULL;

    SetPrecision(precision);

    // Number of nodes
	rdim = pow(2, rdimLog2);

    ppmU = dvector(0,rdim-1);
    ppmTh = dvector(0,rdim-1);
    ppmSm = dvector(0,rdim-1);

	totalHe = 0.0;
	heModelAge = 0.0;
	heCorrModelAge = 0.0;

    for (int i=0;i<rdim;i++) {
        ppmU[i] = ppm_U[i];
        ppmTh[i] = ppm_Th[i];
        ppmSm[i] = ppm_Sm[i];
     }

	radius = grainRadius;

}

extern "C" void RDAAM_Init(int precision, double grainRadius, double *ppm_U, double *ppm_Th, double *ppm_Sm)
{

	GeneralInit(precision, grainRadius, ppm_U, ppm_Th, ppm_Sm);

// Hardwire RDAAM parameters
	E = 29.23;            	// Activation energy (kcal/mol)
	dInf = 0.6071;         	// Diffusivity at infinite temperature (cm^2/s)
	Et = 8.126;           	// Trapping activation energy (kcal/mol)
	psi = 1.0e-13;          // Trapping model extra parameter
	polyA = 1.0e-22;				// Extra polynomial term for psi
	trapRmr0 = 0.83;     		// FT rmr0 parameter for trap annealing
	psiUnits = PSI_CMSQ_TRACK;

// Alpha stopping distances in apatite (from Ketcham et al 2011)
	asDist.asU238 = 18.81;
	asDist.asU235 = 21.80;
	asDist.asTh232 = 22.25;
	asDist.asSm147 = 5.93;

// most significant stopping distances (>99% alphas) - CG 221115
	/*
	sdU238 = {21.7845,18.2043,31.6377,19.1253,15.7282,13.1324,15.2829,15.6834,15.4442,14.9819,12.9333};
    frU238 = {0.1260,0.1260,0.1260,0.1259,0.1190,0.0995,0.0961,0.0899,0.0358,0.0295,0.0263};
    maxSDU238 = 32;
    sdU235 = {29.7535640771331,25.2148975052404,26.3479012722287,13.9936175999724,20.2812870385912,19.7190718493755,16.8006006168443,21.9753667008956,21.6531929933677,16.5053473038250,20.4917923288944,16.8699119544318,13.8553719384429,23.2806878841381,24.8145948848071,17.0136908330447,19.3789958453760,20.4405763604049,15.5063328027966,20.2429093806895,24.0956452351219,13.2034842934578,14.8735459473538,20.2656323632956,14.6926160479959,20.2016248441497,16.4240383727413,21.5573253461306,21.8190018955796,16.8870222207504,21.0651531443760,18.8449424247089,14.0645012961151,20.0327626476417,14.4523194996807,15.2556195685931,20.1614096629705,16.0457495082610,16.6689804322044,20.7554398051794,15.4012666745273,19.1861125641568,21.0898740924305,12.9291286366060,13.2218077914455,21.3256533785943,14.1566236400475,16.5145705902897,16.4555064972044,13.4442789714073,16.6170294779009,21.0173271301945,20.6928116768470,30.1507368698292,20.5199940008232,19.7526004044279,19.6129429824526,21.2934065280688,19.6878122637146,18.1203992052928,18.3734810169975,18.5065669422490,24.6808813098626};
	frU235 = {0.142919557831915,0.120545970993926,0.114573622609470,0.0793646637869762,0.0759014156115611,0.0370849121904395,0.0366520447270475,0.0344385498277550,0.0334423934277786,0.0329002606211292,0.0290308436564546,0.0288598777378326,0.0245308960796108,0.0233551523126587,0.0186146061922187,0.0158729327558080,0.0132755327685620,0.0132755327685620,0.0121211486498897,0.0118115687425771,0.0108224454605923,0.00822506515610481,0.00721496943517966,0.00695886399412075,0.00606057432555091,0.00516578247416325,0.00432898166067490,0.00426924171418451,0.00412693365704502,0.00360748471722908,0.00344385498277550,0.00327559341137345,0.00303028716277546,0.00293154597707336,0.00245308960796108,0.00216449083033745,0.00213462085709225,0.00202019144164828,0.00202019144164828,0.00180731232567144,0.00144299388689163,0.00144299269223500,0.00144299269223500,0.00129869449833234,0.00129869449833234,0.00110288744283100,0.00101009572092515,0.000949865155985286,0.000788567299308539,0.000577197554814373,0.000577197554756653,0.000461757661515201,0.000442578057703794,0.000393845556749594,0.000324462370278022,0.000307385403421284,0.000250462180565491,0.000247616019422701,0.000241923697137122,0.000216448903835250,0.000187589049990550,0.000187589049990550,0.000173159127369476};
    maxSDU235 = 31;
    sdTh232 = {26.1103759422357,23.3351277183811,20.1221065436592,12.3536439669681,18.7927603382688,38.9804885989926,18.3816810930971,22.0438881172962,12.0855634751228,22.2543042353269,18.9199603023222};
    frTh232 = {0.167193871842947,0.167006443951481,0.158703438518657,0.130748091994933,0.120716269079720,0.107106429324749,0.0454775972156288,0.0420093519261490,0.0362817595433509,0.0162965759438873,0.00846017065849564};
    maxSDTh232 = 39;
	*/

//  simplified stopping distances - CG 240312

	sdU238={13.0907, 15.5028, 19.7048, 31.6360};
    frU238={0.1250,0.3749,0.3748,0.1250};
    maxSDU238 = 32;
    sdU235={13.9900,16.6537,20.6106,25.4396,29.7532};
    frU235={0.1338,0.1477,0.2828,0.2853,0.1420};
    maxSDU235 = 30;
    sdTh232={12.2949,18.6849,20.1233,23.0162,26.1104,38.9805};
    frTh232={0.1666,0.1753,0.1603,0.2246,0.1666,0.1068};
    maxSDTh232 = 39;

	RDAAM_CheckParameters();

	RDAAM_PrepModel();
}

void ZRDAAM_Init(int precision, double grainRadius, double *ppm_U, double *ppm_Th, double *ppm_Sm)
{
	GeneralInit(precision, grainRadius, ppm_U, ppm_Th, ppm_Sm);

// Hardwire ZRDAAM parameters
	E = 39.44;            	// Activation energy (kcal/mol)
	dInf = 193188.;         	// Diffusivity at infinite temperature (cm^2/s)
	Et = 16.97;           	// Trapping activation energy (kcal/mol)
	psi = 6.367e-3;          // Trapping model extra parameter
	polyA = 3.;				// Extra polynomial term for psi
	trapRmr0 = 0.0;     		// FT rmr0 parameter for trap annealing
	psiUnits = PSI_D0N17_1_SEC;

// Alpha stopping distances in zircon (from Ketcham et al 2011)
	asDist.asU238 = 15.55;
	asDist.asU235 = 18.05;
	asDist.asTh232 = 18.43;
	asDist.asSm147 = 4.76;

	RDAAM_CheckParameters();
	RDAAM_PrepModel();
}


void SetPrecision(int newPrecision)
{
  precision = newPrecision;
  switch (precision) {
    case HE_PREC_BEST:
			rdimLog2 = 9;
			maxTempStep = 0.5;
			break;
    case HE_PREC_BETTER:
			rdimLog2 = 8;
			maxTempStep = 2;
			break;
	case HE_PREC_GOOD:
			rdimLog2 = 7;
      		maxTempStep = 10;
			break;
	case HE_PREC_MEDIUM:
			rdimLog2 = 6;
			maxTempStep = 10;
	}
}

bool RDAAM_CheckParameters()
{
	paramsOK = true;
	paramsOK = paramsOK && (radius > 0);

	return paramsOK;
}


void RDAAM_PrepModel()
{
	if (!paramsOK) return;

// Number of nodes
	rdim = pow(2, rdimLog2);

// Romberg integration parameters
	rombLimit = rdim;
	rombInit = rombLimit/2;

// ------------ Allocation -------------
// Get rid of the old
  RDAAM_FreeCalcArrays();
// Matrix solution arrays
  diag = dvector(0,rdim-1);
	b = dvector(0,rdim-1);
  gam = dvector(0,rdim);
// The data
  u = dvector(0,rdim);  // Transformed variable = He * r
// He production rate
  prodHe = dvector(0,rdim-1);
// Alpha depletion from ejection and zoning (redistribution)
  aDepl238 = dvector(0,rdim-1);
  aDepl235 = dvector(0,rdim-1);
  aDepl232 = dvector(0,rdim-1);
	aDepl147 = dvector(0,rdim-1);
// Alpha ejection only
	aEjOnly238 = dvector(0,rdim-1);
	aEjOnly235 = dvector(0,rdim-1);
	aEjOnly232 = dvector(0,rdim-1);
	aEjOnly147 = dvector(0,rdim-1);
// Composition
  nmpg238 = dvector(0,rdim-1);
  nmpg235 = dvector(0,rdim-1);
  nmpg232 = dvector(0,rdim-1);
  nmpg147 = dvector(0,rdim-1);
// Alpha damage
  alphaDamage = dvector(0,rdim-1);
  betaTrap = dvector(0,rdim-1);

// ----------- Initialization -----------

// Initialize U, Th arrays

    max238 = 0.0;
    max235 = 0.0;
    max232 = 0.0;
    max147 = 0.0;

    for (int i=0;i<rdim;i++) {
                nmpg238[i] = ppmU[i]*(U238U235RATIO-1)/U238U235RATIO;
                nmpg235[i] = ppmU[i]/U238U235RATIO;
                nmpg232[i] = ppmTh[i];
                nmpg147[i] = ppmSm[i];
                }


// convert from ppm to nmpg
	for (int i=0;i<rdim;i++) {
		nmpg147[i] *= PPM_NMPG_SM147;
        nmpg232[i] *= PPM_NMPG_TH232;
		nmpg235[i] *= PPM_NMPG_U235;
		nmpg238[i] *= PPM_NMPG_U238;
        if (nmpg238[i]>max238) max238 = nmpg238[i];
        if (nmpg235[i]>max235) max235 = nmpg235[i];
        if (nmpg232[i]>max232) max232 = nmpg232[i];
        if (nmpg147[i]>max147) max147 = nmpg147[i];
  }

	gridSpacing = (radius/1.e4)/(rdim+0.5);  // in cm

    int int_w238, int_w235, int_w232, int_w147;

    // Calculate integration window CG - 220706
    int_w238 = (int)round(maxSDU238/(gridSpacing*1.e4));
    int_w235 = (int)round(maxSDU235/(gridSpacing*1.e4));
    int_w232 = (int)round(maxSDTh232/(gridSpacing*1.e4));
    int_w147 = (int)round(asDist.asSm147/(gridSpacing*1.e4));

    // Composition CG - 220706
    nmpg238z = dvector(0,rdim-1+int_w232);
    nmpg235z = dvector(0,rdim-1+int_w232);
    nmpg232z = dvector(0,rdim-1+int_w232);
    nmpg147z = dvector(0,rdim-1+int_w232);

  // Now initialize totals, alpha stopping
    double xi, xs;
	double innerVol, outerVol, rad;
    double sum, sum_all, sumpc, x, xp, jp, frac;

    // Initialize for zoning - added by CG
    double a; //area of the crossing circle
    double d; // distance between center of grain and point along profile
    double alpha=0; // angle between d and radius of zoning
    double beta=0; // angle between d and stopping distance radius
    double gamma=0; // angle between stopping distance radius and radius from zoned area
    double beta_new=0; // same as beta, re-calculated
    double r_cc; // radius of the circle from two crossing spheres
    double l; // length of the circle
    double width; // width of the crossing zoned layer and the sphere of the stopping distance
    double lengthc_sum=0; // sum of intersections on sphere
    double t;

  // Set up integration variables
  int aDeplRad = 100 * ceil(20.0/radius);   // in grid nodes
  if (aDeplRad < 50) aDeplRad = 50.0;
	double spAdepl238 = 2.*asDist.asU238/(2.*aDeplRad + 1);  // in microns
	double spAdepl235 = 2.*asDist.asU235/(2.*aDeplRad + 1);  // in microns
	double spAdepl232 = 2.*asDist.asTh232/(2.*aDeplRad + 1);  // in microns
	double spAdepl147 = 2.*asDist.asSm147/(2.*aDeplRad + 1);  // in microns


// set values of nmpgXXXz vector and normalize
        for (int i=0;i<rdim;i++) {
            nmpg238z[i] = nmpg238[i];
            nmpg235z[i] = nmpg235[i];
            nmpg232z[i] = nmpg232[i];
            nmpg147z[i] = nmpg147[i];
      }
        for (int i=rdim;i<rdim+int_w232;i++) {
            nmpg238z[i] = 0.0;
            nmpg235z[i] = 0.0;
            nmpg232z[i] = 0.0;
            nmpg147z[i] = 0.0;
        }

    double xc[rdim+int_w232][50];
    double yc[rdim+int_w232][50];
    double zc[rdim+int_w232][50];
    double distc[rdim+int_w232][50];
    double lengthc[rdim+int_w232];

	total238 = 0.0;
	total235 = 0.0;
    total232 = 0.0;
    total147 = 0.0;

    rad = 0.0;
	innerVol = 0.0;

    // set nuclide concentration of center of grain to the value one stopping distance away
    aEjOnly238[0]=nmpg238[int_w238]/max238;
    aDepl238[0]=aEjOnly238[0];
    aEjOnly235[0]=nmpg235[int_w235]/max235;
    aDepl235[0]=aEjOnly235[0];
    aEjOnly232[0]=nmpg232[int_w232]/max232;
    aDepl232[0]=aEjOnly232[0];
    aEjOnly147[0]=nmpg147[int_w147]/max147;
    aDepl147[0]=aEjOnly147[0];

    // x-y coordinates of the cylinder
    for (int i=0; i<rdim+int_w232; i++) {
        for (int j = 0; j<=50; j++) {
           t=double(j)/50*2*PI;
           xc[i][j] = (i*gridSpacing*1.e4) * cos(t);
           yc[i][j] = (i*gridSpacing*1.e4) * sin(t);
        }
    }

    for (int i=1;i<rdim;i++) {

        // Sum total volume of isotopes
        rad += gridSpacing;
        outerVol = rad*rad*rad;
        total238 += nmpg238[i]*(outerVol-innerVol);
        total235 += nmpg235[i]*(outerVol-innerVol);
        total232 += nmpg232[i]*(outerVol-innerVol);
        total147 += nmpg147[i]*(outerVol-innerVol);
        innerVol = outerVol;

        // Calculate the weighted mean nuclide concentration. Does take into account zoning and most important stopping distances - CG 220718

        // 238U

        int n_sd = sdU238.size();
        sum_all = 0;

        for (int k=0;k<n_sd;k++) {
            asDist.asU238 = sdU238[k];
            d=(i*gridSpacing*1.e4);
            lengthc_sum=0;
            for (int j=0; j<rdim+int_w232; j++) {
                for (int o = 0; o <=50; o++) {
                    // 1. calculate z of intersection
                    zc[j][o]=pow(asDist.asU238,2)-pow(xc[j][o]-d,2)-pow(yc[j][o],2);
                    // 2. identify cylinder coordinates within the sphere coordinates
                    if (o>0 && zc[j][o]>0) {
                        distc[j][o]=sqrt(pow(xc[j][o]-xc[j][o-1],2)+pow(yc[j][o]-yc[j][o-1],2)+pow(zc[j][o]-zc[j][o-1],2));
                    }
                    else if (o==50 && zc[j][o]>0) {
                        distc[j][0]=sqrt(pow(xc[j][o]-xc[j][0],2)+pow(yc[j][o]-yc[j][0],2)+pow(zc[j][o]-zc[j][0],2));
                    }
                    else {
                        distc[j][o]=0;
                    }
                }
                // calculate the length
                lengthc[j]=0;
                for (int o = 0; o <=50; o++) {
                    lengthc[j]=lengthc[j]+2*distc[j][o];
                }
                lengthc_sum+=lengthc[j];
            }
            sum=0;
            for (int j=0; j<rdim+int_w232; j++) {
                // normalize the length
                lengthc[j]=lengthc[j]/lengthc_sum;
                sum+=lengthc[j]*nmpg238z[j];
            }
            sum=sum*frU238[k];
            sum_all+=sum;
        }

        aEjOnly238[i] = sum_all / max238;

        // U235

        n_sd = sdU235.size();
        sum_all = 0;

        for (int k=0;k<n_sd;k++) {
            asDist.asU235 = sdU235[k];
            d=(i*gridSpacing*1.e4);
            lengthc_sum=0;
            for (int j=0; j<rdim+int_w232; j++) {
                for (int o = 0; o <=50; o++) {
                    // 1. calculate z of intersection
                    zc[j][o]=pow(asDist.asU235,2)-pow(xc[j][o]-d,2)-pow(yc[j][o],2);
                    // 2. identify cylinder coordinates within the sphere coordinates
                    if (o>0 && zc[j][o]>0) {
                        distc[j][o]=sqrt(pow(xc[j][o]-xc[j][o-1],2)+pow(yc[j][o]-yc[j][o-1],2)+pow(zc[j][o]-zc[j][o-1],2));
                    }
                    else if (o==50 && zc[j][o]>0) {
                        distc[j][0]=sqrt(pow(xc[j][o]-xc[j][0],2)+pow(yc[j][o]-yc[j][0],2)+pow(zc[j][o]-zc[j][0],2));
                    }
                    else {
                        distc[j][o]=0;
                    }
                }
                // calculate the length
                lengthc[j]=0;
                for (int o = 0; o <=50; o++) {
                    lengthc[j]=lengthc[j]+2*distc[j][o];
                }
                lengthc_sum+=lengthc[j];
            }
            sum=0;
            for (int j=0; j<rdim+int_w232; j++) {
                // normalize the length
                lengthc[j]=lengthc[j]/lengthc_sum;
                sum+=lengthc[j]*nmpg235z[j];
            }
            sum=sum*frU235[k];
            sum_all+=sum;
        }

        aEjOnly235[i] = sum_all / max235;

        // 232Th

        n_sd = sdTh232.size();
        sum_all = 0;

        for (int k=0;k<n_sd;k++) {
            asDist.asTh232 = sdTh232[k];
            d=(i*gridSpacing*1.e4);
            lengthc_sum=0;
            for (int j=0; j<rdim+int_w232; j++) {
                for (int o = 0; o <=50; o++) {
                    // 1. calculate z of intersection
                    zc[j][o]=pow(asDist.asTh232,2)-pow(xc[j][o]-d,2)-pow(yc[j][o],2);
                    // 2. identify cylinder coordinates within the sphere coordinates
                    if (o>0 && zc[j][o]>0) {
                        distc[j][o]=sqrt(pow(xc[j][o]-xc[j][o-1],2)+pow(yc[j][o]-yc[j][o-1],2)+pow(zc[j][o]-zc[j][o-1],2));
                    }
                    else if (o==50 && zc[j][o]>0) {
                        distc[j][0]=sqrt(pow(xc[j][o]-xc[j][0],2)+pow(yc[j][o]-yc[j][0],2)+pow(zc[j][o]-zc[j][0],2));
                    }
                    else {
                        distc[j][o]=0;
                    }
                }
                // calculate the length
                lengthc[j]=0;
                for (int o = 0; o <=50; o++) {
                    lengthc[j]=lengthc[j]+2*distc[j][o];
                }
                lengthc_sum+=lengthc[j];
            }
            sum=0;
            for (int j=0; j<rdim+int_w232; j++) {
                // normalize the length
                lengthc[j]=lengthc[j]/lengthc_sum;
                sum+=lengthc[j]*nmpg232z[j];
            }
            sum=sum*frTh232[k];
            sum_all+=sum;
        }

        aEjOnly232[i] = sum_all / max232;

        // 147Sm

        n_sd = sdSm147.size();
        sum_all = 0;

        for (int k=0;k<n_sd;k++) {
            asDist.asSm147 = sdSm147[k];
            d=(i*gridSpacing*1.e4);
            lengthc_sum=0;
            for (int j=0; j<rdim+int_w147; j++) {
                for (int o = 0; o <=50; o++) {
                    // 1. calculate z of intersection
                    zc[j][o]=pow(asDist.asU238,2)-pow(xc[j][o]-d,2)-pow(yc[j][o],2);
                    // 2. identify cylinder coordinates within the sphere coordinates
                    if (o>0 && zc[j][o]>0) {
                        distc[j][o]=sqrt(pow(xc[j][o]-xc[j][o-1],2)+pow(yc[j][o]-yc[j][o-1],2)+pow(zc[j][o]-zc[j][o-1],2));
                    }
                    else if (o==50 && zc[j][o]>0) {
                        distc[j][0]=sqrt(pow(xc[j][o]-xc[j][0],2)+pow(yc[j][o]-yc[j][0],2)+pow(zc[j][o]-zc[j][0],2));
                    }
                    else {
                        distc[j][o]=0;
                    }
                }
                // calculate the length
                lengthc[j]=0;
                for (int o = 0; o <=50; o++) {
                    lengthc[j]=lengthc[j]+2*distc[j][o];
                }
                lengthc_sum+=lengthc[j];
            }
            sum=0;
            for (int j=0; j<rdim+int_w232; j++) {
                // normalize the length
                lengthc[j]=lengthc[j]/lengthc_sum;
                sum+=lengthc[j]*nmpg147z[j];
            }
            sum=sum*frSm147[k];
            sum_all+=sum;
        }

        aEjOnly147[i] = sum_all / max147;

// Compile alpha depletion
		aDepl238[i] = aEjOnly238[i];
		aDepl235[i] = aEjOnly235[i];
		aDepl232[i] = aEjOnly232[i];
		aDepl147[i] = aEjOnly147[i];
	}

// Add last drib of isotopes at edge
	rad += gridSpacing*0.5;
	outerVol = rad*rad*rad;

	total238 += nmpg238[rdim-1]*(outerVol-innerVol);
	total235 += nmpg235[rdim-1]*(outerVol-innerVol);
	total232 += nmpg232[rdim-1]*(outerVol-innerVol);
	total147 += nmpg147[rdim-1]*(outerVol-innerVol);

// Scale each by He production
	total238 *= 8.0;
	total235 *= 7.0;
	total232 *= 6.0;


  heProfile.resize(rdim+1);
  He.resize(rdim+1);

// Calculate the alpha correction factors
	RDAAM_CalcAlphaCorrectionFactor();

// Check to see if there is any net production
	nonzeroProduction = ((ppmU[0] > 0.0) || (ppmTh[0] > 0.0) || (ppmSm[0] > 0.0));
}

/*  RDAAM_CalcAlphaCorrectionFactor
		Calculates alpha correction factors.  This routine requires that the
		alpha depletion arrays be properly initialized (to cover cases in which
		there is zoning).
 */
void RDAAM_CalcAlphaCorrectionFactor()
{
  double innerVol, outerVol, rad, vol;
  double rate238, rate235, rate232, rate147;
  double dTotal,ndTotal;
  double d147, d232, d235, d238, nd147, nd232, nd235, nd238;

  if (!paramsOK) { // || (heModelAge <= 0.0)) {
    alphaCorrFactor = 1.0;
    ft232 = 1.0;
    ft235 = 1.0;
    ft238 = 1.0;
    ft147 = 1.0;
    return;
	}

  rate238 = 8.0*U238SEC;
  rate235 = 7.0*U235SEC;
  rate232 = 6.0*TH232SEC;
  rate147 = SM147SEC;

  rad = 0.0;
  innerVol = 0.0;
  dTotal = 0.0;
  ndTotal = 0.0;
  d147 = 0.0;
  d232 = 0.0;
  d235 = 0.0;
  d238 = 0.0;
  nd147 = 0.0;
  nd232 = 0.0;
  nd235 = 0.0;
  nd238 = 0.0;

  for (int i=0;i<rdim;i++) {
// Sum total volume of isotopes
    rad += gridSpacing;
    outerVol = rad*rad*rad;
	vol = outerVol - innerVol;
    d147 += vol*aDepl147[i]*max147;
    d232 += vol*aDepl232[i]*max232;
    d235 += vol*aDepl235[i]*max235;
    d238 += vol*aDepl238[i]*max238;
    nd147 += vol*nmpg147[i];
    nd232 += vol*nmpg232[i];
    nd235 += vol*nmpg235[i];
    nd238 += vol*nmpg238[i];

    dTotal += vol*(aDepl238[i]*max238*rate238+aDepl235[i]*max235*rate235+aDepl232[i]*max232*rate232+aDepl147[i]*max147*rate147);
    ndTotal += vol*(nmpg238[i]*rate238+nmpg235[i]*rate235+nmpg232[i]*rate232+nmpg147[i]*rate147);

    innerVol = outerVol;
  }
  alphaCorrFactor = (ndTotal == 0.0) ? 0.0 : dTotal/ndTotal;
  ft147 = (nd147 == 0.0) ? 0.0 : d147/nd147;
  ft232 = (nd232 == 0.0) ? 0.0 : d232/nd232;
  ft235 = (nd235 == 0.0) ? 0.0 : d235/nd235;
  ft238 = (nd238 == 0.0) ? 0.0 : d238/nd238;
}


/*	InterpolateTTPath
		Takes the time-temperature path specification and subdivides it into
    constant-rate time steps of approriate duration and temperature change
    for adequate estimation of diffusion.
 */
int RDAAM_InterpolateTTPath(TTPath * tTDef,double startTime)
{
	int dN,n;
  TTPathPoint nextPt;
	double  rate,absRate;   /* Rate of temperature change (K/m.y.) */
	double  timeStep;       /* Size of individual time step (m.y.) */
	double  defTimeStep;		/* Overall default time step (m.y.) */
	double	tempPerTimeStep;/* Temperature change per default time step (K) */
    double  currDefTimeStep;/* Default time step for the current path segment (m.y.) */
    double  endTemp;        /* Temperature at end of current t-T segment */

	double	maxRateAccel = 1.5;	/* maximum step-to-step increase in duration */
	double 	prevTimeStep;
	bool	  truncateStart = false;

	tTPath.clear();

  if (tTDef->size() < 2) return(0);

	nextPt.temperature = tTDef->back().temperature + KELVINS_AT_0C;
	nextPt.time = tTDef->back().time;
	tTPath.push_back(nextPt);

	if ((startTime == 0.0) || (startTime > nextPt.time)) startTime = nextPt.time;
	else truncateStart = true;

/* Default time step = 1% of model duration */
	defTimeStep = startTime * 0.01;
	prevTimeStep = defTimeStep;
	for (dN=tTDef->size()-1;dN>0;dN--) {
/* Calculate rate for this t-T segment: at least 5 steps (probably not required) */
		double currMaxTimeStep = (tTDef->at(dN).time-tTDef->at(dN-1).time) * 0.2;
		if (currMaxTimeStep == 0.0) return(0);  // Bad time step

		rate = (tTDef->at(dN).temperature-tTDef->at(dN-1).temperature)/(tTDef->at(dN).time-tTDef->at(dN-1).time);
		absRate = fabs(rate);
		tempPerTimeStep = absRate*defTimeStep;
		currDefTimeStep = (tempPerTimeStep <= maxTempStep) ? defTimeStep : maxTempStep/absRate;

		if (currDefTimeStep > currMaxTimeStep) currDefTimeStep = currMaxTimeStep;
// Check to make sure time step is large enough to register...
		if (currDefTimeStep < tTDef->at(dN).time*1.e-14)
			currDefTimeStep = tTDef->at(dN).time*1.e-14;

		endTemp = tTDef->at(dN-1).temperature + KELVINS_AT_0C;
		while (tTPath.back().time > tTDef->at(dN-1).time) {
			timeStep = currDefTimeStep;
			if (timeStep > prevTimeStep*maxRateAccel) timeStep = prevTimeStep * maxRateAccel;

/* Check to see if this is final step for this segment.  A small factor
	 is added to account for the possibility of roundoff. NOTE: This factor must
	 be significantly shorter than any time step. */
			if (timeStep*1.01 > tTPath.back().time - tTDef->at(dN-1).time) {
				nextPt.time = tTDef->at(dN-1).time;
				nextPt.temperature = endTemp;
			} else {
				nextPt.time = tTPath.back().time - timeStep;
				nextPt.temperature = tTPath.back().temperature - rate*timeStep;
			}
			tTPath.push_back(nextPt);
			if (truncateStart) {
				if (nextPt.time < startTime) {
					tTPath.clear();
					tTPath.push_back(nextPt);
					truncateStart = false;
				}
			}
			prevTimeStep = timeStep;
		}
	}
/* Convert Ma to seconds */
	for (n=0; n < (int)(tTPath.size()); n++) {
        tTPath[n].time *= SECS_PER_MA;
    }


	return(1);
}

// RDAAM_ExtractHeProfile
// Extracts the He profile and total He from the "u" array used in the
// 1D-FD solution.
// Solve the integral He = (He 4 pi r^2)dr using Romberg integration
void RDAAM_ExtractHeProfile()
{
// Convert u array to He; use scratch array to hold result
	double rad;
	double minHe=u[0]/(0.5*gridSpacing);
	int i,j;


	for (i=0; i<rdim; i++) {
		rad = (0.5+i)*gridSpacing;
		heProfile[i] = u[i]/rad;                // Convert to He
		if (heProfile[i] < minHe) minHe = heProfile[i];
		gam[i] = heProfile[i]*4.0*PI*rad*rad;   // Convert to integration quantity
	}


	if (minHe < -heProfile[0]*0.05) {
		totalHe = 0.0;
		return;
	}
	gam[rdim] = 0.0;

// Get total He using Romberg integration
	double step[20], h[20];
	double sum, tnm;
	double heTot, heErr;
	int fact, start, it;
// Generate trapezoidal rule points
	step[0] = 0.5*(gam[rdim]+gam[0])*(gridSpacing*rdim);
	h[0] = 1.0;
	for (i=1, fact=rombInit, start=rombInit, it=1;i<=rdimLog2;i++) {
		for (j=start, sum=0.0; j<rombLimit; j += fact) {
            sum += gam[j];
        }
		tnm = it;
		step[i] = 0.5*(step[i-1]+(gridSpacing*rdim)*sum/tnm);
		it *= 2;
		start /= 2;
		if (i>1) fact /= 2;
		h[i] = 0.25*h[i-1];
	}
	dpolint2(h+i-5, step+i-5, 5, 0.0, &heTot, &heErr);
// Fill in crystal center (a half-node spacing before first node); NR formula 4.1.10
	heTot += 0.5*gridSpacing*(gam[0]*55./24.-gam[1]*59./24.+gam[2]*37./24.-gam[3]*9./24.);

// Convert to same basis as total238, etc.
	heTot /= 1.333333*PI;
	totalHe = heTot;
}


/*  RDAAM_InitFTAnnealingTraps
		Precalculates the state of all traps at all times in preparation for using
		in the diffusion model.
 */
void RDAAM_InitFTAnnealingTraps(double ** &annealingTraps, bool optimize)
{
	bool isZircon = (psiUnits == PSI_D0N17_1_SEC);
  int numTTNodes = tTPath.size();
  int otNode=0; // Oldest-trap (remaining at present day) node
  int i,j,k;
	int node, tsNode;
  double *fInit, **fState, **heState;
	double equivTime, tempCalc, timeInt, x1, x2;
  double trapKappa = 1.04-trapRmr0;
	double invTrapKappa = 1.0/trapKappa;
	double trapTotAnnealLen = 0.55;
	double trapTotAnnealDivisor = 1.0-trapTotAnnealLen;
  double geomFactor = 1.0; // 0.5;
  double equivTotAnnealLen = pow(trapTotAnnealLen,invTrapKappa)*(1.0-trapRmr0)+trapRmr0;
	annealParamRec zirAnnealParams = {6.24354,-0.11977,-314.93688,-14.286838,-0.057206897,0.0,0.0};
	annealParamRec apAnnealParams = {0.39528,0.01073,-65.12969,-7.91715,0.04672,0.0,0.0};  // K07
	annealParamRec annealParams = isZircon ? zirAnnealParams : apAnnealParams;
	double heU238, heU235, heTh232, heSm147, globalHeState, accum;

	double etaQ = 0.91;
	double R = isZircon ? 0.000552 : 0.000815;   // Etchable range of one fission fragment (in cm)
	double densityConv = isZircon ? 6.022e14 : N_PER_NMOL*DENSITY_APATITE*(U238YR_SF/U238YR)*etaQ*R/8.0;	// Density conversion factor

	int tsNode1, initEndpoint;

	endNode = 0;
	if (optimize) {
	// In the first step, find age of the "oldest trap" persisting at present day.
		fInit = dvector(0,numTTNodes-1);
		equivTime = 0;
		tempCalc = log(2.0/(tTPath[numTTNodes-2].temperature + tTPath[numTTNodes-1].temperature));
		for (node = numTTNodes-2; node >= 0; node--) {
			timeInt = tTPath[node].time - tTPath[node+1].time + equivTime;
			if (timeInt <= 0.0) continue;  // For the occasional zero-length time step
			x1 = (log(timeInt) - annealParams.c2)/(tempCalc - annealParams.c3);
			x2 = annealParams.c0 + annealParams.c1 * x1;
			fInit[node] = pow(x2,1.0/annealParams.a) + 1.0;
			fInit[node] = (fInit[node] <= 0) ? 0.0 : 1.0/fInit[node];
			if (fInit[node] < equivTotAnnealLen) {
				fInit[node] = 0.0;
				otNode = node;
				break;
			}
			if (node == 0) break;
			else if (fInit[node] < 0.999) {   // *** Change to 2/sum
				tempCalc = log(2.0/(tTPath[node-1].temperature + tTPath[node].temperature));
				equivTime = pow(1.0/fInit[node]-1.0,annealParams.a);
				equivTime = (equivTime - annealParams.c0)/annealParams.c1;
				equivTime = exp(equivTime*(tempCalc-annealParams.c3)+annealParams.c2);
			}
		}
	// Next, find age of the oldest trap when the present-day oldest trap formed.
	// This ensures that we start early enough that all traps are represented.
		if (otNode > 0) {
			equivTime = 0;
			for (node = otNode; node >= 0; node--) {
				timeInt = tTPath[node].time - tTPath[node+1].time + equivTime;
				if (timeInt <= 0.0) continue;  // For the occasional zero-length time step
				x1 = (log(timeInt) - annealParams.c2)/(tempCalc - annealParams.c3);
				x2 = annealParams.c0 + annealParams.c1 * x1;
				fInit[node] = pow(x2,1.0/annealParams.a) + 1.0;
				fInit[node] = (fInit[node] <= 0) ? 0.0 : 1.0/fInit[node];

				if (fInit[node] <= equivTotAnnealLen) {
					fInit[node] = 0.0;
					endNode = node;		// final node to use
					break;
				}
				if (node == 0) break;
					else if (fInit[node] < 0.999) {   // *** Change to 2/sum
					tempCalc = log(2.0/(tTPath[node-1].temperature + tTPath[node].temperature));
					equivTime = pow(1.0/fInit[node]-1.0,annealParams.a);
					equivTime = (equivTime - annealParams.c0)/annealParams.c1;
					equivTime = exp(equivTime*(tempCalc-annealParams.c3)+annealParams.c2);
				}
			}
		}
		initEndpoint = (fInit[otNode] >= 0) ? otNode+1 : otNode;
		tsNode1 = numTTNodes-3;
	} else tsNode1 = numTTNodes-2;

// Now create fstate array, holding f of traps formed in each previous time step
	fState = dmatrix(0,numTTNodes-1,0,numTTNodes-1);

	if (optimize) {
		if (psiUnits == PSI_CMSQ_TRACK) {
			for (node=numTTNodes-2; node >= initEndpoint; node--) {
				fState[numTTNodes-2][node] = pow((fInit[node]-trapRmr0)/(1.0-trapRmr0),trapKappa);
				if (fState[numTTNodes-2][node] > 0.757) fState[numTTNodes-2][node] = 1.600*fState[numTTNodes-2][node]-0.599;
				else fState[numTTNodes-2][node] = (9.205*fState[numTTNodes-2][node]*fState[numTTNodes-2][node]-9.157*fState[numTTNodes-2][node]+2.269);
			}
		} else if (isZircon) {
			for (node=numTTNodes-2; node >= initEndpoint; node--) {
				fState[numTTNodes-2][node] = fInit[node];
				fState[numTTNodes-2][node] = 1.25*fState[numTTNodes-2][node]-0.25;
			}
		}
	//
		if (fInit != NULL) free_dvector(fInit,0);
		for (; node >= endNode; node--) fState[numTTNodes-2][node] = 0.0;
	}

// Now do the calculation again for all other time steps
	for (tsNode = tsNode1; tsNode >= endNode; tsNode--) {
		equivTime = 0;
		tempCalc = log(2.0/(tTPath[tsNode].temperature + tTPath[tsNode+1].temperature));
		for (node = tsNode; node >= endNode; node--) {
			timeInt = tTPath[node].time - tTPath[node+1].time + equivTime;
			if (timeInt <= 0.0) continue;  // For the occasional zero-length time step
			x1 = (log(timeInt) - annealParams.c2)/(tempCalc - annealParams.c3);
			x2 = annealParams.c0 + annealParams.c1 * x1;
			fState[tsNode][node] = pow(x2,1.0/annealParams.a) + 1.0;
			fState[tsNode][node] = (fState[tsNode][node] <= 0) ? 0.0 : 1.0/fState[tsNode][node];
			if (fState[tsNode][node] < equivTotAnnealLen) {
				fState[tsNode][node] = 0.0;
				node++;
				break;
			}
			if (node == 0) break;
			if (fState[tsNode][node] < 0.999) {   // *** Change to 2/sum
				tempCalc = log(2.0/(tTPath[node-1].temperature + tTPath[node].temperature));
				equivTime = pow(1.0/fState[tsNode][node]-1.0,annealParams.a);
				equivTime = (equivTime - annealParams.c0)/annealParams.c1;
				equivTime = exp(equivTime*(tempCalc-annealParams.c3)+annealParams.c2);
			}
		}
		if (node < endNode) node = endNode;  // In case we went past the end.
		// Do rmr0 and volume conversions.
		if (psiUnits == PSI_CMSQ_TRACK) {
			for (int tempNode=tsNode; tempNode >= node; tempNode--) {
				fState[tsNode][tempNode] = pow((fState[tsNode][tempNode]-trapRmr0)/(1.0-trapRmr0),trapKappa);
				if (fState[tsNode][tempNode] > 0.757) fState[tsNode][tempNode] = 1.600*fState[tsNode][tempNode]-0.599;
				else fState[tsNode][tempNode] = (9.205*fState[tsNode][tempNode]*fState[tsNode][tempNode]-9.157*fState[tsNode][tempNode]+2.269);
			}
		} else if (isZircon) {
			for (int tempNode=tsNode; tempNode >= node; tempNode--) {
				fState[tsNode][tempNode] = 1.25*fState[tsNode][tempNode]-0.25;
			}
		}
		// Zero out the rest
		for (node--; node >= endNode; node--) fState[tsNode][node] = 0.0;
	}
// OK, that takes care of f.  Now for He
	heState = dmatrix(0,numTTNodes-1,0,rdim);
	for (tsNode=numTTNodes-2; tsNode >= endNode; tsNode--) {
// First calculate baseline He for each isotope
		heU238 = 8. * (exp(U238SEC*tTPath[tsNode].time) - exp(U238SEC*tTPath[tsNode+1].time));
		heU235 = 7. * (exp(U235SEC*tTPath[tsNode].time) - exp(U235SEC*tTPath[tsNode+1].time));
		heTh232 = 6. * (exp(TH232SEC*tTPath[tsNode].time) - exp(TH232SEC*tTPath[tsNode+1].time));
		heSm147 = 1. * (exp(SM147SEC*tTPath[tsNode].time) - exp(SM147SEC*tTPath[tsNode+1].time));
// Then combine for each node

		//globalHeState = densityConv*(nmpg238[0]*heU238 + nmpg235[0]*heU235 + nmpg232[0]*heTh232 + nmpg147[0]*heSm147);
		for (int radNode=0; radNode < rdim; radNode++) {
            globalHeState = densityConv*(nmpg238[radNode]*heU238 + nmpg235[radNode]*heU235 + nmpg232[radNode]*heTh232 + nmpg147[radNode]*heSm147);
            heState[tsNode][radNode] = globalHeState;
        }
	}
// Finally, combine f and He into traps at each node at each time
	annealingTraps = dmatrix(0,numTTNodes-1,0,rdim);
	for (tsNode=numTTNodes-2; tsNode >= endNode; tsNode--) {
		annealingTraps[tsNode][0] = 0.0;
		for (node=tsNode; node >= endNode; node--)
			annealingTraps[tsNode][0] += heState[node][0]*fState[tsNode][node];
		for (int radNode=1; radNode < rdim; radNode++)
			 annealingTraps[tsNode][radNode] = annealingTraps[tsNode][0];
	}

	if (fState != NULL) free_dmatrix(fState,0,numTTNodes-1,0,numTTNodes-1);
  if (heState != NULL) free_dmatrix(heState,0,numTTNodes-1,0,rdim);
}





// RDAAM_CalcHeAge
// The principal FD solver
void RDAAM_CalcHeAge(bool optimize)
{
	double dt;    // Time step length (s)
	double diff;  // Diffusivity at current time step (cm^2/s)
	double r;     // Radial position (cm)
	double beta, p2mb, n2mb;
	double preBeta, trapExpTerm, trapDiffTerm, diffTrap;
	double A, new238, new235, new232, new147, exp238, exp235, exp232, exp147, t1, newHe;
	double **annealingTraps = NULL; // 2D matrix to hold trap info if there's annealing
	double invRadCubed;
    double d,w;

    double bb[rdim][rdim]; // new variable for cylindric diffusion (CG 23/03)
    double AA[rdim]; // new variable for cylindric diffusion (CG 23/03)
    double diag2[rdim]; // new variable for cylindric diffusion (CG 23/03)

// Stuff for Guenthner et al model
	bool isZircon = (psiUnits == PSI_D0N17_1_SEC);
	double tortTerm = 4.2/1.669;
//	double fa_lint0 = 0.0004;
	double fa_lint0 = 0.0000548;  // = 1.0-exp(1.e14*B_ALPHA);
//	double lint0sq = 625.76*625.76;
	double lint0sq = 45920.*45920.;
	double fa, fc, falint, lint, tortuosity;
	double diffN17;
	double radCmSq = radius/1.e4;   // squared radius in cm
	radCmSq *= radCmSq;

	int i,j;
	unsigned int node;

	for (i=0;i<rdim;i++) u[i] = 0.0;
	totalHe = 0.0;

	if (!nonzeroProduction) {
		heModelAge = 0.0;
		heCorrModelAge = 0.0;
		RDAAM_ExtractHeProfile();
		return;
	}

	endNode = 0;
	RDAAM_InitFTAnnealingTraps(annealingTraps, optimize);

	if (optimize && (endNode > tTPath.size()*0.7)) {  // Redo if not enough nodes are used (70% of path unused)
		if (annealingTraps != NULL) free_dmatrix(annealingTraps,0,tTPath.size()-1,0,rdim);
		RDAAM_InterpolateTTPath(tTDef, tTPath[endNode].time/SECS_PER_MA);
		RDAAM_InitFTAnnealingTraps(annealingTraps, optimize);
	}

// Initialize variables for U, Th decay
	t1 = tTPath[endNode].time;
	exp238 = exp(U238SEC*t1);
	exp235 = exp(U235SEC*t1);
	exp232 = exp(TH232SEC*t1);
	exp147 = exp(SM147SEC*t1);

// We're set -- start to run

	for (node=endNode;node < tTPath.size()-1;node++) {
		dt = tTPath[node].time - tTPath[node+1].time;
// Occasionally nodes will be spaced below floating-point resolution limit...
		if (dt <= 0.0) continue;

		diff = dInf * exp(-E*1000.0/(UNIV_GAS_CONST*(tTPath[node].temperature+tTPath[node+1].temperature)*0.5));
		if (isZircon)
			diffN17 = psi * exp(-Et*1000.0/(UNIV_GAS_CONST*(tTPath[node].temperature+tTPath[node+1].temperature)*0.5));
		else
			trapExpTerm = exp(Et*1000.0/(UNIV_GAS_CONST*(tTPath[node].temperature+tTPath[node+1].temperature)*0.5));

		preBeta = 2.0*gridSpacing*gridSpacing/dt;
		beta = 2.0*gridSpacing*gridSpacing/(diff * dt);
		p2mb = 2.0 - beta;
		n2mb = -2.0 - beta;

	// He production
		t1 = tTPath[node+1].time;
		new238 = exp(U238SEC*t1);
		new235 = exp(U235SEC*t1);
		new232 = exp(TH232SEC*t1);
		new147 = exp(SM147SEC*t1);

	// Load arrays
		for (i=0;i<rdim;i++) {   // Main tridiagonal components
			A = 8.*aDepl238[i]*max238*(exp238-new238)+
					6.*aDepl232[i]*max232*(exp232-new232)+
					7.*aDepl235[i]*max235*(exp235-new235)+
					aDepl147[i]*max147*(exp147-new147);
            AA[i]=A;
			if (isZircon) {
				falint = 1.-exp(-annealingTraps[node][i]*B_ALPHA);
				if (falint > fa_lint0) {
					lint = tortTerm/falint - 2.5;
					tortuosity = lint0sq/(lint*lint);
				} else tortuosity = 1.0;
				fa = 1.-exp(-annealingTraps[node][i]*B_ALPHA*polyA);
				fc = 1.-fa;
				diffTrap = (1./(tortuosity*fc*fc*fc*radCmSq/diff + fa*fa*fa*radCmSq/diffN17))*radCmSq;
			} else {
				trapDiffTerm = (psi*annealingTraps[node][i] + polyA*pow(annealingTraps[node][i],3))*trapExpTerm+1.0;
				diffTrap = diff/trapDiffTerm;
			}
			betaTrap[i] = preBeta/diffTrap;
			prodHe[i] = betaTrap[i]*A*(i+0.5)*gridSpacing;
		}

  // Set production variables for next loop
    exp238 = new238;
    exp235 = new235;
	exp232 = new232;
    exp147 = new147;

	// Neumann BC at center
		b[0] = (3.0-betaTrap[0])*u[0] - u[1] - prodHe[0];
		diag[0] = -3.0-betaTrap[0];
	// Zero Dirichlet BC at right
		b[rdim-1] = (2.0-betaTrap[rdim-1])*u[rdim-1] - u[rdim-2] - prodHe[rdim-1];
		diag[rdim-1] = -2.0-betaTrap[rdim-1];
	// Load main array
		for (i=1; i<rdim-1; i++) {
			b[i] = (2.0-betaTrap[i])*u[i] - u[i+1] - u[i-1] - prodHe[i];
			diag[i] = -2.0-betaTrap[i];
		}

  // Solve it
		dtridag2(diag, b, u, gam, rdim);

        // solve 1D diffusion equation with FD using Implicit Euler Method
        // good source: http://hplgit.github.io/Programming-for-Computations/pub/p4c/p4c-sphinx-Matlab/._pylight006.html
        // assuming an infinite cylindrical geometry and symmetry

        // define d
        d=diffTrap*dt/pow(gridSpacing,2);

        // create b
        bb[0][0] = 1+2*d;
        bb[0][1] = -2*d;
        for (i=1; i<rdim; i++) {
            bb[i][i-1] = -(d*gridSpacing)/(radius/1.e4)-d;
            bb[i][i]=1+2*d;
            bb[i][i+1]=(d*gridSpacing)/(radius/1.e4)-d;
        }
        bb[rdim][rdim]=1;

        // create diag
        for (i=0; i<rdim; i++) {
            diag2[i]=He[i]+AA[i];
        }
        diag2[rdim]=0;

        // solve tridiagonal matrix

        // forward elimination
        for (i=1; i<rdim+1; i++) {
            w = bb[i][i-1]/bb[i-1][i-1];
            bb[i][i] = bb[i][i]-w*bb[i-1][i];
            diag2[i] = diag2[i]-w*diag2[i-1];
        }

        // backward substitution
        He[rdim] = diag2[rdim]/bb[rdim][rdim];
        for (i=rdim-1; i>=0; i--) {
            He[i] = (diag2[i]-bb[i][i+1]*He[i+1])/bb[i][i];
        }

	}
    // FD iteration done
	RDAAM_ExtractHeProfile();
	heProfile[rdim] = 0.0;

    // Iterate to find uncorrected age
	double leftSum = total232 + total235 + total238 + total147 + totalHe;
	double midVal, hiAge, loAge, midAge;
	loAge = 0;
	hiAge = tTPath[0].time/SECS_PER_YR;  // Convert to years
	while (hiAge - loAge > ageConv) {
		midAge = (hiAge + loAge)/2.0;
		midVal = total238*exp(U238YR*midAge) +
						 total235*exp(U235YR*midAge) +
						 total232*exp(TH232YR*midAge) +
						 total147*exp(SM147YR*midAge);
		if (midVal < leftSum) loAge = midAge;
		else hiAge = midAge;
	}
	heModelAge = (hiAge+loAge)/2.0;
	heModelAge /= 1.e6;   // Convert from years to Ma

// Calculate corrected He age
	leftSum = total232*ft232 + total235*ft235 + total238*ft238 + total147*ft147 + totalHe;
	loAge = 0;
	hiAge = 1.2*tTPath[0].time/SECS_PER_YR;  // Convert to years
	while (hiAge - loAge > ageConv) {
		midAge = (hiAge + loAge)/2.0;
		midVal = total238*ft238*exp(U238YR*midAge) +
						 total235*ft235*exp(U235YR*midAge) +
						 total232*ft232*exp(TH232YR*midAge) +
						 total147*ft147*exp(SM147YR*midAge);
		if (midVal < leftSum) loAge = midAge;
		else hiAge = midAge;
	}
	heCorrModelAge = (hiAge+loAge)/2.0;
	heCorrModelAge /= 1.e6;   // Convert from years to Ma

	if (annealingTraps != NULL) free_dmatrix(annealingTraps,0,tTPath.size()-1,0,rdim);
}

// RDAAM_FreeCalcArrays
// Use this to clean up memory after you're done with all of your RDAAM calculations
void RDAAM_FreeCalcArrays()
{
	if (diag != NULL) free_dvector(diag,0);
  if (b != NULL) free_dvector(b,0);
  if (gam != NULL) free_dvector(gam,0);
  if (prodHe != NULL) free_dvector(prodHe,0);
  if (u != NULL) free_dvector(u,0);
	if (aDepl238 != NULL) free_dvector(aDepl238,0);
	if (aDepl235 != NULL) free_dvector(aDepl235,0);
  if (aDepl232 != NULL) free_dvector(aDepl232,0);
  if (aDepl147 != NULL) free_dvector(aDepl147,0);
	if (aEjOnly238 != NULL) free_dvector(aEjOnly238,0);
	if (aEjOnly235 != NULL) free_dvector(aEjOnly235,0);
	if (aEjOnly232 != NULL) free_dvector(aEjOnly232,0);
	if (aEjOnly147 != NULL) free_dvector(aEjOnly147,0);
	if (nmpg238 != NULL) free_dvector(nmpg238,0);
  if (nmpg235 != NULL) free_dvector(nmpg235,0);
  if (nmpg232 != NULL) free_dvector(nmpg232,0);
  if (nmpg147 != NULL) free_dvector(nmpg147,0);
  if (alphaDamage != NULL) free_dvector(alphaDamage,0);
	if (betaTrap != NULL) free_dvector(betaTrap,0);
}

void nrerror(char *error_text)
{
	fprintf(stderr,"Run-time error: %s\n", error_text);

  exit(1);
}

// dtridag2
// Solves tridiagonal matrix.
// Based on Numerical Recipes, with a couple of changes
void dtridag2(double diag[], double b[], double u[], double gam[], int n)
{
	int j;
	double bet;

	/* Error checking for diag[0]=0 left out */
	u[0] = b[0]/(bet=diag[0]);
	for (j=1;j<n;j++) {
		gam[j] = 1.0/bet;
		bet = diag[j]-gam[j];
		/* Error-checking for bet=0 left out */
		u[j] = (b[j]-u[j-1])/bet;
	}
	for (j=(n-2); j >= 0; j--) {
		u[j] -= gam[j+1]*u[j+1];
    }
}

/*  dpolint2
		Given arrays xa[1..n] and ya[1..n], and given a value x, this routine
		returns a value y and an error estimate dy.  If P(x) is a polynomial of
		degree n-1 such that P(xai)=yai, i=[1..n], then the returned value y = P(x).

		Converted to 0-base arrays
 */
void dpolint2(double xa[], double ya[], int n, double x, double *y, double *dy)
{
	int i, m, ns=1;
	double den, dif, dift, ho, hp, w;
	double *c, *d;

	dif = fabs(x-xa[0]);
	c=dvector(0,n-1);
	d=dvector(0,n-1);
	for (i=0; i<n;i++) {
		if ( (dift=fabs(x-xa[i])) < dif) {
			ns=i;
			dif=dift;
		}
		c[i]=ya[i];
		d[i]=ya[i];
	}
	*y=ya[ns--];
	for (m=1;m<n;m++) {
        for (i=0;i<n-m;i++) {
	        ho=xa[i]-x;
            hp=xa[i+m]-x;
		    w=c[i+1]-d[i];
		    if ( (den=ho-hp) == 0.0) nrerror("Error with routine DPOLINT2");
            den=w/den;
		    d[i]=hp*den;
            c[i]=ho*den;
        }
		*y += (*dy=(2*ns < (n-m) ? c[ns+1] : d[ns--]));
  }
	free_dvector(d,0);
	free_dvector(c,0);
}

/* dvector
    Creates a 1D array of double with indices from nl to nh
    Must be deallocated using free_dvector()
 */
double *dvector(int nl, int nh)
{
  double *v;
  v = (double *)malloc((unsigned) (nh-nl+1)*sizeof(double));
	if (!v) nrerror("allocation failure in dvector()");
  return v-nl;
}

/* free_dvector
    Frees a vector created by dvector()
 */
void free_dvector(double *v, int nl)
{
  free((char *) (v+nl));
}

/* dmatrix
    Creates 2D array of double with indices [nrl..nrh][ncl..nch]
    Must be deallocated with free_dmatrix()
 */
double **dmatrix(int nrl, int nrh, int ncl, int nch)
{
  int i;
  double **m;

  m = (double **) malloc((unsigned) (nrh-nrl+1)*sizeof(double*));
  if (!m) nrerror("allocation failure 1 in dmatrix()");
  m -= nrl;

  for (i=nrl; i <= nrh; i++) {
    m[i] = (double *) malloc((unsigned) (nch-ncl+1)*sizeof(double));
    if (!m[i]) nrerror("allocation failure 2 in dmatrix()");
    m[i] -= ncl;
  }
  return m;
}

/* free_dmatrix
    Frees a matrix created by dmatrix()
 */
void free_dmatrix(double **m, int nrl, int nrh, int ncl, int nch)
{
  int i;

	for (i=nrh; i >= nrl; i--) free((char *) m[i]+ncl);
  free((char *) (m+nrl));
}

// RDAAM_Calculate
// This is the principal routine for determining the AHe age from an input t-T
// path.  It presumes that RDAAM_Init has been called first.
// It provides values for variables heModelAge and heCorrModelAge in Ma, and totHe
// in nmol/g, and overall returns 1 if successful, 0 if not.
// Set optimize to true for geological paths to make them calculate faster, and
// false for degassing (i.e. any history that has short, high-temperature steps
// at the end).
extern "C" int RDAAM_Calculate(int ntime, double radius, double *ppm_U, double *ppm_Th, double *ppm_Sm, double time[], double temp[], double &age, double &corrAge, double &totHe, double *heProCyl, bool optimize)
{
	//printf("Hello, here RDAAM");
	RDAAM_Init(HE_PREC_MEDIUM, radius, ppm_U, ppm_Th, ppm_Sm);

	//vector<double> *heProCyl;

	age = 0.0;
	corrAge = 0.0;
	totHe = 0.0;

	TTPath path;
    path.clear();
    for (int i=0;i<ntime;i++) {
        TTPathPoint timetemp = {time[i],temp[i]};
        path.push_back(timetemp);
        }

	if (!paramsOK) return(0);

	if (RDAAM_InterpolateTTPath(&path, 0.0)) {

		RDAAM_CalcHeAge(optimize);

		age = heModelAge;
		corrAge = heCorrModelAge;

		for (int i=0;i<rdim;i++){
			heProCyl[i] = He[i];
			//printf("%f \n", heProCyl[i]);
		}

		float rad = radius/1.e4; // Convert to cm;
		totHe = totalHe/(rad*rad*rad);  // Convert from calc units  to nmol/g

		return(1);
	}
	return(1);
}
