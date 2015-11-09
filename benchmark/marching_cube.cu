/*
 * marching_cube.cu
 *
 *  Created on: Sep 7, 2012
 *      Author: ollie
 */

#include <sys/time.h>

#include <vtkImageData.h>
#include <vtkRTAnalyticSource.h>

#include <thrust/extrema.h>

#include "piston/marching_cube.h"
#include "piston/util/tangle_field.h"

#include "Stats.h"

static const float ISO_START=16;
static const float ISO_STEP=-1;
static const int ISO_NUM=15;
static const size_t DIMENSIONS = 128;

//#define SPACE thrust::host_space_tag
#define SPACE thrust::device_space_tag

using namespace piston;

int main()
{
  const int grid_size = DIMENSIONS + 1;

  tangle_field<SPACE>* tangle;
  marching_cube<tangle_field<SPACE>, tangle_field<SPACE> > *isosurface;

  tangle = new tangle_field<SPACE>(grid_size, grid_size, grid_size);
  isosurface = new marching_cube<tangle_field<SPACE>,  tangle_field<SPACE> >(*tangle, *tangle, ISO_START);

  std::vector<double> samples;

  const double MAX_RUNTIME = 30;
  const size_t MAX_ITERATIONS = 1;
  samples.reserve(MAX_ITERATIONS);

  size_t iter = 0;
  Timer timer;
  for (double el = 0.0; el < MAX_RUNTIME && iter < MAX_ITERATIONS; el += samples.back(), ++iter)
  {
    float isovalue = ISO_START;
    timer.Reset();
    for (int j=0; j<ISO_NUM; j++)
    {
      isovalue += ISO_STEP;
      isosurface->set_isovalue(isovalue);
      (*isosurface)();
      std::cout << isovalue << " " << isosurface->num_total_vertices << std::endl;
    }
    samples.push_back(timer.GetElapsedTime());
  }

  std::sort(samples.begin(), samples.end());
  stats::Winsorize(samples, 5.0);
  std::cout << "Benchmark \'VTK MPI Isosurface\' results:\n"
      << "\tmedian = " << stats::PercentileValue(samples, 50.0) << "s\n"
      << "\tmedian abs dev = " << stats::MedianAbsDeviation(samples) << "s\n"
      << "\tmean = " << stats::Mean(samples) << "s\n"
      << "\tstd dev = " << stats::StandardDeviation(samples) << "s\n"
      << "\tmin = " << samples.front() << "s\n"
      << "\tmax = " << samples.back() << "s\n"
      << "\t# of runs = " << samples.size() << "\n";


    return 0;
}



