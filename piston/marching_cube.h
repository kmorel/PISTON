/*
Copyright (c) 2011, Los Alamos National Security, LLC
All rights reserved.
Copyright 2011. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL),
which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software.

NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.

If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.

Additionally, redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
·         Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
·         Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other
          materials provided with the distribution.
·         Neither the name of Los Alamos National Security, LLC, Los Alamos National Laboratory, LANL, the U.S. Government, nor the names of its contributors may be used
          to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef MARCHING_CUBE_H_
#define MARCHING_CUBE_H_

#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <piston/image3d.h>
#include <piston/piston_math.h>
#include <piston/choose_container.h>
#include <piston/hsv_color_map.h>

#define MIN_VALID_VALUE -500.0

namespace piston {

template <typename InputDataSet1, typename InputDataSet2>
class marching_cube
{
public:
    typedef typename InputDataSet1::PointDataIterator InputPointDataIterator;
    typedef typename InputDataSet1::GridCoordinatesIterator InputGridCoordinatesIterator;
    typedef typename InputDataSet1::PhysicalCoordinatesIterator InputPhysCoordinatesIterator;
    typedef typename InputDataSet2::PointDataIterator ScalarSourceIterator;

    typedef typename thrust::iterator_difference<InputPointDataIterator>::type	diff_type;
    typedef typename thrust::iterator_space<InputPointDataIterator>::type	space_type;
    typedef typename thrust::iterator_value<InputPointDataIterator>::type	value_type;

    typedef typename thrust::counting_iterator<int, space_type>	CountingIterator;

    typedef typename detail::choose_container<InputPointDataIterator, int>::type  TableContainer;
    typedef typename detail::choose_container<InputPointDataIterator, int>::type  IndicesContainer;

    typedef typename detail::choose_container<InputPointDataIterator, float4>::type 	VerticesContainer;
    typedef typename detail::choose_container<InputPointDataIterator, float3>::type	NormalsContainer;
    typedef typename detail::choose_container<ScalarSourceIterator, float>::type	ScalarContainer;

    typedef typename TableContainer::iterator	 TableIterator;
    typedef typename VerticesContainer::iterator VerticesIterator;
    typedef typename IndicesContainer::iterator  IndicesIterator;
    typedef typename NormalsContainer::iterator  NormalsIterator;
    typedef typename ScalarContainer::iterator   ScalarIterator;

    static const int triTable_array[256][16];
    static const int numVerticesTable_array[256];

    InputDataSet1 &input;		// scalar field for generating isosurface/cut geometry
    InputDataSet2 &source;		// scalar field for generating interpolated scalar values

    value_type isovalue;
    bool discardMinVals;
    bool useInterop;

    TableContainer	triTable;	// a copy of triangle edge indices table in host|device_vector
    TableContainer	numVertsTable;	// a copy of number of vertices per cell table in host|device_vector

    IndicesContainer	case_index;	// classification of cells as indices into triTable and numVertsTable
    IndicesContainer	num_vertices;	// number of vertices will be generated by the cell

    IndicesContainer 	valid_cell_enum;	// enumeration of valid cells
    IndicesContainer	valid_cell_indices;	// a sequence of indices to valid cells

    IndicesContainer 	output_vertices_enum;	// enumeration of output vertices, only valid ones

#ifdef USE_INTEROP
    value_type minIso, maxIso;
    bool colorFlip;
    float4 *vertexBufferData;
    float3 *normalBufferData;
    float4 *colorBufferData;
    int vboSize;
    GLuint vboBuffers[3];
    struct cudaGraphicsResource* vboResources[3]; // vertex buffers for interop
#endif

    VerticesContainer	vertices; 	// output vertices, only valid ones
    NormalsContainer	normals;	// surface normal computed by cross product of triangle edges
    ScalarContainer	scalars;	// interpolated scalar output

    unsigned int num_total_vertices;

    marching_cube(InputDataSet1 &input, InputDataSet2 &source,
                  value_type isovalue = value_type()) :
	input(input), source(source), isovalue(isovalue),
	discardMinVals(true), useInterop(false),
	triTable((int*) triTable_array, (int*) triTable_array+256*16),
	numVertsTable((int *) numVerticesTable_array, (int *) numVerticesTable_array+256)
#ifdef USE_INTEROP
    , colorFlip(false), vboSize(0)
#endif
	{}

    void freeMemory(bool includeInput=true)
    {
	if (includeInput) {
	    case_index.clear();
	    num_vertices.clear();
	    valid_cell_enum.clear();
	}
	valid_cell_indices.clear();
	output_vertices_enum.clear();
	vertices.clear();
	normals.clear();
	scalars.clear();
    }

    void operator()()
    {
	const int NCells = input.NCells;

	case_index.resize(NCells);
	num_vertices.resize(NCells);
        //thrust::copy(input.point_data_begin(), input.point_data_begin()+20, std::ostream_iterator<float>(std::cout, " "));  std::cout << std::endl;
        //std::cout << std::endl;

	// classify all cells, generate indices into triTable and numVertsTable,
	// we also use numVertsTable to generate numVertices for each cell
	thrust::transform(CountingIterator(0), CountingIterator(0)+NCells,
	                  thrust::make_zip_iterator(thrust::make_tuple(case_index.begin(), num_vertices.begin())),
	                  classify_cell(input, isovalue, discardMinVals,
	                                numVertsTable.begin()));
	
        // enumerating valid cells
	valid_cell_enum.resize(NCells);
	thrust::transform_inclusive_scan(num_vertices.begin(), num_vertices.end(),
	                                 valid_cell_enum.begin(),
	                                 is_valid_cell(),
	                                 thrust::plus<int>());
	// the total number of valid cells is the last element of the enumeration.
	unsigned int num_valid_cells = valid_cell_enum.back();
	
        // no valid cells at all, return with empty vectors.
	if (num_valid_cells == 0) {
	    vertices.clear();
	    normals.clear();
	    scalars.clear();
	    return;
	}

	// find indices to valid cells
	valid_cell_indices.resize(num_valid_cells);
	thrust::upper_bound(valid_cell_enum.begin(), valid_cell_enum.end(),
	                    CountingIterator(0), CountingIterator(0)+num_valid_cells,
	                    valid_cell_indices.begin());
	
        // use indices to valid cells to fetch number of vertices generated by
	// valid cells and do an enumeration to get the output indices for
	// the first vertex generated by the valid cells.
	output_vertices_enum.resize(num_valid_cells);
	thrust::exclusive_scan(thrust::make_permutation_iterator(num_vertices.begin(), valid_cell_indices.begin()),
	                       thrust::make_permutation_iterator(num_vertices.begin(), valid_cell_indices.begin()) + num_valid_cells,
	                       output_vertices_enum.begin());

	// get the total number of vertices,
	num_total_vertices = num_vertices[valid_cell_indices.back()] + output_vertices_enum.back();

	if (useInterop) {
#if USE_INTEROP
	    if (num_total_vertices > vboSize)
	    {
              glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[0]);
              glBufferData(GL_ARRAY_BUFFER, num_total_vertices*sizeof(float4), 0, GL_DYNAMIC_DRAW);
              if (glGetError() == GL_OUT_OF_MEMORY) { std::cout << "Out of VBO memory" << std::endl; exit(-1); }
              glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[1]);
              glBufferData(GL_ARRAY_BUFFER, num_total_vertices*sizeof(float4), 0, GL_DYNAMIC_DRAW);
              if (glGetError() == GL_OUT_OF_MEMORY) { std::cout << "Out of VBO memory" << std::endl; exit(-1); }
              glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[2]);
              glBufferData(GL_ARRAY_BUFFER, num_total_vertices*sizeof(float3), 0, GL_DYNAMIC_DRAW);
              if (glGetError() == GL_OUT_OF_MEMORY) { std::cout << "Out of VBO memory" << std::endl; exit(-1); }
              glBindBuffer(GL_ARRAY_BUFFER, 0);
              vboSize = num_total_vertices;
	    }
	    size_t num_bytes;
	    cudaGraphicsMapResources(1, &vboResources[0], 0);
	    cudaGraphicsResourceGetMappedPointer((void **) &vertexBufferData,
						 &num_bytes, vboResources[0]);

	    if (vboResources[1]) {
		cudaGraphicsMapResources(1, &vboResources[1], 0);
		cudaGraphicsResourceGetMappedPointer((void **) &colorBufferData,
						     &num_bytes, vboResources[1]);
	    }

	    cudaGraphicsMapResources(1, &vboResources[2], 0);
	    cudaGraphicsResourceGetMappedPointer((void **) &normalBufferData,
						 &num_bytes, vboResources[2]);
#endif
	} else {
	    vertices.resize(num_total_vertices);
	    normals.resize(num_total_vertices);
	}
	//scalars.resize(num_total_vertices);

	// do edge interpolation for each valid cell
	if (useInterop) {
#if USE_INTEROP
	    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(valid_cell_indices.begin(), output_vertices_enum.begin(),
	                                                                  thrust::make_permutation_iterator(case_index.begin(), valid_cell_indices.begin()),
	                                                                  thrust::make_permutation_iterator(num_vertices.begin(), valid_cell_indices.begin()))),
	                     thrust::make_zip_iterator(thrust::make_tuple(valid_cell_indices.end(), output_vertices_enum.end(),
	                                                                  thrust::make_permutation_iterator(case_index.begin(), valid_cell_indices.begin()) + num_valid_cells,
	                                                                  thrust::make_permutation_iterator(num_vertices.begin(), valid_cell_indices.begin()) + num_valid_cells)),
	                     isosurface_functor(input, source, isovalue,
	                                        triTable.begin(),
	                                        vertexBufferData,
	                                        normalBufferData,
	                                        thrust::raw_pointer_cast(&*scalars.begin())));
	    if (vboResources[1])
		thrust::transform(scalars.begin(), scalars.end(),
		                  thrust::device_ptr<float4>(colorBufferData),
		                  color_map<float>(minIso, maxIso, colorFlip));
	    for (int i = 0; i < 3; i++)
		cudaGraphicsUnmapResources(1, &vboResources[i], 0);
#endif
	} else {
	    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(valid_cell_indices.begin(), output_vertices_enum.begin(),
	                                                                  thrust::make_permutation_iterator(case_index.begin(),   valid_cell_indices.begin()),
	                                                                  thrust::make_permutation_iterator(num_vertices.begin(), valid_cell_indices.begin()))),
	                     thrust::make_zip_iterator(thrust::make_tuple(valid_cell_indices.end(),   output_vertices_enum.end(),
	                                                                  thrust::make_permutation_iterator(case_index.begin(),   valid_cell_indices.begin()) + num_valid_cells,
	                                                                  thrust::make_permutation_iterator(num_vertices.begin(), valid_cell_indices.begin()) + num_valid_cells)),
	                     isosurface_functor(input, source,isovalue,
	                                        triTable.begin(),
	                                        thrust::raw_pointer_cast(&*vertices.begin()),
	                                        thrust::raw_pointer_cast(&*normals.begin())/*,
	                                        thrust::raw_pointer_cast(&*scalars.begin())*/));
	}
    }

    struct classify_cell : public thrust::unary_function<int, thrust::tuple<int, int> >
    {
	// FixME: constant iterator and/or iterator to const problem.
	InputPointDataIterator	point_data;
	const float		isovalue;
	const bool 		discardMinVals;
	TableIterator		numVertsTable;

	const int xdim;
	const int ydim;
	const int zdim;
	const int cells_per_layer;
	const int points_per_layer;

	classify_cell(InputDataSet1 &input,
	              float isovalue, bool discardMinVals,
	              TableIterator numVertsTable) :
	        	  point_data(input.point_data_begin()),
	        	  isovalue(isovalue),
	        	  discardMinVals(discardMinVals),
	        	  numVertsTable(numVertsTable),
	        	  xdim(input.dim0), ydim(input.dim1), zdim(input.dim2),
	        	  cells_per_layer((xdim - 1) * (ydim - 1)),
	        	  points_per_layer (xdim*ydim) {}

	__host__ __device__
	thrust::tuple<int, int> operator() (int cell_id) const {
	    // FIXME: this integer division/modulus is repeated at every
	    // instance of the input iterator when the scalars are computed
	    // on the fly.
	    const int x = cell_id % (xdim - 1);
	    const int y = (cell_id / (xdim - 1)) % (ydim -1);
	    const int z = cell_id / cells_per_layer;

	    // indices to the eight vertices of the voxel
	    const int i0 = x    + y*xdim + z * points_per_layer;
	    const int i1 = i0   + 1;
	    const int i2 = i0   + 1	+ xdim;
	    const int i3 = i0   + xdim;

	    const int i4 = i0   + points_per_layer;
	    const int i5 = i1   + points_per_layer;
	    const int i6 = i2   + points_per_layer;
	    const int i7 = i3   + points_per_layer;

	    // FIXME: there is too much redundant computation to get
	    // triple (col, row, layer) in the input iterator when data
	    // is calculated on the fly
	    const float f0 = *(point_data + i0);
	    const float f1 = *(point_data + i1);
	    const float f2 = *(point_data + i2);
	    const float f3 = *(point_data + i3);
	    const float f4 = *(point_data + i4);
	    const float f5 = *(point_data + i5);
	    const float f6 = *(point_data + i6);
	    const float f7 = *(point_data + i7);

	    unsigned int cubeindex = (f0 > isovalue);
	    cubeindex += (f1 > isovalue)*2;
	    cubeindex += (f2 > isovalue)*4;
	    cubeindex += (f3 > isovalue)*8;
	    cubeindex += (f4 > isovalue)*16;
	    cubeindex += (f5 > isovalue)*32;
	    cubeindex += (f6 > isovalue)*64;
	    cubeindex += (f7 > isovalue)*128;

	    //bool valid = (!discardMinVals) || ((f0 > MIN_VALID_VALUE) && (f1 > MIN_VALID_VALUE) && (f2 > MIN_VALID_VALUE) && (f3 > MIN_VALID_VALUE) &&
	    //	    	    	               (f4 > MIN_VALID_VALUE) && (f5 > MIN_VALID_VALUE) && (f6 > MIN_VALID_VALUE) && (f7 > MIN_VALID_VALUE));

	    return thrust::make_tuple(cubeindex, /*valid*/numVertsTable[cubeindex]);
	}
    };

    struct is_valid_cell : public thrust::unary_function<int, bool>
    {
	__host__ __device__
	bool operator()(int numVertices) const {
	    return numVertices != 0;
	}
    };

    struct isosurface_functor : public thrust::unary_function<thrust::tuple<int, int, int, int>, void>
    {
	// FixME: constant iterator and/or iterator to const problem.
	InputPointDataIterator	point_data;
	InputPhysCoordinatesIterator physical_coord;
	ScalarSourceIterator	scalar_source;
	const float		isovalue;
	TableIterator		triangle_table;

	typedef typename InputPhysCoordinatesIterator::value_type	grid_tuple_type;

	float4 *vertices_output;
	float3 *normals_output;
	float  *scalars_output;

	const int xdim;
	const int ydim;
	const int zdim;
	const int cells_per_layer;

	__host__ __device__
	isosurface_functor(InputDataSet1 &input,
	                   InputDataSet2 &source,
	                   const float isovalue,
	                   TableIterator triangle_table,
	                   float4 *vertices,
	                   float3 *normals/*,
	                   float  *scalars*/)
	    : point_data(input.point_data_begin()),
	      physical_coord(input.physical_coordinates_begin()),
	      scalar_source(source.point_data_begin()),
	      isovalue(isovalue),
	      triangle_table(triangle_table),
	      vertices_output(vertices), normals_output(normals), /*scalars_output(scalars),*/
	      xdim(input.dim0), ydim(input.dim1), zdim(input.dim2),
	      cells_per_layer((xdim - 1) * (ydim - 1)) {}

	__host__ __device__
	float3 vertex_interp(float3 p0, float3 p1, float t) const {
	    return lerp(p0, p1, t);
	}
	__host__ __device__
	float scalar_interp(float s0, float s1, float t) const {
	    return lerp(s0, s1, t);
	}

	// FixME: the type of the grid coordinates may not be 3-tuple of ints
	template <typename Tuple>
	__host__ __device__
	float3 tuple2float3(Tuple xyz) {
	    return make_float3((float) thrust::get<0>(grid_tuple_type(xyz)),
	                       (float) thrust::get<1>(grid_tuple_type(xyz)),
	                       (float) thrust::get<2>(grid_tuple_type(xyz)));
	}


	__host__ __device__
	void operator()(thrust::tuple<int, int, int, int> indices_tuple) {
	    const int cell_id  = thrust::get<0>(indices_tuple);
	    const int outputVertId = thrust::get<1>(indices_tuple);
	    const int cubeindex    = thrust::get<2>(indices_tuple);
	    const int numVertices  = thrust::get<3>(indices_tuple);

	    const int verticesForEdge[] = { 0, 1, 1, 2, 3, 2, 0, 3,
	                                    4, 5, 5, 6, 7, 6, 4, 7,
	                                    0, 4, 1, 5, 2, 6, 3, 7 };

	    const int x = cell_id % (xdim - 1);
	    const int y = (cell_id / (xdim - 1)) % (ydim -1);
	    const int z = cell_id / cells_per_layer;

	    // indices to the eight vertices of the voxel
	    int i[8];
	    i[0] = x      + y*xdim + z * xdim * ydim;
	    i[1] = i[0]   + 1;
	    i[2] = i[0]   + 1	+ xdim;
	    i[3] = i[0]   + xdim;

	    i[4] = i[0]   + xdim * ydim;
	    i[5] = i[1]   + xdim * ydim;
	    i[6] = i[2]   + xdim * ydim;
	    i[7] = i[3]   + xdim * ydim;

	    float f[8];
	    f[0] = *(point_data + i[0]);
	    f[1] = *(point_data + i[1]);
	    f[2] = *(point_data + i[2]);
	    f[3] = *(point_data + i[3]);
	    f[4] = *(point_data + i[4]);
	    f[5] = *(point_data + i[5]);
	    f[6] = *(point_data + i[6]);
	    f[7] = *(point_data + i[7]);

	    // TODO: Reconsider what GridCoordinates should be (tuple or float3)
	    float3 p[8];
	    p[0] = tuple2float3(*(physical_coord + i[0]));
	    p[1] = tuple2float3(*(physical_coord + i[1]));
	    p[2] = tuple2float3(*(physical_coord + i[2]));
	    p[3] = tuple2float3(*(physical_coord + i[3]));
	    p[4] = tuple2float3(*(physical_coord + i[4]));
	    p[5] = tuple2float3(*(physical_coord + i[5]));
	    p[6] = tuple2float3(*(physical_coord + i[6]));
	    p[7] = tuple2float3(*(physical_coord + i[7]));

	    /*float s[8];
	    s[0] = *(scalar_source + i[0]);
	    s[1] = *(scalar_source + i[1]);
	    s[2] = *(scalar_source + i[2]);
	    s[3] = *(scalar_source + i[3]);
	    s[4] = *(scalar_source + i[4]);
	    s[5] = *(scalar_source + i[5]);
	    s[6] = *(scalar_source + i[6]);
	    s[7] = *(scalar_source + i[7]);*/

	    // interpolation for vertex positions and associated scalar values
	    for (int v = 0; v < numVertices; v++) {
		const int edge = triangle_table[cubeindex*16 + v];
		const int v0   = verticesForEdge[2*edge];
		const int v1   = verticesForEdge[2*edge + 1];
		const float t  = (isovalue - f[v0]) / (f[v1] - f[v0]);
		*(vertices_output + outputVertId + v) = make_float4(vertex_interp(p[v0], p[v1], t), 1.0f);
		//*(scalars_output  + outputVertId + v) = scalar_interp(s[v0], s[v1], t);
	    }

	    // generate normal vectors by cross product of triangle edges
	    for (int v = 0; v < numVertices; v += 3) {
		const float4 *vertex = (vertices_output + outputVertId + v);
		const float3 edge0 = make_float3(vertex[1] - vertex[0]);
		const float3 edge1 = make_float3(vertex[2] - vertex[0]);
		const float3 normal = normalize(cross(edge0, edge1));
		*(normals_output + outputVertId + v) =
		*(normals_output + outputVertId + v + 1) =
		*(normals_output + outputVertId + v + 2) = normal;
	    }
	}
    };

    VerticesIterator vertices_begin() {
	return vertices.begin();
    }
    VerticesIterator vertices_end() {
	return vertices.end();
    }

    NormalsIterator normals_begin() {
	return normals.begin();
    }
    NormalsIterator normals_end() {
	return normals.end();
    }

    ScalarIterator scalars_begin() {
	return scalars.begin();
    }
    ScalarIterator scalars_end() {
	return scalars.end();
    }

    void set_isovalue(value_type val) {
	isovalue = val;
    }
};

template <typename InputDataSet1, typename InputDataSet2>
const int marching_cube<InputDataSet1, InputDataSet2>::triTable_array[256][16] =
{
#define X -1
     {X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X},
     {0, 8, 3, X, X, X, X, X, X, X, X, X, X, X, X, X},
     {0, 1, 9, X, X, X, X, X, X, X, X, X, X, X, X, X},
     {1, 8, 3, 9, 8, 1, X, X, X, X, X, X, X, X, X, X},
     {1, 2, 10, X, X, X, X, X, X, X, X, X, X, X, X, X},
     {0, 8, 3, 1, 2, 10, X, X, X, X, X, X, X, X, X, X},
     {9, 2, 10, 0, 2, 9, X, X, X, X, X, X, X, X, X, X},
     {2, 8, 3, 2, 10, 8, 10, 9, 8, X, X, X, X, X, X, X},
     {3, 11, 2, X, X, X, X, X, X, X, X, X, X, X, X, X},
     {0, 11, 2, 8, 11, 0, X, X, X, X, X, X, X, X, X, X},
     {1, 9, 0, 2, 3, 11, X, X, X, X, X, X, X, X, X, X},
     {1, 11, 2, 1, 9, 11, 9, 8, 11, X, X, X, X, X, X, X},
     {3, 10, 1, 11, 10, 3, X, X, X, X, X, X, X, X, X, X},
     {0, 10, 1, 0, 8, 10, 8, 11, 10, X, X, X, X, X, X, X},
     {3, 9, 0, 3, 11, 9, 11, 10, 9, X, X, X, X, X, X, X},
     {9, 8, 10, 10, 8, 11, X, X, X, X, X, X, X, X, X, X},
     {4, 7, 8, X, X, X, X, X, X, X, X, X, X, X, X, X},
     {4, 3, 0, 7, 3, 4, X, X, X, X, X, X, X, X, X, X},
     {0, 1, 9, 8, 4, 7, X, X, X, X, X, X, X, X, X, X},
     {4, 1, 9, 4, 7, 1, 7, 3, 1, X, X, X, X, X, X, X},
     {1, 2, 10, 8, 4, 7, X, X, X, X, X, X, X, X, X, X},
     {3, 4, 7, 3, 0, 4, 1, 2, 10, X, X, X, X, X, X, X},
     {9, 2, 10, 9, 0, 2, 8, 4, 7, X, X, X, X, X, X, X},
     {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, X, X, X, X},
     {8, 4, 7, 3, 11, 2, X, X, X, X, X, X, X, X, X, X},
     {11, 4, 7, 11, 2, 4, 2, 0, 4, X, X, X, X, X, X, X},
     {9, 0, 1, 8, 4, 7, 2, 3, 11, X, X, X, X, X, X, X},
     {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, X, X, X, X},
     {3, 10, 1, 3, 11, 10, 7, 8, 4, X, X, X, X, X, X, X},
     {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, X, X, X, X},
     {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, X, X, X, X},
     {4, 7, 11, 4, 11, 9, 9, 11, 10, X, X, X, X, X, X, X},
     {9, 5, 4, X, X, X, X, X, X, X, X, X, X, X, X, X},
     {9, 5, 4, 0, 8, 3, X, X, X, X, X, X, X, X, X, X},
     {0, 5, 4, 1, 5, 0, X, X, X, X, X, X, X, X, X, X},
     {8, 5, 4, 8, 3, 5, 3, 1, 5, X, X, X, X, X, X, X},
     {1, 2, 10, 9, 5, 4, X, X, X, X, X, X, X, X, X, X},
     {3, 0, 8, 1, 2, 10, 4, 9, 5, X, X, X, X, X, X, X},
     {5, 2, 10, 5, 4, 2, 4, 0, 2, X, X, X, X, X, X, X},
     {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, X, X, X, X},
     {9, 5, 4, 2, 3, 11, X, X, X, X, X, X, X, X, X, X},
     {0, 11, 2, 0, 8, 11, 4, 9, 5, X, X, X, X, X, X, X},
     {0, 5, 4, 0, 1, 5, 2, 3, 11, X, X, X, X, X, X, X},
     {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, X, X, X, X},
     {10, 3, 11, 10, 1, 3, 9, 5, 4, X, X, X, X, X, X, X},
     {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, X, X, X, X},
     {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, X, X, X, X},
     {5, 4, 8, 5, 8, 10, 10, 8, 11, X, X, X, X, X, X, X},
     {9, 7, 8, 5, 7, 9, X, X, X, X, X, X, X, X, X, X},
     {9, 3, 0, 9, 5, 3, 5, 7, 3, X, X, X, X, X, X, X},
     {0, 7, 8, 0, 1, 7, 1, 5, 7, X, X, X, X, X, X, X},
     {1, 5, 3, 3, 5, 7, X, X, X, X, X, X, X, X, X, X},
     {9, 7, 8, 9, 5, 7, 10, 1, 2, X, X, X, X, X, X, X},
     {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, X, X, X, X},
     {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, X, X, X, X},
     {2, 10, 5, 2, 5, 3, 3, 5, 7, X, X, X, X, X, X, X},
     {7, 9, 5, 7, 8, 9, 3, 11, 2, X, X, X, X, X, X, X},
     {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, X, X, X, X},
     {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, X, X, X, X},
     {11, 2, 1, 11, 1, 7, 7, 1, 5, X, X, X, X, X, X, X},
     {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, X, X, X, X},
     {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, X},
     {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, X},
     {11, 10, 5, 7, 11, 5, X, X, X, X, X, X, X, X, X, X},
     {10, 6, 5, X, X, X, X, X, X, X, X, X, X, X, X, X},
     {0, 8, 3, 5, 10, 6, X, X, X, X, X, X, X, X, X, X},
     {9, 0, 1, 5, 10, 6, X, X, X, X, X, X, X, X, X, X},
     {1, 8, 3, 1, 9, 8, 5, 10, 6, X, X, X, X, X, X, X},
     {1, 6, 5, 2, 6, 1, X, X, X, X, X, X, X, X, X, X},
     {1, 6, 5, 1, 2, 6, 3, 0, 8, X, X, X, X, X, X, X},
     {9, 6, 5, 9, 0, 6, 0, 2, 6, X, X, X, X, X, X, X},
     {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, X, X, X, X},
     {2, 3, 11, 10, 6, 5, X, X, X, X, X, X, X, X, X, X},
     {11, 0, 8, 11, 2, 0, 10, 6, 5, X, X, X, X, X, X, X},
     {0, 1, 9, 2, 3, 11, 5, 10, 6, X, X, X, X, X, X, X},
     {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, X, X, X, X},
     {6, 3, 11, 6, 5, 3, 5, 1, 3, X, X, X, X, X, X, X},
     {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, X, X, X, X},
     {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, X, X, X, X},
     {6, 5, 9, 6, 9, 11, 11, 9, 8, X, X, X, X, X, X, X},
     {5, 10, 6, 4, 7, 8, X, X, X, X, X, X, X, X, X, X},
     {4, 3, 0, 4, 7, 3, 6, 5, 10, X, X, X, X, X, X, X},
     {1, 9, 0, 5, 10, 6, 8, 4, 7, X, X, X, X, X, X, X},
     {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, X, X, X, X},
     {6, 1, 2, 6, 5, 1, 4, 7, 8, X, X, X, X, X, X, X},
     {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, X, X, X, X},
     {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, X, X, X, X},
     {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, X},
     {3, 11, 2, 7, 8, 4, 10, 6, 5, X, X, X, X, X, X, X},
     {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, X, X, X, X},
     {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, X, X, X, X},
     {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, X},
     {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, X, X, X, X},
     {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, X},
     {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, X},
     {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, X, X, X, X},
     {10, 4, 9, 6, 4, 10, X, X, X, X, X, X, X, X, X, X},
     {4, 10, 6, 4, 9, 10, 0, 8, 3, X, X, X, X, X, X, X},
     {10, 0, 1, 10, 6, 0, 6, 4, 0, X, X, X, X, X, X, X},
     {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, X, X, X, X},
     {1, 4, 9, 1, 2, 4, 2, 6, 4, X, X, X, X, X, X, X},
     {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, X, X, X, X},
     {0, 2, 4, 4, 2, 6, X, X, X, X, X, X, X, X, X, X},
     {8, 3, 2, 8, 2, 4, 4, 2, 6, X, X, X, X, X, X, X},
     {10, 4, 9, 10, 6, 4, 11, 2, 3, X, X, X, X, X, X, X},
     {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, X, X, X, X},
     {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, X, X, X, X},
     {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, X},
     {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, X, X, X, X},
     {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, X},
     {3, 11, 6, 3, 6, 0, 0, 6, 4, X, X, X, X, X, X, X},
     {6, 4, 8, 11, 6, 8, X, X, X, X, X, X, X, X, X, X},
     {7, 10, 6, 7, 8, 10, 8, 9, 10, X, X, X, X, X, X, X},
     {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, X, X, X, X},
     {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, X, X, X, X},
     {10, 6, 7, 10, 7, 1, 1, 7, 3, X, X, X, X, X, X, X},
     {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, X, X, X, X},
     {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, X},
     {7, 8, 0, 7, 0, 6, 6, 0, 2, X, X, X, X, X, X, X},
     {7, 3, 2, 6, 7, 2, X, X, X, X, X, X, X, X, X, X},
     {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, X, X, X, X},
     {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, X},
     {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, X},
     {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, X, X, X, X},
     {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, X},
     {0, 9, 1, 11, 6, 7, X, X, X, X, X, X, X, X, X, X},
     {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, X, X, X, X},
     {7, 11, 6, X, X, X, X, X, X, X, X, X, X, X, X, X},
     {7, 6, 11, X, X, X, X, X, X, X, X, X, X, X, X, X},
     {3, 0, 8, 11, 7, 6, X, X, X, X, X, X, X, X, X, X},
     {0, 1, 9, 11, 7, 6, X, X, X, X, X, X, X, X, X, X},
     {8, 1, 9, 8, 3, 1, 11, 7, 6, X, X, X, X, X, X, X},
     {10, 1, 2, 6, 11, 7, X, X, X, X, X, X, X, X, X, X},
     {1, 2, 10, 3, 0, 8, 6, 11, 7, X, X, X, X, X, X, X},
     {2, 9, 0, 2, 10, 9, 6, 11, 7, X, X, X, X, X, X, X},
     {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, X, X, X, X},
     {7, 2, 3, 6, 2, 7, X, X, X, X, X, X, X, X, X, X},
     {7, 0, 8, 7, 6, 0, 6, 2, 0, X, X, X, X, X, X, X},
     {2, 7, 6, 2, 3, 7, 0, 1, 9, X, X, X, X, X, X, X},
     {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, X, X, X, X},
     {10, 7, 6, 10, 1, 7, 1, 3, 7, X, X, X, X, X, X, X},
     {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, X, X, X, X},
     {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, X, X, X, X},
     {7, 6, 10, 7, 10, 8, 8, 10, 9, X, X, X, X, X, X, X},
     {6, 8, 4, 11, 8, 6, X, X, X, X, X, X, X, X, X, X},
     {3, 6, 11, 3, 0, 6, 0, 4, 6, X, X, X, X, X, X, X},
     {8, 6, 11, 8, 4, 6, 9, 0, 1, X, X, X, X, X, X, X},
     {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, X, X, X, X},
     {6, 8, 4, 6, 11, 8, 2, 10, 1, X, X, X, X, X, X, X},
     {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, X, X, X, X},
     {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, X, X, X, X},
     {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, X},
     {8, 2, 3, 8, 4, 2, 4, 6, 2, X, X, X, X, X, X, X},
     {0, 4, 2, 4, 6, 2, X, X, X, X, X, X, X, X, X, X},
     {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, X, X, X, X},
     {1, 9, 4, 1, 4, 2, 2, 4, 6, X, X, X, X, X, X, X},
     {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, X, X, X, X},
     {10, 1, 0, 10, 0, 6, 6, 0, 4, X, X, X, X, X, X, X},
     {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, X},
     {10, 9, 4, 6, 10, 4, X, X, X, X, X, X, X, X, X, X},
     {4, 9, 5, 7, 6, 11, X, X, X, X, X, X, X, X, X, X},
     {0, 8, 3, 4, 9, 5, 11, 7, 6, X, X, X, X, X, X, X},
     {5, 0, 1, 5, 4, 0, 7, 6, 11, X, X, X, X, X, X, X},
     {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, X, X, X, X},
     {9, 5, 4, 10, 1, 2, 7, 6, 11, X, X, X, X, X, X, X},
     {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, X, X, X, X},
     {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, X, X, X, X},
     {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, X},
     {7, 2, 3, 7, 6, 2, 5, 4, 9, X, X, X, X, X, X, X},
     {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, X, X, X, X},
     {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, X, X, X, X},
     {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, X},
     {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, X, X, X, X},
     {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, X},
     {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, X},
     {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, X, X, X, X},
     {6, 9, 5, 6, 11, 9, 11, 8, 9, X, X, X, X, X, X, X},
     {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, X, X, X, X},
     {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, X, X, X, X},
     {6, 11, 3, 6, 3, 5, 5, 3, 1, X, X, X, X, X, X, X},
     {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, X, X, X, X},
     {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, X},
     {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, X},
     {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, X, X, X, X},
     {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, X, X, X, X},
     {9, 5, 6, 9, 6, 0, 0, 6, 2, X, X, X, X, X, X, X},
     {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, X},
     {1, 5, 6, 2, 1, 6, X, X, X, X, X, X, X, X, X, X},
     {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, X},
     {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, X, X, X, X},
     {0, 3, 8, 5, 6, 10, X, X, X, X, X, X, X, X, X, X},
     {10, 5, 6, X, X, X, X, X, X, X, X, X, X, X, X, X},
     {11, 5, 10, 7, 5, 11, X, X, X, X, X, X, X, X, X, X},
     {11, 5, 10, 11, 7, 5, 8, 3, 0, X, X, X, X, X, X, X},
     {5, 11, 7, 5, 10, 11, 1, 9, 0, X, X, X, X, X, X, X},
     {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, X, X, X, X},
     {11, 1, 2, 11, 7, 1, 7, 5, 1, X, X, X, X, X, X, X},
     {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, X, X, X, X},
     {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, X, X, X, X},
     {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, X},
     {2, 5, 10, 2, 3, 5, 3, 7, 5, X, X, X, X, X, X, X},
     {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, X, X, X, X},
     {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, X, X, X, X},
     {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, X},
     {1, 3, 5, 3, 7, 5, X, X, X, X, X, X, X, X, X, X},
     {0, 8, 7, 0, 7, 1, 1, 7, 5, X, X, X, X, X, X, X},
     {9, 0, 3, 9, 3, 5, 5, 3, 7, X, X, X, X, X, X, X},
     {9, 8, 7, 5, 9, 7, X, X, X, X, X, X, X, X, X, X},
     {5, 8, 4, 5, 10, 8, 10, 11, 8, X, X, X, X, X, X, X},
     {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, X, X, X, X},
     {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, X, X, X, X},
     {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, X},
     {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, X, X, X, X},
     {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, X},
     {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, X},
     {9, 4, 5, 2, 11, 3, X, X, X, X, X, X, X, X, X, X},
     {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, X, X, X, X},
     {5, 10, 2, 5, 2, 4, 4, 2, 0, X, X, X, X, X, X, X},
     {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, X},
     {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, X, X, X, X},
     {8, 4, 5, 8, 5, 3, 3, 5, 1, X, X, X, X, X, X, X},
     {0, 4, 5, 1, 0, 5, X, X, X, X, X, X, X, X, X, X},
     {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, X, X, X, X},
     {9, 4, 5, X, X, X, X, X, X, X, X, X, X, X, X, X},
     {4, 11, 7, 4, 9, 11, 9, 10, 11, X, X, X, X, X, X, X},
     {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, X, X, X, X},
     {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, X, X, X, X},
     {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, X},
     {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, X, X, X, X},
     {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, X},
     {11, 7, 4, 11, 4, 2, 2, 4, 0, X, X, X, X, X, X, X},
     {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, X, X, X, X},
     {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, X, X, X, X},
     {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, X},
     {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, X},
     {1, 10, 2, 8, 7, 4, X, X, X, X, X, X, X, X, X, X},
     {4, 9, 1, 4, 1, 7, 7, 1, 3, X, X, X, X, X, X, X},
     {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, X, X, X, X},
     {4, 0, 3, 7, 4, 3, X, X, X, X, X, X, X, X, X, X},
     {4, 8, 7, X, X, X, X, X, X, X, X, X, X, X, X, X},
     {9, 10, 8, 10, 11, 8, X, X, X, X, X, X, X, X, X, X},
     {3, 0, 9, 3, 9, 11, 11, 9, 10, X, X, X, X, X, X, X},
     {0, 1, 10, 0, 10, 8, 8, 10, 11, X, X, X, X, X, X, X},
     {3, 1, 10, 11, 3, 10, X, X, X, X, X, X, X, X, X, X},
     {1, 2, 11, 1, 11, 9, 9, 11, 8, X, X, X, X, X, X, X},
     {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, X, X, X, X},
     {0, 2, 11, 8, 0, 11, X, X, X, X, X, X, X, X, X, X},
     {3, 2, 11, X, X, X, X, X, X, X, X, X, X, X, X, X},
     {2, 3, 8, 2, 8, 10, 10, 8, 9, X, X, X, X, X, X, X},
     {9, 10, 2, 0, 9, 2, X, X, X, X, X, X, X, X, X, X},
     {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, X, X, X, X},
     {1, 10, 2, X, X, X, X, X, X, X, X, X, X, X, X, X},
     {1, 3, 8, 9, 1, 8, X, X, X, X, X, X, X, X, X, X},
     {0, 9, 1, X, X, X, X, X, X, X, X, X, X, X, X, X},
     {0, 3, 8, X, X, X, X, X, X, X, X, X, X, X, X, X},
     {X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X}
#undef X
};

template <typename InputDataSet1, typename InputDataSet2>
const int marching_cube<InputDataSet1, InputDataSet2>::numVerticesTable_array[256] = {
    0,
    3,
    3,
    6,
    3,
    6,
    6,
    9,
    3,
    6,
    6,
    9,
    6,
    9,
    9,
    6,
    3,
    6,
    6,
    9,
    6,
    9,
    9,
    12,
    6,
    9,
    9,
    12,
    9,
    12,
    12,
    9,
    3,
    6,
    6,
    9,
    6,
    9,
    9,
    12,
    6,
    9,
    9,
    12,
    9,
    12,
    12,
    9,
    6,
    9,
    9,
    6,
    9,
    12,
    12,
    9,
    9,
    12,
    12,
    9,
    12,
    15,
    15,
    6,
    3,
    6,
    6,
    9,
    6,
    9,
    9,
    12,
    6,
    9,
    9,
    12,
    9,
    12,
    12,
    9,
    6,
    9,
    9,
    12,
    9,
    12,
    12,
    15,
    9,
    12,
    12,
    15,
    12,
    15,
    15,
    12,
    6,
    9,
    9,
    12,
    9,
    12,
    6,
    9,
    9,
    12,
    12,
    15,
    12,
    15,
    9,
    6,
    9,
    12,
    12,
    9,
    12,
    15,
    9,
    6,
    12,
    15,
    15,
    12,
    15,
    6,
    12,
    3,
    3,
    6,
    6,
    9,
    6,
    9,
    9,
    12,
    6,
    9,
    9,
    12,
    9,
    12,
    12,
    9,
    6,
    9,
    9,
    12,
    9,
    12,
    12,
    15,
    9,
    6,
    12,
    9,
    12,
    9,
    15,
    6,
    6,
    9,
    9,
    12,
    9,
    12,
    12,
    15,
    9,
    12,
    12,
    15,
    12,
    15,
    15,
    12,
    9,
    12,
    12,
    9,
    12,
    15,
    15,
    12,
    12,
    9,
    15,
    6,
    15,
    12,
    6,
    3,
    6,
    9,
    9,
    12,
    9,
    12,
    12,
    15,
    9,
    12,
    12,
    15,
    6,
    9,
    9,
    6,
    9,
    12,
    12,
    15,
    12,
    15,
    15,
    6,
    12,
    9,
    15,
    12,
    9,
    6,
    12,
    3,
    9,
    12,
    12,
    15,
    12,
    15,
    9,
    12,
    12,
    15,
    15,
    6,
    9,
    12,
    6,
    3,
    6,
    9,
    9,
    6,
    9,
    12,
    6,
    3,
    9,
    6,
    12,
    3,
    6,
    3,
    3,
    0,
};

}


#endif /* MARCHING_CUBE_H_ */
