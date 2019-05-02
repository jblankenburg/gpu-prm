#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>

#include <iostream>
#include <random>

int NUM_NNS = 5;

struct vec2 {
  float x;
  float y;

  __host__ __device__
  vec2(float x, float y) : x{x}, y{y} { }

  __host__ __device__
  vec2() : x{0.0}, y{0.0} { }

};

struct FarEnough {

  float x;
  float y;
  float thresh;

  FarEnough(float x, float y, float thresh)
    : x{x}, y{y}, thresh{thresh} {

    }
  
  __host__ __device__ bool operator()(const vec2 &v) const {
    return (v.x - x)*(v.x - x) + (v.y - y)*(v.y - y) > thresh * thresh;
  }
};

struct DistTo {

  float x;
  float y;

  DistTo(float x, float y) : x{x}, y{y} { }

  __host__ __device__ float operator()(const vec2 &v) const {
    return (v.x - x)*(v.x - x) + (v.y - y)*(v.y - y);
  }
};

vec2 rand_vec(std::mt19937 &gen, std::uniform_real_distribution<> &dist) {
  return vec2(dist(gen), dist(gen));
}

// TODO: need to swap this to be on the GPU instead
/*
  Return a host_vector of N random vec2's
 */
thrust::host_vector<vec2> rand_vecs(std::mt19937 &gen, std::uniform_real_distribution<> &dist, int N) {
  thrust::host_vector<vec2> v(N);
  for (int i = 0; i < N; ++i) {
    v[i] = rand_vec(gen, dist);
  }

  return v;
}

/*
  Return the 5 nearest neighbors for each vec2 in v.
  
  out[i] is an int representing the k nearest neighbors of v[i].
 */
thrust::device_vector<int> get_neighbors(const thrust::device_vector<vec2> &v) {
  thrust::device_vector<int> out;

  // do stuff

  return out;  
}



int main(int argc, char **argv)
{

  if (argc != 2) {
    std::cerr << "Need N" << std::endl;
    return -1;
  }
  
  int N = std::atoi(argv[1]);
  
  std::random_device rd;
  std::mt19937 gen(rd()); 
  std::uniform_real_distribution<> dis(0.0, 1.0);

  // H has storage for N integers
  thrust::host_vector<vec2> H = rand_vecs(gen, dis, N);
    
  // print contents of H
  for(int i = 0; i < H.size(); i++) {
    auto tmp = H[i];
    std::cout << "H[" << i << "] = "
        << tmp.x << " "
        << tmp.y << std::endl;
  }

  // Copy host_vector H to device_vector D
  thrust::device_vector<vec2> D = H;

  // thrust::device_vector<float> dists_to_center(D.size());

  // // TODO: this finds the distance for all points to the center point, 
  // //       need to put this in a loop to get the distance to all points from
  // //       each and every point, and then copy the smallest 5 from gpu to cpu
  // //       to store the graph on the cpu.
  // thrust::transform(D.begin(), D.end(), dists_to_center.begin(), DistTo(0.5, 0.5));
  // thrust::sort_by_key(dists_to_center.begin(), dists_to_center.end(), D.begin());

  // for(int i = 0; i < dists_to_center.size(); i++) {
  //   float tmp = dists_to_center[i];
  //   std::cout << "dists_to_center[" << i << "] = "
  //       << tmp
  //       << std::endl;
  // }
 
  // filter based on distance to (0.5, 0.5)
  int N_close = thrust::count_if(D.begin(), D.end(), FarEnough(0.5, 0.5, 0.15));
  
  thrust::device_vector<vec2> filter_target(N_close);
  
  thrust::copy_if(D.begin(), D.end(), filter_target.begin(), FarEnough(0.5, 0.5, 0.15));

  std::cout << std::endl;
  
  // print contents of filter_target
  for(int i = 0; i < filter_target.size(); i++) {
    vec2 tmp = filter_target[i];
    std::cout << "D[" << i << "] = "
        << tmp.x << " " 
        << tmp.y
        << std::endl;
  }

  // define thing to store the graph
  // vec2[NUM_NNS] nnlist;
  thrust::host_vector<std::vector<vec2>> graph(filter_target.size());

  //--------------------
  // find the 5 nearest neighbors from the filtered points
  thrust::device_vector<float> dists_to_point(filter_target.size());

  // loop through the filtered points
  for(int i = 0; i < filter_target.size(); i++ ) {

    // save array so we can sort them each step
    thrust::device_vector<vec2> target_points = filter_target;

    // sort all points by distance to point
    vec2 pnt = filter_target[i];
    thrust::transform(filter_target.begin(), filter_target.end(), dists_to_point.begin(), DistTo(pnt.x, pnt.y));
    thrust::sort_by_key(dists_to_point.begin(), dists_to_point.end(), target_points.begin());

    std::cout << "Point looking for: (" << pnt.x << ", " << pnt.y << ")" << std::endl;

    // // print out all distances and the respective points
    // for(int j = 0; j < dists_to_point.size(); j++) {
    //   float tmp = dists_to_point[j];
    //   vec2 tmp2 = target_points[j];
    //   std::cout << "dists_to_center[" << j << "] = "
    //       << tmp
    //       << "\t pnt: (" << tmp2.x << ", " << tmp2.y << ")"
    //       << std::endl;
    // }

    // store the closest NUM_NNS neighbors (not including itself, so start at indx 1)
    for( int l = 1; l < NUM_NNS + 1; l++) {
      graph[i].push_back(target_points[l]);
    }

    // print out the closest NUM_NNS that were stored
    std::cout << "\t Stored: ";
    for (auto k = graph[i].begin(); k != graph[i].end(); ++k)
      std::cout << " (" << (*k).x << ", " << (*k).y << ") ";
    std::cout << std::endl;

  //--------------------

  }



  // H and D are automatically deleted when the function returns
  return 0;
}
