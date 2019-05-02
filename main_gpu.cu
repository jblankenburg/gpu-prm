#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>

#include <iostream>
#include <random>

struct vec2 {
  float x;
  float y;

  __host__ __device__
  vec2(float x, float y) : x{x}, y{y} { }

  __host__ __device__
  vec2() : x{0.0}, y{0.0} { }

};

struct CloseEnough {

  float x;
  float y;
  float thresh;

  CloseEnough(float x, float y, float thresh)
    : x{x}, y{y}, thresh{thresh} {

    }
  
  __host__ __device__ bool operator()(const vec2 &v) const {
    return (v.x - x)*(v.x - x) + (v.y - y)*(v.y - y) < thresh * thresh;
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

  // H has storage for 4 integers
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

  thrust::device_vector<float> dists_to_center(D.size());

  thrust::transform(D.begin(), D.end(), dists_to_center.begin(), DistTo(0.5, 0.5));
  thrust::sort_by_key(dists_to_center.begin(), dists_to_center.end(), D.begin());

  for(int i = 0; i < dists_to_center.size(); i++) {
    float tmp = dists_to_center[i];
    std::cout << "dists_to_center[" << i << "] = "
	      << tmp
	      << std::endl;
  }

  
  // filter based on distance to (0.5, 0.5)
  int N_close = thrust::count_if(D.begin(), D.end(), CloseEnough(0.5, 0.5, 0.15));
  
  thrust::device_vector<vec2> filter_target(N_close);
  
  thrust::copy_if(D.begin(), D.end(), filter_target.begin(), CloseEnough(0.5, 0.5, 0.15));

  std::cout << std::endl;
  
  // print contents of filter_target
  for(int i = 0; i < filter_target.size(); i++) {
    vec2 tmp = filter_target[i];
    std::cout << "D[" << i << "] = "
	      << tmp.x << " " 
	      << tmp.y
	      << std::endl;
  }

  // H and D are automatically deleted when the function returns
  return 0;
}
