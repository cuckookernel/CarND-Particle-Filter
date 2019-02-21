/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"
// #include "map.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::cout; 
using std::endl; 
typedef Map::single_landmark_s MapLandmark;
typedef unsigned int uint;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 1000;  // TODO_: Set the number of particles
  particles = vector<Particle>(num_particles);

  std::default_random_engine gen;
  normal_distribution<double> noise_x(0.0, std[0]);
  normal_distribution<double> noise_y(0.0, std[1]);
  normal_distribution<double> noise_th(0.0, std[2]);

  for( int i=0; i < num_particles; i++) {
     Particle& p = particles[i];
     p.x = x + noise_x(gen) ;
     p.y = y + noise_y(gen); 
     p.theta = theta + noise_th(gen);
     
  }

}

void ParticleFilter::prediction(double dt, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  normal_distribution<double> noise_x(0.0, std_pos[0]);
  normal_distribution<double> noise_y(0.0, std_pos[1]);
  normal_distribution<double> noise_th(0.0, std_pos[2]);

  if( fabs(yaw_rate) > 1e-6 ) {
    double vel_over_yaw  = velocity / yaw_rate;
    for( int i= 0; i< num_particles; i++ ) {
      Particle& p = particles[i];
      double new_theta = p.theta + yaw_rate * dt;
      p.x += vel_over_yaw * ( sin(new_theta) - sin(p.theta) ) + noise_x(gen);
      p.y += vel_over_yaw * ( cos(p.theta)   - cos(new_theta) ) + noise_y(gen);
      p.theta = new_theta + noise_th(gen); 
    }    
  } else { // case in which yaw_rate is really small, better not divide by it, use derivative approximation instead.....  
    for( int i= 0; i< num_particles; i++ ) {
      Particle& p = particles[i];      
      p.x += velocity * cos(p.theta) + noise_x(gen);
      p.y += velocity * sin(p.theta) + noise_y(gen);
      p.theta += yaw_rate * dt + noise_th(gen);
    }
  }

}

void ParticleFilter::dataAssociation( const vector<MapLandmark>& predicted, 
                                      vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for( auto& obs : observations ) {
    double closest_dist = std::numeric_limits<double>::infinity();

    for( const auto& prd :  predicted ){
        double d = dist( obs.x, obs.y, prd.x_f, prd.y_f );
        if( d < closest_dist ) {
          obs.id = prd.id_i;
          closest_dist = d;
        }
     } // for each prd
  } // for each obs

}

// Filter Landmarks from map by sensor_range. These are in map coordinates.     
vector<MapLandmark> get_predicted( const Particle& p, double sensor_range,
                                   const vector<MapLandmark>& landmarks );

vector<LandmarkObs> get_observations_map( const Particle& p, const vector<LandmarkObs> &observations );

double get_weight( const vector<LandmarkObs>& obs_map, const vector<MapLandmark>& map_lmarks, double std_landmark[] );


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html   * 
   */      
   for( auto& p : particles )  {          
     // 1. Filter Landmarks from map by sensor_range. These are in map coordinates.     
     auto predicted = get_predicted( p, sensor_range, map_landmarks.landmark_list);     
     // 2. transform observations to map coordinates
     auto obs_map = get_observations_map( p, observations );
     // 3. assign each (transformed) observation to landmark
     dataAssociation( predicted, obs_map ); 
     // 4. compute new weight 
     p.weight = get_weight( obs_map, map_landmarks.landmark_list, std_landmark );      
   }
}

vector<MapLandmark> get_predicted( const Particle& p, double sensor_range,
                                   const vector<MapLandmark>& landmarks ) {
     vector<MapLandmark> predicted;      
     predicted.reserve( landmarks.size() );

     for( auto& lm : landmarks ) {
       if ( dist(lm.x_f, lm.y_f, p.x, p.y) < sensor_range ) {         
         predicted.push_back( lm );
       }
     } 
     
     return predicted;
}

// transform observations in map coordinate systems...
vector<LandmarkObs> get_observations_map( const Particle& p, const vector<LandmarkObs> &observations ) {
  
     vector<LandmarkObs> obs_map; 
     obs_map.reserve( observations.size() );
     
     for( auto& obs : observations ) {
       // if ( dist(obs.x, obs.y, p.x, p.y) < sensor_range ) {
          LandmarkObs obs_m; 

          double th = p.theta; 
          obs_m.x = cos(th) * obs.x - sin(th) * obs.y + p.x;
          obs_m.y = sin(th) * obs.x + cos(th) * obs.y + p.y; 
          obs_map.push_back( obs_m );          
       // }
     }

     return obs_map;
} 

double get_weight( const vector<LandmarkObs>& obs_map, const vector<MapLandmark>& map_lmarks, double std_landmark[] ) {

  double sigma_x = std_landmark[0];
  double sigma_y = std_landmark[1];
  
  double w0 = 1.0 / ( 2 * M_PI * sigma_x * sigma_y ); 
  double twice_var_x = 2 * sigma_x * sigma_x;
  double twice_var_y = 2 * sigma_y * sigma_y;

  double weight = 1.0;  // accumulate product here 
  for( auto& obs_m : obs_map ){
    
    const MapLandmark& mlm = map_lmarks[obs_m.id - 1]; // WARNING: this uses the fact that landmark ids in map are number in increasing order, starting from 1...
    double dx = obs_m.x - mlm.x_f;
    double dy = obs_m.y - mlm.y_f;
    
    weight *= (w0  * exp( -dx*dx / twice_var_x  - dy * dy / twice_var_y ));
  }; 

  return weight; 
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<double> weights;
  weights.reserve( num_particles );
  for ( auto& p : particles )  { weights.push_back( p.weight ); }

  std::default_random_engine gen;
  std::discrete_distribution<> dd{weights.begin(), weights.end()};

  vector<Particle> new_particles;
  new_particles.resize(num_particles);
  for ( auto& p : new_particles)  {
    int idx = dd(gen);
    p = particles[idx];
  }

  particles = new_particles; 
  // print_std_devs( particles ); 
}

void print_std_devs( const vector<Particle>& particles ) {
  /* check std_dev in x and y ... */
  
  double total_weight = sum( particles, [](const Particle& p ) -> double {  return p.weight; });
  double mean_x = sum( particles, [](const Particle& p ) -> double {  return p.weight * p.x; } )  / total_weight;
  double mean_y = sum( particles, [](const Particle& p ) -> double {  return p.weight * p.y; } ) / total_weight;
  double mean_x2 = sum( particles, [](const Particle& p ) -> double {  return p.weight * p.x * p.x; }  ) / total_weight;
  double mean_y2 = sum( particles, [](const Particle& p ) -> double {  return p.weight * p.y * p.y; }  ) / total_weight;

  double stddev_x = sqrt( mean_x2 - mean_x * mean_x );
  double stddev_y = sqrt( mean_y2 - mean_y * mean_y );

  cout << "stddev_x = " << stddev_x << " stddev_y = " << stddev_y << endl;     
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}