/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include "particle_filter.h"
#include <assert.h>
#include <map>
#include "Eigen/Dense"


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);
	num_particles = 1000;
	for (int i = 0; i < num_particles; i++)
	{
		particles.push_back(Particle{ i,dist_x(gen) ,dist_y(gen),dist_theta(gen),1 });
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	std::random_device rd;
	std::mt19937 gen(rd());
	for (auto particle : particles)
	{
			particle.x = yaw_rate == 0? velocity*cos(particle.theta)*delta_t:
				particle.x + velocity / yaw_rate*(sin(particle.theta + yaw_rate*delta_t) - sin(particle.theta));
			std::normal_distribution<double> dist_x(particle.x, std_pos[0]);
			particle.x += dist_x(gen);
			particle.y = yaw_rate == 0 ? velocity*sin(particle.theta)*delta_t:
				particle.y + velocity / yaw_rate*(cos(particle.theta) - cos(particle.theta + yaw_rate*delta_t));
			std::normal_distribution<double> dist_y(particle.y, std_pos[1]);
			particle.y += dist_y(gen);
			particle.theta = particle.theta + yaw_rate * delta_t;
			std::normal_distribution<double> dist_theta(particle.theta, std_pos[2]);
			particle.theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(const std::vector<Map::single_landmark_s>& transformedLandmarks, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (auto landmark : transformedLandmarks)
	{
		unsigned int minDistance = std::numeric_limits<unsigned int>::max();
		for (auto observation : observations)
		{
			auto dist = calculateDistance(landmark.x_f, landmark.y_f, observation.x, observation.y);
			if (dist < minDistance)
			{
				observation.id = landmark.id_i;
				minDistance = dist;
			}
		}
		assert(minDistance < std::numeric_limits<unsigned int>::max());
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	for (auto particle : particles)
	{
		//ToDo: Refactor dataAssociation to a generic templace function working on iterators
		//to prevent doubling of landmark containers
		std::vector<Map::single_landmark_s> v_LandmarksWithinSensorRange;
		std::map<int, Map::single_landmark_s> m_LandmarksWithinSensorRange;
		std::vector<LandmarkObs> v_transformedObservations; 
		for (auto observation : observations)
		{
			transformObservationsToMapSpace(observation, particle);
			v_transformedObservations.push_back(observation);
		}
		//Only consider landmarks within sensor range of the particle
		for (auto landmark : map_landmarks.landmark_list)
		{
			auto dist = calculateDistance(particle.x, particle.y, landmark.x_f, landmark.y_f);
			if (dist <= sensor_range)
			{
				v_LandmarksWithinSensorRange.push_back(landmark);
				m_LandmarksWithinSensorRange.insert(std::make_pair(landmark.id_i, landmark));
			}
		}
		dataAssociation(v_LandmarksWithinSensorRange, v_transformedObservations);

		for (auto observation : observations)
		{
			Eigen::VectorXd x(2);
			x << observation.x,
				observation.y;
			Eigen::VectorXd mu(2);
			mu << m_LandmarksWithinSensorRange[observation.id].x_f,
				m_LandmarksWithinSensorRange[observation.id].y_f;
			Eigen::MatrixXd Sigma(2, 2);
			Sigma << std_landmark[0], 0,
				0, std_landmark[1];
			auto diff = x - mu;
			auto numerator = exp(-1 / 2 * diff.transpose()*Sigma.inverse()*diff);
			auto denominator = sqrt((2 * M_PI * Sigma).determinant());
			particle.weight *= numerator / denominator;
			weights.push_back(particle.weight);
		}
	}
	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::vector<double> new_weights;
	std::vector<Particle> sampled_particles;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<> discrete_distribution(weights.begin(),weights.end());
	for (int n = 0; n<num_particles; ++n) 
	{
		 auto sampled_weight = discrete_distribution(gen);
		 for (auto particle : particles)
		 {
			 if (sampled_weight == particle.weight)
			 {
				 sampled_particles.push_back(particle);
				 break;
			 }
		 }
	}
	weights.clear();
	particles = sampled_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
