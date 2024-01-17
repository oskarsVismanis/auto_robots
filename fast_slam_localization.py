import numpy as np
import hashlib
import matplotlib.pyplot as plt

"""
    • Objekti nav unikāli atpazīstami (+)
    • Robots veic mērījumus, kuros objekti ir sastapti atkārtoti (+)
    • Ir jābūt iespējai izgūt datus par atsevišķām daļiņām (+)
"""

class FastSLAM:
    def __init__(self):

        # daļiņu skaits
        self.num_particles = 10
        # faktiskās šķēršļu pozīcijas
        self.landmark_positions = np.array([9, 11, 14])
        # sastaptie šķēršļi
        self.encountered_landmarks = {}
        # pārvietošanās kļūda
        self.motion_noise_std = 0.2
        # mērījumu kļūda
        self.measurement_noise_std = 0.2
        # maksimālais redzes attālums
        self.max_vision_distance = 4.0

        # sākotnējā robota pozīcija
        self.initial_robot_position = 6.0

        # inicializē daļiņas
        self.particles = {
            f'particle_{i}': {
                'pos': self.initial_robot_position,
                'weight': 1,
                'landmarks': {}
            } for i in range(self.num_particles)
        }

        self.landmark_template = {
            'status': '', 
            'position': 0.0, 
            'covariance': 0.0,
            'weight': 0.0}

        # print(self.particles)

    def detect_landmarks(self, robot_position, movement = 0):
        
        # nosaka kurus šķēršļus iespējams redzēt
        visible_landmarks = self.landmark_positions[self.landmark_positions - robot_position <= self.max_vision_distance]
        true_distances = []
        
        # nosaka reālos attālumus līdz tiem
        for landmark in visible_landmarks:
            true_distances.append(landmark - robot_position)
             
        # Compare new true_distances with existing ones
        for i, new_landmark in enumerate(true_distances):

            for particle_id, particle_info in self.particles.items():
                num_keys = len(particle_info['landmarks'])
                # print(f"Number of landmarks for {particle_id}: {num_keys}")

            similar_landmarks = [(key, value) for particle_info in self.particles.values()
                                for key, value in particle_info['landmarks'].items()
                                if isinstance(value['distance'], (int, float))
                                and np.isclose(value['distance'], new_landmark, atol=self.max_vision_distance/2)]

            if not similar_landmarks:
                name = "landmark_" + str(num_keys+1)
                # If there are no similar landmarks, add the new one to the dictionary
                for particle_id, particle_info in self.particles.items():
                    particle_info['landmarks'][name] = {'status': 'new',
                                                        'distance': new_landmark,
                                                        'position': new_landmark + robot_position,
                                                        'covariance': round(np.abs(new_landmark) * self.measurement_noise_std, 4),
                                                        'weight': 0.0}
                    # print(f"New landmark {name} found at distance {new_landmark}")
            else:
                similar_name, _ = similar_landmarks[0]
                for particle_id, particle_info in self.particles.items():
                    particle_info['landmarks'][similar_name]['status'] = "existing"
                    particle_info['landmarks'][similar_name]['distance'] = new_landmark

                    # print(f'Robot position: {robot_position}\n')
                    # print(particle_info['landmarks'][similar_name]['distance'])
                    """                    
                    Z = particle_position + movement
                    Y = Z - 1 * landmark_position
                    S = landmark_covariance + measumenent_error
                    K = Y * 1/S
                    landmark_position = previous_landmark_position + K * Y
                    landmark_covariance = 1 - K * 1                                     # p
                    landmark_weight = np.abs(2*np.pi*S) * np.exp(-0.5 * Y * (1/S) * Y)  # w
                    """
                    # Z = robot_position + movement #particle_info['landmarks'][similar_name]['distance']
                    # print(movement)
                    Z = particle_info['pos'] + movement #particle_info['landmarks'][similar_name]['distance']
                    # print(particle_info['landmarks'][similar_name]['position'])
                    Y = Z - 1 * particle_info['landmarks'][similar_name]['position']
                    Q = self.measurement_noise_std * movement
                    S = particle_info['landmarks'][similar_name]['covariance'] + Q
                    K = Y * 1/S

                    # print(f'Z:{Z}, Y:{Y}, S:{S}, K:{K} \n')

                    particle_info['landmarks'][similar_name]['position'] = particle_info['landmarks'][similar_name]['position'] + K * Y
                    # print(particle_info['landmarks'][similar_name]['position'])
                    particle_info['landmarks'][similar_name]['covariance'] = 1 - K * 1 # p
                    particle_info['landmarks'][similar_name]['weight'] = np.abs(2*np.pi*S)**(-1/2) * np.exp(-0.5 * Y**2 / S)#np.exp(-0.5 * Y * (1/S) * Y) # w

                    # print(f"Similar landmark {similar_name} found. Updating distance to {new_landmark}.")

        # for particle_id, particle_info in self.particles.items():
        #     print(self.particles[particle_id])

    def update_robot_position(self, robot_position, movement_direction, movement_amount):
        
        movement = movement_direction * movement_amount
        # atjaunina robota pozīciju
        robot_position += movement + np.random.normal(scale=self.motion_noise_std)

        for particle_id, particle_info in self.particles.items():
            # atjaunina pozīciju katrā daļiņā
            particle_info['pos'] += movement + np.random.normal(scale=self.motion_noise_std)
        return robot_position, movement
    
def get_all_landmark_values(particles):
    all_landmark_values = {}

    for particle_id, particle_info in particles.items():
        for landmark_name, landmark_data in particle_info['landmarks'].items():
            if landmark_name not in all_landmark_values:
                all_landmark_values[landmark_name] = []

            all_landmark_values[landmark_name].append({
                'status': landmark_data['status'],
                'position': landmark_data['position'],
                'covariance': landmark_data['covariance'],
                'weight': landmark_data['weight']
            })

    return all_landmark_values

def get_landmark_color(landmark_name):
    hashed_color = int(hashlib.sha256(landmark_name.encode()).hexdigest(), 16) % 0xFFFFFF
    return "#{:06x}".format(hashed_color)

def plot_particle_and_landmarks(particle_id, particle_info, true_landmark_positions):
    plt.figure(figsize=(10, 5))
    plt.title(f'{particle_id} - Landmarks')

    # reālās objektu pozīcijas
    plt.scatter(true_landmark_positions, np.zeros_like(true_landmark_positions), marker='o', color='green', label='True Landmarks')

    # daļiņu uzskats par objektu pozīciju
    for landmark_name, landmark_data in particle_info['landmarks'].items():
        color = get_landmark_color(landmark_name)
        
        if landmark_data['weight'] > 0 and landmark_data['weight'] > 0.1:
            size = landmark_data['weight'] * 150
        else:
            size = 100

        plt.scatter(landmark_data['position'], 0, s=size, alpha=0.5,
                    label=f'{landmark_name}', color=color)

    # robota pozīcija
    plt.scatter(particle_info['pos'], 0, marker='x', color='red', label=f'True Robot Position')

    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    fastSlam = FastSLAM()

    # first iteration
    fastSlam.detect_landmarks(fastSlam.initial_robot_position)
    robot_position, movement = fastSlam.update_robot_position(fastSlam.initial_robot_position, movement_direction=1, movement_amount=2.5)
    print(fastSlam.particles["particle_0"])
    print(fastSlam.particles["particle_3"])
    print(fastSlam.particles["particle_7"])

    # second iteration
    fastSlam.detect_landmarks(robot_position, movement)
    robot_position, movement = fastSlam.update_robot_position(robot_position, movement_direction=1, movement_amount=2)
    print(fastSlam.particles["particle_0"])
    print(fastSlam.particles["particle_3"])
    print(fastSlam.particles["particle_7"])

    # third iteration
    fastSlam.detect_landmarks(robot_position, movement)
    robot_position, movement = fastSlam.update_robot_position(robot_position, movement_direction=-1, movement_amount=3)
    print(fastSlam.particles["particle_0"])
    print(fastSlam.particles["particle_3"])
    print(fastSlam.particles["particle_7"])

    all_landmark_values = get_all_landmark_values(fastSlam.particles)

    for landmark_name, values_list in all_landmark_values.items():
        print(f"Landmark: {landmark_name}")
        for i, values in enumerate(values_list):
            print(f"Particle {i+1}:")
            # print(f"  Status: {values['status']}")
            print(f"  Position: {values['position']}")
            print(f"  Covariance: {values['covariance']}")
            print(f"  Weight: {values['weight']}")
            print()

    for particle_id, particle_info in fastSlam.particles.items():
        plot_particle_and_landmarks(particle_id, particle_info, fastSlam.landmark_positions)

if __name__ == "__main__":
    main()