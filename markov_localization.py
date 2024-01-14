import numpy as np
import matplotlib.pyplot as plt

"""
    • Robots atrodas vidē, kuras izmērs ir vismaz 10 x 10 šūnas. (+)
    • Robots pašlokalizācijai izmanto Markova pašlokalizācijas metodi.
    • Robota pozīcija tiek atspoguļota kā (x, y, θ), kur θ var būt viens no 4 stāvokļiem - 0deg, 90deg, 180deg un 270deg. (+)
    • Kā sākotnējo faktisko robota pozīciju var izvēlēties jebkuru šūnu un virzienu, 
      bet tiek pieņemts, ka robots par savu atrašanās vietu neko nezina. (+)
    • Robota sensori var pieļaut kļūdas un sensoru modelī ir jāparedz kļūdas iespēja.
    • Robota odometrijas mērījumi arī var būt kļūdaini - ir jāparedz odometrijas kļūdas iespēja.
"""

class Environment:
    def __init__(self):
        self.obstacle_value = -1
        self.setup()
        self.init_robot()


    def setup(self):
        
        self.ws = np.array([[1,1,1,1,1,1,1,1,1,1,1,1],
                            [1,0,0,0,0,1,0,0,0,1,1,1],
                            [1,0,1,0,0,1,1,0,1,0,0,1],
                            [1,0,0,0,0,0,0,0,0,0,1,1],
                            [1,1,1,1,1,1,0,0,0,1,0,1],
                            [1,0,0,1,0,0,0,0,0,0,0,1],
                            [1,0,0,1,0,0,0,1,1,0,0,1],
                            [1,0,0,0,0,0,0,1,0,0,1,1],
                            [1,0,0,1,0,0,0,1,0,0,0,1],
                            [1,0,1,1,1,0,0,0,1,0,0,1],
                            [1,0,0,0,0,0,1,0,0,0,0,1],
                            [1,1,1,1,1,1,1,1,1,1,1,1]])
        
        self.ws[self.ws == 1] = self.obstacle_value #-1#2

        self.rbt_ws = np.copy(self.ws)
        self.probability_ws = np.copy(self.ws)

        self.show_env(self.ws)

    def init_robot(self):

        # Define four predefined numbers
        possible_theta = [0, 90, 180, 270]

        # nosaka brīvās vietas kartē
        self.clear_space = np.where(self.rbt_ws == 0)
        # print(self.clear_space)
        
        # izvēlās nenoteiktu brīvo lauciņu un rotāciju
        random_index = np.random.choice(range(len(self.clear_space[0])))
        theta = np.random.choice(possible_theta)
        
        # iestata to par robota (x, y)
        self.start_setting = [self.clear_space[1][random_index],self.clear_space[0][random_index],theta]

        # pievieno to kartei
        self.rbt_ws[self.start_setting[1], self.start_setting[0]] = 1.5

        print(f"Robot starting position and rotation: ({self.start_setting[0]}, {self.start_setting[1]}, {self.start_setting[2]})")

        self.show_env(self.rbt_ws, self.start_setting, 'Visualization of robot in environment')

    def show_env(self, ws, robot_setting=[0,0,0], title="Visualization of environment"):

        plt.imshow(ws, cmap='PiYG', interpolation='nearest')
        plt.xticks(np.arange(-0.5, len(ws[0]), 1), [])
        plt.yticks(np.arange(-0.5, len(ws), 1), [])
        plt.grid(True, color='black', linewidth=0.5)
        
        if robot_setting[0] != 0:
            if robot_setting[2] == 0:
                tx = 0
                ty = 0.25
                dx = 0
                dy = -0.4
            elif robot_setting[2] == 90:
                tx = -0.25
                ty = 0
                dx = 0.4
                dy = 0
            elif robot_setting[2] == 180:
                tx = 0
                ty = -0.25
                dx = 0
                dy = 0.4
            elif robot_setting[2] == 270:
                tx = 0.25
                ty = 0
                dx = -0.4
                dy = 0

            # Ar bultu vizualizē robota skatīšanās virzienu
            arrow_properties = dict(facecolor='red', edgecolor='red', width=0.1, head_width=0.4)
            plt.arrow(robot_setting[0]+tx, robot_setting[1]+ty, dx, dy, **arrow_properties)  # Adjust the arrow direction if needed

        plt.title(title)
        plt.show()

    def normalize(self, ws):
        # Apply normalization only to non-2 values
        non_2_indices = (ws != self.obstacle_value)
        sum_non_2 = np.sum(ws[non_2_indices])
        ws[non_2_indices] /= sum_non_2
        return ws

class Sensors:
    def __init__(self, env):
        self.env = env
        self.ws = env.ws
        self.rbt_ws = env.rbt_ws
        self.start_setting = env.start_setting
        self.obstacle_value = env.obstacle_value
        self.init_sensors()
        # self.first_sensor_reading()

    def init_sensors(self):

        # Count the number of occurrences of 0 in the array
        # count_zeros = np.count_nonzero(self.ws == 0)
        count_zeros = np.count_nonzero(self.env.probability_ws == 0)
        print(count_zeros)

        # self.ws = self.ws.astype(float)
        self.env.probability_ws = self.env.probability_ws.astype(float)
        # Avoid division by zero
        if count_zeros > 0:
            # Replace each 0 with 1 divided by the count
            # self.ws[self.ws == 0] = 1 / count_zeros
            self.env.probability_ws[self.env.probability_ws == 0] = 1 / count_zeros

        # self.env.show_env(self.ws, title="Environment with equal occupancy possibility")
        self.env.show_env(self.env.probability_ws, title="Environment with equal occupancy possibility")

    def get_direction_values(self, grid, theta):

        error = True
        error_probabilities = [0.25, 0.2, 0.1, 0.2, 0.25]

        if theta == 0:
            grid[2, 0] = grid[2, 1] = grid[2, 2] = 0#-2
        elif theta == 90:
            grid[0, 0] = grid[1, 0] = grid[2, 0] = 0#-2
            grid = np.rot90(grid, k=1)
        elif theta == 180:
            grid[0, 0] = grid[0, 1] = grid[0, 2] = 0#-2
            grid = np.rot90(grid, k=2)
        elif theta == 270:
            grid[0, 2] = grid[1, 2] = grid[2, 2] = 0#-2
            grid = np.rot90(grid, k=3)

        left = grid[1,0]
        diag_left = grid[0,0]
        front = grid[0,1]
        diag_right = grid[0,2]
        right = grid[1,2]

        readings = [left, diag_left, front, diag_right, right]

        # Introduce errors based on the error_probabilities
        if error_probabilities is not None and error == True:
            print(f"Values before introducing errors {readings}")
            readings = self.introduce_errors(readings, error_probabilities)
            print(f"Values after introducing errors {readings}")

            grid[1,0] = readings[0]
            grid[0,0] = readings[1]
            grid[0,1] = readings[2]
            grid[0,2] = readings[3]
            grid[1,2] = readings[4]

        if theta == 90:
            grid = np.rot90(grid, k=3)
        elif theta == 180:
            grid = np.rot90(grid, k=2)
        elif theta == 270:
            grid = np.rot90(grid, k=1)

        return readings, grid

    def read_sensors(self, robot_position, ws):
        
        row, col, theta = robot_position

        # Create a 3x3 subgrid
        subgrid = ws[col-1:col+2, row-1:row+2]
        readings, subgrid = self.get_direction_values(subgrid, theta)

        readings.append(theta)
        print(readings)

        print("\nValues around the robot:")
        print(subgrid)

        plt.imshow(subgrid, cmap='gray_r', interpolation='nearest')
        plt.xticks(np.arange(-0.5, len(subgrid[0]), 1), [])
        plt.yticks(np.arange(-0.5, len(subgrid), 1), [])
        plt.grid(True, color='black', linewidth=0.5)
        plt.title('Robot sensor readings')
        plt.show()

        return readings
    
    def introduce_errors(self, readings, error_probabilities):
        # Introduce errors to each sensor reading separately
        for i in range(len(readings)):  # Exclude the last element (theta)
            if error_probabilities[i] is not None and np.random.rand() < error_probabilities[i]:
                # Introduce an error by randomly selecting from [0, 2]
                readings[i] = np.random.choice([0, self.obstacle_value])
        return readings
    
    def rotate_indices(self, i, j, rotation):
        if rotation == 0:
            views = [[i , j-1],[i-1, j-1],[i-1, j],[i-1, j+1],[i, j+1]]
            return views
        elif rotation == 90:
            views = [[i-1, j], [i-1, j+1], [i, j+1], [i+1, j+1], [i+1, j]]
            return views
        elif rotation == 180: 
            views = [[i, j+1], [i+1, j+1], [i+1, j], [i+1, j-1], [i, j-1]]
            return views
        elif rotation == 270: 
            views = [[i+1, j], [i+1, j-1], [i, j-1], [i-1, j-1], [i-1, j]]
            return views

    def estimate_position(self, readings):
        
        # varbutibu tabula
        pos_left = 0.1
        pos_diag_left = 0.2
        pos_front = 0.4
        pos_diag_right = 0.2
        pos_right = 0.1
        beliefs = [pos_left,pos_diag_left,pos_front,pos_diag_right,pos_right]

        # Loop through all possible positions in the environment
        possible_positions = []
        for i, j in zip(self.env.clear_space[0], self.env.clear_space[1]):
            views = self.rotate_indices(i, j, readings[5])
            similarity_score = (
                (beliefs[0] if readings[0] == self.env.ws[views[0][0], views[0][1]] else 0) + # left
                (beliefs[1] if readings[1] == self.env.ws[views[1][0], views[1][1]] else 0) + # diag_left
                (beliefs[2] if readings[2] == self.env.ws[views[2][0], views[2][1]] else 0) + # front
                (beliefs[3] if readings[3] == self.env.ws[views[3][0], views[3][1]] else 0) + # diag_right
                (beliefs[4] if readings[4] == self.env.ws[views[4][0], views[4][1]] else 0) # right
            )
            possible_positions.append(((i, j), 1 * similarity_score))

        # Sort the possible positions based on the similarity score
        possible_positions.sort(key=lambda x: x[1], reverse=True)

        # print("\nPossible robot positions based on sensor readings:")
        # for position, score in possible_positions:
        #     print(f"Position: {position[1], position[0]}, Similarity Score: {score}")
        
        return possible_positions
    
    def update_environment(self, estimated_positions, ws):

        weight_possible = 0.8
        weight_other = 0.2

        mask = (self.env.ws == 0) & (self.env.ws != self.obstacle_value)

        for position, score in estimated_positions:
            i, j = position
            if mask[i, j]:
                if score > 0:
                    if score > weight_other:
                        ws[i, j] *= weight_possible * score
                    else:
                        ws[i, j] *= weight_other
                else:
                    ws[i, j] *= weight_other
                        
        # Apply normalization only to non-2 values
        ws = self.env.normalize(ws)

        return ws

    def show_updated_environment(self):
        self.env.show_env(self.ws, title="Updated Environment")

    def first_sensor_reading(self):

        readings = self.read_sensors(self.start_setting, self.env.ws)
        possible_positions = self.estimate_position(readings)
        updated_ws = self.update_environment(possible_positions)
        self.show_updated_environment()

        return updated_ws

class Robot:
    def __init__(self, env):
        self.env = env
        self.rbt_ws = env.rbt_ws
        self.robot_position = env.start_setting
        self.obstacle_value = env.obstacle_value

    def get_next_step(self, current_position):
        row, col, theta = current_position

        robot_position = current_position

        # print(row, col)
        action = False

        # Introduce a 10% chance that neither movement nor rotation happens
        if np.random.rand() < 0.1:
            print("No movement or rotation (1/10 chance)")
            action = "stay"
            return robot_position, action, current_position

        # Calculate the next position based on the robot's current direction
        next_row, next_col = row, col

        if theta == 0:
            next_col -= 1
        elif theta == 90:
            next_row += 1
        elif theta == 180:
            next_col += 1
        elif theta == 270:
            next_row -= 1

        # print(next_row, next_col)
        # print(self.rbt_ws[next_col, next_row])
        print(f"For coordinates {next_row},{next_col} value is {self.rbt_ws[next_col,next_row]}")

        # Check if the next position is not an obstacle
        if (next_row < 11 and next_row > 0 and 
            next_col < 11 and next_col > 0 and 
            self.rbt_ws[next_col, next_row] == 0):
            print(f"Moving to ({next_row}, {next_col})")
            robot_position = [next_row, next_col, theta]
            action = "move"
        else:
            # If not, rotate 90 degrees
            print("Rotating 90 degrees")
            robot_position[2] = (theta + 90) % 360
            action = "rotate"

        print(robot_position)

        return robot_position, action, current_position
    
    def update_robot_pos(self, robot_position, action, current_position):

        if action == "move":
            self.rbt_ws[current_position[1], current_position[0]] = 0
            self.rbt_ws[robot_position[1], robot_position[0]] = 1.5
        
        self.env.show_env(self.rbt_ws, robot_position, 'Robots movement in the environment')

    def update_robot_position_probability(self, ws, current_position, action):
        
        obst_pos = np.where(ws == self.env.obstacle_value)

        mask = np.copy(ws)
        mask[mask == self.env.obstacle_value] = 0

        # Extract current position information
        row, col, theta = current_position

        print(action)
        ws[ws != self.env.obstacle_value] *= 0.2

        if not action:
            # self.env.show_env(ws, title="ws after no movement")
            ws = self.env.normalize(ws)
            return ws

        # Specify the shift for each axis (negative values for left/up, positive values for right/down)
        elif action == "move":
            if theta == 0:
                shift_rows = -1  # shift one position down
                shift_cols = 0
            elif theta == 180:
                shift_rows = 1  # shift one position up
                shift_cols = 0
            elif theta == 90:
                shift_rows = 0
                shift_cols = +1  # shift one position to the left
            elif theta == 270:
                shift_rows = 0
                shift_cols = -1  # shift one position to the right
        
            mask_shifted = np.roll(mask, (shift_rows, shift_cols), axis=(0, 1))
            
            mask_shifted = mask_shifted * 0.8
            mask_other = np.where(mask_shifted == 0, mask, mask_shifted)
            # mask = mask * mask_shifted
            mask = mask * mask_other

            mask[obst_pos] = self.env.obstacle_value
            mask = self.env.normalize(mask)

            return mask
        
        elif action == "rotate":
            ws[ws != self.env.obstacle_value] *= 0.8
            ws = self.env.normalize(ws)
            # self.env.show_env(ws, title="Mask after rotate")
            return ws

        return ws

def sensor_step(env, sensor, robot_position, ws):
    readings = sensor.read_sensors(robot_position, ws)
    possible_positions = sensor.estimate_position(readings)
    env.probability_ws = sensor.update_environment(possible_positions, env.probability_ws)

def robot_step(robot, env, robot_position):
    robot_position, action, current_position = robot.get_next_step(robot_position)
    robot.update_robot_pos(robot_position, action, current_position)
    env.probability_ws = robot.update_robot_position_probability(env.probability_ws, robot_position, action)

    return robot_position
    
def main():
    env = Environment()
    sensor = Sensors(env)
    robot = Robot(env)

    # first sensor step
    sensor_step(env, sensor, env.start_setting, env.ws)
    env.show_env(env.probability_ws, title="First sensor update")

    # first robot movement
    robot_position = robot_step(robot, env, env.start_setting)
    env.show_env(env.probability_ws, title="First robot movement update")

    # second sensor step
    sensor_step(env, sensor, robot_position, env.ws)
    env.show_env(env.probability_ws, title="Second sensor update")

    # second_robot_movement
    robot_position = robot_step(robot, env, robot_position)
    env.show_env(env.probability_ws, title="Second robot movement update")

    # third sensor step
    sensor_step(env, sensor, robot_position, env.ws)
    env.show_env(env.probability_ws, title="Third sensor update")

    # third robot movement
    robot_position = robot_step(robot, env, robot_position)
    env.show_env(env.probability_ws, title="Third robot movement update")

    # fourth sensor step
    sensor_step(env, sensor, robot_position, env.ws)
    env.show_env(env.probability_ws, title="Fourth sensor update")

    # fourth robot movement
    robot_position = robot_step(robot, env, robot_position)
    env.show_env(env.probability_ws, title="Fourth robot movement update")

    # fifth sensor step
    sensor_step(env, sensor, robot_position, env.ws)
    env.show_env(env.probability_ws, title="Fifth sensor update")

    # fifth robot movement
    robot_position = robot_step(robot, env, robot_position)
    env.show_env(env.probability_ws, title="Fifth robot movement update")

if __name__ == "__main__":
    main()