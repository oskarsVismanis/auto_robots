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

        # definē karti
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
        
        self.ws[self.ws == 1] = self.obstacle_value

        self.rbt_ws = np.copy(self.ws)
        self.probability_ws = np.copy(self.ws)

        self.show_env(self.ws)

    def init_robot(self):

        # definē rotācijas
        possible_theta = [0, 90, 180, 270]

        # nosaka brīvās vietas kartē
        self.clear_space = np.where(self.rbt_ws == 0)
        
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
        # normalizē vērtības, kas nav šķērslis
        free_values = (ws != self.obstacle_value)
        sum_free = np.sum(ws[free_values])
        ws[free_values] /= sum_free
        return ws

class Sensors:
    def __init__(self, env):
        self.env = env
        self.ws = env.ws
        self.rbt_ws = env.rbt_ws
        self.start_setting = env.start_setting
        self.obstacle_value = env.obstacle_value
        self.init_sensors()

    def init_sensors(self):

        # saskaita brīvos laukus
        count_zeros = np.count_nonzero(self.env.probability_ws == 0)
        # print(count_zeros)

        self.env.probability_ws = self.env.probability_ws.astype(float)
        # iedod visiem brīvajiem lauciņiem vienādu sākuma varbūtību
        if count_zeros > 0:
            self.env.probability_ws[self.env.probability_ws == 0] = 1 / count_zeros

        self.env.show_env(self.env.probability_ws, title="Environment with equal occupancy possibility")

    # iegūst sensora rādījumus atkarībā no pagrieziena leņķa
    def get_direction_values(self, grid, theta):

        error = True
        # kad atļauta kļūda, kļūdas varbūtības lasījumiem |pa kreisi|diag. pa kreisi|taisni|diag. pa labi|pa labi|
        error_probabilities = [0.25, 0.2, 0.1, 0.2, 0.25]

        # nogriež liekos lasījumus un pēc vajadzības parotē lasījumu vertikāli
        if theta == 0:
            grid[2, 0] = grid[2, 1] = grid[2, 2] = 0
        elif theta == 90:
            grid[0, 0] = grid[1, 0] = grid[2, 0] = 0
            grid = np.rot90(grid, k=1)
        elif theta == 180:
            grid[0, 0] = grid[0, 1] = grid[0, 2] = 0
            grid = np.rot90(grid, k=2)
        elif theta == 270:
            grid[0, 2] = grid[1, 2] = grid[2, 2] = 0
            grid = np.rot90(grid, k=3)

        # nolasa vērtības
        left = grid[1,0]
        diag_left = grid[0,0]
        front = grid[0,1]
        diag_right = grid[0,2]
        right = grid[1,2]

        readings = [left, diag_left, front, diag_right, right]

        # atkarībā no kļūdu varbūtības ievieš tās
        if error_probabilities is not None and error == True:
            print(f"Values before introducing errors {readings}")
            readings = self.introduce_errors(readings, error_probabilities)
            print(f"Values after introducing errors {readings}")

            grid[1,0] = readings[0]
            grid[0,0] = readings[1]
            grid[0,1] = readings[2]
            grid[0,2] = readings[3]
            grid[1,2] = readings[4]

        # atgriež lasījumu atpakaļ oriģinālajā rotācijā, lai to var salīdzināt kartē
        if theta == 90:
            grid = np.rot90(grid, k=3)
        elif theta == 180:
            grid = np.rot90(grid, k=2)
        elif theta == 270:
            grid = np.rot90(grid, k=1)

        return readings, grid

    def read_sensors(self, robot_position, ws):
        
        row, col, theta = robot_position

        # izveido 3x3 režģi
        subgrid = ws[col-1:col+2, row-1:row+2]
        readings, subgrid = self.get_direction_values(subgrid, theta)

        readings.append(theta)
        print(readings)

        print("\nValues around the robot:")
        print(subgrid)

        # vizualizē sensora mērījumu
        plt.imshow(subgrid, cmap='gray_r', interpolation='nearest')
        plt.xticks(np.arange(-0.5, len(subgrid[0]), 1), [])
        plt.yticks(np.arange(-0.5, len(subgrid), 1), [])
        plt.grid(True, color='black', linewidth=0.5)
        plt.title('Robot sensor readings')
        plt.show()

        return readings
    
    def introduce_errors(self, readings, error_probabilities):
        # katram sensora lasījumam atsevišķi ieviest kļūdu
        for i in range(len(readings)):
            if error_probabilities[i] is not None and np.random.rand() < error_probabilities[i]:
                # Ieviest kļūdu, pēc nenoteiktības izvēloties, vai lasījums rādīs brīvu lauciņu vai šķērsli 
                readings[i] = np.random.choice([0, self.obstacle_value])
        return readings
    
    # palīgfunkcija pozīcijas novērtēšanai
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

        # pārbauda katru brīvo lauciņu kartē
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

        # sašķiro vērtības pēc ticamības vērtējuma
        possible_positions.sort(key=lambda x: x[1], reverse=True)

        # For debugging
        # print("\nPossible robot positions based on sensor readings:")
        # for position, score in possible_positions:
        #     print(f"Position: {position[1], position[0]}, Similarity Score: {score}")
        
        return possible_positions
    
    def update_environment(self, estimated_positions, ws):

        # ticamības vērtības; ja pozīcija ir ar augstāku ticamību nekā kļūdains mērījums, to pareizina ar 0.8
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
                        
        # normalizē
        ws = self.env.normalize(ws)

        return ws

    def show_updated_environment(self):
        self.env.show_env(self.ws, title="Updated Environment")

class Robot:
    def __init__(self, env):
        self.env = env
        self.rbt_ws = env.rbt_ws
        self.robot_position = env.start_setting
        self.obstacle_value = env.obstacle_value

    def get_next_step(self, current_position):
        row, col, theta = current_position

        robot_position = current_position
        action = False

        # 10% iespēja, ka nenotiks darbība, atgriež esošo vērtību
        if np.random.rand() < 0.1:
            print("No movement or rotation (1/10 chance)")
            action = "stay"
            return robot_position, action, current_position

        # atkarībā no pagrieziena leņķa, aprēķina iespējamo nākamo pozīciju
        next_row, next_col = row, col

        if theta == 0:
            next_col -= 1
        elif theta == 90:
            next_row += 1
        elif theta == 180:
            next_col += 1
        elif theta == 270:
            next_row -= 1

        print(f"For coordinates {next_row},{next_col} value is {self.rbt_ws[next_col,next_row]}")

        # pārbauda, vai nākamā pozīcija nav šķērslis
        if (next_row < 11 and next_row > 0 and 
            next_col < 11 and next_col > 0 and 
            self.rbt_ws[next_col, next_row] == 0):
            print(f"Moving to ({next_row}, {next_col})")
            robot_position = [next_row, next_col, theta]
            action = "move"
        else:
            # ja ir, rotē par 90grādiem pulksteņrādītāja virzienā
            print("Rotating 90 degrees")
            robot_position[2] = (theta + 90) % 360
            action = "rotate"

        print(robot_position)

        return robot_position, action, current_position
    
    def update_robot_pos(self, robot_position, action, current_position):

        # ja notika kustība, atjauno robota kartē robota faktisko lokāciju un veco atbrīvo
        if action == "move":
            self.rbt_ws[current_position[1], current_position[0]] = 0
            self.rbt_ws[robot_position[1], robot_position[0]] = 1.5
        
        self.env.show_env(self.rbt_ws, robot_position, 'Robots movement in the environment')

    def update_robot_position_probability(self, ws, current_position, action):
        
        obst_pos = np.where(ws == self.env.obstacle_value)

        # izveido kartes masku un nonullē šķēršļus
        mask = np.copy(ws)
        mask[mask == self.env.obstacle_value] = 0

        _, _, theta = current_position

        print(action)
        # pareizina lauciņus ar varbūtībām ar 0.2, iegūstot nepārvietošanās situāciju
        ws[ws != self.env.obstacle_value] *= 0.2
        mask *= 0.2

        # ja nenotiek darbība, atgriež iepriekš iegūto varbūtību karti
        if not action:
            ws = self.env.normalize(ws)
            return ws

        # nosaka kustības virzienu, lai pārvietotu varbūtību vērtības
        elif action == "move":
            if theta == 0:
                shift_rows = -1  # pastumt uz leju
                shift_cols = 0
            elif theta == 180:
                shift_rows = 1  # pastumt uz augšu
                shift_cols = 0
            elif theta == 90:
                shift_rows = 0
                shift_cols = +1  # pastumt pa kreisi
            elif theta == 270:
                shift_rows = 0
                shift_cols = -1  # pastumt pa labi
        
            # pabīda varbūtību masku kustības virzienā
            mask_shifted = np.roll(mask, (shift_rows, shift_cols), axis=(0, 1))
            
            # sareizina pakustinātās vērtības ar 0.8 (veiksmīgo pārvietojumu)
            mask_shifted = mask_shifted * 0.8
            # no oriģinālās maskas atgūst nozaudētās vērtības, kas tika nonullētas pabīdot karti
            mask_other = np.where(mask_shifted == 0, mask, mask_shifted)
            # sareizina masku ar pabīdīto masku
            mask = mask * mask_other

            # atgriež kartē šķēršļus
            mask[obst_pos] = self.env.obstacle_value
            mask = self.env.normalize(mask)

            return mask
        
        elif action == "rotate":
            ws[ws != self.env.obstacle_value] *= 0.8
            ws = self.env.normalize(ws)
            return ws

        return ws

def sensor_step(env, sensor, robot_position, ws):
    # nolasa sensoru rādījumus
    readings = sensor.read_sensors(robot_position, ws)
    # iegūst iespējamās pozīcijas
    possible_positions = sensor.estimate_position(readings)
    # atjaunina karti
    env.probability_ws = sensor.update_environment(possible_positions, env.probability_ws)

def robot_step(robot, env, robot_position):
    # iegūst nākamo robota soli
    robot_position, action, current_position = robot.get_next_step(robot_position)
    # atjaunina robota stāvokli
    robot.update_robot_pos(robot_position, action, current_position)
    # atjaunina karti
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