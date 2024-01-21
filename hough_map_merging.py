import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift

class MapMerge:
    def __init__(self):
        self.theta_val = np.arange(0, 181, 5)

    # izveido divas kartes
    def create_maps(self):
        
        # izveido divas 20x20 kartes (masīvus)
        map_size = 20
        map_1 = np.zeros((map_size, map_size), dtype=int)
        map_2 = np.zeros((map_size, map_size), dtype=int)

        # pievieno šķēršļus
        map_1[1, 1:4] = 1
        map_1[5:8, 4:7] = 1
        map_1[10:15, 9] = 1
        map_1[14:16, 10:14] = 1
        map_1[9, 9:14] = 1
        map_1[15:18, 2] = 1
        map_1[18, 2:7] = 1
        map_1[3:7, 15:18] = 1

        # map_2[2:7, 2] = 1
        # map_2[6, 2:8] = 1
        # map_2[2:6, 7:9] = 1
        # map_2[13, 8:11] = 1
        # map_2[9:14, 11] = 1
        # map_2[9:12, 0] = 1
        # map_2[4:6, 13:17] = 1
        # map_2[9:14, 14] = 1
        # map_2[13, 14:18] = 1

        # map_2 = np.rot90(map_2, k=2)
        translation = [3, -8]
        map_2 = shift(map_1, shift=translation)
        map_2 = np.rot90(map_2, k=-1)

        # centieni atrast otro karti, kas derētu veiksmīgam scenārijam
        # map_2[13:17, 3:6] = 1
        # map_2[6:9, 7] = 1
        # map_2[11:16, 12] = 1
        # map_2[14:18, 13] = 1
        # map_2[9:14, 13] = 1
        # map_2[16:19, 6] = 1
        # map_2[18, 2:7] = 1
        # map_2[15:18, 15:18] = 1

        return map_1, map_2

    # aprēķina Hju spektru
    def calculate_hough_spectrum(self, map):

        theta_val = self.theta_val
        results = {}

        # atrod iezīmju lokācijas
        indices = np.where(map == 1)

        for theta in theta_val:

            # aprēķina attālumus
            p_values = indices[1] * np.cos(np.radians(theta)) + indices[0] * np.sin(np.radians(theta))

            # noapaļo vērtības
            rounded_p_values = np.round(p_values).astype(int)

            unique_values, counts = np.unique(rounded_p_values, return_counts=True)

            sum_of_squared_counts = np.sum(counts**2)

            results[theta] = sum_of_squared_counts

        normalized_results = self.normalize_results(results)

        return normalized_results

    # normalizē spektru vērtības
    # def normalize_results(self, results):

    #     min_value = min(results.values())
    #     max_value = max(results.values())

    #     normalized_results = {key: (value - min_value) / (max_value - min_value) for key, value in results.items()}

    #     return normalized_results

    def normalize_results(self, data):
        if isinstance(data, dict):
            # For dictionaries
            min_value = min(data.values())
            max_value = max(data.values())

            normalized_data = {key: (value - min_value) / (max_value - min_value) for key, value in data.items()}
        elif isinstance(data, np.ndarray):
            # For arrays
            min_value = np.min(data)
            max_value = np.max(data)

            normalized_data = (data - min_value) / (max_value - min_value)
        else:
            raise ValueError("Unsupported data type. Supported types: dict, np.ndarray")

        return normalized_data

    # vizualizē spektru
    def visualize_spectrum(self, results_1, results_2=None, results_3=None, title=''):
        # definē X vērtības, lai varētu vizualizēt spektrus 360 grādos
        theta_values = list(results_1.keys())
        mirrored_thetas = [theta + 180 for theta in theta_values]
        all_thetas = np.concatenate([theta_values, mirrored_thetas])
        
        # duplicē Y vērtības, lai varētu vizualizēt spektrus 360 grādos
        all_values_1 = [results_1[theta] for theta in theta_values] + [results_1[theta] for theta in theta_values]

        # vizualizēt spektru
        fig, ax1 = plt.subplots()
        ax1.plot(all_thetas, all_values_1, marker='o', linestyle='-', color='b', label='Spectrum 1')

        if results_2 is not None:
            all_values_2 = [results_2[theta] for theta in theta_values] + [results_2[theta] for theta in theta_values]
            ax1.plot(all_thetas, all_values_2, marker='o', linestyle='-', color='g', label='Spectrum 2')

        if results_3 is not None:
            all_values_3 = [results_3[theta] for theta in theta_values] + [results_3[theta] for theta in theta_values]
            ax1.plot(all_thetas, all_values_3, marker='o', linestyle='-', color='r', label='Spectrum 3')

        ax1.set_xlabel('Theta')
        ax1.set_ylabel('Normalized Spectrum Values', color='black')
        ax1.tick_params('y', colors='black')
        ax1.legend(loc='upper left')

        plt.title(title)

        x_ticks = np.arange(0, 360, 45)

        plt.xticks(x_ticks)

        plt.grid(True)
        plt.show()

    # vizualizē karti
    def visualize_map(self, map, title=''):

        height, width = map.shape

        plt.imshow(map, cmap='binary', interpolation='none', extent=[0, width, height, 0])
        plt.title(title)

        x_ticks = np.arange(0, width + 1, 1)
        y_ticks = np.arange(0, height + 1, 1)

        plt.xticks(x_ticks)
        plt.yticks(y_ticks)

        plt.grid(True)
        plt.show()
    
    # aprēķina cirkulāro korelāciju starp diviem spektriem
    def calculate_circular_correlation(self, spectrum_1, spectrum_2):
        
        # pārvērš spektrus no dict uz list
        spectrum_1 = list(spectrum_1.values())
        spectrum_2 = list(spectrum_2.values())

        cc = {}

        # iterē katrai rotācijai
        for i, theta in enumerate(self.theta_val):

            r_spectrum = np.roll(spectrum_2, shift=-i)
            cc[theta] = np.round(np.sum(spectrum_1 * r_spectrum), decimals=2)

        # normalizē rezultātus
        normalized_results = self.normalize_results(cc)

        # nosaka maksimālos punktus
        max_theta, max_value = max(normalized_results.items(), key=lambda x: x[1])
        print(f"The theta with the maximum value is '{max_theta}' with a value of {max_value}.")

        return normalized_results, max_theta
    
    # rotēt karti
    def rotate_map(self, map, theta):

        rotated_map = rotate(map, angle=theta, reshape=False)

        return rotated_map
    
    # translēt karti
    def translate_map(self, map, translation):

        translated_map = shift(map, shift=translation)

        return translated_map
    
    # def translate_map(self, map, translation):
    #     map_size = np.array(map.shape) + np.abs(translation)
    #     merged_map = np.zeros(map_size)

    #     start_position = np.maximum(0, np.array(translation))

    #     merged_map[start_position[0]:start_position[0]+map.shape[0], start_position[1]:start_position[1]+map.shape[1]] = map

    #     self.visualize_map(merged_map)

    #     return merged_map
    
    # aprēķina kartes X un Y spektrus
    def get_XY_spectrums(self, map):

        x_sums = np.sum(map, axis=0)  # iezīmes uz X ass

        y_sums = np.sum(map, axis=1)  # iezīmes uz Y ass

        return x_sums, y_sums
    
    # vizualizē XY spektrus    
    def visualize_XY_spectrums(self, x_spectrum, y_spectrum, title=''):

        # salīdzina abus spektrus, lai to garumi sakristu
        min_length = min(len(x_spectrum), len(y_spectrum))
        x_values = np.arange(-10, 11)[:min_length]

        plt.plot(x_values, x_spectrum, label='X spectrum')
        plt.plot(x_values, y_spectrum, label='Y spectrum')

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title(title)
        plt.legend() 

        plt.xticks(np.arange(-10, 11, step=1))
        plt.grid(True)
        plt.show()
    
    # aprēķina X un Y spektru korelācijas
    def get_XY_correlation(self, map1_sums, map2_sums):

        ccx = {}
        ccy = {}

        # translāciju iespējamie galējie punkti
        max_translation = 11
        min_translation = -10

        for i in range(min_translation, max_translation):
            rx = np.roll(map2_sums[0], shift=i)
            ry = np.roll(map2_sums[1], shift=i)

            ccx[i] = np.round(np.sum(map1_sums[0] * rx), decimals=2)
            ccy[i] = np.round(np.sum(map1_sums[1] * ry), decimals=2)

        # normalizē rezultātus
        normalized_ccx = self.normalize_results(ccx)
        normalized_ccy = self.normalize_results(ccy)

        max_ccx, max_ccx_value = max(normalized_ccx.items(), key=lambda x: x[1])
        print(f"The X translation with the maximum value is '{max_ccx}' with a value of {max_ccx_value}.")

        max_ccy, max_ccy_value = max(normalized_ccy.items(), key=lambda x: x[1])
        print(f"The Y translation with the maximum value is '{max_ccy}' with a value of {max_ccy_value}.")

        return max_ccx, max_ccy, normalized_ccx, normalized_ccy
    
    # apvieno kartes
    def merge_maps(self, map1, map2, translation, rotation):

        r_map2 = self.rotate_map(map2, rotation)
        transformed_map2 = self.translate_map(r_map2, translation)

        final_size = np.maximum(map1.shape, np.array(transformed_map2.shape) + np.abs(translation))

        # izveido tukšu karti
        merged_map = np.zeros(final_size)

        # aprēķina otrās kartes sākuma punktu
        start_position = np.maximum(0, np.array(translation))

        # ievieto pirmo karti jaunajā tukšajā kartē
        merged_map[start_position[0]:start_position[0] + map1.shape[0], start_position[1]:start_position[1] + map1.shape[1]] = map1

        # ievieto transformēto karti jaunajā kartē
        merged_map[:transformed_map2.shape[0], :transformed_map2.shape[1]] = np.maximum(merged_map[:transformed_map2.shape[0], :transformed_map2.shape[1]], transformed_map2)

        return merged_map
   
def main():

    mapmerge = MapMerge()

    # pirmais uzdevums
    map_1, map_2 = mapmerge.create_maps()
    mapmerge.visualize_map(map_1, title='map 1')
    mapmerge.visualize_map(map_2, title='map 2')

    # otrais uzdevums
    map1_hough = mapmerge.calculate_hough_spectrum(map_1)
    mapmerge.visualize_spectrum(map1_hough, title='map1 spectrum')

    map2_hough = mapmerge.calculate_hough_spectrum(map_2)
    mapmerge.visualize_spectrum(map2_hough, title='map2 spectrum')

    # trešais uzdevums
    cc, theta = mapmerge.calculate_circular_correlation(map1_hough, map2_hough)
    # theta = 90
    mapmerge.visualize_spectrum(map1_hough, map2_hough, cc, title='All spectrums')

    # ceturtais uzdevums
    rotated_map = mapmerge.rotate_map(map_2, theta)
    mapmerge.visualize_map(rotated_map, title='Rotated map')

    # piektais uzdevums
    map1_x, map1_y = mapmerge.get_XY_spectrums(map_1)
    map2_x, map2_y = mapmerge.get_XY_spectrums(rotated_map)

    mapmerge.visualize_XY_spectrums(mapmerge.normalize_results(map1_x), mapmerge.normalize_results(map1_y), title='map_1 X, Y spectrum')
    mapmerge.visualize_XY_spectrums(mapmerge.normalize_results(map2_x), mapmerge.normalize_results(map2_y), title='map_2 X, Y spectrum')

    tx, ty, ccx, ccy = mapmerge.get_XY_correlation([map1_x, map1_y],[map2_x, map2_y])
    _, ccx = zip(*ccx.items())
    _, ccy = zip(*ccy.items())
    # mapmerge.visualize_XY_spectrums(ccx.items(), ccy.items(), title='map_1 and map_2 X,Y correlation spectrum')
    mapmerge.visualize_XY_spectrums(ccx, ccy, title='map_1 and map_2 X,Y correlation spectrum')
    translated_map = mapmerge.translate_map(map_2, [tx, ty])
    mapmerge.visualize_map(translated_map, title='Translated map')

    # sestais uzdevums
    final_map = mapmerge.merge_maps(map_1, map_2, [tx, ty], theta)
    mapmerge.visualize_map(final_map, title='Final map')

if __name__ == "__main__":
    main()