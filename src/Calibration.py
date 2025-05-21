import pandas as pd
import os
import glob
import numpy as np

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

class Calibration:
    def __init__(self, file_path: str, sensor_number: int, sensor_type: str):
        self.file_path = file_path
        self.sensor_number = sensor_number
        self.sensor_type = sensor_type

        self.gain = 512
        self.int_time = 1057

        
        self.A_pm = np.pi*((2.5E-3)**2)      # Area of 5mm aperture on TLPM
        self.planck = (6.626e-34)            # Planck's constant (J⋅s)
        self.c_light = (3.0e8)               # Speed of light (m/s)
        self.avogadro = 6.02214076e23        # Avogadro's constant (mol⁻¹)

        self.df_raw = pd.DataFrame()
        self.df_norm = pd.DataFrame()
        self.df_norm_mean = pd.DataFrame()
        self.df_beamsplitter = pd.DataFrame()

        self.coefficients = []
        self.channels = []

    def _load_raw_data(self, file = "", **kwargs):
        try:
            if file == "":
                path = self.file_path
                # Define the pattern to match the files
                pattern = os.path.join(path, f"{self.sensor_type}.20*.sensor{self.sensor_number}.csv")

                # Use glob to find all files that match the pattern
                matching_files = glob.glob(pattern)

                # Check if any files are found and print their names
                if matching_files:
                    print("Found matching files:")
                    for file in matching_files:
                        print(os.path.basename(file))  # Print only the file name
                        file = os.path.basename(file)
                else:
                    print("No matching file found")
                    return False
                    
                self.df_raw = pd.read_csv(path + '/' + file, delimiter=",")
                self.df_raw = self.df_raw.reset_index(drop=True)

                return True
            else:
                self.df_raw = pd.read_csv(self.file_path+file, delimiter=",")
                self.df_raw = self.df_raw.reset_index(drop=True)
                return True

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def _basic_value(self):
        columns_to_modify = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'clear', 'nir']
        self.df_raw[columns_to_modify] = self.df_raw[columns_to_modify].div(self.gain * self.int_time)

    def _load_beamsplitter(self, beamsplitter_file = "BeamSplitterData.csv",**kwargs):
        try:
            self.df_beamsplitter = pd.read_csv(beamsplitter_file, delimiter=",")
            #self.df_beamsplitter['ratio'] = self.df_beamsplitter['trans_adj_bs']/self.df_beamsplitter['refl_adj_bs']  # ratio of transmitted (DUT) to reflected (PM)
            #print("Beamsplitter data loaded")
        except Exception as e:
            print(f"Error loading beamsplitter data: {e}")

    
    def _calculate_irradiance(self):
        self.df_raw = pd.merge(self.df_raw, self.df_beamsplitter[['wavelength', 'ratio']], on='wavelength', how='left')
        P_pm = self.df_raw['power']                          # Power (W)

        I_dut = P_pm * self.df_raw['ratio'] / self.A_pm      # W/m²

        self.df_raw['I_dut'] = I_dut
        self.df_raw['P_dut'] = P_pm * self.df_raw['ratio']
        self.df_raw = self.df_raw[self.df_raw["clear"] > 0]    # remove measurements which may have failed, with a clear channel reading of 0

    def _calculate_iterations(self):
        self.df_raw['cycle'] = 0

        # Find the indices where 'wavelength' is 390, as this was the starting wavelength for each cycle
        start_indices = self.df_raw.index[self.df_raw['wavelength'] == 390].tolist()

        # Assign cycle numbers (to make a unique data set for each measurement)
        for cycle_number in range(len(start_indices) - 1):
            start_idx = start_indices[cycle_number]
            end_idx = start_indices[cycle_number + 1]
            self.df_raw.loc[start_idx:end_idx - 1, 'cycle'] = cycle_number + 1

        # for the last cycle, if needed
        if len(start_indices) > 1:
            self.df_raw.loc[start_indices[-1]:, 'cycle'] = len(start_indices)

        #print(str(self.df_raw['cycle'].max()), " iterations are in the data.")      # to see how many cycles are in this data. Total new rows of data will be 8* this value (for the number of bins)

    def _normalization(self):
        # normalize to incident light 
        df_normalize = self.df_raw.copy()
        df_normalize["f1"] = self.df_raw["f1"]/self.df_raw["I_dut"]
        df_normalize["f2"] = self.df_raw["f2"]/self.df_raw["I_dut"]
        df_normalize["f3"] = self.df_raw["f3"]/self.df_raw["I_dut"]
        df_normalize["f4"] = self.df_raw["f4"]/self.df_raw["I_dut"]
        df_normalize["f5"] = self.df_raw["f5"]/self.df_raw["I_dut"]
        df_normalize["f6"] = self.df_raw["f6"]/self.df_raw["I_dut"]
        df_normalize["f7"] = self.df_raw["f7"]/self.df_raw["I_dut"]
        df_normalize["f8"] = self.df_raw["f8"]/self.df_raw["I_dut"]
        df_normalize["clear"] = self.df_raw["clear"]/self.df_raw["I_dut"]
        df_normalize["nir"] = self.df_raw["nir"]/self.df_raw["I_dut"]
        df_normalize["I_dut"] = self.df_raw["I_dut"]/self.df_raw["I_dut"]

        # store scaled data in df_norm
        self.df_norm = df_normalize.copy()
        self.df_norm = self.df_norm.sort_values(by="wavelength").reset_index(drop=True)
        self.df_norm_mean = self.df_norm.groupby("wavelength").mean(numeric_only=True).reset_index().copy()

    def _ppfd_calculation(self):
        # Calculate the PPFD (Photosynthetic Photon Flux Density) using the formula
        start = 400
        end = 700

        self.df_norm["PAR"] = (self.df_norm['I_dut'] * self.df_norm['wavelength']*1E-9).astype(float) / (self.planck * self.c_light) * ((1e6)/(self.avogadro))
        self.df_norm.loc[(self.df_norm['wavelength'] < start) | (self.df_norm['wavelength'] > end), 'PAR'] = 0

        self.df_norm_mean["PAR"] = (self.df_norm_mean['I_dut'] * self.df_norm_mean['wavelength']*1E-9).astype(float) / (self.planck * self.c_light) * ((1e6)/(self.avogadro))
        self.df_norm_mean.loc[(self.df_norm_mean['wavelength'] < start) | (self.df_norm_mean['wavelength'] > end), 'PAR'] = 0

    def _save_data(self):
        try:
            # save the resulting data to a new dataframe and file
            file_name_normmean = ('/' + self.sensor_type + f'.Sensor{self.sensor_number}_preprocessed_normed+mean.csv')
            self.df_norm_mean.to_csv(self.file_path+file_name_normmean, index=False)

            file_name_norm = ('/' + self.sensor_type + f'.Sensor{self.sensor_number}_preprocessed_normed.csv')
            self.df_norm.to_csv(self.file_path+file_name_norm, index=False)

            print("Data saved to: ", self.file_path)

        except Exception as e:
            print(f"Error save data: {e}")

    def preprocessing(self, save_data = True, PAR = True, **kwargs):
        if self._load_raw_data(**kwargs):
            self._basic_value()
            self._load_beamsplitter(**kwargs)
            self._calculate_irradiance()
            self._calculate_iterations()
            self._normalization()
            self._ppfd_calculation(**kwargs)
            if not PAR:
                self._irradiance_transform(**kwargs)
            if save_data:
                self._save_data()
            return True
        else:
            return False

    def _load_norm_data(self):
        try:
            path = self.file_path
            # Define the pattern to match the files
            pattern = os.path.join(path, self.sensor_type + f".Sensor{self.sensor_number}_preprocessed_normed.csv")

            # Use glob to find all files that match the pattern
            matching_files = glob.glob(pattern)

            # Check if any files are found and print their names
            if matching_files:
                print("Found matching files:")
                for file in matching_files:
                    print(os.path.basename(file))  # Print only the file name
                    file = os.path.basename(file)
            else:
                print("No matching file found")
                
            self.df_norm = pd.read_csv(path + '/' + file, delimiter=",")
            self.df_norm = self.df_norm.reset_index(drop=True)

            return self.df_norm

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def _execute_pls(self, channels = None, **kwargs):
        if self.df_norm['cycle'].max() > 1:
            df_pls = self.df_norm[self.df_norm['cycle'].between(1, self.df_norm['cycle'].max() - 1)].copy()
        else:
            df_pls = self.df_norm.copy()

        df_pls = df_pls.dropna()
        if channels is None:    
            self.channels = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"]
        else:
            self.channels = channels
        X_train =  df_pls[self.channels]
        y_train =  df_pls["PAR"]

        components = np.arange(1,len(self.channels)+1)
        xticks = np.arange(1,len(self.channels)+1)

        r2s = []
        rmses = []

        for n_comp in components:
            pls = PLSRegression(n_components=n_comp)
            y_pred = cross_val_predict(pls, X_train, y_train, cv=10)

            # Calculate scores
            r2s.append(r2_score(y_train, y_pred))
            mse = mean_squared_error(y_train, y_pred)
            rmses.append(np.sqrt(mse))

        max_r2 = max(r2s)
        min_rmse = min(rmses)
        max_r2_index = r2s.index(max_r2)
        min_rmse_index = rmses.index(min_rmse)
        #print("Max R2 is reached with %d PLS components" %(max_r2_index+1))
        #print('R2: %0.4f, RMSE: %0.8f' % (max_r2, min_rmse))

        pls2 = PLSRegression(n_components=max_r2_index+1)
        pls2.fit(X_train, y_train)
        self.coefficients = pls2.coef_

        print("Coefficients:", ", ".join(map(str, self.coefficients.flatten())))

    def _save_coefficients(self, **kwargs):
        try:
            # Define the file name and path
            file_name = ('/' + self.sensor_type + f'_coefficients.csv')
            full_path = self.file_path + file_name

            # Create a DataFrame for the current coefficients
            coeff_data = pd.DataFrame([{
            "sensor_type": self.sensor_type,
            "sensor_number": self.sensor_number,
            **{f"coef_{i+1}": coef for i, coef in enumerate(self.coefficients.flatten()[:-2])},
            "coef_clear": self.coefficients.flatten()[-2],
            "coef_nir": self.coefficients.flatten()[-1]
            }])

            # Check if the file already exists
            if os.path.exists(full_path):
                # Load the existing file
                existing_data = pd.read_csv(full_path)
                # Check if the sensor_number already exists
                if self.sensor_number in existing_data["sensor_number"].values:
                    # Overwrite the row with the same sensor_number
                    existing_data = existing_data[existing_data["sensor_number"] != self.sensor_number]
                # Append the new data
                updated_data = pd.concat([existing_data, coeff_data], ignore_index=True)
            else:
                # If the file doesn't exist, create it with the new data
                updated_data = coeff_data

            # Sort the data by sensor_number before saving
            updated_data = updated_data.sort_values(by="sensor_number")
            # Save the updated data back to the file
            updated_data.to_csv(full_path, index=False)

            print("Coefficients saved to: ", full_path)

        except Exception as e:
            print(f"Error saving coefficients: {e}")

    def pls(self, save_coef = True, channels = None, **kwargs):
        #self._load_norm_data()
        self._execute_pls(channels=channels, **kwargs)
        if save_coef:
            self._save_coefficients(**kwargs)

    def _load_norm_mean_data(self):
        try:
            path = self.file_path
            # Define the pattern to match the files
            pattern = os.path.join(path, self.sensor_type + f".Sensor{self.sensor_number}_preprocessed_normed+mean.csv")

            # Use glob to find all files that match the pattern
            matching_files = glob.glob(pattern)

            # Check if any files are found and print their names
            if matching_files:
                print("Found matching files:")
                for file in matching_files:
                    print(os.path.basename(file))  # Print only the file name
                    file = os.path.basename(file)
            else:
                print("No matching file found")
                
            self.df_norm_mean = pd.read_csv(path + '/' + file, delimiter=",")
            self.df_norm_mean = self.df_norm_mean.reset_index(drop=True)


        except Exception as e:
            print(f"Error loading data: {e}")

        
    def _repeatability(self):
        df_repeatability = self.df_norm.copy()

        # Calculate mean and standard deviation for each wavelength in the range 390 to 439
        df_repeatability["PAR_calculated"] = df_repeatability[self.channels].mul(self.coefficients, axis=1).sum(axis=1)
        stats_390_399 = df_repeatability[df_repeatability['wavelength'].between(390, 399)].groupby('wavelength')["PAR_calculated"].agg(['mean', 'std']).reset_index()
        stats_400_439 = df_repeatability[df_repeatability['wavelength'].between(400, 439)].groupby('wavelength')["PAR_calculated"].agg(['mean', 'std']).reset_index()
        stats_440_700 = df_repeatability[df_repeatability['wavelength'].between(440, 700)].groupby('wavelength')["PAR_calculated"].agg(['mean', 'std']).reset_index()
        stats_701_1100 = df_repeatability[df_repeatability['wavelength'].between(701, 1100)].groupby('wavelength')["PAR_calculated"].agg(['mean', 'std']).reset_index()

        # Combine the results into a single DataFrame with an additional column indicating the range
        stats_390_399["range"] = "390_399"
        stats_400_439["range"] = "400_439"
        stats_440_700["range"] = "440_700"
        stats_701_1100["range"] = "701_1100"
        stats = pd.concat([stats_390_399, stats_400_439, stats_440_700, stats_701_1100], ignore_index=True)
                
        return stats
    
    # Define Gaussian function
    def _gaussian(self, x, A, mu, sigma):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

    def plot_gaussian(self, save = False, spectral_range = [390, 1100]):
        df_plot = self.df_norm_mean.copy()

        # Channel Colours (based on https://405nm.com/wavelength-to-color/)
        f1_colour = '#7600ed'       # λ = 415nm
        f1_colour_dark = '#5300a5'  # λ = 415nm, 30% darker
        f1_colour_light = '#a020ff' # λ = 415nm, 30% lighter
        f2_colour = '#0028ff'       # λ = 445nm
        f2_colour_dark = '#001eb2'  # λ = 445nm, 30% darker
        f2_colour_light = '#0036ff' # λ = 445nm, 30% lighter
        f3_colour = '#00d5ff'       # λ = 480nm
        f3_colour_dark = '#0098b2'  # λ = 480nm, 30% darker
        f3_colour_light = '#19ffff' # λ = 480nm, 30% lighter
        f4_colour = '#1fff00'       # λ = 515nm
        f4_colour_dark = '#17b200'  # λ = 515nm, 30% darker
        f4_colour_light = '#29ff00' # λ = 515nm, 30% lighter
        f5_colour = '#b3ff00'       # λ = 555nm
        f5_colour_dark = '#7eb200'  # λ = 555nm, 30% darker
        f5_colour_light = '#dcff19' # λ = 555nm, 30% lighter
        f6_colour = '#ffdf00'       # λ = 590nm
        f6_colour_dark = '#b2a500'  # λ = 590nm, 30% darker
        f6_colour_light = '#fff719' # λ = 590nm, 30% lighter
        f7_colour = '#ff4f00'       # λ = 630nm
        f7_colour_dark = '#b23800'  # λ = 630nm, 30% darker
        f7_colour_light = '#ff6733' # λ = 630nm, 30% lighter
        f8_colour = '#df0000'       # λ = 680nm
        f8_colour_dark = '#9f0000'  # λ = 680nm, 30% darker
        f8_colour_light = '#ff3333' # λ = 680nm, 30% lighter

        # Measurement Markers {'none' none | 'o' circle | '^' triangle | 's' square |'X' X | '+' plus}
        f1_mark = 'o'       # λ = 415nm
        f2_mark = 'o'       # λ = 445nm
        f3_mark = 'o'       # λ = 480nm
        f4_mark = 'o'       # λ = 515nm
        f5_mark = 'o'       # λ = 555nm
        f6_mark = 'o'       # λ = 590nm
        f7_mark = 'o'       # λ = 630nm
        f8_mark = 'o'       # λ = 680nm
        
        # Gaussian fit for F1 to F8
        channels = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
        colors = [f1_colour, f2_colour, f3_colour, f4_colour, f5_colour, f6_colour, f7_colour, f8_colour]
        fits = {}
        mark_sz = 1.5
        plt.rcParams.update({'font.size': 16})

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "Times New Roman"

        df_plot = df_plot[(df_plot["wavelength"] >= spectral_range[0]) & (df_plot["wavelength"] <= spectral_range[1])].copy()

        df_plot['PAR_slope'] = (df_plot['wavelength']/10E9).astype(float) / (self.planck * self.c_light) * ((1e6)/(self.avogadro))
        df_plot.loc[(df_plot['wavelength'] < 401) | (df_plot['wavelength'] > 700), 'PAR_slope'] = 0
        df_plot['PAR_slope'] = df_plot['PAR_slope']/df_plot['PAR_slope'].max()

        #fig, ax = plt.subplots()
        plt.figure(figsize=([8,4]))
        for channel, color in zip(channels, colors):
            x = df_plot["wavelength"].values
            y = df_plot[channel].values/df_plot["f8"].max()

            A_guess = max(y)
            mu_guess = x[np.argmax(y)]  # x corresponding to max y
            sigma_guess = (max(x) - min(x)) / 6
            p0 = [A_guess, mu_guess, sigma_guess]
            
            try:
                # Gaussian fit with bounds and higher maxfev
                popt, _ = curve_fit(
                    self._gaussian, x, y, p0=p0, 
                    bounds=([0, min(x), 0], [np.inf, max(x), np.inf]),  # A > 0, mu in [min, max], sigma > 0
                    maxfev=5000  # Increase maximum iterations
                )
                fits[channel] = popt
                # Plot scatter
                plt.scatter(x, y,  color=color, s=mark_sz, marker=f1_mark)
                # Plot Gaussian fit
                x_fit = np.linspace(min(x), max(x), 500)
                plt.plot(x_fit, self._gaussian(x_fit, *popt), label=channel.upper(), color=color, linestyle='--',linewidth=0.7)
            except RuntimeError as e:
                print(f"Gaussian fit failed for {channel}: {e}")
                plt.scatter(x, y, label=f"{channel.upper()} (fit failed)", color=color, s=mark_sz, marker=f1_mark)

        # Moving average for Clear and NIR
        for channel, color in zip(["clear", "nir"], ["#b4c9ff", "#670505"]):
            x = df_plot["wavelength"].values
            y = df_plot[channel].values/df_plot["f8"].max()
            y_smooth = pd.Series(y).rolling(window=30, center=True).mean()  # Moving average with a window of 10
            # Plot scatter
            plt.scatter(x, y, color=color, s=mark_sz, marker='+', label=channel.upper())

        plt.step(df_plot['wavelength'], df_plot['PAR_slope'], linestyle='--',linewidth=0.7, label=f'Ideal\nPPFD', color="black")
        plt.grid()
        plt.xlabel('Wavelength (nm)', fontsize=14)
        plt.ylabel('Normalized Channel Response', fontsize=14)
        plt.title(f'Normalized Channel Response of AS7341', fontsize=16)
        #plt.title(f'Normalized Channel Response of AS7341\n{self.sensor_type} Sensor {self.sensor_number}', fontsize=18)
        #plt.xlim([395,755])
        plt.legend(loc='lower left' , bbox_to_anchor=[0.99, -0.050], fontsize=14)
        plt.tight_layout()

        if save:
            plt.savefig(self.file_path + '/' + self.sensor_type + f'.Sensor{self.sensor_number}_gaussian_fit.png', dpi=600, bbox_inches='tight')

    def plot_quantum_response(self, save = False, plot = True):
        fr1  = '#344a9a'    # freiburg logo blue
        fr1a = '#868dc2'    # freiburg logo blue (lighter)
        fr1b = '#afb1d8'    # freiburg logo blue (lightest)
        fr1c = '#00004a'    # freiburg logo blue (darker)
        fr2  = '#93bc3c'    # ecosense green
        fr2a = '#c1e653'    # ecosense green (lighter)
        fr2a = '#659023'    # ecosense green (darker)
        fr3  = '#00a082'    # freiburg teal
        fr4  = '#f5c2cd'    # freiburg pink
        fr5  = '#ffe863'    # freiburg yellow
        fr6  = '#8f6b30'    # freiburg brown

        # Measurement Markers {'none' none | 'o' circle | '^' triangle | 's' square |'X' X | '+' plus}
        f1_mark = 'o'       # λ = 415nm
        f2_mark = 'o'       # λ = 445nm
        f3_mark = 'o'       # λ = 480nm
        f4_mark = 'o'       # λ = 515nm
        f5_mark = 'o'       # λ = 555nm
        f6_mark = 'o'       # λ = 590nm
        f7_mark = 'o'       # λ = 630nm
        f8_mark = 'o'       # λ = 680nm

        # Marker Size
        mark_sz = 1

        df_plot = self.df_norm[self.df_norm['cycle'] == self.df_norm['cycle'].max()].copy()

        df_plot["PAR_calculated"] = df_plot[self.channels].mul(self.coefficients, axis=1).sum(axis=1)

        if plot:
            plt.figure(figsize=[8,4])

            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = "Times New Roman"


            plt.scatter(df_plot["wavelength"], df_plot["PAR_calculated"]/df_plot['PAR'].max(), label='Calculated PAR', color=fr1, s=mark_sz, marker=f1_mark)
            plt.step(df_plot['wavelength'], df_plot['PAR']/df_plot['PAR'].max(), linestyle='--', label=f'Ideal Quantum Response', color=fr2)

            plt.rcParams['axes.labelsize'] = 14
            plt.rcParams['xtick.labelsize'] = 14
            plt.rcParams['ytick.labelsize'] = 14
            plt.rcParams['axes.titlesize'] = 16

            plt.grid()
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Normalized Response')
            # Remove rows with NaN values in 'PAR' or 'PAR_calculated'
            df_plot_filtered = df_plot.dropna(subset=['PAR', 'PAR_calculated'])
            plt.text(820, 0.23, f"R2: {r2_score(df_plot_filtered['PAR'], df_plot_filtered['PAR_calculated']):.3f}", fontsize = 14)
            plt.title(f'Quantum Response')
            #plt.title(f'{self.sensor_type} Sensor {self.sensor_number} Quantum Response')
            plt.legend(loc = "lower left", bbox_to_anchor=(0.5,0.5), fontsize = 14)

            plt.subplots_adjust(bottom=0.15)

        if save:
            plt.savefig(self.file_path + '/' + self.sensor_type + f'.Sensor{self.sensor_number}_quantum_response.png', dpi=600, bbox_inches='tight')

        return df_plot["wavelength"], df_plot["PAR_calculated"]/df_plot['PAR'].max(), df_plot['PAR']/df_plot['PAR'].max()