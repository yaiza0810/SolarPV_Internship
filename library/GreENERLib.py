import os
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score as R2
from matplotlib import rcParams
rcParams['figure.figsize'] = 20, 6
rcParams['lines.linewidth'] = 2.5

import pvlib
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

# from pso import PSO, CPSO
# from tqdm import tqdm
# from IPython.display import clear_output

from abc import ABC
from pyswarm import pso


class physicalform():
    
    """This class contains all the necessary functions to create a clear sky model from any PV panel"""

    def __init__(self, latitude = 45.20, longitude = 5.70, altitude = 212, tilt = 6, azimuth = 139, tz='Europe/Paris', start = '2022-09-10',
    end = '2023-06-01'):

        """ Initialization

            # Parameters :
            latitude : Int, 45.20 (default)
                Latitude of the wanted location for the computation

            longitude : Int, 5.70 (default)
                Longitude of the wanted location for the computation

            altitude : Int, 45.20 (default)
                Altitude of the wanted location for the computation

            tz : String 'Europe/Paris (default)
                Location timezone 
                
            site_location : None (default)
                Location objects are convenient containers for latitude, longitude, timezone, and altitude data associated with a particular geographic location. 

            solar_position : None (default)
                The solar zenith, azimuth, etc. at this location.

            airmass_relative : None (default)
                Relative (not pressure-adjusted) airmass at sea level.

            airmass_absolute : None (default)
                Absolute (pressure-adjusted) airmass from relative airmass and pressure.

            incidence_angle : None (default)
                Angle of incidence of the solar vector on a surface. This is the angle between the solar vector and the surface normal.

            Init site location to grenoble by default lat = 45.20, long = 5.70, alt = 212, timezone = Europe/Paris
            Init tilt = 6 and azimuth = 0 of the panels
            Init time stamp from 2022-09-10 to 2023-06-01
            Init solar position, relative and absolute airmass, incidence angle
        """

        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.tz = tz
        self.start = start
        self.end = end
        self.times = pd.date_range(self.start, self.end, freq='1H',tz = 'Europe/Paris')
        self.site_location = None
        self.solar_position = None
        self.airmass_relative = None
        self.airmass_absolute = None 
        self.incidence_angle = None
        #  dataframes 
        self.df_prod_toit = None
        self.df_inverters  = None
        self.create_df()

        
    def create_df(self, df):
         # creates the dataframe with the roof production
        self.df_prod_toit = pd.read_csv('data/prod.csv', parse_dates=True, index_col='Unnamed: 0')[' Production PV toiture instantanee reelle (kW)']
        self.df_prod_toit = self.df_prod_toit.resample('1H').mean().interpolate(method='time')
        self.df_prod_toit = self.df_prod_toit.loc[self.df_prod_toit.index <= self.end]
        self.df_prod_toit = self.df_prod_toit.loc[self.df_prod_toit.index >= self.start]

         # creates the dataframe with the inverters of the roof production
        self.df_inverters = pd.read_csv('data/pv_inverters_energy.csv',parse_dates=True, index_col='Unnamed: 0')
        self.df_inverters = self.df_inverters[self.df_inverters <= 1000]
        self.df_inverters = self.df_inverters[self.df_inverters != 0]
        self.df_inverters = self.df_inverters.interpolate('pad')
        self.df_inverters = self.df_inverters.resample('1H').mean().interpolate(method='time')
        self.df_inverters.loc["2023-01-05 13:00:00",'Onduleur: energie totale (MWh)'] = self.df_inverters.loc["2023-01-05 14:00:00",'Onduleur: energie totale (MWh)']
        self.df_inverters.loc["2022-12-22 11:00:00",'Onduleur: energie totale (MWh)'] = self.df_inverters.loc["2022-12-22 12:00:00",'Onduleur: energie totale (MWh)']
        self.df_inverters = self.df_inverters.diff().fillna(0) 
        self.df_inverters*=1000
        self.df_inverters = self.df_inverters.loc[self.df_inverters.index <= self.end]
        self.df_inverters = self.df_inverters.loc[self.df_inverters.index >= self.start]

        if df == None : 
            df = self.neb.copy()
            df['ghi_m'] = self.ghi_m['ghi_m']
            df['ghi_cs'] = self.ghi_cs['ghi']
        return df 



    def physical_val(self, tilt, azimuth):

        # create location object and get clearsky data
        self.site_location = pvlib.location.Location(latitude = self.latitude, longitude = self.longitude,tz = self.tz, altitude = self.altitude)

        #compute solar position trough times 
        self.solar_position = self.site_location.get_solarposition(self.times)

        # Compute absolute airmass
        self.airmass_relative  = pvlib.atmosphere.get_relative_airmass(self.solar_position['zenith'])
        self.airmass_absolute = pvlib.atmosphere.get_absolute_airmass(self.airmass_relative)

        # Compute aoi
        self.incidence_angle = pvlib.irradiance.aoi(surface_tilt=tilt, surface_azimuth= azimuth, solar_zenith= self.solar_position['zenith'], 
                                solar_azimuth= self.solar_position['azimuth'])
        


    def clear_sky_model(self, model ):
        """ create the wanted clear_sky model to use 
        ineichen or simplified solis """

        if model=="ineichen":
                linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(self.times, self.latitude, self.longitude, filepath=None, interp_turbidity=True)

                return pvlib.clearsky.ineichen(self.solar_position['apparent_zenith'], self.airmass_absolute, linke_turbidity, altitude=self.altitude, dni_extra=1364.0, perez_enhancement=False)

        elif model == "simplified solis":
                return pvlib.clearsky.simplified_solis(self.solar_position['apparent_elevation'], aod700=0.1, precipitable_water=1.0, pressure=101325.0, dni_extra=1364.0)
        else :
            raise "ce model n'existe pas "


    def calcule_pred(self, clear_sky = 'ineichen', model_irrad = 'haydavies', power=183, tilt = 6, azimuth = 139): 

        """ Compute the clearsky model based on the PV power
        Uses pvlib libary. 

        # Parameters
        Clear_sky : String, ineichen (default) | simplified solis
            Clear sky model

        Power : Int, 183 (default)
            Reference power of the PV panel

        model_irrad : String, haydavies (default) | perez | isotropic | klucher 
            Irradiance model

        tilt : Int, 6 (default)
            Inclination of the photovoltaique panels in degrees

        azimuth : Int, 139 (default)
            Orientation of the photovoltaique panels in degrees

        # Returns 
        df_pred : DataFrame 
            A dataframe containing the production under a clearsky of PV module of 183 KWp 

        """
        self.physical_val(tilt, azimuth)
        
        module = pvlib.pvsystem.retrieve_sam('SandiaMod')['Schott_Solar_ASE_300_DGF_50__320___2007__E__']
        cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

        temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        
        sapm_system = PVSystem(
            surface_tilt= tilt, 
            surface_azimuth= azimuth,
            module_parameters=module,
            inverter_parameters=cec_inverters,
            temperature_model_parameters=temperature_model_parameters)
        
        model_clear_sky = self.clear_sky_model(clear_sky)
       
        
        irradiance = PVSystem.get_irradiance(sapm_system, self.solar_position['zenith'], self.solar_position['azimuth'], model_clear_sky['dni'], model_clear_sky['ghi'], model_clear_sky['dhi'], model=model_irrad)

    
        # Compute effective irradiance
        effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(irradiance['poa_direct'], irradiance['poa_diffuse'], self.airmass_absolute, 
                                                    self.incidence_angle, module )

        # celltemp = pvlib.pvsystem.sapm_celltemp(irradiance['poa_direct'], wind, temperature)
        m_pow = (module["Impo"] * module["Vmpo"]) 
        
        df_pred=self.solar_position.copy()

        pv_produced = pvlib.pvsystem.sapm(effective_irradiance, 25, module)
        pv_produced ["p_mp"] = (pv_produced ["p_mp"]* power)/(m_pow/1000) 

        pv_produced ["p_mp"] /=1000  # kWh

        df_pred['power'] = pv_produced['p_mp']

        df_pred = df_pred.loc[df_pred.index >= self.start]
        df_pred = df_pred.loc[df_pred.index <= self.end]
        df_pred = df_pred['power']
        df_pred.index = pd.to_datetime(df_pred.index, format ='%Y-%m-%d %H:%M:%S') 
        df_pred =df_pred.tz_convert('Europe/Paris').tz_localize(None)
        

        return df_pred


    def inverters(self, power_angles, clear_sky = 'ineichen', model_irrad = 'haydavies'):
        """ Given the power on the pannel at different angles 139 and 41, connected to the same inverter
        It compute the clear sky for such configuration
        # Parameters

        power_angles : List, 
            The ith litst is the power and corresponding angle, power and angles to 1 inverter.
                Example : [[p1, a1], [p2,a2]] where p1 and a1 are the power and inclination of panel 1 (same for p2 and a2 with panel 2)
                    panel 1 and 2 are both linked to the same inverter
                    if only 1 panel then [p1, a1]
        
        # Returns
            df_pred : DataFrame 
            A dataframe containing the production under a clearsky the panel connected to the given inverters power and angles 
        """
        i = 0
        if type(power_angles) == list and type(power_angles[0]) == list:
            for p,a in power_angles:
                if i == 0:
                    df_pred = self.calcule_pred(clear_sky, model_irrad, power =p, azimuth=a)
                    i+=1
                else:
                    df_pred += self.calcule_pred(clear_sky, model_irrad, power =p, azimuth=a)
        elif type(power_angles) == list and type(power_angles[0]) == int or float:
            df_pred = self.calcule_pred(clear_sky, model_irrad, power =power_angles[0], azimuth=power_angles[1])
        
        else : 
            raise "power_angles is not of the correct type"

        return df_pred, df_true



class eval():
    def __init__(self) -> None:
        pass

    def plot_result(self, df_real, df_pred, path = 'figs/output.png'):

        """
            Plots and saves the figure in path
            
            df_real : Dataframe or Array
                Real values 
            df_pred : Dataframe or Array
                Predictions
            path : String, 'figs/output.png' (default) 
                Path were to save the figure

        """

        plt.figure()
        plt.plot(df_pred, label = 'pred')
        plt.plot(df_real, label = 'real')
        plt.legend()
        plt.savefig(path)
        # plt.show()


    def evaluation(self, df_real, df_pred):
        """ Given a datframe with the real values and another with the computed ones 
        It prints different evaluation metrics like RMSE, RMSE in %, MAE, MAPE, and R2"""
     

        real  = df_real.values
        pred = df_pred.values

        testScore = np.sqrt(MSE(real, pred))
        print('\n')
        print('Test Score: %.2f RMSE' % (testScore))
        print('RMSE en % : ', testScore*100/real.max())
        test_mae = MAE(real, pred) 
        print("MAE : ",test_mae)
        test_mape = MAPE(real, pred) 
        print("MAPE : ",test_mape)
        test_ape = R2(real, pred) 
        print("R2 : ",test_ape)
        print('\n')


class Models(ABC):

    # abstract method
    def predictions(self):
        """
            Returns the prediction and the truth values
        """
        pass
    
    def testing(self):
        pass

    def plot(self):
        pass


class XGBoost(Models):
    
    def XGBoost (self):
        """
         XGBoost model 
        """
        return True



class PSO(Models):

    def __init__(self, start = '2022-09-10 04', end = '2022-10-11 04') -> None:
        """
                start et end with form yyyy-mm-dd hh:mm:ss
        """
        super().__init__()
        
        self.start =start
        self.end = end
        # self.dates = None
        # self.index = None
        # self.i = None
        self.ghi_cs = None
        self.ghi_m = None
        self.neb = None
        self.df = self.create_df()

    def create_df(self):

        """
            Creates a dataframe with the nebulosity, the measured irradiance and the physical irraidance
        """
        # creates dataframe for the PSO parallelized 
        self.ghi_cs = pd.read_csv('data/Ineichen_clear-sky_model.csv', index_col='Unnamed: 0')[:'2023-06-01 01']
        self.ghi_cs = self.ghi_cs.drop('dhi', axis = 1)
        self.ghi_cs = self.ghi_cs.drop('dni', axis = 1)
        self.ghi_cs.index = pd.to_datetime(self.ghi_cs.index, format ='%Y-%m-%d %H:%M:%S')
        self.ghi_cs /= 1000
        # print('taille GHI_CS',len(self.ghi_cs))
        # print(self.ghi_cs)

        self.ghi_m = pd.read_csv('data/Rayonnement solaire-data-2023-06-13 09_46_08.csv', index_col='Time')[:'2023-06-01 01']
        self.ghi_m.columns = ['ghi_m']
        self.ghi_m.index = pd.to_datetime(self.ghi_m.index, format ='%Y-%m-%d %H:%M:%S')
        self.ghi_m/=1000
        # print('taille GHI_m',len(self.ghi_m))
        # print(self.ghi_m)
        
        self.neb =pd.read_csv("data/neb.csv", index_col='Date')
        self.neb.index = pd.to_datetime(self.neb.index, format ='%Y-%m-%d %H:%M:%S')
        self.neb= self.neb ['2022-09-10':'2023-06-01 00']
        # print('taille neb',len(self.neb))
        # print(self.neb)
        self.neb.index = self.ghi_cs.index
        self.ghi_m.index = self.ghi_cs.index
       

        df = self.neb.copy()
        df['ghi_m'] = self.ghi_m['ghi_m']
        df['ghi_cs'] = self.ghi_cs['ghi']
      
        return df 
        

    def loss_func(self, x):
        a = x[0]
        n = x[1]
        diff = np.square(self.df['ghi_m'][self.index:self.dates[self.i]] - (1- (a * np.power(self.df[' nebulosity'][self.index:self.dates[self.i]], n)))* self.df['ghi_cs'][self.index:self.dates[self.i]])
        loss = np.sum(diff)
        return loss

    def predictions(self, lag = 24):
        """ 
            Ompitizes a and n using the loss function based on a lag time and computes the predictions

            # Parameters
            lag : int 

        """
        lb = [0, 0]
        ub = [1, 10]
        xa = []
        xn = []
        self.i = 1
        lag_time = lag

        hour = (int(self.end[12])+lag)%24
        day = int(self.end[9])+(lag//24)
        end = self.end

        if lag >= 24 :
            if day < 10:
                end = end[:9] + str(day) + end[10:]
            else : 
                end = end[:8] + str(day) + end[10:]
            lag %= 24

        if lag < 24 :
            if hour < 10:
                end = end[:12] + str(hour) + end[13:]
            else : 
                end = end[:11] + str(hour) + end[13:]
        elif lag % 24 == 0 : 
            end = end[:9] + str((int(end[9])+1)%24) + end[10:]

        # print(end, self.end)

        self.dates = pd.date_range(self.start, end, freq="H")
        # print(self.dates)
        # print(self.df[self.start:self.end])
        
        
        for self.index, _ in self.df[self.start:self.end].iterrows():
            if ((self.i-1) % lag_time) == 0 :
                # print(index, self.dates[i])
                xopt, _ = pso(self.loss_func, lb, ub)
                

            xa.append(xopt[0])
            xn.append(xopt[1])
            self.i+=1

        df_pred = self.ghi_cs[self.start:self.end].copy()
        df_pred.columns = ['pred_']
        # print(df_pred)

        df_pred['pred_'] = ((1-xa*np.power(self.df[' nebulosity'][self.start:self.end].values, xn))* self.df['ghi_cs'][self.start:self.end].to_numpy())*1000
        # print(df_pred['pred_'])
        df_pred['pred_'] = [data if data >= 0 else 0 for data in pred['pred_']] 
        df_pred['pred_'] = [data if data <= 1000 else 1000 for data in pred['pred_']] 

        df_true =self.ghi_m[self.start:self.end]*1000

        return df_pred, df_true




class parallel_PSO(Models):

    def __init__(self) -> None:
        super().__init__()

    def loss_function(self, x):
        a = x[0]
        n = x[1]
        ghi_m = torch.Tensor(df['ghi_m'][index:index].to_numpy()).to(device)#.cuda()
        neb = torch.Tensor(df[' nebulosity'][index:index].to_numpy()).to(device)#.cuda()
        ghi_cs = torch.Tensor(df['ghi_cs'][index:index].to_numpy()).to(device)#.cuda()

        diff = torch.square(ghi_m - (1- (a * torch.pow(neb, n)))* ghi_cs)
        loss = diff.sum()
        return loss


    def PSO_parallelized (self, df):
        """"
            Particule Swarm Optimization model with parallalization
            df : Dataframe 
                Contains nebulosity, ghi_cs, ghi_m

        """

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu")
        print(f"Using {device} device")

        
        w = 0.5  # inertia
        c1 = 0.5  # cognitive (particle)
        c2 = 0.5  # social (swarm)

    
        xa = []
        xn = []
        for index, _ in df[:'2022-10-01'].iterrows():
            print(index)
            # print(df[index:index])
            _, xopt = PSO(self.loss_function, 100, 100, 2, w, c1, c2, -1, 1, device)
            xa.append(xopt[0][0])
            xn.append(xopt[0][1])
            clear_output(wait=False)
        
        pred = df[:'2022-10-01'].copy()
        pred = pred.drop(' nebulosity', axis =1)
        pred = pred.drop('ghi_m', axis =1)

        pred.columns = ['pred_']
        xa_ = torch.tensor(xa).detach().cpu()
        xn_ = torch.tensor(xn).detach().cpu()

        pred['pred_'] = ((1-xa_*np.power(df[' nebulosity'][:'2022-10-01'].values, xn_))* df['ghi_cs'][:'2022-10-01'].to_numpy())
        pred['pred_'] = [data if data >= 0 else 0 for data in pred['pred_']] 
        pred['pred_'] = [data if data <= 1 else 1 for data in pred['pred_']] #don't know if I keep it like this or not

        # this should go in data class 

        true = df[:'2022-10-01'].copy()
        true = true.drop(' nebulosity', axis =1)
        true = true.drop('ghi_cs', axis =1)
        true = true[:'2022-10-01']
        return pred


class LSTM():
    def __init__(self):
        self.LSTM_model = None
        
        # creates the dataframe for the training of the model 2020-2022
        self.weather_data = pd.read_csv('data/Weather_data_2020-2022.csv', parse_dates=True, index_col='DateHeure')


    def LSTM(self, train_X, train_y):
        self.LSTM_model = Sequential()
        self.LSTM_model.add(LSTM(128))
        self.LSTM_model.add(Dense(1, activation = 'relu'))
        self.LSTM_model.compile(loss= ['mse'], optimizer= 'Adam', metrics= tf.keras.metrics.RootMeanSquaredError())
        my_callbacks = tf.keras.callbacks.EarlyStopping(patience=10)
        self.LSTM_model.fit(train_X, train_y, epochs = 50, batch_size =32,validation_split=.2, shuffle = True, verbose =1, callbacks = my_callbacks)


    def create_X_y_data(self, dataset, time_lag):
        """
            Used in prediction to create the X and y data for training and testing. 
        """
        X, y = [],[]
        for i in range(len(dataset)- time_lag):
            X.append(dataset[i:(i+time_lag),0])
            y.append(dataset[i+time_lag,0])
        return np.array(X), np.array(y)

    # train the model one and save if then just load it, leave a function so it can be retrained still 

        
    def prediction(self, df, time_lag = 24, split_size = 0.7):
        """
            creates the data split

            # Parameters 

            df : Dataframe
                Used for training on testing based on slpit_size
        """
        data = df.values.reshape(-1,1)
        data = data.astype('float')
        scaler = MinMaxScaler(feature_range=(0,1))
        data = scaler.fit_transform(data)

        train_size = int(len(data)*split_size)

        train = data[0:train_size,:]
        test = data[train_size:-1,:]

        train_X, train_y = self.create_X_y_data(train, time_lag)
        test_X, test_y = self.create_X_y_data(test, time_lag)

        train_X = np.reshape(train_X, (train_X.shape[0],train_X.shape[1], -1))
        test_X = np.reshape(test_X, (test_X.shape[0],test_X.shape[1], -1))

        self.LSTM(train_X, train_y)
        test_predict = self.LSTM_model.predict(test_X)

        #invert prediction data
        test_predict = scaler.inverse_transform(test_predict)
        testY = scaler.inverse_transform([test_y])

        return  testY[0], test_predict


    def plot_train_test(data, test_predict, max, time_lag):
        test_predict_plot = np.empty_like(data)
        test_predict_plot[:, :] = np.nan
        test_predict_plot[(len(data) - len(test_predict))+(time_lag*2)+1:len(data), :] = test_predict
        plt.plot(data*max, label = 'True values')
        plt.plot(test_predict_plot, label = 'prediction on test set')
        plt.xlabel('Time')
        plt.ylabel('KiloWatts')
        plt.legend()
        plt.show()
    
class hybrid():

    def hybrid(self):
        """
            hybrid model 
        """
        return True

