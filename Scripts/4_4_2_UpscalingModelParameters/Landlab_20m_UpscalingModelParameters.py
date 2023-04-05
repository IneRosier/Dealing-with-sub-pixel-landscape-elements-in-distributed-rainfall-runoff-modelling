# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 18:34:50 2021

@author: u0117123
"""
### 1 RUN 


####################### Overland Flow ############################
######## STEP 1 - Import libraries
## Landlab components
import landlab
from landlab.components import OverlandFlow, SoilInfiltrationGreenAmpt, SinkFiller # SinkFiller is optional
## Landlab utilities
from landlab.io import read_esri_ascii 
from landlab.plot import imshow_grid  # plotter functions are optional
from landlab.components import SinkFillerBarnes
## Additional Python packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt # plotter functions are optional
import os
import datetime
from osgeo import gdal
from osgeo import osr
import cv2
import matplotlib.pyplot as plt
import rasterio

#%%create output directory
infiltration = "inf" #"inf" or "Ksat0"
DTM_slope = "velm" #flat or steep or flat_scaled
DTM_type = "heulengracht_burn" #plane or mpd_plane
fill_sinks = "fill" #fill or no_fill
psi_value  = 0.1727
rainfall_events = ["20060614", "20060821","20070618", "20070809", "20070821", 
                   "20070903", "20130507", "20130522", "20130818", "20160530",
                   "20160601", "20160623", "20190610",  "20190727", "20190817"]
outlet_id = 10171
scaling = "FlLength_manning"# "FlLength_Ks" "FlLength_norm_mean_manning" , "none", "gamma", "std"
Ke_range = [0.72] #[19.2, 1.92, 0.96, 0.72, 0.48, 0.192]
output_df = pd.DataFrame(columns=['rainfall_event','Theta0','KSAT', 'discharge_volume','peak_discharge', 'peak_discharge_time', 'infiltration_sum', 'h', 'balance'])

#%%
### 6 RUN (save as tiff)

# configimport os
GDAL_DATA_TYPE = gdal.GDT_Float64 
GEOTIFF_DRIVER_NAME = r'GTiff'
SPATIAL_REFERENCE_SYSTEM_WKID = 31370
bottom_left = (201810, 159320)
upper_left = (201810, 162270)
cell_resolution = 20
NO_DATA = -9999


# functions to write arrays as raster
def create_raster(output_path,
                  columns,
                  rows,
                  nband = 1,
                  gdal_data_type = GDAL_DATA_TYPE,
                  driver = GEOTIFF_DRIVER_NAME):
    ''' returns gdal data source raster object

    '''
    # create driver
    driver = gdal.GetDriverByName(driver)

    output_raster = driver.Create(output_path,
                                  int(columns),
                                  int(rows),
                                  nband,
                                  eType = gdal_data_type)    
    return output_raster

os.environ['PROJ_LIB'] = 'C:\\Users\\u0117123\\Anaconda3\\envs\\landlab_dev\\Library\\share\\proj'
os.environ['GDAL_DATA'] = 'C:\\Users\\u0117123\\Anaconda3\\envs\\landlab_dev\\Library\\share'
def numpy_array_to_raster(output_path,
                          numpy_array,
                          upper_left_tuple,
                          cell_resolution,
                          nband = 1,
                          no_data = NO_DATA,
                          gdal_data_type = GDAL_DATA_TYPE,
                          spatial_reference_system_wkid = SPATIAL_REFERENCE_SYSTEM_WKID,
                          driver = GEOTIFF_DRIVER_NAME):
    ''' returns a gdal raster data source

    keyword arguments:

    output_path -- full path to the raster to be written to disk
    numpy_array -- numpy array containing data to write to raster
    upper_left_tuple -- the upper left point of the numpy array (should be a tuple structured as (x, y))
    cell_resolution -- the cell resolution of the output raster
    nband -- the band to write to in the output raster
    no_data -- value in numpy array that should be treated as no data
    gdal_data_type -- gdal data type of raster (see gdal documentation for list of values)
    spatial_reference_system_wkid -- well known id (wkid) of the spatial reference of the data
    driver -- string value of the gdal driver to use

    '''

    #print ('UL: (%s, %s)' % (upper_left_tuple[0], upper_left_tuple[1]))

    rows, columns = numpy_array.shape
    #print ('ROWS: %s\n COLUMNS: %s\n' % (rows, columns))

    # create output raster
    output_raster = create_raster(output_path,
                                  int(columns),
                                  int(rows),
                                  nband,
                                  gdal_data_type) 
    geotransform = (upper_left_tuple[0],
                    cell_resolution,
                    0,
                    upper_left_tuple[1] + cell_resolution,
                    0,
                    cell_resolution)
                   

    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(spatial_reference_system_wkid)
    output_raster.SetProjection(spatial_reference.ExportToWkt())
    output_raster.SetGeoTransform(geotransform)
    output_band = output_raster.GetRasterBand(1)
    output_band.SetNoDataValue(no_data)
    output_band.WriteArray(numpy_array)          
    output_band.FlushCache()
    output_band.ComputeStatistics(False)

    if os.path.exists(output_path) == False:
        raise Exception('Failed to create raster: %s' % output_path)

    return  output_raster
#
run = 1

for rainfall_event in rainfall_events:
    for Ke_value in Ke_range:
        print('run ' + str(run))
        ######## STEP 1 - Determining inputs
        
        if rainfall_event == "20060614":
            year = 2006
            Theta0_values = [0.169798]
            elapsed_time = 0.0
            model_run_time = 65000
            storm_duration = 9660 #single storm event
            rainfall_intensity_df = pd.read_csv('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\RainfallEvent\\Evrard2008\\20060614.csv', sep = ";", header=0)
        if rainfall_event == "20060821":
            year = 2006
            Theta0_values = [0.398803]
            elapsed_time = 0.0
            model_run_time = 100000
            storm_duration = 17520 #single storm event
            rainfall_intensity_df = pd.read_csv('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\RainfallEvent\\Evrard2008\\20060821.csv', sep = ";", header=0)
        if rainfall_event == "20070809":
            year = 2007
            Theta0_values = [0.305709]
            elapsed_time = 0.0
            model_run_time = 130000
            storm_duration = 76980 #single storm event
            rainfall_intensity_df = pd.read_csv('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\RainfallEvent\\Evrard2008\\20070809.csv', sep = ";", header=0)
        if rainfall_event == "20070821":
            year = 2007
            Theta0_values = [0.371615]
            elapsed_time = 0.0
            model_run_time = 100000
            storm_duration = 70260 #single storm event
            rainfall_intensity_df = pd.read_csv('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\RainfallEvent\\Evrard2008\\20070821.csv', sep = ";", header=0)
        if rainfall_event == "20190610":
            year = 2019
            Theta0_values = [0.224417]
            elapsed_time = 0.0
            model_run_time = 70000
            storm_duration = 7200 #single storm event
            rainfall_intensity_df = pd.read_csv('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\RainfallEvent\\Velm_June_2019_2hours.csv', sep = ";", header=0)

        if rainfall_event == "20070618":
            year = 2007
            Theta0_values = [0.296993012]
            elapsed_time = 0.0
            model_run_time = 50000
            storm_duration = 6660 #single storm event
            rainfall_intensity_df = pd.read_csv('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\RainfallEvent\\Extra\\20070618.csv', sep = ";", header=0)
        if rainfall_event == "20190619":
            year = 2019
            Theta0_values = [0.317663646]
            elapsed_time = 0.0
            model_run_time = 45000
            storm_duration = 22020 #single storm event
            rainfall_intensity_df = pd.read_csv('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\RainfallEvent\\Extra\\20190619.csv', sep = ";", header=0)
        if rainfall_event == "20190727":
            year = 2019
            Theta0_values = [0.12719103]
            elapsed_time = 0.0
            model_run_time = 160000
            storm_duration = 88020 #single storm event
            rainfall_intensity_df = pd.read_csv('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\RainfallEvent\\Extra\\20190727.csv', sep = ";", header=0)
        if rainfall_event == "20190817":
            year = 2019
            Theta0_values = [0.261501464]
            elapsed_time = 0.0
            model_run_time = 110000
            storm_duration = 56820 #single storm event
            rainfall_intensity_df = pd.read_csv('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\RainfallEvent\\Extra\\20190817.csv', sep = ";", header=0)

        if rainfall_event == "20160530":
            year = 2016
            Theta0_values = [0.343115666]
            elapsed_time = 0.0
            model_run_time = 150000
            storm_duration = 72960 #single storm event
            rainfall_intensity_df = pd.read_csv('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\RainfallEvent\\Extra\\20160530.csv', sep = ";", header=0)
        if rainfall_event == "20160601":
            year = 2016
            Theta0_values = [0.411745538]
            elapsed_time = 0.0
            model_run_time = 130000
            storm_duration = 79140 #single storm event
            rainfall_intensity_df = pd.read_csv('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\RainfallEvent\\Extra\\20160601.csv', sep = ";", header=0)
        if rainfall_event == "20160623":
            year = 2016
            Theta0_values = [0.378715824]
            elapsed_time = 0.0
            model_run_time = 80000
            storm_duration = 27240 #single storm event
            rainfall_intensity_df = pd.read_csv('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\RainfallEvent\\Extra\\20160623.csv', sep = ";", header=0)        


        if rainfall_event == "20130507":
            year = 2013
            Theta0_values = [0.174247985283]
            elapsed_time = 0.0
            model_run_time = 185000
            storm_duration = 138600 #single storm event
            rainfall_intensity_df = pd.read_csv('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\RainfallEvent\\Extra\\20130507.csv', sep = ";", header=0)        
        if rainfall_event == "20130522":
            year = 2013
            Theta0_values = [0.402523974439]
            elapsed_time = 0.0
            model_run_time = 125000
            storm_duration = 76440 #single storm event
            rainfall_intensity_df = pd.read_csv('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\RainfallEvent\\Extra\\20130522.csv', sep = ";", header=0)        
        if rainfall_event == "20130619":
            year = 2013
            Theta0_values = [0.184634128]
            elapsed_time = 0.0
            model_run_time = 12000
            storm_duration = 9300 #single storm event
            rainfall_intensity_df = pd.read_csv('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\RainfallEvent\\Extra\\20130619.csv', sep = ";", header=0)        
        if rainfall_event == "20130818":
            year = 2013
            Theta0_values = [0.138218234]
            elapsed_time = 0.0
            model_run_time = 125000
            storm_duration = 100020 #single storm event
            rainfall_intensity_df = pd.read_csv('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\RainfallEvent\\Extra\\20130818.csv', sep = ";", header=0)        
        if rainfall_event == "20130727":
            year = 2013
            Theta0_values = [0.112175136]
            elapsed_time = 0.0
            model_run_time = 75000
            storm_duration = 55860 #single storm event
            rainfall_intensity_df = pd.read_csv('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\RainfallEvent\\Extra\\20130727.csv', sep = ";", header=0)        
        if rainfall_event == "20070525":
            year = 2007
            Theta0_values = [0.275377839]
            elapsed_time = 0.0
            model_run_time = 22000
            storm_duration = 8700 #single storm event
            rainfall_intensity_df = pd.read_csv('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\RainfallEvent\\Extra\\20070525.csv', sep = ";", header=0)        
        if rainfall_event == "20070903":
            year = 2007
            Theta0_values = [0.27004686]
            elapsed_time = 0.0
            model_run_time = 66000
            storm_duration = 25080 #single storm event
            rainfall_intensity_df = pd.read_csv('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\RainfallEvent\\Extra\\20070903.csv', sep = ";", header=0)        
                                                             
        print("run: " + str(run) + 
              "; rainfall event: " + str(rainfall_event) + 
              ", theta_0: " + str(Theta0_values[0]) +
              "; Ke: " + str(Ke_value)   )                                                           
        #create output folder
        main = "D:\\1_phd\\WP3\\Landlab_output\\OutletAtFlume_8_9\\adapted_h0_adapted_manning_adaptedRain\\20m\\"

        output_folder = os.path.join(main + scaling + "\\",
                                     "20m_" + 
                                     str(Ke_value) +
                                     "scaling_" + scaling +
                                     #"_Theta0" + str(Theta0_values[0]) +
                                     "_Ke" + str(Ke_value) +
                                     #DTM_type + "_" + DTM_slope + 
                                     "_" + 
                                     str(rainfall_event) + "_" + 
                                     #+ "_outlet_id" + str(outlet_id) + "_Flume89_h0_adapt_" +
                                     datetime.datetime.now().strftime('%Y%m%d_%Hh%M')
                                     )
        os.makedirs(output_folder)
        
        ###### STEP 2 -Defining the model domain
        #Reading ESRI grid DEM
        if DTM_slope =="velm":
            if DTM_type == "heulengracht_burn":
                watershed_DEM = "C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\DTM\\Heulengracht\\20m\\heulengracht_20m_burnin_flume.asc"
        (rmg, z) = read_esri_ascii(watershed_DEM, name="topographic__elevation")
        rmg.set_watershed_boundary_condition_outlet_id(10171, z, nodata_value=-9999.0) # outlet = node 10171
     
        ######## STEP 3 - Pre-processing the DEM - fill sinks
        if fill_sinks == "fill":
            sfb = SinkFillerBarnes(rmg, method='Steepest', fill_flat=False)
            sfb.run_one_step()
        
        ######## STEP 4 - Setting boundary conditions
        rmg.at_node['surface_water__depth'] = np.ones(14994)* 1 * (10**-8) 
        h0 = sum(rmg.at_node['surface_water__depth'])
        
        # ######## STEP 5 - vle configuration#
        ## Not applicable 

        ######## STEP 6 - Define mannings roughness coefficient, hydraulic conductivity, wetting front capillary pressyre head and soil infiltration at t=0
        ### mannings roughtness coefficient
        if year == 2006:
            manning = rasterio.open('D:\\1_phd\\WP3\\Heulengracht\\landcover\\manning_rasters\\manning_2006_20m.tif')
        if year == 2007:
            manning = rasterio.open('D:\\1_phd\\WP3\\Heulengracht\\landcover\\manning_rasters\\manning_2007_20m.tif')
        if year == 2013:
            manning = rasterio.open('D:\\1_phd\\WP3\\Heulengracht\\landcover\\manning_rasters\\manning_2013_20m.tif')
        if year == 2016:
            manning = rasterio.open('D:\\1_phd\\WP3\\Heulengracht\\landcover\\manning_rasters\\manning_2016_20m.tif')
        if year == 2019:
            manning = rasterio.open('D:\\1_phd\\WP3\\Heulengracht\\landcover\\manning_rasters\\manning_2019_vLE_20m.tif')            
        print(manning)
        config = manning
        manning = manning.read()
        manning = np.flip(manning, 1)

        fllength_scaling = rasterio.open('D:\\1_phd\\WP3\\Heulengracht\\Ks_scaling_flowlength\\flowlenght_scaling_20m.tif')
        fllength_scaling = fllength_scaling.read()
        fllength_scaling[fllength_scaling<0] = 1
        fllength_scaling = np.flip(fllength_scaling, 1)
        if scaling == "FlLength_manning":
            manning = manning * fllength_scaling
        manning = manning.flatten()
        manning = manning.astype(np.float64)            
        rmg.at_node['mannings_n__coefficients'] = manning
        ### saturated hydraulic conductivity 
        rows, columns = config.shape
        tau_gamma = rasterio.open('D:\\1_phd\\WP3\\Heulengracht\\Fang_et_al\\gamma_20m_con.tif')
        tau_gamma = tau_gamma.read()
        tau_gamma[tau_gamma<0] = 1
        tau_gamma = np.flip(tau_gamma, 1)
        tau_std = rasterio.open('D:\\1_phd\\WP3\\Heulengracht\\Fang_et_al\\stdev_20m_con.tif')
        tau_std = tau_std.read()
        tau_std[tau_std<0] = 1
        tau_std = np.flip(tau_std, 1)
        fllength_scaling = rasterio.open('D:\\1_phd\\WP3\\Heulengracht\\Ks_scaling_flowlength\\flowlenght_scaling_20m.tif')
        fllength_scaling = fllength_scaling.read()
        fllength_scaling[fllength_scaling<0] = 1
        fllength_scaling = np.flip(fllength_scaling, 1)

        if scaling == "none":
            hydraulic_conductivity_rmg = np.ones((rows, columns))*Ke_value/3600000 #Van den Putte et al. (2013); average value summer
            hydraulic_conductivity_rmg = hydraulic_conductivity_rmg.flatten()
        if scaling == "gamma":
            hydraulic_conductivity_rmg = (tau_gamma *Ke_value)/3600000 #Van den Putte et al. (2013); average value summer
            hydraulic_conductivity_rmg = hydraulic_conductivity_rmg.flatten()
        if scaling == "std":
            hydraulic_conductivity_rmg = (tau_std *Ke_value)/3600000 #Van den Putte et al. (2013); average value summer
            hydraulic_conductivity_rmg = hydraulic_conductivity_rmg.flatten()
        if scaling == "FlLength_Ks":
            hydraulic_conductivity_rmg = (fllength_scaling *Ke_value)/3600000 #Van den Putte et al. (2013); average value summer
            hydraulic_conductivity_rmg = hydraulic_conductivity_rmg.flatten()

        if scaling == "FlLength_manning":
            hydraulic_conductivity_rmg = np.ones((rows, columns))*Ke_value/3600000 #Van den Putte et al. (2013); average value summer
            hydraulic_conductivity_rmg = hydraulic_conductivity_rmg.flatten()            
        #no infiltration in culverts
        #indices = [58458, 58459, 49705, 46855, 46652, 40137, 29760, 28540, 28539]
        indices = [10171, 7507, 7200]
        for index_id in indices:
            hydraulic_conductivity_rmg[index_id] = 0
        ### wetting front capillary pressure head
        wetting_front_capillary_pressure_head_rmg = rmg.ones('node')*psi_value # [m]
        ### soil infiltration at t=0
        d = rmg.add_ones("soil_water_infiltration__depth", at="node", dtype=float)
        d *= 0.00000001
        
        ######## STEP 7 - Initializing the OverlandFlow and SoilInfiltration component
        of = OverlandFlow(rmg,
                          steep_slopes=True, 
                          theta=0.9,
                          h_init = 10**-10,
                          culvert_links=[20068, 19967, #culvert 7
                                         14676, 14677,14778, #culvert 8
                                         14168#culvert 9
                                         ]
                          )
        SI = SoilInfiltrationGreenAmpt(rmg, 
                                       hydraulic_conductivity = hydraulic_conductivity_rmg,
                                       wetting_front_capillary_pressure_head = wetting_front_capillary_pressure_head_rmg, 
                                       soil_type='loam',
                                       soil_bulk_density =1230,
                                       initial_soil_moisture_content = Theta0_values[0])
   

        ######## STEP 8 - RUN (runoff & infiltration)
        ### dataframes for output
        dataframe_int = pd.DataFrame(columns=['time', 'timestep', 'discharge_sum', 'depth_sum', 'max_h', 
                                              'infiltration_depth', 'actual_infiltration','h_beforeInf_sum', 
                                              'h_afterInf_sum', 'Boundary_discharge_sum'])
        dataframe_c10171= pd.DataFrame(columns=['time', 'timestep', 'depth', 'depth_before_inf', 
                                              'depth_after_inf', 'infiltration_depth', 'actual_infiltration',
                                              'h_beforeInf_10171', 'h_afterInf_10171', 'discharge_10171', 'surface_water_discharge_10171', 'rainfall_intensity'])
           
        inf_t0 = sum(rmg.at_node['soil_water_infiltration__depth'])
        inf_t0_c10171 = (rmg.at_node['soil_water_infiltration__depth'])[10171]
        hydrograph_time = []
        discharge_at_outlet = []
        
        a=0
        time_step_raster = 300
        while elapsed_time < model_run_time:
            of.dt = of.calc_time_step()
            timestep = of.dt
            if elapsed_time < (storm_duration):
                if rainfall_intensity_df['time'].iloc[a+1]>elapsed_time:
                    if rainfall_intensity_df['time'].iloc[a+1]>=elapsed_time + timestep:
                        rainfall_mmhr = rainfall_intensity_df['P_int'].iloc[a]
                        of.rainfall_intensity = rainfall_mmhr * (2.777778 * 10**-7) #[m/s]
                    else:
                        rainfall_mmhr = (((elapsed_time + timestep - rainfall_intensity_df['time'].iloc[a+1])/timestep) 
                                          * rainfall_intensity_df['P_int'].iloc[a+1]) + ((1-((elapsed_time + timestep - rainfall_intensity_df['time'].iloc[a+1])/timestep))*rainfall_intensity_df['P_int'].iloc[a])
                        of.rainfall_intensity = rainfall_mmhr * (2.777778 * 10**-7) #[m/s]
                else:
                    a=a+1
                    if rainfall_intensity_df['time'].iloc[a+1]>=elapsed_time + timestep:
                        rainfall_mmhr = rainfall_intensity_df['P_int'].iloc[a]
                        of.rainfall_intensity = rainfall_mmhr * (2.777778 * 10**-7) #[m/s]
                    else:
                        rainfall_mmhr = (((elapsed_time + timestep - rainfall_intensity_df['time'].iloc[a+1])/timestep) 
                                          * rainfall_intensity_df['P_int'].iloc[a+1]) + ((1-((elapsed_time + timestep - rainfall_intensity_df['time'].iloc[a+1])/timestep))*rainfall_intensity_df['P_int'].iloc[a])
                        of.rainfall_intensity = rainfall_mmhr * (2.777778 * 10**-7) #[m/s]
            else: 
                rainfall_mmhr = 0.0
                of.rainfall_intensity = 0.0

            # model overland flow
            of.overland_flow(
                culvert_links=[20068, 19967, #culvert 7
                               14676, 14677,14778, #culvert 8
                               14168#culvert 9
                               ]
                )

            discharge = of._q
            discharge_10171 = rmg.calc_flux_div_at_node(of._q)[10171]
            depth_before_inf = rmg.at_node['surface_water__depth']
            rmg.at_node['surface_water__discharge'] = of.discharge_mapper(of._q, convert_to_volume=True)
            surface_water_discharge_10171 = rmg.at_node['surface_water__discharge'][10171]
            # model infiltration
            h_beforeInf_sum = sum(rmg.at_node['surface_water__depth'])
            h_beforeInf_10171 = rmg.at_node['surface_water__depth'][10171]
            SI.run_one_step(of.dt)
            h_afterInf_sum = sum(rmg.at_node['surface_water__depth'])
            h_afterInf_10171 = rmg.at_node['surface_water__depth'][10171]
            depth_after_inf = rmg.at_node['surface_water__depth']
    
            # write output for time step
            dataframe_int = dataframe_int.append(pd.DataFrame({'time': elapsed_time,
                                                               'timestep': of.dt,
                                                               'discharge_sum':sum(rmg.at_node['surface_water__discharge']),
                                                               'depth_sum': sum(rmg.at_node['surface_water__depth']),
                                                               'max_h': np.amax(rmg.at_node["surface_water__depth"]),
                                                               'infiltration_depth':sum(rmg.at_node['soil_water_infiltration__depth']),
                                                               'actual_infiltration':sum(rmg.at_node['soil_water_infiltration__depth']) - inf_t0, 
                                                               'h_beforeInf_sum': h_beforeInf_sum,
                                                               'h_afterInf_sum': h_afterInf_sum},
                                                              index=[0]), ignore_index=True)
    
    
            dataframe_c10171 = dataframe_c10171.append(pd.DataFrame({'time': elapsed_time,
                                                                 'timestep': of.dt,
                                                                 'depth': (rmg.at_node['surface_water__depth'])[10171],
                                                                 'depth_before_inf': depth_before_inf[10171], 
                                                                 'depth_after_inf': depth_after_inf[10171],
                                                                 'infiltration_depth':(rmg.at_node['soil_water_infiltration__depth'])[10171],
                                                                 'actual_infiltration':(rmg.at_node['soil_water_infiltration__depth'])[10171] - inf_t0_c10171,
                                                                 'h_beforeInf_10171': h_beforeInf_10171,
                                                                 'h_afterInf_10171': h_afterInf_10171,
                                                                 'discharge_10171': discharge_10171,
                                                                 'surface_water_discharge_10171':surface_water_discharge_10171,
                                                                 'rainfall_intensity': of.rainfall_intensity}, 
                                                                index=[0]), ignore_index=True)
            #check water balance
            (unique, counts) = np.unique(rmg.status_at_node, return_counts=True)
            #water out
            dataframe_int['Q_volume'] = dataframe_int['timestep'] * dataframe_c10171['surface_water_discharge_10171']
            dataframe_int['cumm_Q_volume'] = dataframe_int['Q_volume'].cumsum()
            dataframe_int['h_volume'] = dataframe_int['depth_sum']*(rmg.dx**2)
            dataframe_int['cumm_I_vol'] = dataframe_int['infiltration_depth']*(rmg.dx**2)
            #water in
            dataframe_int['P_volume'] = dataframe_int['timestep'] * dataframe_c10171['rainfall_intensity'] * counts[0] * (rmg.dx**2)
            dataframe_int['cumm_P_volume'] = dataframe_int['P_volume'].cumsum()
            dataframe_int['h0_vol'] = len(rmg.status_at_node) * (rmg.dx**2) * (10**-8) 
            dataframe_int['I0_vol'] = len(rmg.status_at_node) * (rmg.dx**2) * (10**-8) 
            dataframe_int['balance'] = dataframe_int['cumm_P_volume'] + dataframe_int['h0_vol'] + dataframe_int['I0_vol'] - dataframe_int['cumm_Q_volume'] - dataframe_int['h_volume'] - dataframe_int['cumm_I_vol']
          

            hydrograph_time.append(elapsed_time / 3600.) # convert seconds to hours
            discharge_at_outlet.append(np.abs(of._q[10171]) * rmg.dx) # append discharge in m^3/s
            
            #add infiltrated water column to total infiltrated water column
            inf_t0 = sum(rmg.at_node['soil_water_infiltration__depth'])
    
            inf_t0_c10171 = (rmg.at_node['soil_water_infiltration__depth'])[10171]
            #create output for output rasters 
            if elapsed_time <= time_step_raster:
                h_t = (rmg.at_node['surface_water__depth']).copy()
                inf_t = (rmg.at_node['soil_water_infiltration__depth']).copy()
                q_t = (rmg.at_node['surface_water__discharge']).copy()
            else:
                # Run to save rasters as tif
                #save water depth on surface as tiff
                output_folder_h = os.path.join(output_folder,"h")
                if not os.path.exists(output_folder_h):
                        os.makedirs(output_folder_h)
    
                h_reshape = h_t.reshape((147,102))
                numpy_array_to_raster(output_folder_h + '\\h_' + str(int(time_step_raster/60)) + '.tif',
                                      h_reshape,
                                      bottom_left,
                                      cell_resolution,
                                      nband = 1,
                                      no_data = NO_DATA,gdal_data_type = GDAL_DATA_TYPE,
                                      spatial_reference_system_wkid = SPATIAL_REFERENCE_SYSTEM_WKID,
                                      driver = GEOTIFF_DRIVER_NAME)
    
    
                #save infiltration as tiff
                output_folder_inf = os.path.join(output_folder,"inf")
                if not os.path.exists(output_folder_inf):
                        os.makedirs(output_folder_inf)
    
                inf_reshape = inf_t.reshape((147,102))
                numpy_array_to_raster(output_folder_inf  + '\\inf_' + str(int(time_step_raster/60)) + '.tif',
                                      inf_reshape,
                                      bottom_left,
                                      cell_resolution,
                                      nband = 1,
                                      no_data = NO_DATA,gdal_data_type = GDAL_DATA_TYPE,
                                      spatial_reference_system_wkid = SPATIAL_REFERENCE_SYSTEM_WKID,
                                      driver = GEOTIFF_DRIVER_NAME)
    
    
                #save discharge as tiff
                output_folder_q = os.path.join(output_folder,"q")
                if not os.path.exists(output_folder_q):
                        os.makedirs(output_folder_q)
    
                q_reshape = q_t.reshape((147,102))
                numpy_array_to_raster(output_folder_q  + '\\q_' + str(int(time_step_raster/60)) + '.tif',
                                      q_reshape,
                                      bottom_left,
                                      cell_resolution,
                                      nband = 1,
                                      no_data = NO_DATA,gdal_data_type = GDAL_DATA_TYPE,
                                      spatial_reference_system_wkid = SPATIAL_REFERENCE_SYSTEM_WKID,
                                      driver = GEOTIFF_DRIVER_NAME)
    
                time_step_raster = time_step_raster + 300
                h = (rmg.at_node['surface_water__depth']).copy()
                inf = (rmg.at_node['soil_water_infiltration__depth']).copy()
                q = (rmg.at_node['surface_water__discharge']).copy()
                
                
        
            elapsed_time += of.dt
            
        #print("stop")
        run = run + 1
      
        def save_xls(list_dfs, list_dfs_names, xls_path):
            with pd.ExcelWriter(xls_path) as writer:
                for n, df in enumerate(list_dfs):
                    df.to_excel(writer,list_dfs_names[n])
                writer.save()
    
    
        list_dfs = [dataframe_int, dataframe_c10171]
        list_dfs_names = ["int","c10171"]
        xls_path = output_folder + "\\dataframes_" + str(rainfall_event) + ".xlsx"
    
        save_xls(list_dfs, list_dfs_names, xls_path)
        
        
        #discharge graph
        ax = dataframe_c10171.plot(x='time', y='surface_water_discharge_10171', style='.', ms=1)
        ax.set_ylabel("discharge (m³/sec)")
        ax.get_figure().savefig(output_folder + '\\discharge.png')
        
        
        #Output file
        #discharge volume (m³) sum
        dataframe_c10171['discharge_volume'] = dataframe_c10171['timestep'] * dataframe_c10171['surface_water_discharge_10171']
        sum_discharge_10171  = dataframe_c10171['discharge_volume'].sum()
        
        #peak discharge (m³/sec)
    
        peak_discharge10171 = dataframe_c10171['surface_water_discharge_10171'].max()
                            
        #time of peak discharge
        peak_discharge_df10171 = dataframe_c10171[dataframe_c10171['surface_water_discharge_10171']==dataframe_c10171['surface_water_discharge_10171'].max()]
        peak_discharge_time10171 = peak_discharge_df10171['time'].iloc[0]
        #infiltration sum (m³)
        sum_infiltration = dataframe_int['actual_infiltration'].sum()
    
    
        output_df = output_df.append(pd.DataFrame({
            'rainfall_event': rainfall_event,
            'Theta0': Theta0_values[0],
            'KSAT': Ke_value,
            'discharge_volume': dataframe_int['cumm_Q_volume'][len(dataframe_int)-1],
            'peak_discharge': peak_discharge10171, 
            'peak_discharge_time': peak_discharge_time10171, 
            'infiltration_sum': dataframe_int['cumm_I_vol'][len(dataframe_int)-1],
            'h': dataframe_int['h_volume'][len(dataframe_int)-1], 
            'balance': dataframe_int['balance'][len(dataframe_int)-1],
            'P_Vol': dataframe_int['cumm_P_volume'][len(dataframe_int)-1]},
            index=[0]), ignore_index=True)
 
  
    output_df.to_csv(main 
                     + "\\output_df_velm" 
                     + "_Resolution20m_" 
                     + scaling + "_" 
                     + rainfall_event + "_"  
                     + datetime.datetime.now().strftime('%Y%m%d_%Hh%M') 
                     + ".csv" )
output_df.to_csv(main 
                 + "\\output_df_velm" 
                 + "_Resolution20m_" + scaling  
                 + datetime.datetime.now().strftime('%Y%m%d_%Hh%M') 
                 + ".csv" )
