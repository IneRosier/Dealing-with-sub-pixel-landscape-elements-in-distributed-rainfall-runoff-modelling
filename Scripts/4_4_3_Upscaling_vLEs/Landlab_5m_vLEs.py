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
#vle_type = "parallel_250m" #no_vle  #hedge  #grassedbuffer  #grassedbuffer_wide  #grassedbuffer_wider #rect_hedge_100 #rect_hedge_50_1 #rect_hedge_50_6 #rect_hedge_50_10
DTM_slope = "velm" #flat or steep or flat_scaled
DTM_type = "heulengracht_burn" #plane or mpd_plane
fill_sinks = "fill" #fill or no_fill
psi_value  = 0.1727
rainfall_events = ["20060614", "20060821","20070618", "20070809", "20070821", 
                   "20070903", "20130507", "20130522", "20130818", "20160530",
                   "20160601", "20160623", "20190610",  "20190727", "20190817"]

outlet_id = 9715
scaling = "none" #none, gamma, std
Ke_range = [0.72]
Ke_vLE = 102.4
manning_vLE = 0.43
output_df = pd.DataFrame(columns=['rainfall_event','Theta0','KSAT', 'discharge_volume','peak_discharge', 'peak_discharge_time', 'infiltration_sum', 'h', 'balance'])
vLE = ['no_vLE', 'FA305', 'FA400', 'FA500', 'FA600', 'FA700', 'FA800'] 


#%%
### 6 RUN (save as tiff)

# configimport os
GDAL_DATA_TYPE = gdal.GDT_Float64 
GEOTIFF_DRIVER_NAME = r'GTiff'
SPATIAL_REFERENCE_SYSTEM_WKID = 31370
bottom_left = (201810, 159550)
upper_left = (201810, 162270)
cell_resolution = 5
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

os.environ['PROJ_LIB'] = 'C:\\Users\\IneR\\anaconda3\\envs\\landlab_dev\\Library\\share\\proj'
os.environ['GDAL_DATA'] = 'C:\\Users\\IneR\\anaconda3\\envs\\landlab_dev\\Library\\share'
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
        for vLE_type in vLE:
            print('run ' + str(run))
            ######## STEP 1 - Determining inputs
            
            if rainfall_event == "20060614":
                year = 2006
                Theta0_values = [0.169798]
                elapsed_time = 0.0
                model_run_time = 65000
                storm_duration = 9660 #single storm event
                rainfall_intensity_df = pd.read_csv('I:\\WP3\\Precipitation\\20060614.csv', sep = ";", header=0)
            if rainfall_event == "20060821":
                year = 2006
                Theta0_values = [0.398803]
                elapsed_time = 0.0
                model_run_time = 100000
                storm_duration = 17520 #single storm event
                rainfall_intensity_df = pd.read_csv('I:\\WP3\\Precipitation\\20060821.csv', sep = ";", header=0)
            if rainfall_event == "20070809":
                year = 2007
                Theta0_values = [0.305709]
                elapsed_time = 0.0
                model_run_time = 130000
                storm_duration = 76980 #single storm event
                rainfall_intensity_df = pd.read_csv('I:\\WP3\\Precipitation\\20070809.csv', sep = ";", header=0)
            if rainfall_event == "20070821":
                year = 2007
                Theta0_values = [0.371615]
                elapsed_time = 0.0
                model_run_time = 100000
                storm_duration = 70260 #single storm event
                rainfall_intensity_df = pd.read_csv('I:\\WP3\\Precipitation\\20070821.csv', sep = ";", header=0)
            if rainfall_event == "20190610":
                year = 2019
                Theta0_values = [0.224417]
                elapsed_time = 0.0
                model_run_time = 70000
                storm_duration = 7200 #single storm event
                rainfall_intensity_df = pd.read_csv('I:\\WP3\\Precipitation\\Velm_June_2019_2hours.csv', sep = ";", header=0)
            if rainfall_event == "20070618":
                year = 2007
                Theta0_values = [0.296993012]
                elapsed_time = 0.0
                model_run_time = 50000
                storm_duration = 6660 #single storm event
                rainfall_intensity_df = pd.read_csv('I:\\WP3\\Precipitation\\20070618.csv', sep = ";", header=0)
            if rainfall_event == "20190619":
                year = 2019
                Theta0_values = [0.317663646]
                elapsed_time = 0.0
                model_run_time = 45000
                storm_duration = 22020 #single storm event
                rainfall_intensity_df = pd.read_csv('I:\\WP3\\Precipitation\\20190619.csv', sep = ";", header=0)
            if rainfall_event == "20190727":
                year = 2019
                Theta0_values = [0.12719103]
                elapsed_time = 0.0
                model_run_time = 160000
                storm_duration = 88020 #single storm event
                rainfall_intensity_df = pd.read_csv('I:\\WP3\\Precipitation\\20190727.csv', sep = ";", header=0)
            if rainfall_event == "20190817":
                year = 2019
                Theta0_values = [0.261501464]
                elapsed_time = 0.0
                model_run_time = 110000
                storm_duration = 56820 #single storm event
                rainfall_intensity_df = pd.read_csv('I:\\WP3\\Precipitation\\20190817.csv', sep = ";", header=0)
    
            if rainfall_event == "20160530":
                year = 2016
                Theta0_values = [0.343115666]
                elapsed_time = 0.0
                model_run_time = 150000
                storm_duration = 72960 #single storm event
                rainfall_intensity_df = pd.read_csv('I:\\WP3\\Precipitation\\20160530.csv', sep = ";", header=0)
            if rainfall_event == "20160601":
                year = 2016
                Theta0_values = [0.411745538]
                elapsed_time = 0.0
                model_run_time = 130000
                storm_duration = 79140 #single storm event
                rainfall_intensity_df = pd.read_csv('I:\\WP3\\Precipitation\\20160601.csv', sep = ";", header=0)
            if rainfall_event == "20160623":
                year = 2016
                Theta0_values = [0.378715824]
                elapsed_time = 0.0
                model_run_time = 80000
                storm_duration = 27240 #single storm event
                rainfall_intensity_df = pd.read_csv('I:\\WP3\\Precipitation\\20160623.csv', sep = ";", header=0)        

            if rainfall_event == "20130507":
                year = 2013
                Theta0_values = [0.174247985283]
                elapsed_time = 0.0
                model_run_time = 185000
                storm_duration = 138600 #single storm event
                rainfall_intensity_df = pd.read_csv('I:\\WP3\\Precipitation\\20130507.csv', sep = ";", header=0)        
            if rainfall_event == "20130522":
                year = 2013
                Theta0_values = [0.402523974439]
                elapsed_time = 0.0
                model_run_time = 125000
                storm_duration = 76440 #single storm event
                rainfall_intensity_df = pd.read_csv('I:\\WP3\\Precipitation\\20130522.csv', sep = ";", header=0)        
            # if rainfall_event == "20130619":
            #     year = 2013
            #     Theta0_values = [0.184634128]
            #     elapsed_time = 0.0
            #     model_run_time = 12000
            #     storm_duration = 9300 #single storm event
            #     rainfall_intensity_df = pd.read_csv('I:\\WP3\\Precipitation\\20130619.csv', sep = ";", header=0)        
            if rainfall_event == "20130818":
                year = 2013
                Theta0_values = [0.138218234]
                elapsed_time = 0.0
                model_run_time = 125000
                storm_duration = 100020 #single storm event
                rainfall_intensity_df = pd.read_csv('I:\\WP3\\Precipitation\\20130818.csv', sep = ";", header=0)        
            # if rainfall_event == "20130727":
            #     year = 2013
            #     Theta0_values = [0.112175136]
            #     elapsed_time = 0.0
            #     model_run_time = 75000
            #     storm_duration = 55860 #single storm event
            #     rainfall_intensity_df = pd.read_csv('I:\\WP3\\Precipitation\\20130727.csv', sep = ";", header=0)        
            # if rainfall_event == "20070525":
            #     year = 2007
            #     Theta0_values = [0.275377839]
            #     elapsed_time = 0.0
            #     model_run_time = 22000
            #     storm_duration = 8700 #single storm event
            #     rainfall_intensity_df = pd.read_csv('I:\\WP3\\Precipitation\\20070525.csv', sep = ";", header=0)        
            if rainfall_event == "20070903":
                year = 2007
                Theta0_values = [0.27004686]
                elapsed_time = 0.0
                model_run_time = 66000
                storm_duration = 25080 #single storm event
                rainfall_intensity_df = pd.read_csv('I:\\WP3\\Precipitation\\20070903.csv', sep = ";", header=0)        
                              



            if vLE_type == "FA305":
                vLE_config = rasterio.open('I:\\WP3\Subcatchment1\\Rasters_vLE_landscape\\subc1_5m_FA305.tif')
                vLE_config = vLE_config.read()
                vLE_config = np.flip(vLE_config, 1)
                vLE_config = vLE_config.flatten()
                vLE_config = vLE_config.astype(np.float64)
            if vLE_type == "FA400":
                vLE_config = rasterio.open('I:\\WP3\Subcatchment1\\Rasters_vLE_landscape\\subc1_5m_FA400.tif')
                vLE_config = vLE_config.read()
                vLE_config = np.flip(vLE_config, 1)
                vLE_config = vLE_config.flatten()
                vLE_config = vLE_config.astype(np.float64)
            if vLE_type == "FA500":
                vLE_config = rasterio.open('I:\\WP3\Subcatchment1\\Rasters_vLE_landscape\\subc1_5m_FA500.tif')
                vLE_config = vLE_config.read()
                vLE_config = np.flip(vLE_config, 1)
                vLE_config = vLE_config.flatten()
                vLE_config = vLE_config.astype(np.float64)                
            if vLE_type == "FA600":
                vLE_config = rasterio.open('I:\\WP3\Subcatchment1\\Rasters_vLE_landscape\\subc1_5m_FA600.tif')
                vLE_config = vLE_config.read()
                vLE_config = np.flip(vLE_config, 1)
                vLE_config = vLE_config.flatten()
                vLE_config = vLE_config.astype(np.float64)
            if vLE_type == "FA700":
                vLE_config = rasterio.open('I:\\WP3\Subcatchment1\\Rasters_vLE_landscape\\subc1_5m_FA700.tif')
                vLE_config = vLE_config.read()
                vLE_config = np.flip(vLE_config, 1)
                vLE_config = vLE_config.flatten()
                vLE_config = vLE_config.astype(np.float64)                
            if vLE_type == "FA800":
                vLE_config = rasterio.open('I:\\WP3\Subcatchment1\\Rasters_vLE_landscape\\subc1_5m_FA800.tif')
                vLE_config = vLE_config.read()
                vLE_config = np.flip(vLE_config, 1)
                vLE_config = vLE_config.flatten()
                vLE_config = vLE_config.astype(np.float64)
            if vLE_type == "no_vLE":
                vLE_config = np.zeros(16359)
                
            
            print("run: " + str(run) + 
                  "; rainfall event: " + str(rainfall_event) + 
                  ", theta_0: " + str(Theta0_values[0]) +
                  "; Ke: " + str(Ke_value) +
                  "vLE config:" + str(vLE_type))                                                           
            #create output folder
            main = "I:\\WP3\\Landlab_output\\Subcatchment1\\5m\\"
            output_folder = os.path.join(main,
                                         "5m_subc1_vLEType_" + vLE_type + "_Ke" +
                                         str(Ke_value) +
                                         "scaling_" + scaling +
                                         "_Theta0" + str(Theta0_values[0]) +
                                         DTM_type + "_" + DTM_slope +  
                                         str(rainfall_event) + 
                                         "_outlet_id" + str(outlet_id) + "_Flume89_h0_adapt_" +
                                         datetime.datetime.now().strftime('%Y%m%d_%Hh%M'))
            os.makedirs(output_folder)
            
            ###### STEP 2 -Defining the model domain
            #Reading ESRI grid DEM
            if DTM_slope =="velm":
                if DTM_type == "heulengracht_burn":
                    watershed_DEM = "I:\\WP3\\Subcatchment1\\subc1_5m_dtm_halo.asc"
            (rmg, z) = read_esri_ascii(watershed_DEM, name="topographic__elevation")
            rmg.set_watershed_boundary_condition_outlet_id(9715, z, nodata_value=-9999.0) # outlet = node 9715
         
            ######## STEP 3 - Pre-processing the DEM - fill sinks
            if fill_sinks == "fill":
                sfb = SinkFillerBarnes(rmg, method='Steepest', fill_flat=False)
                sfb.run_one_step()
            
            ######## STEP 4 - Setting boundary conditions
            rmg.at_node['surface_water__depth'] = np.ones(len(z))* 1 * (10**-8) 
            h0 = sum(rmg.at_node['surface_water__depth'])
            
            # ######## STEP 5 - vle configuration
            # if DTM_type == "heulengracht_burn":
            #     config = rasterio.open('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\\WP3\\DTM\\Heulengracht\\10m\\catchment_heulengracht_10m_burnin2.tif')
            #     array_config = config.read()
            #     array_config=np.flip(array_config, 1)
            #     fields=array_config.flatten()
            #     fields = fields.astype(np.float64)
            # if DTM_type == "heulengracht_original":
            #     config = rasterio.open('D:\\1_phd\\WP3\\Heulengracht\\DTM\\10m\\Heulengracht_10m.tif')
            #     array_config = config.read()
            #     array_config=np.flip(array_config, 1)
            #     fields=array_config.flatten()
            #     fields = fields.astype(np.float64)
            
    
            ######## STEP 6 - Define mannings roughness coefficient, hydraulic conductivity, wetting front capillary pressyre head and soil infiltration at t=0
            ### mannings roughtness coefficient
            if year == 2006:
                manning = rasterio.open('I:\\WP3\\Subcatchment1\\landcover\\manning_2006_5m.tif')
            if year == 2007:
                manning = rasterio.open('I:\\WP3\\Subcatchment1\\landcover\\manning_2007_5m.tif')
            if year == 2013:
                manning = rasterio.open('I:\\WP3\\Subcatchment1\\landcover\\manning_2013_5m.tif')
            if year == 2019:
                manning = rasterio.open('I:\\WP3\\Subcatchment1\\landcover\\manning_2019_vLE_5m.tif')
            if year == 2016:
                manning = rasterio.open('I:\\WP3\\Subcatchment1\\landcover\\manning_2016_5m.tif')            
            print(manning)
            config = manning
            manning = manning.read()
            manning = np.flip(manning, 1)
            manning = manning.flatten()
            manning = manning.astype(np.float64)
            vLE_config_inv = np.ones(len(z)) - vLE_config
            manning_vLE =  (vLE_config_inv * manning) + (vLE_config*manning_vLE)
            
            rmg.at_node['mannings_n__coefficients'] = manning_vLE
            ### saturated hydraulic conductivity 
            rows, columns = config.shape
            # tau_gamma = rasterio.open('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\WP3\\Subcatchment1\\fang_et_al\\gamma_con_10m.tif')
            # tau_gamma = tau_gamma.read()
            # tau_gamma[tau_gamma<0] = 1 
            # tau_std = rasterio.open('C:\\Users\\u0117123\\OneDrive - KU Leuven\\FWO\WP3\\Subcatchment1\\fang_et_al\\stdev_con_10m.tif')
            # tau_std = tau_std.read()
            # tau_std[tau_std<0] = 1
            if scaling == "none":
                hydraulic_conductivity_rmg = np.ones((rows, columns))*Ke_value/3600000 #Van den Putte et al. (2013); average value summer
                hydraulic_conductivity_rmg = hydraulic_conductivity_rmg.flatten()
            # if scaling == "gamma":
            #     hydraulic_conductivity_rmg = (tau_gamma *Ke_value)/3600000 #Van den Putte et al. (2013); average value summer
            #     hydraulic_conductivity_rmg = hydraulic_conductivity_rmg.flatten()
            # if scaling == "std":
            #     hydraulic_conductivity_rmg = (tau_std *Ke_value)/3600000 #Van den Putte et al. (2013); average value summer
            #     hydraulic_conductivity_rmg = hydraulic_conductivity_rmg.flatten()   

            vLE_config_inv = np.ones(len(z)) - vLE_config
            hydraulic_conductivity_rmg_vLE =  (vLE_config_inv * hydraulic_conductivity_rmg) + (vLE_config*(Ke_vLE/3600000))            
            #no infiltration IN culverts
            # indices = [40137, 29760, 28540, 28539]
            # for index_id in indices:
            #     hydraulic_conductivity_rmg[index_id] = 0
            ### wetting front capillary pressure head
            wetting_front_capillary_pressure_head_rmg = rmg.ones('node')*psi_value # [m]
            ### soil infiltration at t=0
            d = rmg.add_ones("soil_water_infiltration__depth", at="node", dtype=float)
            d *= 0.00000001
            
            ######## STEP 7 - Initializing the OverlandFlow and SoilInfiltration component
            of = OverlandFlow(rmg,
                              steep_slopes=True, 
                              theta=0.9,
                              h_init = 10**-8,
                              # culvert_links=[80133, 80336, 80335, 80336, #culvert 7
                              #                59454, 59252, 59251, #culvert 8
                              #                56616, 56819, 57022, 56818, 57021, 56820, 56617#culvert 9
                              #                ]
                              )
            SI = SoilInfiltrationGreenAmpt(rmg, 
                                           hydraulic_conductivity = hydraulic_conductivity_rmg_vLE,
                                           wetting_front_capillary_pressure_head = wetting_front_capillary_pressure_head_rmg, 
                                           soil_type='loam',
                                           soil_bulk_density =1230,
                                           initial_soil_moisture_content = Theta0_values[0])
       
    
            ######## STEP 8 - RUN (runoff & infiltration)
            ### dataframes for output
            dataframe_int = pd.DataFrame(columns=['time', 'timestep', 'discharge_sum', 'depth_sum', 'max_h', 
                                                  'infiltration_depth', 'actual_infiltration','h_beforeInf_sum', 
                                                  'h_afterInf_sum', 'Boundary_discharge_sum'])
            dataframe_c9715= pd.DataFrame(columns=['time', 'timestep', 'depth', 'depth_before_inf', 
                                                  'depth_after_inf', 'infiltration_depth', 'actual_infiltration',
                                                  'h_beforeInf_9715', 'h_afterInf_9715', 'discharge_9715', 'surface_water_discharge_9715', 'rainfall_intensity'])
               
            inf_t0 = sum(rmg.at_node['soil_water_infiltration__depth'])
            inf_t0_c9715 = (rmg.at_node['soil_water_infiltration__depth'])[9715]
            hydrograph_time = []
            discharge_at_outlet = []
            
            a=0
            time_step_raster = 300
            while elapsed_time < model_run_time:
                #print('run ' + str(run) + 'elapsed_time ' + str(elapsed_time))
        
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
                    # culvert_links=[80133, 80336, 80335, 80336, #culvert 7
                    #                59454, 59252, 59251, #culvert 8
                    #                56616, 56819, 57022, 56818, 57021, 56820, 56617#culvert 9
                    #                ]
                    )
                # #Rainfall event
                # timestep = of.dt
                # if elapsed_time < (storm_duration):
                #     if rainfall_intensity_df['time'].iloc[a+1]>elapsed_time:
                #         if rainfall_intensity_df['time'].iloc[a+1]>=elapsed_time + timestep:
                #             rainfall_mmhr = rainfall_intensity_df['P_int'].iloc[a]
                #             of.rainfall_intensity = rainfall_mmhr * (2.777778 * 10**-7) #[m/s]
                #         else:
                #             rainfall_mmhr = (((elapsed_time + timestep - rainfall_intensity_df['time'].iloc[a+1])/timestep) 
                #                               * rainfall_intensity_df['P_int'].iloc[a+1]) + ((1-((elapsed_time + timestep - rainfall_intensity_df['time'].iloc[a+1])/timestep))*rainfall_intensity_df['P_int'].iloc[a])
                #             of.rainfall_intensity = rainfall_mmhr * (2.777778 * 10**-7) #[m/s]
                #     else:
                #         a=a+1
                #         if rainfall_intensity_df['time'].iloc[a+1]>=elapsed_time + timestep:
                #             rainfall_mmhr = rainfall_intensity_df['P_int'].iloc[a]
                #             of.rainfall_intensity = rainfall_mmhr * (2.777778 * 10**-7) #[m/s]
                #         else:
                #             rainfall_mmhr = (((elapsed_time + timestep - rainfall_intensity_df['time'].iloc[a+1])/timestep) 
                #                               * rainfall_intensity_df['P_int'].iloc[a+1]) + ((1-((elapsed_time + timestep - rainfall_intensity_df['time'].iloc[a+1])/timestep))*rainfall_intensity_df['P_int'].iloc[a])
                #             of.rainfall_intensity = rainfall_mmhr * (2.777778 * 10**-7) #[m/s]
                # else: 
                #     rainfall_mmhr = 0.0
                #     of.rainfall_intensity = 0.0
                
                discharge = of._q
                discharge_9715 = rmg.calc_flux_div_at_node(of._q)[9715]
                depth_before_inf = rmg.at_node['surface_water__depth']
                rmg.at_node['surface_water__discharge'] = of.discharge_mapper(of._q, convert_to_volume=True)
                surface_water_discharge_9715 = rmg.at_node['surface_water__discharge'][9715]
                # model infiltration
                h_beforeInf_sum = sum(rmg.at_node['surface_water__depth'])
                h_beforeInf_9715 = rmg.at_node['surface_water__depth'][9715]
                SI.run_one_step(of.dt)
                h_afterInf_sum = sum(rmg.at_node['surface_water__depth'])
                h_afterInf_9715 = rmg.at_node['surface_water__depth'][9715]
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
        
        
                dataframe_c9715 = dataframe_c9715.append(pd.DataFrame({'time': elapsed_time,
                                                                     'timestep': of.dt,
                                                                     'depth': (rmg.at_node['surface_water__depth'])[9715],
                                                                     'depth_before_inf': depth_before_inf[9715], 
                                                                     'depth_after_inf': depth_after_inf[9715],
                                                                     'infiltration_depth':(rmg.at_node['soil_water_infiltration__depth'])[9715],
                                                                     'actual_infiltration':(rmg.at_node['soil_water_infiltration__depth'])[9715] - inf_t0_c9715,
                                                                     'h_beforeInf_9715': h_beforeInf_9715,
                                                                     'h_afterInf_9715': h_afterInf_9715,
                                                                     'discharge_9715': discharge_9715,
                                                                     'surface_water_discharge_9715':surface_water_discharge_9715,
                                                                     'rainfall_intensity': of.rainfall_intensity}, 
                                                                    index=[0]), ignore_index=True)
                #check water balance
                (unique, counts) = np.unique(rmg.status_at_node, return_counts=True)
                #water out
                dataframe_int['Q_volume'] = dataframe_int['timestep'] * dataframe_c9715['surface_water_discharge_9715']
                dataframe_int['cumm_Q_volume'] = dataframe_int['Q_volume'].cumsum()
                dataframe_int['h_volume'] = dataframe_int['depth_sum']*(rmg.dx**2)
                dataframe_int['cumm_I_vol'] = dataframe_int['infiltration_depth']*(rmg.dx**2)
                #water in
                dataframe_int['P_volume'] = dataframe_int['timestep'] * dataframe_c9715['rainfall_intensity'] * counts[0] * (rmg.dx**2)
                dataframe_int['cumm_P_volume'] = dataframe_int['P_volume'].cumsum()
                dataframe_int['h0_vol'] = len(rmg.status_at_node) * (rmg.dx**2) * (10**-8) 
                dataframe_int['I0_vol'] = len(rmg.status_at_node) * (rmg.dx**2) * (10**-8) 
                dataframe_int['balance'] = dataframe_int['cumm_P_volume'] + dataframe_int['h0_vol'] + dataframe_int['I0_vol'] - dataframe_int['cumm_Q_volume'] - dataframe_int['h_volume'] - dataframe_int['cumm_I_vol']
              
    
                hydrograph_time.append(elapsed_time / 3600.) # convert seconds to hours
                discharge_at_outlet.append(np.abs(of._q[9715]) * rmg.dx) # append discharge in m^3/s
                
                #add infiltrated water column to total infiltrated water column
                inf_t0 = sum(rmg.at_node['soil_water_infiltration__depth'])
        
                inf_t0_c9715 = (rmg.at_node['soil_water_infiltration__depth'])[9715]
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
        
                    h_reshape = h_t.reshape((rmg.shape[0],rmg.shape[1]))
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
        
                    inf_reshape = inf_t.reshape((rmg.shape[0],rmg.shape[1]))
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
        
                    q_reshape = q_t.reshape((rmg.shape[0],rmg.shape[1]))
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
        
        
            list_dfs = [dataframe_int, dataframe_c9715]
            list_dfs_names = ["int","c9715"]
            xls_path = output_folder + "\\dataframes_" + str(rainfall_event) + ".xlsx"
        
            save_xls(list_dfs, list_dfs_names, xls_path)
            
            
            #discharge graph
            ax = dataframe_c9715.plot(x='time', y='surface_water_discharge_9715', style='.', ms=1)
            ax.set_ylabel("discharge (m³/sec)")
            ax.get_figure().savefig(output_folder + '\\discharge.png')
            
            #Output file
            #discharge volume (m³) sum
            dataframe_c9715['discharge_volume'] = dataframe_c9715['timestep'] * dataframe_c9715['surface_water_discharge_9715']
            sum_discharge_9715  = dataframe_c9715['discharge_volume'].sum()
            
            #peak discharge (m³/sec)
        
            peak_discharge9715 = dataframe_c9715['surface_water_discharge_9715'].max()
                                
            #time of peak discharge
            peak_discharge_df9715 = dataframe_c9715[dataframe_c9715['surface_water_discharge_9715']==dataframe_c9715['surface_water_discharge_9715'].max()]
            peak_discharge_time9715 = peak_discharge_df9715['time'].iloc[0]
            #infiltration sum (m³)
            sum_infiltration = dataframe_int['actual_infiltration'].sum()
        
        
            output_df = output_df.append(pd.DataFrame({
                'rainfall_event': rainfall_event,
                'Theta0': Theta0_values[0],
                'KSAT': Ke_value,
                'discharge_volume': dataframe_int['cumm_Q_volume'][len(dataframe_int)-1],
                'peak_discharge': peak_discharge9715, 
                'peak_discharge_time': peak_discharge_time9715, 
                'infiltration_sum': dataframe_int['cumm_I_vol'][len(dataframe_int)-1],
                'h': dataframe_int['h_volume'][len(dataframe_int)-1], 
                'balance': dataframe_int['balance'][len(dataframe_int)-1],
                'vLE': vLE_type},
                index=[0]), ignore_index=True)
        
output_df.to_csv(main 
                 + "\\output_df_velm" 
                 + "_RainfallEvent_" 
                 + rainfall_event 
                 + datetime.datetime.now().strftime('%Y%m%d_%Hh%M') 
                 + ".csv" )
