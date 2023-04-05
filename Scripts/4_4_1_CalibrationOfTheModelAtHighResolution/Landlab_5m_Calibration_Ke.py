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

###############################################################################
########### adapt rainfall event and corresponding Theta0_value!!! ############
rainfall_event = "20130507"
Theta0_values = [0.174247985283]
###############################################################################

#%%create output directory
infiltration = "inf" #"inf" or "Ksat0"
DTM_slope = "velm" #flat or steep or flat_scaled
DTM_type = "heulengracht_burn" #plane or mpd_plane
fill_sinks = "fill" #fill or no_fill
psi_value  = 0.1727
outlet_id = 159863
Ke_range = [19.2, 1.92, 0.72, 0.96, 0.48] 
output_df = pd.DataFrame(columns=['Theta0','KSAT','manning','vle_orientation', 'vle_density','discharge_volume','peak_discharge', 'peak_discharge_time', 'infiltration_sum'])
#%%
### 6 RUN (save as tiff)

# configimport os
GDAL_DATA_TYPE = gdal.GDT_Float64 
GEOTIFF_DRIVER_NAME = r'GTiff'
SPATIAL_REFERENCE_SYSTEM_WKID = 31370
bottom_left = (201810, 159340)
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
for Ke_value in Ke_range:
    print('run ' + str(run))
                    
    #create output folder
    main = "I:\\WP3\\Landlab_output\\5m\\"
    output_folder = os.path.join(main,
                                 str(Ke_value)
                                 +"_Theta0" + str(Theta0_values[0]) +
                                 DTM_type + "_" + DTM_slope + "_Res5m" + 
                                 "_RP" + str(rainfall_event) + 
                                 "_outlet_id" + str(outlet_id) + "_Flume89_h0_adapt_" +
                                 datetime.datetime.now().strftime('%Y%m%d_%Hh%M'))
    os.makedirs(output_folder)
    
    ###### STEP 1 -Defining the model domain
    #Reading ESRI grid DEM
    if DTM_slope =="velm":
        if DTM_type == "heulengracht_burn":
            watershed_DEM = "I:\\WP3\\DTM\\heulengracht_5m_burn_flume.asc"
    (rmg, z) = read_esri_ascii(watershed_DEM, name="topographic__elevation")
    rmg.set_watershed_boundary_condition_outlet_id(159863, z, nodata_value=-9999.0) # outlet = node 159863
    
    ######## STEP 2 - Pre-processing the DEM - fill sinks
    if fill_sinks == "fill":
        sfb = SinkFillerBarnes(rmg, method='Steepest', fill_flat=False)
        sfb.run_one_step()
    
    ######## STEP 3 - Setting boundary conditions
    rmg.at_node['surface_water__depth'] = np.ones(236925)* 1 * (10**-8) 
    h0 = sum(rmg.at_node['surface_water__depth'])
    
    ######## STEP 4 - vle configuration
    if DTM_type == "heulengracht_burn":
        config= rasterio.open('I:\\WP3\\DTM\\Heulengracht_5m_burn.tif')
        array_config = config.read()
        array_config=np.flip(array_config, 1)
        fields=array_config.flatten()
        fields = fields.astype(np.float64)
    if DTM_type == "heulengracht_original":
        config = rasterio.open('I:\\WP3\\DTM\\Heulengracht_5m.tif')
        array_config = config.read()
        array_config=np.flip(array_config, 1)
        fields=array_config.flatten()
        fields = fields.astype(np.float64)
    ######## STEP 5 - Define mannings roughness coefficient, hydraulic conductivity, wetting front capillary pressyre head and soil infiltration at t=0
    ### mannings roughtness coefficient
    manning = rasterio.open('I:\\WP3\\Landcover\\manning_2013_5m.tif')
    manning = manning.read()
    manning = np.flip(manning, 1)
    manning = manning.flatten()
    manning = manning.astype(np.float64)
    rmg.at_node['mannings_n__coefficients'] = manning
    ### saturated hydraulic conductivity 
    rows, columns = config.shape
    hydraulic_conductivity_rmg = np.ones((rows, columns))*Ke_value/3600000 #Van den Putte et al. (2013); average value summer
    hydraulic_conductivity_rmg = hydraulic_conductivity_rmg.flatten()
    #no infiltration in culverts
    #indices = [232461, 197986, 185821, 159863, 118505, 113639, 113640]
    indices = [159863, 118505, 113639, 113640]
    for index_id in indices:
        hydraulic_conductivity_rmg[index_id] = 0

    ### wetting front capillary pressure head
    wetting_front_capillary_pressure_head_rmg = rmg.ones('node')*psi_value # [m]
    ### soil infiltration at t=0
    d = rmg.add_ones("soil_water_infiltration__depth", at="node", dtype=float)
    d *= 0.00000001
    
    ######## STEP 6 - Initializing the OverlandFlow and SoilInfiltration component
    of = OverlandFlow(rmg,
                      steep_slopes=True, 
                      theta=0.9,
                      h_init = 10**-8,

                       culvert_links = [319038, 319039, 319443, 318634,
                                        236473, 236877, 236068, 235664, 235663, 
                                        226760, 227164, 226759, 226355, 227163, 226758, 226354]
                      )
    SI = SoilInfiltrationGreenAmpt(rmg, 
                                   hydraulic_conductivity = hydraulic_conductivity_rmg,
                                   wetting_front_capillary_pressure_head = wetting_front_capillary_pressure_head_rmg, 
                                   soil_type='loam',
                                   soil_bulk_density =1230,
                                   initial_soil_moisture_content = Theta0_values[0])
 #%    
    ######## STEP 7 - Determining Precipitation inputs
    #single storm event
    elapsed_time = 0.0
    model_run_time = 185000
    storm_duration = 138600

    if rainfall_event == "20130507":
        rainfall_intensity_df = pd.read_csv('I:\\WP3\\Precipitation\\20130507.csv', sep = ";", header=0)
    ######## STEP 8 - RUN (runoff & infiltration)
    ### dataframes for output
    dataframe_int = pd.DataFrame(columns=['time', 'timestep', 'discharge_sum', 'depth_sum', 'max_h', 
                                          'infiltration_depth', 'actual_infiltration','h_beforeInf_sum', 
                                          'h_afterInf_sum', 'Boundary_discharge_sum'])

    dataframe_c159863= pd.DataFrame(columns=['time', 'timestep', 'depth', 'depth_before_inf', 
                                          'depth_after_inf', 'infiltration_depth', 'actual_infiltration',
                                          'h_beforeInf_159863', 'h_afterInf_159863', 'discharge_159863', 'surface_water_discharge_159863', 'rainfall_intensity'])
              
    inf_t0 = sum(rmg.at_node['soil_water_infiltration__depth'])

    inf_t0_c159863 = (rmg.at_node['soil_water_infiltration__depth'])[159863]
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

            culvert_links = [319038, 319039, 319443, 318634,
                             236473, 236877, 236068, 235664, 235663, 
                             226760, 227164, 226759, 226355, 227163, 226758, 226354]
            )

        discharge = of._q

        discharge_159863 = rmg.calc_flux_div_at_node(of._q)[159863]
        depth_before_inf = rmg.at_node['surface_water__depth']
        rmg.at_node['surface_water__discharge'] = of.discharge_mapper(of._q, convert_to_volume=True)

        surface_water_discharge_159863 = rmg.at_node['surface_water__discharge'][159863]        
        # model infiltration
        h_beforeInf_sum = sum(rmg.at_node['surface_water__depth'])

        h_beforeInf_159863 = rmg.at_node['surface_water__depth'][159863]
        SI.run_one_step(of.dt)
        h_afterInf_sum = sum(rmg.at_node['surface_water__depth'])

        h_afterInf_159863 = rmg.at_node['surface_water__depth'][159863]
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
        dataframe_c159863 = dataframe_c159863.append(pd.DataFrame({'time': elapsed_time,
                                                             'timestep': of.dt,
                                                             'depth': (rmg.at_node['surface_water__depth'])[159863],
                                                             'depth_before_inf': depth_before_inf[159863], 
                                                             'depth_after_inf': depth_after_inf[159863],
                                                             'infiltration_depth':(rmg.at_node['soil_water_infiltration__depth'])[159863],
                                                             'actual_infiltration':(rmg.at_node['soil_water_infiltration__depth'])[159863] - inf_t0_c159863,
                                                             'h_beforeInf_159863': h_beforeInf_159863,
                                                             'h_afterInf_159863': h_afterInf_159863,
                                                             'discharge_159863': discharge_159863,
                                                             'surface_water_discharge_159863':surface_water_discharge_159863,
                                                             'rainfall_intensity': of.rainfall_intensity}, 
                                                            index=[0]), ignore_index=True)
        #check water balans
        (unique, counts) = np.unique(rmg.status_at_node, return_counts=True)
        #water out
        dataframe_int['Q_volume'] = dataframe_int['timestep'] * dataframe_c159863['surface_water_discharge_159863']
        dataframe_int['cumm_Q_volume'] = dataframe_int['Q_volume'].cumsum()
        dataframe_int['h_volume'] = dataframe_int['depth_sum']*(rmg.dx**2)
        dataframe_int['cumm_I_vol'] = dataframe_int['infiltration_depth']*(rmg.dx**2)
        #water in
        dataframe_int['P_volume'] = dataframe_int['timestep'] * dataframe_c159863['rainfall_intensity'] * counts[0] * (rmg.dx**2)
        dataframe_int['cumm_P_volume'] = dataframe_int['P_volume'].cumsum()
        dataframe_int['h0_vol'] = len(rmg.status_at_node) * (rmg.dx**2) * (10**-8) 
        dataframe_int['I0_vol'] = len(rmg.status_at_node) * (rmg.dx**2) * (10**-8) 
        dataframe_int['balance'] = dataframe_int['cumm_P_volume'] + dataframe_int['h0_vol'] + dataframe_int['I0_vol'] - dataframe_int['cumm_Q_volume'] - dataframe_int['h_volume'] - dataframe_int['cumm_I_vol']


        hydrograph_time.append(elapsed_time / 3600.) # convert seconds to hours
        discharge_at_outlet.append(np.abs(of._q[159863]) * rmg.dx) # append discharge in m^3/s
        
        #add infiltrated water column to total infiltrated water column
        inf_t0 = sum(rmg.at_node['soil_water_infiltration__depth'])

        inf_t0_c159863 = (rmg.at_node['soil_water_infiltration__depth'])[159863]       
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

            h_reshape = h_t.reshape((585,405))
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

            inf_reshape = inf_t.reshape((585,405))
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

            q_reshape = q_t.reshape((585,405))
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


    list_dfs = [dataframe_int, dataframe_c159863]
    list_dfs_names = ["int", "c159863"]
    xls_path = output_folder + "\\dataframes_" + "20130507" + ".xlsx"

    save_xls(list_dfs, list_dfs_names, xls_path)
    
    
    #discharge graph
    ax = dataframe_c159863.plot(x='time', y='surface_water_discharge_159863', style='.', ms=1)
    ax.set_ylabel("discharge (m³/sec)")
    ax.get_figure().savefig(output_folder + '\\discharge.png')
    
    
    #Output file
    #discharge volume (m³) sum
    dataframe_c159863['discharge_volume'] = dataframe_c159863['timestep'] * dataframe_c159863['surface_water_discharge_159863']
    sum_discharge  = dataframe_c159863['discharge_volume'].sum()
    
    #peak discharge (m³/sec)
    peak_discharge = dataframe_c159863['surface_water_discharge_159863'].max()
                        
    #time of peak discharge
    peak_discharge_df = dataframe_c159863[dataframe_c159863['surface_water_discharge_159863']==dataframe_c159863['surface_water_discharge_159863'].max()]
    peak_discharge_time = peak_discharge_df['time'].iloc[0]
    
    #infiltration sum (m³)
    sum_infiltration = dataframe_int['actual_infiltration'].sum()


    output_df = output_df.append(pd.DataFrame({
        'rainfall_event': rainfall_event,
        'Theta0': Theta0_values[0],
        'KSAT': Ke_value,
        'discharge_volume': dataframe_int['cumm_Q_volume'][len(dataframe_int)-1],
        'peak_discharge': peak_discharge, 
        'peak_discharge_time': peak_discharge_time, 
        'infiltration_sum': dataframe_int['cumm_I_vol'][len(dataframe_int)-1],
        'h': dataframe_int['h_volume'][len(dataframe_int)-1], 
        'balance': dataframe_int['balance'][len(dataframe_int)-1],
        'P_Vol': dataframe_int['cumm_P_volume'][len(dataframe_int)-1]},
        index=[0]), ignore_index=True)   


    output_df.to_csv(main 
                 + "\\output_df_velm" 
                 + "_Resolution5m_" 
                 + rainfall_event + "_"  
                 + datetime.datetime.now().strftime('%Y%m%d_%Hh%M') 
                 + ".csv" )
output_df.to_csv(main 
                 + "\\output_df_velm" 
                 + "_RainfallEvent_" 
                 + rainfall_event 
                 + datetime.datetime.now().strftime('%Y%m%d_%Hh%M') 
                 + ".csv" )
