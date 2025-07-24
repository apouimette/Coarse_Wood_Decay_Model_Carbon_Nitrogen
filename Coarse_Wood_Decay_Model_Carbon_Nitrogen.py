# -*- coding: utf-8 -*-
'''
Created on Thu Feb 20 15:13:38 2025
@author: Andrew Ouimette
'''

from pathlib import Path
import pandas as pd
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba









'''This defines a function that opens a dialog for the user to select a working directory. ***EXPAND FOR MORE INFO***
    This folder should have subfolders called "inputs", "outputs", and "scripts".
    This script should be stored in the "scripts" subfolder.
    Tkinter (short for "Tk Interface") is Python’s standard GUI (Graphical User Interface) library. 
    It provides tools to create windows, buttons, menus, dialogs, and other interactive elements for desktop applications.
    It is built on Tk, a GUI toolkit originally developed for the Tcl programming language, but now widely used in Python.'''
def select_directory():
    root = tk.Tk() # createsng a new instance of the Tkinter root window, which acts as the main application window for any GUI elements we want to display.
    root.withdraw()  # root = tk.Tk() opens a full GUI window by default. However, we only need the file selection dialog—we don’t want a visible blank window. So we use withdraw to prevent the default Tkinter window from appearing while still allowing the file selection dialog to work.
    root.attributes('-topmost', True)  # Bring the dialog window to the front
    selected_path = filedialog.askdirectory(title="Select Your Working Directory")
    root.destroy()  # Ensure the window closes properly
    
    if not selected_path:  # If the user cancels, raise an error
        raise ValueError("No directory selected. Exiting.")
    
    return Path(selected_path)  # Return as a Path object

#Runs the select directory function to get user-selected directory# 
base_path = select_directory()

#Define paths for input and output folders# 
input_path = base_path / 'inputs'
output_path = base_path / 'outputs'

#Print paths for verification# 
print(f"Selected Base Path: {base_path}")
print(f"Input Path: {input_path}")
print(f"Output Path: {output_path}")


#Start time measurement# 
st = time.perf_counter()

#Read input parameter CSV into a DataFrame#
input_parameters = pd.read_csv(input_path / 'CWD_input_parameters.csv')

#Read disturbance parameter CSV into a DataFrame#
disturbance_parameters = pd.read_csv(input_path / 'CWD_disturbance_inputs.csv')

# Keep only the columns you need (in correct order)
cols_to_keep = ['year', 'disturbance_mortality', 'disturbance_biomass_removal', 'variation_mortality', 'variation_biomass_removal']
disturbance_parameters = disturbance_parameters[cols_to_keep]

disturbance_parameters = disturbance_parameters.dropna(subset=['year'])
disturbance_parameters['year'] = disturbance_parameters['year'].astype(int)

#Convert input parameters into a dictionary indexed by parameter name# 
input_params_dict = input_parameters.set_index('parameter').to_dict()

#Convert disturbance parameters into a dictionary indexed by year#
#disturbance_params_dict = disturbance_parameters.set_index('year').to_dict()

#Number of simulations - defined by user in the input file# 
num_simulations = int(input_params_dict['value'].get('num_simulations', 1))  # Default to 1 if missing


'''Create lists for the possible model structures that can be used. ***EXPAND FOR MORE INFO***
    These model structures are selected/entered by the user on the input csv file
    labeled "CWD_Input_Parameters.csv" located in the inputs subfolder.
    Not included in these lists is the last option "MC", for Monte Carlo type simulations. This is addressed later on'''
model_options = {
    'model_version': ['default', 'no_snags', 'no_bark', 'no_snags_no_bark'],
    'wood_mort_mode': ['constant', 'random', 'age_related'],
    'snag_fall_mode': ['constant', 'exponential', 'random'], "write_cohort_results": ['yes', 'no']}



'''Defines a function that retrieves a parameter by name, ***EXPAND FOR MORE INFO***
    using the "default" column in the input csv if value is missing, 
    and applies proportional Monte Carlo variation if specified.'''
def get_parameter(param_dict, key, apply_random=False):
    value = param_dict['value'].get(key)  # Check if value exists
    if value is None or pd.isna(value):  # If missing or NaN, use default which is a column in the input file
        value = param_dict['default'].get(key, 0)  # Default to 0 if "default" is also missing
    value = float(value)  # Ensure it's a float
    variation = float(param_dict['variation'].get(key, 0))    # Get variation (default to 0 if missing)
    if apply_random and variation != 0:     # Apply proportional variation if specified and nonzero
        value *= np.random.uniform(1 - variation, 1 + variation)  
    return value


'''Defines a function that retrieves a categorical parameters by name that control model structure, ***EXPAND FOR MORE INFO***
    using the "default" column in the input csv if value is missing, 
    and applies Monte Carlo variation type run that chooses between possible 'model options' if 'MC' is specified.
        Retrieves a categorical parameter from the input CSV.
        - If "MC", randomly selects from predefined options.
        - If missing, falls back on the "default" column.
        - If specified, uses the provided value.'''
def get_categorical_choice(param_dict, key, options_list):

    value = param_dict['value'].get(key)
    if value == "MC":
        return np.random.choice(options_list)  # Monte Carlo mode (random selection)
    if value is None or pd.isna(value):  # Use default if missing
        return str(param_dict['default'].get(key, options_list[0]))  # Default to first option if none provided
    return str(value)  # Use user-specified value if available


'''Defines a function that retrieves the disturbance mortality and biomass removal for a given year. ***EXPAND FOR MORE INFO***
    Checks for a corresponding variation (variation_mortality, variation_removal) and
    applies proportional variation if needed. Ensures values never exceed 100% (1.0).'''
# def get_disturbance_parameter(year, key, apply_random=False):
#     value = disturbance_params_dict[key].get(year, 0)  # Default to 0 if missing
#     variation = disturbance_params_dict[key.replace('disturbance_', 'variation_')].get(year, 0)  # Get variation
#     if apply_random and variation != 0:
#         value *= np.random.uniform(1 - variation, 1 + variation)  # Apply proportional variation
#     return min(1, value)  # Ensure max value is 1

def get_disturbance_parameter(year, column, apply_random=False):
    row = disturbance_parameters[disturbance_parameters['year'] == year]
    if row.empty:
        return 0.0

    value = row.iloc[0][column]
    if pd.isnull(value):
        return 0.0

    variation_col = 'variation_' + column.split('_')[-1]
    variation = row.iloc[0][variation_col] if variation_col in row.columns else 0

    if apply_random and variation > 0:
        value *= np.random.uniform(1 - variation, 1 + variation)  # ✅ proportional

    # Final safety
    return max(0.0, min(1.0, value))


'''########################################################################################################
            Define functions to adjust parameter values over time, calculate NPP, etc.
########################################################################################################'''

'''This function calculates a dynamic LAI scalar based on stand age. ***EXPAND FOR MORE INFO*** 
    It is used to adjust Net Primary Productivity (NPP) based on canopy development.
    - If the stand age is greater than or equal to `years_to_full_canopy`, the canopy is fully developed, 
      and the LAI scalar is set to 1 (no reduction in productivity).
    - Otherwise, the scalar increases linearly from 0 to 1 as the stand age approaches `years_to_full_canopy`.
Parameters:
    stand_age (int): The current stand age in years.
    years_to_full_canopy (int): The number of years required to reach full canopy closure.
Returns: float: The LAI scalar (range: 0 to 1).'''

def lai_scalar_eqn(stand_age, years_until_full_canopy, reestablishment_phase):
    if stand_age < reestablishment_phase:
        return 0  # No canopy development yet
    adjusted_age = stand_age - reestablishment_phase
    return 1 if adjusted_age >= years_until_full_canopy else (adjusted_age + 1) / years_until_full_canopy


''' Calculate wood production based on stand age with a transition from ***EXPAND FOR MORE INFO***
initial to final production values over a defined age range.
    Vectorized version of wood_production_eqn.
Applies element-wise operations on NumPy arrays instead of row-by-row processing.
    Parameters:
        stand_age (int): Current stand age.
        initial (float): Initial wood production (early stand age).
        final (float): Final wood production (older stand age).
        start_age (int): Age when wood production starts to decline.
        end_age (int): Age when wood production stabilizes at the final value.
    Returns: float: Estimated wood production for the given stand age.'''
def wood_production_vectorized(stand_ages, initial, final, decline_start_age, decline_end_age, reestablishment_phase):
    conditions = [
        stand_ages < reestablishment_phase,                         # No production during reestablishment
        (stand_ages >= reestablishment_phase) & (stand_ages < decline_start_age),  # Rising and peak productivity phase
        stand_ages >= decline_end_age,                              # After decline plateau
        (stand_ages >= decline_start_age) & (stand_ages < decline_end_age)  # Linear decline
    ]

    choices = [
        0,  # No NPP during reestablishment
        initial * 0.5,  # Peak productivity
        final * 0.5,    # Final stable productivity
        (initial - ((stand_ages - decline_start_age) / 
                    (decline_end_age - decline_start_age)) * (initial - final)) * 0.5  # Declining phase
    ]

    return np.select(conditions, choices, default=final)




#This initializes an lists before the simulation loop starts. It serves as a container to store the outputs of each simulation run. 
results = []  # Stores the main 'age' DataFrame for each run
cohort_results = []  # Stores cohort (necromass) data for each run
all_summaries = []  # Store all runs' summary data
all_snag_cohorts = []
all_dwd_cohorts = []


print(f"Number of simulations: {num_simulations}")

       
'''*************************************************************************************************************
                                                THE LOOP
****************************************************************************************************************'''

'''This next section of the code uses a for loop to run multiple simulations of the coarse wood decay model. ***EXPAND FOR MORE INFO***
        The model runs for a user-specified number of simulations (num_simulations), allowing for Monte Carlo-type variability.
        Each simulation prints the current run number for tracking progress.'''
for run in range(num_simulations):
    print(f"Running simulation {run + 1} of {num_simulations}")
    

    '''This first section of the loop initializes model parameters for the current simulation run. ***EXPAND FOR MORE INFO*** 
           It retrieves values from a global dictionary (`input_params_dict`) that was initially created outside the loop from the input csv
           and stores them in a local dictionary (`params`) that can change with each simulation run.  
           Each parameter is extracted, assigned to a variable, and optionally varied based on user-defined Monte Carlo settings.
    How Parameters Are Retrieved:
        Step 1: Previously (Outside this Loop) we read in the input CSV and Created a Global Dictionary
            The CSV (`CWD_Input_Parameters.csv`) is read in as a DataFrame ('input_parameters').
            It is then converted outside the loop into a dictionary (`input_params_dict`) where:
               `input_params_dict['value']` stores parameter values.
               `input_params_dict['variation']` stores variation ranges.
               `input_params_dict['default']` stores default values (used when a parameter is missing).
        Step 2: Populate `params` (Inside the Loop)
            Inside the loop, values are retrieved from `input_params_dict` and stored in a new dictionary, `params`,  
            which holds only the necessary values for the current simulation run.
            Some parameters can undergo proportional Monte Carlo variation.
    Why Use This Two-Step Approach?
        Efficiency: Instead of searching through a dataframe every time, `input_params_dict` provides fast lookups.
        Flexibility: Allows for Monte Carlo variation while keeping original values untouched.
        Readability: `params` is a clean dictionary containing only the values relevant to the current run.
    Handling Monte Carlo Variability:
        Random Model Structure Selection  
            - Some parameters (e.g., `model_version`, `wood_mort_mode`) can be randomly selected from predefined lists  
            if specified as `'MC'` in the input file.
            - This allows for structured uncertainty testing in model behavior.
        Applying Proportional Variation to Continuous Parameters  
            - If a parameter has a nonzero variation specified in the input file, we apply a proportional scaling factor:
               value = np.random.uniform(1 - variation, 1 + variation)
            - This means:
               If `variation = 0.1` (10%), the parameter is scaled by a factor between `0.9x` and `1.1x`.
               If `variation = 0`, the parameter remains unchanged.
   Ensuring Biophysical Validity:
       - Parameters won’t become negative since they are scaled proportionally.
       - For disturbance-related parameters (e.g., mortality), values are later capped at `1` or '100%' to maintain biological realism.
   Key Biogeochemical Parameters Being Retrieved
       - Growth and Decay Rates: `wood_mort_rate`, `snag_fall_rate`, `snag_decay_rate`, `dwd_decay_rate`, 
       - Nitrogen Concentrations of wood and bark, and below critical C:N ratios of wood and bark'''

    params = {
        'start_year': int(get_parameter(input_params_dict, 'start_year')),
        'end_year': int(get_parameter(input_params_dict, 'end_year')),
    
        'model_version': get_categorical_choice(input_params_dict, 'model_version', model_options['model_version']),
        'wood_mort_mode': get_categorical_choice(input_params_dict, 'wood_mort_mode', model_options['wood_mort_mode']),
        'snag_fall_mode': get_categorical_choice(input_params_dict, 'snag_fall_mode', model_options['snag_fall_mode']),
        'write_cohort_results': get_categorical_choice(input_params_dict, 'write_cohort_results', model_options['write_cohort_results']),
        
        'reestablishment_phase': round(get_parameter(input_params_dict, 'reestablishment_phase', apply_random=True)),
        'years_until_full_canopy': round(get_parameter(input_params_dict, 'years_until_full_canopy', apply_random=True)),
        'wood_production_initial': get_parameter(input_params_dict, 'wood_production_initial', apply_random=True),
        'wood_production_final': get_parameter(input_params_dict, 'wood_production_final', apply_random=True),
        'wood_production_change_start_age': round(get_parameter(input_params_dict, 'wood_production_change_start_age', apply_random=True)),
        'wood_production_change_end_age': round(get_parameter(input_params_dict, 'wood_production_change_end_age', apply_random=True)),

        'wood_mort_rate': get_parameter(input_params_dict, 'wood_mort_rate', apply_random=True),
        'wood_mort_rate_var': get_parameter(input_params_dict, 'wood_mort_rate_var', apply_random=True),
        'wood_mort_rate_ini': get_parameter(input_params_dict, 'wood_mort_rate_ini', apply_random=True),
        'wood_mort_rate_end': get_parameter(input_params_dict, 'wood_mort_rate_end', apply_random=True),
        'wood_mort_start_age': get_parameter(input_params_dict, 'wood_mort_start_age'),
        'wood_mort_end_age': get_parameter(input_params_dict, 'wood_mort_end_age'),
        
        'initial_fall_fraction': get_parameter(input_params_dict, 'initial_fall_fraction', apply_random=True),
        'snag_fall_rate': get_parameter(input_params_dict, 'snag_fall_rate', apply_random=True),
        'snag_fall_rate_var': get_parameter(input_params_dict, 'snag_fall_rate_var', apply_random=True),
        'lambda_snag_fall': get_parameter(input_params_dict, 'lambda_snag_fall', apply_random=True),

        'k_dwd_decay': get_parameter(input_params_dict, 'k_dwd_decay', apply_random=True),
        'snag_decay_factor': get_parameter(input_params_dict, 'snag_decay_factor', apply_random=True),
        'limit_value': get_parameter(input_params_dict, 'limit_value', apply_random=True),
        'slow_decay_rate': get_parameter(input_params_dict, 'slow_decay_rate', apply_random=True),

        'wood_perc_N': get_parameter(input_params_dict, 'wood_perc_N', apply_random=True),
        'bark_perc_N': get_parameter(input_params_dict, 'bark_perc_N', apply_random=True),
        'bark_frac_wood': get_parameter(input_params_dict, 'bark_frac_wood', apply_random=True),
        'critical_CN_wood': get_parameter(input_params_dict, 'critical_CN_wood', apply_random=False),
        'critical_CN_bark': get_parameter(input_params_dict, 'critical_CN_bark', apply_random=False)

    } 
   
    

    '''This modifies wood_perc_N (Wood Nitrogen %) if the model is run without separate wood and bark pools.  ***EXPAND FOR MORE INFO***
        When model_version is 'no_bark' or 'no_snags_no_bark', bark and wood are combined as a single pool, so:
        The wood nitrogen % is adjusted using a weighted mix of the wood and bark nitrogen %'s.
        In these cases, bark_frac_wood is set to 0 since there is not an explicit bark pool.
    How It Works:
        (1 - params['bark_frac_wood']) represents the fraction of material that is wood (not bark).
        params['bark_perc_N'] * params['bark_frac_wood'] adds the nitrogen that would have come from the bark fraction.
        params['bark_frac_wood'] = 0 ensures the model treats all material as wood-only.'''
   
    if params['model_version'] in ['no_bark', 'no_snags_no_bark']:
        params['wood_perc_N'] = params['wood_perc_N'] * (1 - params['bark_frac_wood']) + params['bark_perc_N'] * params['bark_frac_wood']
        params['bark_frac_wood'] = 0
       
  
    '''Creates a lookup dictionary that links C:N ratios (critical_CN_wood, critical_CN_bark) to the corresponding wood and bark nitrogen contents.
        This makes the next computation more efficient and readable.
    Why is this Needed?
        It eliminates the need for multiple if statements to check whether we're dealing with wood or bark.
        The loop below can now directly retrieve nitrogen values using cn_mapping[key].'''
    
    cn_mapping = {
        'critical_CN_wood': params['wood_perc_N'],
        'critical_CN_bark': params['bark_perc_N']
    }
    
    '''This section calculates the critical C:N values for wood and bark.  ***EXPAND FOR MORE INFO***
        - If a critical C:N is specified in the input file and no potential variation is provided, it uses the input value as-is.
        - If a critical C:N is specified with a variation range, it adds proportional variation.
        - If the critical C:N ratio for wood or bark is **zero**, it computes the ratio using an empirical formula (Manzoni et al. 2008)
        and applies proportional variation if specified in the input file.
    Reference: Manzoni S, Jackson RB, Trofymow JA, Porporato A. The global stoichiometry of litter nitrogen mineralization.
    Science. 2008 Aug 1;321(5889):684-6.'''
    
    for key in ['critical_CN_wood', 'critical_CN_bark']:
        variation_range = input_params_dict['variation'].get(key, 0)  # Get variation from input file
        
        if params[key] == 0:
            # Compute empirical critical C:N value
            base_value = 1 / (0.45 * ((cn_mapping[key] / 0.50) ** 0.76))
        else:
            base_value = params[key]  # Use the parameterized value if not zero
        
        # Apply proportional variation only if a nonzero variation range is specified
        if variation_range != 0:
            params[key] = base_value * np.random.uniform(1 - variation_range, 1 + variation_range)
        else:
            params[key] = base_value  # No variation applied
  

    # '''This section initializes a local dictionary called disturb_params within the loop.  ***EXPAND FOR MORE INFO***
    #     It stores disturbance mortality and biomass removal values for each simulation run, 
    #     applying proportional random variation when specified in the input CSV.
    # How It Works:
    #     - Retrieves values from the global disturbance dictionary (disturbance_params_dict) that was
    #     created outside the loop by reading the disturbance parameters CSV.
    #     - Stores disturbance data in a nested dictionary format:
    #         Outer dictionary: Groups parameters (disturbance_mortality and disturbance_biomass_removal).
    #         Inner dictionary: Stores disturbance values indexed by year.
    #     - Just like params, this dictionary stores values specific to the current simulation 
    #         while keeping the original input data (disturbance_params_dict) untouched.
    # Why a Nested Dictionary?
    #     - Logical Grouping: Keeps disturbance_mortality and disturbance_biomass_removal organized under a single dictionary.
    #     - Year-Based Lookup: Allows faster retrieval of values by year, making it efficient when mapping data into the age DataFrame.
    #     - Consistency with params: We retrieve values from a global dictionary and store them in a local dictionary that applies variation when needed.'''
        
    # disturb_params = {
    #     'disturbance_mortality': {
    #         year: get_disturbance_parameter(year, 'disturbance_mortality', apply_random=True)
    #         for year in disturbance_params_dict['disturbance_mortality']
    #     },
    #     'disturbance_biomass_removal': {
    #         year: get_disturbance_parameter(year, 'disturbance_biomass_removal', apply_random=True)
    #         for year in disturbance_params_dict['disturbance_biomass_removal']
    #     }
    # }

    # Build dictionary using .unique() to avoid repeated years
    disturb_params = {
        'disturbance_mortality': {
            int(year): get_disturbance_parameter(int(year), 'disturbance_mortality', apply_random=True)
            for year in disturbance_parameters['year'].unique()
        },
        'disturbance_biomass_removal': {
            int(year): get_disturbance_parameter(int(year), 'disturbance_biomass_removal', apply_random=True)
            for year in disturbance_parameters['year'].unique()
        }
    }

   
    '''This section initializes the `age` DataFrame to store the year-by-year model state.  ***EXPAND FOR MORE INFO***
        It is re-initialized for each simulation run.
    Why is This Inside the Loop?
        - Each simulation run requires a fresh `age` DataFrame to store independent results.
        - If `age` were defined outside the loop, it would persist across runs, causing overwriting issues.
    Step 1: Create DataFrame with Years
        - A new DataFrame (`age`) is created containing a single column: `'year'`.
        - Uses `range(params['start_year'], params['end_year'] + 1)` to generate a list of years from `start_year` to `end_year`, inclusive.
        - Ensures each year within the simulation period has its own row in the DataFrame.
    Step 2: Assign Run Number Explicitly
        - The `'run_number'` column is inserted at index `0` (making it the first column).
        - The current simulation run number (`run + 1`) is assigned to every row.
        - Using `.insert(0, ...)` ensures `run_number` remains the first column and prevents NaN values.
    Step 3: Assign Timestep Explicitly
        - The `'timestep'` column represents the simulation time step for each year.
        - Uses `range(0, int(end_year) - int(start_year) + 1)`, ensuring:
            - The first timestep is `0` (starting year).
            - The last timestep is `(end_year - start_year)`, matching the number of years simulated.
        - This approach ensures that timestep calculation is explicitly tied to the simulation years, independent of DataFrame indexing.'''

    age = pd.DataFrame({'year': list(range(params['start_year'], params['end_year'] + 1))})
    age = pd.DataFrame({'year': list(range(int(params['start_year']), int(params['end_year']) + 1))})

    age.insert(0, 'run_number', run + 1)  
    age['timestep'] = list(range(0, params['end_year'] - params['start_year'] + 1))
    num_years = len(age)
    
    
    disturbance_parameters['year'] = disturbance_parameters['year'].astype(int)   
    age['disturb_mort'] = age['year'].map(disturb_params['disturbance_mortality']).fillna(0)
    age['disturb_bio_remove'] = age['year'].map(disturb_params['disturbance_biomass_removal']).fillna(0)

    
    '''The section of code below builds on the dataframe (`age`). 
    It maps model settings and biogeochemical parameters from `params` (local dictionary).'''
    
    #Assign model settings to each row in the DataFrame
    age['model_version'] = params['model_version']
    age['wood_mort_mode'] = params['wood_mort_mode']
    age['snag_fall_mode'] = params['snag_fall_mode']
    
    
    #Assign wood productivity parameters
    age['reestablishment_phase'] = params['reestablishment_phase']
    age['years_until_full_canopy'] = params['years_until_full_canopy']
    age['wood_production_initial'] = params['wood_production_initial']
    age['wood_production_final'] = params['wood_production_final']
    age['wood_production_change_start_age'] = params['wood_production_change_start_age']
    age['wood_production_change_end_age'] = params['wood_production_change_end_age']
    
    #Assign wood mortality parameters
    age['wood_mort_rate'] = params['wood_mort_rate']
    age['wood_mort_rate_var'] = params['wood_mort_rate_var']
    age['wood_mort_rate_ini'] = params['wood_mort_rate_ini']
    age['wood_mort_rate_end'] = params['wood_mort_rate_end']
    age['wood_mort_start_age'] = params['wood_mort_start_age']
    age['wood_mort_end_age'] = params['wood_mort_end_age']
    
    # Assign snag fall parameters
    age['initial_fall_fraction'] = params['initial_fall_fraction']
    
    # For model versions that bypass snags entirely
    if params['model_version'] in ['no_snags', 'no_snags_no_bark']:
        params['initial_fall_fraction'] = 1.0  # Ensures all mortality bypasses snag pools
        age['initial_fall_fraction'] = 1.0  # Apply to all years in this case
    
    # Otherwise, set initial_fall_fraction = .8 for year 1998 only
    #elif 1998 in age['year'].values:
    #    age.loc[age['year'] == 1998, 'initial_fall_fraction'] = .8

    age['snag_fall_rate'] = params['snag_fall_rate']
    age['snag_fall_rate_var'] = params['snag_fall_rate_var']
    age['lambda_snag_fall'] = params['lambda_snag_fall']


    #Assign decay parameters for DWD and snags
    age['k_dwd_decay'] = params['k_dwd_decay']
    age['snag_decay_factor'] = params['snag_decay_factor']
    age['limit_value'] = params['limit_value']
    age['slow_decay_rate'] = params['slow_decay_rate']
    
    #Assign wood biochemical parameters
    age['wood_perc_N'] = params['wood_perc_N']
    age['bark_perc_N'] = params['bark_perc_N']
    age['bark_frac_wood'] = params['bark_frac_wood']
    age['critical_CN_wood'] = params['critical_CN_wood']
    age['critical_CN_bark'] = params['critical_CN_bark']


   
    # '''Maps disturbance parameters from `disturb_params` instead of recalculating them. ***EXPAND FOR MORE INFO***
    #     - Uses `.map()` to quickly assign values based on `year`.
    #     - Defaults to `0` for missing years using `.fillna(0)`.  '''
    # # Set 'year' as the index and convert each disturbance column to a dictionary
    # mort_map = disturb_params.set_index('year')['disturbance_mortality'].to_dict()
    # bio_remove_map = disturb_params.set_index('year')['disturbance_biomass_removal'].to_dict()
    
    # # Map values safely to the age DataFrame and fill missing years with 0
    # age['disturb_mort'] = age['year'].map(mort_map).fillna(0)
    # age['disturb_bio_remove'] = age['year'].map(bio_remove_map).fillna(0)

    # #Read stand age reset parameters from input file
    # disturbance_window = int(get_parameter(input_params_dict, 'disturbance_window'))
    # disturbance_threshold = get_parameter(input_params_dict, 'disturbance_threshold')
    
    '''Maps disturbance parameters from `disturb_params` dictionary for this run. ***EXPAND FOR MORE INFO***
        - Applies run-specific disturbance values (with optional random variation).
        - Defaults to 0 for years with no disturbance.
    '''

    
    # Read stand age reset parameters from input file
    disturbance_window = int(get_parameter(input_params_dict, 'disturbance_window'))
    disturbance_threshold = get_parameter(input_params_dict, 'disturbance_threshold')
    

    '''Stand Age Calculation. ***EXPAND FOR MORE INFO***
        Stand age is currently only used for canopy closure calculations which limits NPP to a fraction
        of total NPP. There is a 'years_until_full_canopy' input parameter that allows adjustment.
    Stand Age Calculation based on disturbance history.
        - Uses `.rolling().sum()` to check disturbance accumulation over `disturbance_window` years.
        - If cumulative disturbance within the window exceeds `disturbance_threshold`, stand age resets to 0.
        - Otherwise, it increments by 1 each year.
        - We use .rolling() instead of .sum() because it avoids overlapping resets:
            - If disturbances occur multiple times within the disturbance_window, 
                the stand age only resets once per qualifying event.
            - It is more efficient and readable because rolling().sum() computes the sum over 
                the last disturbance_window years without manually indexing slices.
            - It ensures non-continuous disturbances do not repeatedly trigger resets.'''
 
    # Compute rolling cumulative disturbance over the disturbance window
    age['cumulative_disturbance'] = age['disturb_mort'].rolling(window=disturbance_window, min_periods=1).sum()
    
    # Identify first-time threshold exceedance (resets only when first crossed)
    reset_mask = (age['cumulative_disturbance'] >= disturbance_threshold) & (age['cumulative_disturbance'].shift(1, fill_value=0) < disturbance_threshold)
    
    # Convert to NumPy for efficient computation
    reset_mask_np = reset_mask.to_numpy()
    
    
    # Create an array for stand age that starts at 0 and increments
    stand_age_array = np.arange(num_years)
    
    # Find indices where a reset should occur
    reset_indices = np.where(reset_mask_np)[0]
    
    # Apply reset logic efficiently
    if reset_indices.size > 0:
        last_reset = np.zeros(num_years, dtype=int)  # Track the last reset index for each year
        last_reset[reset_indices] = reset_indices  # Assign reset positions
    
        # Forward fill the last reset position
        last_reset = np.maximum.accumulate(last_reset)
    
        # Compute stand age as years since last reset
        stand_age_array = np.arange(num_years) - last_reset
    
    # Assign computed values back to DataFrame
    age['stand_age'] = stand_age_array
    
    # Drop temporary column
    age.drop(columns=['cumulative_disturbance'], inplace=True)




    # =====================================================
    # Assign Calculated Values to Each Year in the `age` DataFrame
    # =====================================================
    
    # **LAI Scalar Calculation (Canopy Closure)**
    # Adjusted stand age after reestablishment
    adjusted_age = age['stand_age'].values - params['reestablishment_phase']
    
    # LAI scalar: 0 during reestablishment, linear increase after
    age['lai_scalar'] = np.where(
        age['stand_age'].values < params['reestablishment_phase'],
        0,
        np.where(
            adjusted_age >= params['years_until_full_canopy'],
            1,
            (adjusted_age + 1) / params['years_until_full_canopy']
        )
    )

   
    # **Wood Mortality Rate Calculation**
    if params['wood_mort_mode'] == 'constant':
        age['wood_mort_rate'] = params['wood_mort_rate']
    
    elif params['wood_mort_mode'] == 'random':
        variation_factor = np.random.uniform(1 - params['wood_mort_rate_var'], 1 + params['wood_mort_rate_var'], num_years)
        age['wood_mort_rate'] = np.maximum(0, params['wood_mort_rate'] * variation_factor)
    
    elif params['wood_mort_mode'] == 'age_related':
        stand_ages = age['stand_age'].values  # Extract stand ages
    
        age['wood_mort_rate'] = np.where(
            stand_ages < params['wood_mort_start_age'], 
            params['wood_mort_rate_ini'],  # Before mortality decline starts
            np.where(stand_ages >= params['wood_mort_end_age'], 
                params['wood_mort_rate_end'],  # After mortality stabilizes
                params['wood_mort_rate_ini'] - ((stand_ages - params['wood_mort_start_age']) /
                (params['wood_mort_end_age'] - params['wood_mort_start_age'])) * 
                (params['wood_mort_rate_ini'] - params['wood_mort_rate_end'])))  # Linear interpolation


    # **Snag Fall Rate Calculation (Stored in age DataFrame)**
    if params['model_version'] in ['no_snags', 'no_snags_no_bark']:
        age['snag_fall_rate'] = 1  # All snags fall immediately
    elif params['snag_fall_mode'] == 'constant':
        age['snag_fall_rate'] = params['snag_fall_rate']  # Fixed snag fall rate
    elif params['snag_fall_mode'] == 'random':
        # Generate a different snag fall rate per year and store it
        variation_factor = np.random.uniform(1 - params['snag_fall_rate_var'], 1 + params['snag_fall_rate_var'], num_years)
        age['snag_fall_rate'] = np.maximum(0, params['snag_fall_rate'] * variation_factor)
    elif params['snag_fall_mode'] == 'exponential':   # Use the already varied lambda_snag_fall from params
        age['snag_fall_rate'] = 1 - np.exp(-params['lambda_snag_fall'])
    else:
        raise ValueError(f"Invalid snag fall mode: {params['snag_fall_mode']}")

    
    
  # Compute Wood Production Using Vectorized Function BEFORE extracting values
    age['wood_production'] = wood_production_vectorized(
        age['stand_age'].values,  # Use stand ages as input
        params['wood_production_initial'],
        params['wood_production_final'],
        params['wood_production_change_start_age'],
        params['wood_production_change_end_age'],
        params['reestablishment_phase'])

    age['wood_production'] *= age['lai_scalar']   
    
  # Extract required values as NumPy arrays (avoids repeated DataFrame lookups)
    stand_ages = age['stand_age'].values  # Define stand_ages
    wood_production = age['wood_production'].values
    bark_frac_wood = age['bark_frac_wood'].values
    wood_mort_rate = age['wood_mort_rate'].values
    disturb_mort = age['disturb_mort'].values
    wood_perc_N = age['wood_perc_N'].values
    bark_perc_N = age['bark_perc_N'].values
    snag_fall_rates = age['snag_fall_rate'].values

    
    
    # Preallocate arrays for tracking biomass
    bio_array = np.zeros(num_years)
    bark_bio_array = np.zeros(num_years)
    total_mort_wood_array = np.zeros(num_years)  
    total_mort_bark_array = np.zeros(num_years)  
    
    # Compute Live Biomass & Mortality Using a Fast Loop
    for i in range(num_years):
        if i == 0:
            # Initial Year Biomass (No mortality in year 0)
            bio_array[i] = wood_production[i] * (1 - bark_frac_wood[i])
            bark_bio_array[i] = wood_production[i] * bark_frac_wood[i]
            total_mort_wood_array[i] = 0  # No mortality yet
            total_mort_bark_array[i] = 0
        else:
            # **Mortality Calculation**
            total_mort_wood_array[i] = (bio_array[i-1] * wood_mort_rate[i-1]) + (disturb_mort[i] * bio_array[i-1])
            total_mort_bark_array[i] = (bark_bio_array[i-1] * wood_mort_rate[i-1]) + (disturb_mort[i] * bark_bio_array[i-1])
    
            # **Update Biomass (Live Biomass Calculation)**
            bio_array[i] = (bio_array[i-1] +
                            wood_production[i] * (1 - bark_frac_wood[i])) - total_mort_wood_array[i]
    
            bark_bio_array[i] = (bark_bio_array[i-1] +
                                 wood_production[i] * bark_frac_wood[i]) - total_mort_bark_array[i]

    
    # Assign computed biomass & mortality to DataFrame (vectorized)
    age['wood_biomass'] = bio_array
    age['bark_biomass'] = bark_bio_array
    age['total_mort_wood'] = total_mort_wood_array 
    age['total_mort_bark'] = total_mort_bark_array  
    
    # Compute total biomass and nitrogen content (vectorized)
    age['total_biomass'] = age['wood_biomass'] + age['bark_biomass']
    age['wood_biomass_N'] = age['wood_biomass'] * wood_perc_N * 2
    age['bark_biomass_N'] = age['bark_biomass'] * bark_perc_N * 2
    age['total_biomass_N'] = age['wood_biomass_N'] + age['bark_biomass_N']
    
    # Compute nitrogen increase (vectorized)
    biomass_N_incr = np.zeros(num_years)
    biomass_N_incr[1:] = np.diff(age['total_biomass_N'].values)
    age['total_biomass_N_incr'] = biomass_N_incr
    
    # Append results
    results.append(age)




    '''***********************************************************************************************
                                            COHORTS
    *************************************************************************************************'''

    #This section of the code initializes variables that will be used to track deadwood (necromass) cohorts over time. 
    '''Below is a detailed breakdown of both the code functionality and the ecological processes being modeled. ***EXPAND FOR MORE INFO***   
    Coding Explanation:
        This creates an empty dictionary of lists called cohort_data.
        Each key in the dictionary (e.g., 'run_number', 'snag_wood_labile', etc.) corresponds to a list ([]) 
        that will be populated with values as the simulation progresses and represents a column in the final DataFrame.
    Why is it a dictionary of lists and not NumPy arrays?
        Conceptually, this is like a 3D matrix (run × year × cohort), but instead of explicitly using a 3D NumPy array, 
        we're using lists within a dictionary to track everything.
        Why Use Lists Instead of a 3D Matrix?
        Flexibility – A 3D NumPy array would require a fixed shape ([num_runs, num_years, num_cohorts]), but here:
        The number of cohorts changes over time (new cohorts are formed, old ones decay).
        The number of years varies per simulation.
        Using lists allows us to dynamically add cohorts without worrying about preallocating memory.
        Efficient Storage – Instead of storing a huge array filled with empty values (since most years have few cohorts), 
        we only store the data that exists in lists.
        Easier Export to Pandas – Since each column in a Pandas DataFrame is essentially a list, 
        this format makes it easy to convert everything at the end:
        So, even though we don't explicitly create a 3D NumPy array, we still track:
        Run index ('run_number')
        Year index ('year')
        Cohort index ('cohort_number')
        Cohort properties (snag & DWD pools, nitrogen, etc.)
        This is logically like a 3D structure (run_number → year → cohort), but instead of preallocating space, we dynamically grow lists.
    What happens later?
        At this point, the lists are empty. During the simulation, data for each year is appended to the lists.
        Once all data is collected, the lists are converted into a Pandas DataFrame at the end of the simulation.
        We can do something like this to turn the dictionary into a dataframe cohort_df = pd.DataFrame(cohort_data)
        This automatically aligns runs, years, and cohorts into a flat table structure.
    Dictionary keys to track run number, year, cohort number and cohort age                
        run_number: Keeps track of which simulation run this data belongs to (if running multiple simulations).
        year: The simulation year for this cohort.
        cohort_number: A unique ID for each deadwood cohort (assigned when a cohort is created).
        cohort_age: The number of years since the cohort was formed.
    Ecological Meaning:
        Each time a tree dies, a new cohort of deadwood is formed.
        This cohort starts at age 0 and accumulates age each year.
        Cohorts allow us to track how long deadwood persists before it fully decomposes.'''
    
    snag_cohort_data = {
        'run_number': [], 'year': [], 'cohort_number': [], 'cohort_age': [],
        
        # Snag C
        'snag_C_wood_labile': [], 'snag_C_wood_recalc': [], 'snag_C_bark_labile': [], 'snag_C_bark_recalc': [],
                
        # Snag N
        'snag_N_wood_labile': [], 'snag_N_wood_recalc': [], 'snag_N_bark_labile': [], 'snag_N_bark_recalc': [],
                
        # Snag %N
        'snag_N_perc_wood_labile': [], 'snag_N_perc_wood_recalc': [], 'snag_N_perc_bark_labile': [], 'snag_N_perc_bark_recalc': [],
                
         # Snag C:N
        'snag_CN_wood_labile': [], 'snag_CN_wood_recalc': [], 'snag_CN_bark_labile': [], 'snag_CN_bark_recalc': [],
    }
    
    dwd_cohort_data = {
        'run_number': [], 'year': [], 'cohort_number': [], 'cohort_age': [],
        
        # dwd C
        'dwd_C_wood_labile': [], 'dwd_C_wood_recalc': [], 'dwd_C_bark_labile': [], 'dwd_C_bark_recalc': [],
        
        # dwd N
        'dwd_N_wood_labile': [], 'dwd_N_wood_recalc': [], 'dwd_N_bark_labile': [], 'dwd_N_bark_recalc': [],
        
        # dwd initial C
        'initial_dwd_C_wood_labile': [], 'initial_dwd_C_wood_recalc': [], 'initial_dwd_C_bark_labile': [], 'initial_dwd_C_bark_recalc': [],
        
        #dwd initial N
        'initial_dwd_N_wood_labile': [], 'initial_dwd_N_wood_recalc': [], 'initial_dwd_N_bark_labile': [], 'initial_dwd_N_bark_recalc': [],
        
        # dwd %N
        'dwd_N_perc_wood_labile': [], 'dwd_N_perc_wood_recalc': [], 'dwd_N_perc_bark_labile': [], 'dwd_N_perc_bark_recalc': [],
        
         # dwd C:N
        'dwd_CN_wood_labile': [], 'dwd_CN_wood_recalc': [], 'dwd_CN_bark_labile': [], 'dwd_CN_bark_recalc': [],
    }

    summary_data = {
        'run_number': [], 'year': [],
        
        # Total Snag C across all cohorts
        'snag_C_wood_labile': [], 'snag_C_wood_recalc': [], 'snag_C_bark_labile': [], 'snag_C_bark_recalc': [],
        'snag_C_wood_total': [], 'snag_C_bark_total': [], 'snag_C_total': [],
        
        # Total Snag C Lost to Decay (CO₂ loss)
        'snag_C_decay_wood_labile': [], 'snag_C_decay_wood_recalc': [], 'snag_C_decay_bark_labile': [], 'snag_C_decay_bark_recalc': [],
        'snag_C_decay_wood_total': [], 'snag_C_decay_bark_total': [], 'snag_C_decay_total': [], 'snag_C_decay_total_running_sum': [],
        
        # Total Snag C to Microbial Biomass
        'snag_C_to_microbial_biomass_wood_labile': [], 'snag_C_to_microbial_biomass_wood_recalc': [], 'snag_C_to_microbial_biomass_bark_labile': [], 
        'snag_C_to_microbial_biomass_bark_recalc': [], 'snag_C_to_microbial_biomass_wood_total': [], 'snag_C_to_microbial_biomass_bark_total': [],
        'snag_C_to_microbial_biomass_total': [], 'snag_C_to_microbial_biomass_total_running_sum': [],
  
        # Total Snag C Lost to Snag Fall across all cohorts
        'snag_fall_C_wood_labile': [], 'snag_fall_C_wood_recalc': [], 'snag_fall_C_bark_labile': [], 'snag_fall_C_bark_recalc': [],
        'snag_fall_C_wood_total': [], 'snag_fall_C_bark_total': [], 'snag_fall_C_total': [],  
    
        # Total Snag N across all cohorts
        'snag_N_wood_labile': [], 'snag_N_wood_recalc': [], 'snag_N_bark_labile': [], 'snag_N_bark_recalc': [],
        'snag_N_wood_total': [], 'snag_N_bark_total': [], 'snag_N_total': [],       
    
        # Total Snag N Mineralized/Immobilized across all cohorts
        'snag_N_min_imm_wood_labile': [], 'snag_N_min_imm_wood_recalc': [], 'snag_N_min_imm_bark_labile': [], 'snag_N_min_imm_bark_recalc': [], 
        'snag_N_min_imm_wood_total': [], 'snag_N_min_imm_bark_total': [], 'snag_N_min_imm_total': [], 
        
        # Total Snag N Lost to Snag Fall across all cohorts
        'snag_fall_N_wood_labile': [], 'snag_fall_N_wood_recalc': [], 'snag_fall_N_bark_labile': [], 'snag_fall_N_bark_recalc': [],
        'snag_fall_N_wood_total': [], 'snag_fall_N_bark_total': [], 'snag_fall_N_total': [],         
    
        # Total Snag %N across all cohorts
        'snag_N_perc_wood_labile': [], 'snag_N_perc_wood_recalc': [],'snag_N_perc_bark_labile': [], 'snag_N_perc_bark_recalc': [],
        'snag_N_perc_wood_total': [], 'snag_N_perc_bark_total': [], 'snag_N_perc_total': [], 
    
        # === Total Snag C:N across all cohorts
        'snag_CN_wood_labile': [], 'snag_CN_wood_recalc': [], 'snag_CN_bark_labile': [], 'snag_CN_bark_recalc': [], 
        'snag_CN_wood_total': [], 'snag_CN_bark_total': [], 'snag_CN_total': [],


        # === DWD === #
        # Total dwd C across all cohorts
        'dwd_C_wood_labile': [], 'dwd_C_wood_recalc': [], 'dwd_C_bark_labile': [], 'dwd_C_bark_recalc': [],
        'dwd_C_wood_total': [], 'dwd_C_bark_total': [], 'dwd_C_total': [],
        
        # Total dwd C Lost to Decay (CO₂ loss)
        'dwd_C_decay_wood_labile': [], 'dwd_C_decay_wood_recalc': [], 'dwd_C_decay_bark_labile': [], 'dwd_C_decay_bark_recalc': [],
        'dwd_C_decay_wood_total': [], 'dwd_C_decay_bark_total': [], 'dwd_C_decay_total': [], 'dwd_C_decay_total_running_sum': [],
        
        # Total dwd C to Microbial Biomass
        'dwd_C_to_microbial_biomass_wood_labile': [], 'dwd_C_to_microbial_biomass_wood_recalc': [], 'dwd_C_to_microbial_biomass_bark_labile': [], 
        'dwd_C_to_microbial_biomass_bark_recalc': [], 'dwd_C_to_microbial_biomass_wood_total': [], 'dwd_C_to_microbial_biomass_bark_total': [],
        'dwd_C_to_microbial_biomass_total': [], 'dwd_C_to_microbial_biomass_total_running_sum': [],
  

        # Total dwd N across all cohorts
        'dwd_N_wood_labile': [], 'dwd_N_wood_recalc': [], 'dwd_N_bark_labile': [], 'dwd_N_bark_recalc': [],
        'dwd_N_wood_total': [], 'dwd_N_bark_total': [], 'dwd_N_total': [],       
    
        # Total dwd N Mineralized/Immobilized across all cohorts
        'dwd_N_min_imm_wood_labile': [], 'dwd_N_min_imm_wood_recalc': [], 'dwd_N_min_imm_bark_labile': [], 'dwd_N_min_imm_bark_recalc': [], 
        'dwd_N_min_imm_wood_total': [], 'dwd_N_min_imm_bark_total': [], 'dwd_N_min_imm_total': [], 
        
        # Total dwd %N across all cohorts
        'dwd_N_perc_wood_labile': [], 'dwd_N_perc_wood_recalc': [], 'dwd_N_perc_bark_labile': [], 'dwd_N_perc_bark_recalc': [],
        'dwd_N_perc_wood_total': [], 'dwd_N_perc_bark_total': [], 'dwd_N_perc_total': [], 
    
        # === Total dwd C:N across all cohorts
        'dwd_CN_wood_labile': [], 'dwd_CN_wood_recalc': [], 'dwd_CN_bark_labile': [], 'dwd_CN_bark_recalc': [], 
        'dwd_CN_wood_total': [], 'dwd_CN_bark_total': [], 'dwd_CN_total': []
    }


    #Extract Values from the age DataFrame; .values converts each column into a NumPy array (faster than using Pandas directly).
    years = age['year'].values  
    total_mort_wood = age['total_mort_wood'].values
    total_mort_bark = age['total_mort_bark'].values
    wood_perc_N = age['wood_perc_N'].values
    bark_perc_N = age['bark_perc_N'].values
    initial_fall_fraction = age['initial_fall_fraction'].values
    recalcitrant_fraction = params['limit_value']  # Fraction of mass that is recalcitrant
    
    
    '''Initialize Key Simulation Variables ***EXPAND FOR MORE INFO***
        num_years stores how many years the simulation will run (end_year - start_year).
        cohort_counter starts at 1 and increases every time a new cohort is created.'''
    num_years = len(age)
    snag_cohort_counter = 1  # Unique ID for snags
    dwd_cohort_counter = 1  # Unique identifier for DWD cohorts

    '''Initialize an Empty List for Storing Cohorts ***EXPAND FOR MORE INFO***
        existing_cohorts = [] initializes an empty list.
        As new deadwood cohorts form, they are added to this list.
        Every year, the model will loop over this list to update decay, snag fall, and nitrogen transformations.'''
    existing_snag_cohorts = []  # Tracks all previous cohorts
    existing_dwd_cohorts = []  # Tracks all DWD cohorts


    # Track cohort existence for each (run_number, year, cohort_number)
    cohort_tracker = set()

    #I need to initialize and reset these for each run - that's why they are outside the for i in range (num_years) lo0p
    total_snag_decay_C_total_running_sum = 0
    total_snag_C_to_microbial_biomass_total_running_sum = 0
    total_dwd_decay_C_total_running_sum = 0
    total_dwd_C_to_microbial_biomass_total_running_sum = 0
    total_decay_C_total_running_sum = 0
    total_C_to_microbial_biomass_total_running_sum = 0


    '''This section of the code is responsible for: ***EXPAND FOR MORE INFO***
        Calculating intial snag cohort pools for bark, wood, labile, and recalcitrant fractions and then
        allowing for decay (CO2 loss) and continued snag fall.'''
    for i in range(num_years):
        year = years[i]  # Get the current simulation year

        # === Initialize New Cohorts from Mortality ===
        snag_C_wood = 0; snag_C_bark = 0; snag_N_wood = 0; snag_N_bark = 0
        
        # if total_mort_wood[i] > 0 or total_mort_bark[i] > 0:
        #     snag_C_wood = total_mort_wood[i] * (1 - initial_fall_fraction); snag_N_wood = total_mort_wood[i] * (1 - initial_fall_fraction) * params['wood_perc_N'] * 2
        #     snag_C_bark = total_mort_bark[i] * (1 - initial_fall_fraction); snag_N_bark = total_mort_bark[i] * (1 - initial_fall_fraction) * params['bark_perc_N'] * 2
   
        if total_mort_wood[i] > 0 or total_mort_bark[i] > 0:
            frac = initial_fall_fraction[i]  # ✅ use year-specific value
            snag_C_wood = total_mort_wood[i] * (1 - frac)
            snag_N_wood = snag_C_wood * params['wood_perc_N'] * 2
            snag_C_bark = total_mort_bark[i] * (1 - frac)
            snag_N_bark = snag_C_bark * params['bark_perc_N'] * 2

       # Create initial cohorts for labile vs. recalcitrant fractions for C, N, %N, and C:N
        new_snag_cohort = {
            'cohort_number': snag_cohort_counter,  # Assign unique cohort number
            'cohort_age': 0,  # Start new cohort at age 0

            # Store initial mass for snag fall calculations when 'constant' snag fall rate is used
            'initial_snag_C_wood': snag_C_wood, 'initial_snag_C_bark': snag_C_bark,
            
            # === Carbon Pools ===
            'snag_C_wood_labile': snag_C_wood * (1 - recalcitrant_fraction), 'snag_C_wood_recalc': snag_C_wood * recalcitrant_fraction,
            'snag_C_bark_labile': snag_C_bark * (1 - recalcitrant_fraction), 'snag_C_bark_recalc': snag_C_bark * recalcitrant_fraction,
    
            # === Nitrogen Pools ===
            'snag_N_wood_labile': snag_N_wood * (1 - recalcitrant_fraction), 'snag_N_wood_recalc': snag_N_wood * recalcitrant_fraction,
            'snag_N_bark_labile': snag_N_bark * (1 - recalcitrant_fraction), 'snag_N_bark_recalc': snag_N_bark * recalcitrant_fraction,
            
            # === %N ===
            'snag_N_perc_wood_labile': params['wood_perc_N'] if snag_C_wood > 0 else 0,
            'snag_N_perc_wood_recalc': params['wood_perc_N'] if params['limit_value'] > 0 and snag_C_wood > 0 else 0,
            'snag_N_perc_bark_labile': params['bark_perc_N'] if snag_C_bark > 0 else 0,
            'snag_N_perc_bark_recalc': params['bark_perc_N'] if params['limit_value'] > 0 and snag_C_bark > 0 else 0,   
            
            # === C:N ===
            'snag_CN_wood_labile' : 0.5 / params['wood_perc_N'] if snag_C_wood > 0 else 0,
            'snag_CN_wood_recalc' : 0.5 / params['wood_perc_N'] if params['limit_value'] > 0 and snag_C_wood > 0 else 0,
            'snag_CN_bark_labile' : 0.5 / params['bark_perc_N'] if snag_C_bark > 0 else 0,
            'snag_CN_bark_recalc' : 0.5 / params['bark_perc_N'] if params['limit_value'] > 0 and snag_C_bark > 0 else 0
         }
    
        # Store new cohort
        existing_snag_cohorts.append(new_snag_cohort)
        snag_cohort_counter += 1  # Increment cohort number
    
        # === Step 2: Process Snag Decay & Snag Fall for All Cohorts ===
        # === Carbon Loss Tracking for Snags ===
        total_snag_fall_C_wood_labile = 0; total_snag_fall_C_wood_recalc = 0; total_snag_fall_C_bark_labile = 0; total_snag_fall_C_bark_recalc = 0
        total_snag_fall_C_wood_total = 0; total_snag_fall_C_bark_total = 0; total_snag_fall_C_total = 0  
        
        total_snag_decay_C_wood_labile = 0; total_snag_decay_C_wood_recalc = 0; total_snag_decay_C_bark_labile = 0; total_snag_decay_C_bark_recalc = 0
        total_snag_decay_C_wood_total = 0; total_snag_decay_C_bark_total = 0;  total_snag_decay_C_total = 0
        
        total_snag_C_to_microbial_biomass_wood_labile = 0; total_snag_C_to_microbial_biomass_wood_recalc = 0; total_snag_C_to_microbial_biomass_bark_labile = 0
        total_snag_C_to_microbial_biomass_bark_recalc = 0; total_snag_C_to_microbial_biomass_wood_total = 0; total_snag_C_to_microbial_biomass_bark_total = 0
        total_snag_C_to_microbial_biomass_total = 0
        
        # === Snag Nitrogen Immobilization/Mineralization Tracking ===
        total_snag_N_min_imm_wood_labile = 0; total_snag_N_min_imm_wood_recalc = 0; total_snag_N_min_imm_bark_labile = 0; total_snag_N_min_imm_bark_recalc = 0
        total_snag_N_min_imm_wood_total = 0; total_snag_N_min_imm_bark_total = 0; total_snag_N_min_imm_total = 0
        
        # === Nitrogen Loss via Snag Fall ===
        total_snag_fall_N_wood_labile = 0; total_snag_fall_N_wood_recalc = 0; total_snag_fall_N_bark_labile = 0; total_snag_fall_N_bark_recalc = 0
        total_snag_fall_N_wood_total = 0; total_snag_fall_N_bark_total = 0; total_snag_fall_N_total = 0  
        
        # === Total Snag N Tracking (for %N and C:N calculations) ===
        total_snag_N_wood_labile = 0; total_snag_N_wood_recalc = 0; total_snag_N_bark_labile = 0; total_snag_N_bark_recalc = 0
        total_snag_N_wood_total = 0; total_snag_N_bark_total = 0; total_snag_N_total = 0



        for snag_cohort in existing_snag_cohorts:
            snag_cohort['cohort_age'] += 1  # Increment cohort age
    
            # === Compute Snag Decay ===
            labile_decay_factor = 1 - np.exp(-params['k_dwd_decay'] * params['snag_decay_factor'])
            recalcitrant_decay_factor = 1 - np.exp(-params['slow_decay_rate'])  # Slower decay for recalcitrant pools               
            snag_decay_C_wood_labile = min(snag_cohort['snag_C_wood_labile'] * labile_decay_factor, snag_cohort['snag_C_wood_labile'])
            snag_decay_C_wood_recalc = min(snag_cohort['snag_C_wood_recalc'] * recalcitrant_decay_factor, snag_cohort['snag_C_wood_recalc'])
            snag_decay_C_bark_labile = min(snag_cohort['snag_C_bark_labile'] * labile_decay_factor, snag_cohort['snag_C_bark_labile'])
            snag_decay_C_bark_recalc = min(snag_cohort['snag_C_bark_recalc'] * recalcitrant_decay_factor, snag_cohort['snag_C_bark_recalc'])
                                  
            # Compute microbial biomass C for each snag pool
            snag_microbial_C_wood_labile = snag_decay_C_wood_labile * (10 / params['critical_CN_wood'])
            snag_microbial_C_wood_recalc = snag_decay_C_wood_recalc * (10 / params['critical_CN_wood']) if params['limit_value'] > 0 else 0
            snag_microbial_C_bark_labile = snag_decay_C_bark_labile * (10 / params['critical_CN_bark'])
            snag_microbial_C_bark_recalc = snag_decay_C_bark_recalc * (10 / params['critical_CN_bark']) if params['limit_value'] > 0 else 0
        
            # Accumulate for summary tracking
            total_snag_C_to_microbial_biomass_wood_labile += snag_microbial_C_wood_labile
            total_snag_C_to_microbial_biomass_wood_recalc += snag_microbial_C_wood_recalc
            total_snag_C_to_microbial_biomass_bark_labile += snag_microbial_C_bark_labile
            total_snag_C_to_microbial_biomass_bark_recalc += snag_microbial_C_bark_recalc
            total_snag_C_to_microbial_biomass_wood_total = total_snag_C_to_microbial_biomass_wood_labile + total_snag_C_to_microbial_biomass_wood_recalc
            total_snag_C_to_microbial_biomass_bark_total = total_snag_C_to_microbial_biomass_bark_labile + total_snag_C_to_microbial_biomass_bark_recalc
            total_snag_C_to_microbial_biomass_total = total_snag_C_to_microbial_biomass_wood_total + total_snag_C_to_microbial_biomass_bark_total


            # === Compute Nitrogen Mineralization & Immobilization for Labile and Recalcitrant Pools ===
            if snag_cohort['snag_C_wood_labile'] > 0 and snag_cohort['snag_N_wood_labile'] > 0:
                snag_N_immobilized_wood_labile = -snag_decay_C_wood_labile * (1 / (snag_cohort['snag_C_wood_labile'] / snag_cohort['snag_N_wood_labile']) - 1 / params['critical_CN_wood'])
            else: snag_N_immobilized_wood_labile = 0


            if snag_cohort['snag_C_wood_recalc'] > 0 and snag_cohort['snag_N_wood_recalc'] > 0:
                snag_N_immobilized_wood_recalc = -snag_decay_C_wood_recalc * (1 / (snag_cohort['snag_C_wood_recalc'] / snag_cohort['snag_N_wood_recalc']) - 1 / params['critical_CN_wood'])
            else: snag_N_immobilized_wood_recalc = 0
                

            if snag_cohort['snag_C_bark_labile'] > 0 and snag_cohort['snag_N_bark_labile'] > 0:
                snag_N_immobilized_bark_labile = -snag_decay_C_bark_labile * (1 / (snag_cohort['snag_C_bark_labile'] / snag_cohort['snag_N_bark_labile']) - 1 / params['critical_CN_bark'])
            else: snag_N_immobilized_bark_labile = 0


            if snag_cohort['snag_C_bark_recalc'] > 0 and snag_cohort['snag_N_bark_recalc'] > 0:
                snag_N_immobilized_bark_recalc = -snag_decay_C_bark_recalc * (1 / (snag_cohort['snag_C_bark_recalc'] / snag_cohort['snag_N_bark_recalc']) - 1 / params['critical_CN_bark'])
            else: snag_N_immobilized_bark_recalc = 0


            # Track Total Nitrogen Immobilized for Summary Reporting
            total_snag_N_min_imm_wood_labile += snag_N_immobilized_wood_labile
            total_snag_N_min_imm_bark_labile += snag_N_immobilized_bark_labile
            total_snag_N_min_imm_wood_recalc += snag_N_immobilized_wood_recalc
            total_snag_N_min_imm_bark_recalc += snag_N_immobilized_bark_recalc
                

            # === Subtract Decay Loss (Prevent Negative Values) ===
            snag_cohort['snag_C_wood_labile'] = max(0, snag_cohort['snag_C_wood_labile'] - snag_decay_C_wood_labile)
            snag_cohort['snag_C_wood_recalc'] = max(0, snag_cohort['snag_C_wood_recalc'] - snag_decay_C_wood_recalc)
            snag_cohort['snag_C_bark_labile'] = max(0, snag_cohort['snag_C_bark_labile'] - snag_decay_C_bark_labile)
            snag_cohort['snag_C_bark_recalc'] = max(0, snag_cohort['snag_C_bark_recalc'] - snag_decay_C_bark_recalc)

            # Track Total Decay Loss for Summary Data
            total_snag_decay_C_wood_labile += snag_decay_C_wood_labile
            total_snag_decay_C_wood_recalc += snag_decay_C_wood_recalc
            total_snag_decay_C_bark_labile += snag_decay_C_bark_labile
            total_snag_decay_C_bark_recalc += snag_decay_C_bark_recalc
            total_snag_decay_C_wood_total = total_snag_decay_C_wood_labile + total_snag_decay_C_wood_recalc
            total_snag_decay_C_bark_total = total_snag_decay_C_bark_labile + total_snag_decay_C_bark_recalc
            total_snag_decay_C_total = total_snag_decay_C_wood_total + total_snag_decay_C_bark_total

               
            # === Update Snag Nitrogen Pools ===
            snag_cohort['snag_N_wood_labile'] = snag_cohort['snag_N_wood_labile'] + snag_N_immobilized_wood_labile
            snag_cohort['snag_N_bark_labile'] = snag_cohort['snag_N_bark_labile'] + snag_N_immobilized_bark_labile
            snag_cohort['snag_N_wood_recalc'] = snag_cohort['snag_N_wood_recalc'] + snag_N_immobilized_wood_recalc
            snag_cohort['snag_N_bark_recalc'] = snag_cohort['snag_N_bark_recalc'] + snag_N_immobilized_bark_recalc
        
            # === Compute New %N for Each Pool ===
            snag_cohort['snag_N_perc_wood_labile'] = snag_cohort['snag_N_wood_labile'] / (snag_cohort['snag_C_wood_labile'] * 2) if snag_cohort['snag_C_wood_labile'] > 0 else 0
            snag_cohort['snag_N_perc_wood_recalc'] = snag_cohort['snag_N_wood_recalc'] / (snag_cohort['snag_C_wood_recalc'] * 2) if snag_cohort['snag_C_wood_recalc'] > 0 else 0
            snag_cohort['snag_N_perc_bark_labile'] = snag_cohort['snag_N_bark_labile'] / (snag_cohort['snag_C_bark_labile'] * 2) if snag_cohort['snag_C_bark_labile'] > 0 else 0
            snag_cohort['snag_N_perc_bark_recalc'] = snag_cohort['snag_N_bark_recalc'] / (snag_cohort['snag_C_bark_recalc'] * 2) if snag_cohort['snag_C_bark_recalc'] > 0 else 0

            # ==== Compute New C:N RATIOS for each Pool ====           
            snag_cohort['snag_CN_wood_labile'] = snag_cohort['snag_C_wood_labile'] / snag_cohort['snag_N_wood_labile'] if snag_cohort['snag_N_wood_labile'] > 0 else None
            snag_cohort['snag_CN_wood_recalc'] = snag_cohort['snag_C_wood_recalc'] / snag_cohort['snag_N_wood_recalc'] if snag_cohort['snag_N_wood_recalc'] > 0 else None
            snag_cohort['snag_CN_bark_labile'] = snag_cohort['snag_C_bark_labile'] / snag_cohort['snag_N_bark_labile'] if snag_cohort['snag_N_bark_labile'] > 0 else None
            snag_cohort['snag_CN_bark_recalc'] = snag_cohort['snag_C_bark_recalc'] / snag_cohort['snag_N_bark_recalc'] if snag_cohort['snag_N_bark_recalc'] > 0 else None
        


            # === Extract the Precomputed Snag Fall Rate for This Year ===
            snag_fall_rate = snag_fall_rates[i]  # Fast lookup for the current year
            
            # === Compute Snag Fall ===
            if params['snag_fall_mode'] in ['constant', 'random']:
                snag_fall_C_wood = min(snag_cohort['initial_snag_C_wood'] * snag_fall_rate, snag_cohort['snag_C_wood_labile'] + snag_cohort['snag_C_wood_recalc'])
                snag_fall_C_bark = min(snag_cohort['initial_snag_C_bark'] * snag_fall_rate, snag_cohort['snag_C_bark_labile'] + snag_cohort['snag_C_bark_recalc'])
            elif params['snag_fall_mode'] == 'exponential':
                snag_fall_C_wood = min(snag_cohort['snag_C_wood_labile'] * (1 - np.exp(-params['lambda_snag_fall'])), snag_cohort['snag_C_wood_labile'])
                snag_fall_C_bark = min(snag_cohort['snag_C_bark_labile'] * (1 - np.exp(-params['lambda_snag_fall'])), snag_cohort['snag_C_bark_labile'])
            else:
                raise ValueError(f"Invalid snag fall mode: {params['snag_fall_mode']}")



            # Split Snag Fall into Labile & Recalcitrant
            wood_total = snag_cohort['snag_C_wood_labile'] + snag_cohort['snag_C_wood_recalc']
            bark_total = snag_cohort['snag_C_bark_labile'] + snag_cohort['snag_C_bark_recalc']
        
            snag_fall_C_wood_labile = snag_fall_C_wood * (snag_cohort['snag_C_wood_labile'] / wood_total) if wood_total > 0 else 0
            snag_fall_C_wood_recalc = snag_fall_C_wood * (snag_cohort['snag_C_wood_recalc'] / wood_total) if wood_total > 0 else 0
            snag_fall_C_bark_labile = snag_fall_C_bark * (snag_cohort['snag_C_bark_labile'] / bark_total) if bark_total > 0 else 0
            snag_fall_C_bark_recalc = snag_fall_C_bark * (snag_cohort['snag_C_bark_recalc'] / bark_total) if bark_total > 0 else 0
          

            ### ==== COMPUTE SNAG FALL N LOSSES ==== ###
            snag_fall_N_wood_labile = snag_fall_C_wood_labile * (snag_cohort['snag_N_wood_labile'] / snag_cohort['snag_C_wood_labile']) if snag_cohort['snag_C_wood_labile'] > 0 else 0
            snag_fall_N_wood_recalc = snag_fall_C_wood_recalc * (snag_cohort['snag_N_wood_recalc'] / snag_cohort['snag_C_wood_recalc']) if snag_cohort['snag_C_wood_recalc'] > 0 else 0
            snag_fall_N_bark_labile = snag_fall_C_bark_labile * (snag_cohort['snag_N_bark_labile'] / snag_cohort['snag_C_bark_labile']) if snag_cohort['snag_C_bark_labile'] > 0 else 0
            snag_fall_N_bark_recalc = snag_fall_C_bark_recalc * (snag_cohort['snag_N_bark_recalc'] / snag_cohort['snag_C_bark_recalc']) if snag_cohort['snag_C_bark_recalc'] > 0 else 0
                     

            # Apply snag fall (subtract from cohort pools, ensuring values don’t go negative)
            snag_cohort['snag_C_wood_labile'] = max(0, snag_cohort['snag_C_wood_labile'] - snag_fall_C_wood_labile)
            snag_cohort['snag_C_wood_recalc'] = max(0, snag_cohort['snag_C_wood_recalc'] - snag_fall_C_wood_recalc)
            snag_cohort['snag_C_bark_labile'] = max(0, snag_cohort['snag_C_bark_labile'] - snag_fall_C_bark_labile)
            snag_cohort['snag_C_bark_recalc'] = max(0, snag_cohort['snag_C_bark_recalc'] - snag_fall_C_bark_recalc)
            
            # Track total snag fall loss for summary reporting
            total_snag_fall_C_wood_labile += snag_fall_C_wood_labile
            total_snag_fall_C_wood_recalc += snag_fall_C_wood_recalc
            total_snag_fall_C_bark_labile += snag_fall_C_bark_labile
            total_snag_fall_C_bark_recalc += snag_fall_C_bark_recalc


            # Track total snag fall N loss for summary reporting
            total_snag_fall_N_wood_labile += snag_fall_N_wood_labile
            total_snag_fall_N_wood_recalc += snag_fall_N_wood_recalc
            total_snag_fall_N_bark_labile += snag_fall_N_bark_labile
            total_snag_fall_N_bark_recalc += snag_fall_N_bark_recalc

            
            # Update N pools in each cohort for snag fall loss
            snag_cohort['snag_N_wood_labile'] = max(0, snag_cohort['snag_N_wood_labile'] - snag_fall_N_wood_labile)
            snag_cohort['snag_N_wood_recalc'] = max(0, snag_cohort['snag_N_wood_recalc'] - snag_fall_N_wood_recalc)
            snag_cohort['snag_N_bark_labile'] = max(0, snag_cohort['snag_N_bark_labile'] - snag_fall_N_bark_labile)
            snag_cohort['snag_N_bark_recalc'] = max(0, snag_cohort['snag_N_bark_recalc'] - snag_fall_N_bark_recalc)


            ### Accumulate Nitrogen for Summary Tracking
            total_snag_N_wood_labile += snag_cohort['snag_N_wood_labile']
            total_snag_N_wood_recalc += snag_cohort['snag_N_wood_recalc']
            total_snag_N_bark_labile += snag_cohort['snag_N_bark_labile']
            total_snag_N_bark_recalc += snag_cohort['snag_N_bark_recalc']
            total_snag_N_wood_total = total_snag_N_wood_labile + total_snag_N_wood_recalc
            total_snag_N_bark_total = total_snag_N_bark_labile + total_snag_N_bark_recalc
            total_snag_N_total = total_snag_N_wood_total + total_snag_N_bark_total
            
            # === STORE COHORT-LEVEL DATA ===
            snag_cohort_data['run_number'].append(run + 1)
            snag_cohort_data['year'].append(year)
            snag_cohort_data['cohort_number'].append(snag_cohort['cohort_number'])
            snag_cohort_data['cohort_age'].append(snag_cohort['cohort_age'])
            
            # Store Carbon & Nitrogen pools
            snag_cohort_data['snag_C_wood_labile'].append(snag_cohort['snag_C_wood_labile'])
            snag_cohort_data['snag_C_wood_recalc'].append(snag_cohort['snag_C_wood_recalc'])
            snag_cohort_data['snag_C_bark_labile'].append(snag_cohort['snag_C_bark_labile'])
            snag_cohort_data['snag_C_bark_recalc'].append(snag_cohort['snag_C_bark_recalc'])
            
            snag_cohort_data['snag_N_wood_labile'].append(snag_cohort['snag_N_wood_labile'])
            snag_cohort_data['snag_N_wood_recalc'].append(snag_cohort['snag_N_wood_recalc'])
            snag_cohort_data['snag_N_bark_labile'].append(snag_cohort['snag_N_bark_labile'])
            snag_cohort_data['snag_N_bark_recalc'].append(snag_cohort['snag_N_bark_recalc'])
            
            # Store %N
            snag_cohort_data['snag_N_perc_wood_labile'].append(snag_cohort['snag_N_perc_wood_labile'])
            snag_cohort_data['snag_N_perc_wood_recalc'].append(snag_cohort['snag_N_perc_wood_recalc'])
            snag_cohort_data['snag_N_perc_bark_labile'].append(snag_cohort['snag_N_perc_bark_labile'])
            snag_cohort_data['snag_N_perc_bark_recalc'].append(snag_cohort['snag_N_perc_bark_recalc'])
            
            # Store C:N Ratio
            snag_cohort_data['snag_CN_wood_labile'].append(snag_cohort['snag_CN_wood_labile'])
            snag_cohort_data['snag_CN_wood_recalc'].append(snag_cohort['snag_CN_wood_recalc'])
            snag_cohort_data['snag_CN_bark_labile'].append(snag_cohort['snag_CN_bark_labile'])
            snag_cohort_data['snag_CN_bark_recalc'].append(snag_cohort['snag_CN_bark_recalc'])
                    
        
        total_snag_decay_C_total_running_sum += total_snag_decay_C_total
        total_snag_C_to_microbial_biomass_total_running_sum += total_snag_C_to_microbial_biomass_total                                                 
            
        # === Store Summary Data (AFTER Processing Snag Decay & Fall Each Year) ===
        summary_data['run_number'].append(run + 1)
        summary_data['year'].append(year)
        
        # Standing Snag Carbon (After Decay & Snag Fall)
        summary_data['snag_C_wood_labile'].append(sum(snag_cohort['snag_C_wood_labile'] for snag_cohort in existing_snag_cohorts))
        summary_data['snag_C_wood_recalc'].append(sum(snag_cohort['snag_C_wood_recalc'] for snag_cohort in existing_snag_cohorts))
        summary_data['snag_C_bark_labile'].append(sum(snag_cohort['snag_C_bark_labile'] for snag_cohort in existing_snag_cohorts))
        summary_data['snag_C_bark_recalc'].append(sum(snag_cohort['snag_C_bark_recalc'] for snag_cohort in existing_snag_cohorts))
        summary_data['snag_C_wood_total'].append(summary_data['snag_C_wood_labile'][i] + summary_data['snag_C_wood_recalc'][i])
        summary_data['snag_C_bark_total'].append(summary_data['snag_C_bark_labile'][i] + summary_data['snag_C_bark_recalc'][i])
        summary_data['snag_C_total'].append(summary_data['snag_C_wood_total'][i] + summary_data['snag_C_bark_total'][i])
    
        # Snag Decay (CO₂ Loss)
        summary_data['snag_C_decay_wood_labile'].append(total_snag_decay_C_wood_labile)
        summary_data['snag_C_decay_wood_recalc'].append(total_snag_decay_C_wood_recalc)
        summary_data['snag_C_decay_bark_labile'].append(total_snag_decay_C_bark_labile)
        summary_data['snag_C_decay_bark_recalc'].append(total_snag_decay_C_bark_recalc)
        summary_data['snag_C_decay_wood_total'].append(total_snag_decay_C_wood_total)
        summary_data['snag_C_decay_bark_total'].append(total_snag_decay_C_bark_total)
        summary_data['snag_C_decay_total'].append(total_snag_decay_C_total)
        summary_data['snag_C_decay_total_running_sum'].append(total_snag_decay_C_total_running_sum)

        # === Snag Carbon Lost to Microbial Biomass ===
        summary_data['snag_C_to_microbial_biomass_wood_labile'].append(total_snag_C_to_microbial_biomass_wood_labile)
        summary_data['snag_C_to_microbial_biomass_wood_recalc'].append(total_snag_C_to_microbial_biomass_wood_recalc)
        summary_data['snag_C_to_microbial_biomass_bark_labile'].append(total_snag_C_to_microbial_biomass_bark_labile)
        summary_data['snag_C_to_microbial_biomass_bark_recalc'].append(total_snag_C_to_microbial_biomass_bark_recalc)
        summary_data['snag_C_to_microbial_biomass_wood_total'].append(total_snag_C_to_microbial_biomass_wood_total)
        summary_data['snag_C_to_microbial_biomass_bark_total'].append(total_snag_C_to_microbial_biomass_bark_total)
        summary_data['snag_C_to_microbial_biomass_total'].append(total_snag_C_to_microbial_biomass_total)
        summary_data['snag_C_to_microbial_biomass_total_running_sum'].append(total_snag_C_to_microbial_biomass_total_running_sum)

        # Snag Fall C Totals
        summary_data['snag_fall_C_wood_labile'].append(total_snag_fall_C_wood_labile)
        summary_data['snag_fall_C_wood_recalc'].append(total_snag_fall_C_wood_recalc)
        summary_data['snag_fall_C_bark_labile'].append(total_snag_fall_C_bark_labile)
        summary_data['snag_fall_C_bark_recalc'].append(total_snag_fall_C_bark_recalc)
        summary_data['snag_fall_C_wood_total'].append(total_snag_fall_C_wood_labile + total_snag_fall_C_wood_recalc)
        summary_data['snag_fall_C_bark_total'].append(total_snag_fall_C_bark_labile + total_snag_fall_C_bark_recalc)
        summary_data['snag_fall_C_total'].append(summary_data['snag_fall_C_wood_total'][i] + summary_data['snag_fall_C_bark_total'][i])

        # Snag N Totals
        summary_data['snag_N_wood_labile'].append(total_snag_N_wood_labile)
        summary_data['snag_N_wood_recalc'].append(total_snag_N_wood_recalc)
        summary_data['snag_N_bark_labile'].append(total_snag_N_bark_labile)
        summary_data['snag_N_bark_recalc'].append(total_snag_N_bark_recalc)
        summary_data['snag_N_wood_total'].append(total_snag_N_wood_total)
        summary_data['snag_N_bark_total'].append(total_snag_N_bark_total)
        summary_data['snag_N_total'].append(total_snag_N_total)      
        
        # Snag N Immobilization Totals
        summary_data['snag_N_min_imm_wood_labile'].append(total_snag_N_min_imm_wood_labile)
        summary_data['snag_N_min_imm_wood_recalc'].append(total_snag_N_min_imm_wood_recalc)
        summary_data['snag_N_min_imm_bark_labile'].append(total_snag_N_min_imm_bark_labile)
        summary_data['snag_N_min_imm_bark_recalc'].append(total_snag_N_min_imm_bark_recalc)
        summary_data['snag_N_min_imm_wood_total'].append(summary_data['snag_N_min_imm_wood_labile'][i] + summary_data['snag_N_min_imm_wood_recalc'][i])
        summary_data['snag_N_min_imm_bark_total'].append(summary_data['snag_N_min_imm_bark_labile'][i] + summary_data['snag_N_min_imm_bark_recalc'][i])
        summary_data['snag_N_min_imm_total'].append(summary_data['snag_N_min_imm_wood_total'][i] + summary_data['snag_N_min_imm_bark_total'][i])
      
        # Snag Fall N Totals
        summary_data['snag_fall_N_wood_labile'].append(total_snag_fall_N_wood_labile)
        summary_data['snag_fall_N_wood_recalc'].append(total_snag_fall_N_wood_recalc)
        summary_data['snag_fall_N_bark_labile'].append(total_snag_fall_N_bark_labile)
        summary_data['snag_fall_N_bark_recalc'].append(total_snag_fall_N_bark_recalc)
        summary_data['snag_fall_N_wood_total'].append(total_snag_fall_N_wood_labile + total_snag_fall_N_wood_recalc)
        summary_data['snag_fall_N_bark_total'].append(total_snag_fall_N_bark_labile + total_snag_fall_N_bark_recalc)
        summary_data['snag_fall_N_total'].append(summary_data['snag_fall_N_wood_total'][i] + summary_data['snag_fall_N_bark_total'][i])


        # Compute total snag nitrogen and carbon (sum of all pools)
        total_snag_N = sum(snag_cohort['snag_N_wood_labile'] + snag_cohort['snag_N_wood_recalc'] + snag_cohort['snag_N_bark_labile'] + snag_cohort['snag_N_bark_recalc']
            for snag_cohort in existing_snag_cohorts)
        
        total_snag_C = sum(snag_cohort['snag_C_wood_labile'] + snag_cohort['snag_C_wood_recalc'] + snag_cohort['snag_C_bark_labile'] + snag_cohort['snag_C_bark_recalc']
            for snag_cohort in existing_snag_cohorts)
        

        # Compute %N for Each Pool (Ensure No Division by Zero)
        summary_data['snag_N_perc_wood_labile'].append((summary_data['snag_N_wood_labile'][i] / summary_data['snag_C_wood_labile'][i]) * 2
            if summary_data['snag_C_wood_labile'][i] > 0 else 0)
        
        summary_data['snag_N_perc_wood_recalc'].append((summary_data['snag_N_wood_recalc'][i] / summary_data['snag_C_wood_recalc'][i]) * 2
            if summary_data['snag_C_wood_recalc'][i] > 0 else 0)
        
        summary_data['snag_N_perc_bark_labile'].append((summary_data['snag_N_bark_labile'][i] / summary_data['snag_C_bark_labile'][i]) * 2
            if summary_data['snag_C_bark_labile'][i] > 0 else 0)
        
        summary_data['snag_N_perc_bark_recalc'].append((summary_data['snag_N_bark_recalc'][i] / summary_data['snag_C_bark_recalc'][i]) * 2
            if summary_data['snag_C_bark_recalc'][i] > 0 else 0)
        
        summary_data['snag_N_perc_wood_total'].append((summary_data['snag_N_wood_total'][i] / summary_data['snag_C_wood_total'][i]) * 2
            if summary_data['snag_C_wood_total'][i] > 0 else 0)
        
        summary_data['snag_N_perc_bark_total'].append((summary_data['snag_N_bark_total'][i] / summary_data['snag_C_bark_total'][i]) * 2
            if summary_data['snag_C_bark_total'][i] > 0 else 0)
        
        summary_data['snag_N_perc_total'].append((summary_data['snag_N_total'][i] / summary_data['snag_C_total'][i]) * 2
            if summary_data['snag_C_total'][i] > 0 else 0)
        
        
        # === Compute C:N Ratio for Each Pool (Ensure No Division by Zero) ===
        summary_data['snag_CN_wood_labile'].append(summary_data['snag_C_wood_labile'][i] / summary_data['snag_N_wood_labile'][i]
            if summary_data['snag_N_wood_labile'][i] > 0 else 0)
        
        summary_data['snag_CN_wood_recalc'].append(summary_data['snag_C_wood_recalc'][i] / summary_data['snag_N_wood_recalc'][i]
            if summary_data['snag_N_wood_recalc'][i] > 0 else 0)
        
        summary_data['snag_CN_bark_labile'].append(summary_data['snag_C_bark_labile'][i] / summary_data['snag_N_bark_labile'][i]
            if summary_data['snag_N_bark_labile'][i] > 0 else 0)
        
        summary_data['snag_CN_bark_recalc'].append(summary_data['snag_C_bark_recalc'][i] / summary_data['snag_N_bark_recalc'][i]
            if summary_data['snag_N_bark_recalc'][i] > 0 else 0)
        
        summary_data['snag_CN_wood_total'].append(summary_data['snag_C_wood_total'][i] / summary_data['snag_N_wood_total'][i]
            if summary_data['snag_N_wood_total'][i] > 0 else 0)
        
        summary_data['snag_CN_bark_total'].append(summary_data['snag_C_bark_total'][i] / summary_data['snag_N_bark_total'][i]
            if summary_data['snag_N_bark_total'][i] > 0 else 0)

        summary_data['snag_CN_total'].append(summary_data['snag_C_total'][i] / summary_data['snag_N_total'][i]
            if summary_data['snag_N_total'][i] > 0 else 0)






        '''*************************************************************************************************
                                      DWD COHORTS        
        ***************************************************************************************************'''        
        
        year = years[i]  # Get the current simulation year

        # === Initialize New Cohorts from Mortality ===
        dwd_C_wood_labile = 0; dwd_C_wood_recalc = 0; dwd_C_bark_labile = 0; dwd_C_bark_recalc = 0
        dwd_N_wood_labile = 0; dwd_N_wood_recalc = 0; dwd_N_bark_labile = 0; dwd_N_bark_recalc = 0

        # Compute annual input to DWD pools from mortality and snag fall        
        # Separate initialization for labile and recalcitrant fractions
        dwd_C_wood_labile = (total_mort_wood[i] * initial_fall_fraction[i] * (1 - recalcitrant_fraction)) + summary_data['snag_fall_C_wood_labile'][i]
        dwd_C_wood_recalc = (total_mort_wood[i] * initial_fall_fraction[i] * recalcitrant_fraction) + summary_data['snag_fall_C_wood_recalc'][i]
        
        dwd_C_bark_labile = (total_mort_bark[i] * initial_fall_fraction[i] * (1 - recalcitrant_fraction)) + summary_data['snag_fall_C_bark_labile'][i]
        dwd_C_bark_recalc = (total_mort_bark[i] * initial_fall_fraction[i] * recalcitrant_fraction) + summary_data['snag_fall_C_bark_recalc'][i]
            
        dwd_N_wood_labile = (total_mort_wood[i] * 2 * params['wood_perc_N'] * initial_fall_fraction[i] * (1 - recalcitrant_fraction)) + summary_data['snag_fall_N_wood_labile'][i]
        dwd_N_wood_recalc = (total_mort_wood[i] * 2 * params['wood_perc_N'] * initial_fall_fraction[i] * recalcitrant_fraction) + summary_data['snag_fall_N_wood_recalc'][i]
        
        dwd_N_bark_labile = (total_mort_bark[i] * 2 * params['bark_perc_N'] * initial_fall_fraction[i] * (1 - recalcitrant_fraction)) + summary_data['snag_fall_N_bark_labile'][i]
        dwd_N_bark_recalc = (total_mort_bark[i] * 2 * params['bark_perc_N'] * initial_fall_fraction[i] * recalcitrant_fraction) + summary_data['snag_fall_N_bark_recalc'][i]
   

       # Create initial cohorts for labile vs. recalcitrant fractions for C, N, %N, and C:N
        new_dwd_cohort = {
            'cohort_number': dwd_cohort_counter,  # Assign unique cohort number
            'cohort_age': 0,  # Start new cohort at age 0

            # Store initial C mass for dwd fall calculations
            'initial_dwd_C_wood_labile': dwd_C_wood_labile, 'initial_dwd_C_wood_recalc': dwd_C_wood_recalc,
            'initial_dwd_C_bark_labile': dwd_C_bark_labile, 'initial_dwd_C_bark_recalc': dwd_C_bark_recalc,

            # Store initial N mass for dwd fall calculations
            'initial_dwd_N_wood_labile': dwd_N_wood_labile, 'initial_dwd_N_wood_recalc': dwd_N_wood_recalc,
            'initial_dwd_N_bark_labile': dwd_N_bark_labile, 'initial_dwd_N_bark_recalc': dwd_N_bark_recalc,
        
        
            # === Carbon Pools ===
            'dwd_C_wood_labile':  dwd_C_wood_labile, 'dwd_C_wood_recalc': dwd_C_wood_recalc,
            'dwd_C_bark_labile': dwd_C_bark_labile, 'dwd_C_bark_recalc': dwd_C_bark_recalc,
    
            # === Nitrogen Pools ===
            'dwd_N_wood_labile': dwd_N_wood_labile, 'dwd_N_wood_recalc': dwd_N_wood_recalc,
            'dwd_N_bark_labile':dwd_N_bark_labile, 'dwd_N_bark_recalc': dwd_N_bark_recalc,
            
            # === %N ===
            'dwd_N_perc_wood_labile': dwd_N_wood_labile / (2 * dwd_C_wood_labile)  if dwd_C_wood_labile > 0 else 0,
            'dwd_N_perc_wood_recalc': dwd_N_wood_recalc / (2 * dwd_C_wood_recalc) if params['limit_value'] > 0 and dwd_C_wood_recalc > 0 else 0,
            'dwd_N_perc_bark_labile': dwd_N_bark_labile / (2 * dwd_C_bark_labile) if dwd_C_bark_labile > 0 else 0,
            'dwd_N_perc_bark_recalc': dwd_N_bark_recalc / (2 * dwd_C_bark_recalc) if params['limit_value'] > 0 and dwd_C_bark_recalc > 0 else 0,   
            
            # === C:N ===
            'dwd_CN_wood_labile' : dwd_C_wood_labile / dwd_N_wood_labile if dwd_N_wood_labile > 0 else 0,
            'dwd_CN_wood_recalc' : dwd_C_wood_recalc / dwd_N_wood_recalc if params['limit_value'] > 0 and dwd_N_wood_recalc > 0 else 0,
            'dwd_CN_bark_labile' : dwd_C_bark_labile / dwd_N_bark_labile if dwd_N_bark_labile > 0 else 0,
            'dwd_CN_bark_recalc' : dwd_C_bark_recalc / dwd_N_bark_recalc if params['limit_value'] > 0 and dwd_N_bark_recalc > 0 else 0
         }
    
        # Store new cohort
        existing_dwd_cohorts.append(new_dwd_cohort)
        dwd_cohort_counter += 1  # Increment dwd cohort number
    
        # === Process dwd Decay  for All Cohorts ===
        # === Carbon Loss Tracking for Dwd ===
                
        total_dwd_decay_C_wood_labile = 0; total_dwd_decay_C_wood_recalc = 0; total_dwd_decay_C_bark_labile = 0; total_dwd_decay_C_bark_recalc = 0
        total_dwd_decay_C_wood_total = 0; total_dwd_decay_C_bark_total = 0;  total_dwd_decay_C_total = 0
        
        total_dwd_C_to_microbial_biomass_wood_labile = 0; total_dwd_C_to_microbial_biomass_wood_recalc = 0; total_dwd_C_to_microbial_biomass_bark_labile = 0
        total_dwd_C_to_microbial_biomass_bark_recalc = 0; total_dwd_C_to_microbial_biomass_wood_total = 0; total_dwd_C_to_microbial_biomass_bark_total = 0
        total_dwd_C_to_microbial_biomass_total = 0
        
        # === dwd Nitrogen Immobilization/Mineralization Tracking ===
        total_dwd_N_min_imm_wood_labile = 0; total_dwd_N_min_imm_wood_recalc = 0; total_dwd_N_min_imm_bark_labile = 0; total_dwd_N_min_imm_bark_recalc = 0
        total_dwd_N_min_imm_wood_total = 0; total_dwd_N_min_imm_bark_total = 0; total_dwd_N_min_imm_total = 0
        
        # === Total dwd N Tracking (for %N and C:N calculations) ===
        total_dwd_N_wood_labile = 0; total_dwd_N_wood_recalc = 0; total_dwd_N_bark_labile = 0; total_dwd_N_bark_recalc = 0
        total_dwd_N_wood_total = 0; total_dwd_N_bark_total = 0; total_dwd_N_total = 0
               
        
        for dwd_cohort in existing_dwd_cohorts:
            dwd_cohort['cohort_age'] += 1  # Increment cohort age
    
            # === Compute Snag Decay ===
            labile_decay_factor = 1 - np.exp(-params['k_dwd_decay'])
            recalcitrant_decay_factor = 1 - np.exp(-params['slow_decay_rate'])  # Slower decay for recalcitrant pools               
            dwd_decay_C_wood_labile = min(dwd_cohort['dwd_C_wood_labile'] * labile_decay_factor, dwd_cohort['dwd_C_wood_labile'])
            dwd_decay_C_wood_recalc = min(dwd_cohort['dwd_C_wood_recalc'] * recalcitrant_decay_factor, dwd_cohort['dwd_C_wood_recalc'])
            dwd_decay_C_bark_labile = min(dwd_cohort['dwd_C_bark_labile'] * labile_decay_factor, dwd_cohort['dwd_C_bark_labile'])
            dwd_decay_C_bark_recalc = min(dwd_cohort['dwd_C_bark_recalc'] * recalcitrant_decay_factor, dwd_cohort['dwd_C_bark_recalc'])
                                  
            # Compute microbial biomass C for each dwd pool
            dwd_microbial_C_wood_labile = dwd_decay_C_wood_labile * (10 / params['critical_CN_wood'])
            dwd_microbial_C_wood_recalc = dwd_decay_C_wood_recalc * (10 / params['critical_CN_wood']) if params['limit_value'] > 0 else 0
            dwd_microbial_C_bark_labile = dwd_decay_C_bark_labile * (10 / params['critical_CN_bark'])
            dwd_microbial_C_bark_recalc = dwd_decay_C_bark_recalc * (10 / params['critical_CN_bark']) if params['limit_value'] > 0 else 0
        
            # Accumulate for summary tracking
            total_dwd_C_to_microbial_biomass_wood_labile += dwd_microbial_C_wood_labile
            total_dwd_C_to_microbial_biomass_wood_recalc += dwd_microbial_C_wood_recalc
            total_dwd_C_to_microbial_biomass_bark_labile += dwd_microbial_C_bark_labile
            total_dwd_C_to_microbial_biomass_bark_recalc += dwd_microbial_C_bark_recalc
            total_dwd_C_to_microbial_biomass_wood_total = total_dwd_C_to_microbial_biomass_wood_labile + total_dwd_C_to_microbial_biomass_wood_recalc
            total_dwd_C_to_microbial_biomass_bark_total = total_dwd_C_to_microbial_biomass_bark_labile + total_dwd_C_to_microbial_biomass_bark_recalc
            total_dwd_C_to_microbial_biomass_total = total_dwd_C_to_microbial_biomass_wood_total + total_dwd_C_to_microbial_biomass_bark_total
        
                
        
            # === Compute Nitrogen Mineralization & Immobilization for Labile and Recalcitrant Pools ===

                
            if dwd_cohort['dwd_C_wood_labile'] > 0 and dwd_cohort['dwd_N_wood_labile'] > 0:
                dwd_N_immobilized_wood_labile = -dwd_decay_C_wood_labile * (1 / (dwd_cohort['dwd_C_wood_labile'] / dwd_cohort['dwd_N_wood_labile']) - 1 / params['critical_CN_wood'])
            else: dwd_N_immobilized_wood_labile = 0


            if dwd_cohort['dwd_C_wood_recalc'] > 0 and dwd_cohort['dwd_N_wood_recalc'] > 0:
                dwd_N_immobilized_wood_recalc = -dwd_decay_C_wood_recalc * (1 / (dwd_cohort['dwd_C_wood_recalc'] / dwd_cohort['dwd_N_wood_recalc']) - 1 / params['critical_CN_wood'])
            else: dwd_N_immobilized_wood_recalc = 0
                

            if dwd_cohort['dwd_C_bark_labile'] > 0 and dwd_cohort['dwd_N_bark_labile'] > 0:
                dwd_N_immobilized_bark_labile = -dwd_decay_C_bark_labile * (1 / (dwd_cohort['dwd_C_bark_labile'] / dwd_cohort['dwd_N_bark_labile']) - 1 / params['critical_CN_bark'])
            else: dwd_N_immobilized_bark_labile = 0


            if dwd_cohort['dwd_C_bark_recalc'] > 0 and dwd_cohort['dwd_N_bark_recalc'] > 0:
                dwd_N_immobilized_bark_recalc = -dwd_decay_C_bark_recalc * (1 / (dwd_cohort['dwd_C_bark_recalc'] / dwd_cohort['dwd_N_bark_recalc']) - 1 / params['critical_CN_bark'])
            else: dwd_N_immobilized_bark_recalc = 0



            # Track Total Nitrogen Immobilized for Summary Reporting
            total_dwd_N_min_imm_wood_labile += dwd_N_immobilized_wood_labile
            total_dwd_N_min_imm_bark_labile += dwd_N_immobilized_bark_labile
            total_dwd_N_min_imm_wood_recalc += dwd_N_immobilized_wood_recalc
            total_dwd_N_min_imm_bark_recalc += dwd_N_immobilized_bark_recalc

            # === Subtract Decay Loss (Prevent Negative Values) ===
            dwd_cohort['dwd_C_wood_labile'] = max(0, dwd_cohort['dwd_C_wood_labile'] - dwd_decay_C_wood_labile)
            dwd_cohort['dwd_C_wood_recalc'] = max(0, dwd_cohort['dwd_C_wood_recalc'] - dwd_decay_C_wood_recalc)
            dwd_cohort['dwd_C_bark_labile'] = max(0, dwd_cohort['dwd_C_bark_labile'] - dwd_decay_C_bark_labile)
            dwd_cohort['dwd_C_bark_recalc'] = max(0, dwd_cohort['dwd_C_bark_recalc'] - dwd_decay_C_bark_recalc)

            # Track Total Decay Loss for Summary Data
            total_dwd_decay_C_wood_labile += dwd_decay_C_wood_labile
            total_dwd_decay_C_wood_recalc += dwd_decay_C_wood_recalc
            total_dwd_decay_C_bark_labile += dwd_decay_C_bark_labile
            total_dwd_decay_C_bark_recalc += dwd_decay_C_bark_recalc
            total_dwd_decay_C_wood_total = total_dwd_decay_C_wood_labile + total_dwd_decay_C_wood_recalc
            total_dwd_decay_C_bark_total = total_dwd_decay_C_bark_labile + total_dwd_decay_C_bark_recalc
            total_dwd_decay_C_total = total_dwd_decay_C_wood_total + total_dwd_decay_C_bark_total

            
            # === Update Snag Nitrogen Pools ===
            dwd_cohort['dwd_N_wood_labile'] = dwd_cohort['dwd_N_wood_labile'] + dwd_N_immobilized_wood_labile
            dwd_cohort['dwd_N_bark_labile'] = dwd_cohort['dwd_N_bark_labile'] + dwd_N_immobilized_bark_labile
            dwd_cohort['dwd_N_wood_recalc'] = dwd_cohort['dwd_N_wood_recalc'] + dwd_N_immobilized_wood_recalc
            dwd_cohort['dwd_N_bark_recalc'] = dwd_cohort['dwd_N_bark_recalc'] + dwd_N_immobilized_bark_recalc
        
            # === Compute New %N for Each Pool ===
            dwd_cohort['dwd_N_perc_wood_labile'] = dwd_cohort['dwd_N_wood_labile'] / (dwd_cohort['dwd_C_wood_labile'] * 2) if dwd_cohort['dwd_C_wood_labile'] > 0 else 0
            dwd_cohort['dwd_N_perc_wood_recalc'] = dwd_cohort['dwd_N_wood_recalc'] / (dwd_cohort['dwd_C_wood_recalc'] * 2) if dwd_cohort['dwd_C_wood_recalc'] > 0 else 0
            dwd_cohort['dwd_N_perc_bark_labile'] = dwd_cohort['dwd_N_bark_labile'] / (dwd_cohort['dwd_C_bark_labile'] * 2) if dwd_cohort['dwd_C_bark_labile'] > 0 else 0
            dwd_cohort['dwd_N_perc_bark_recalc'] = dwd_cohort['dwd_N_bark_recalc'] / (dwd_cohort['dwd_C_bark_recalc'] * 2) if dwd_cohort['dwd_C_bark_recalc'] > 0 else 0

            # ==== Compute New C:N RATIOS for each Pool ====           
            dwd_cohort['dwd_CN_wood_labile'] = dwd_cohort['dwd_C_wood_labile'] / dwd_cohort['dwd_N_wood_labile'] if dwd_cohort['dwd_N_wood_labile'] > 0 else None
            dwd_cohort['dwd_CN_wood_recalc'] = dwd_cohort['dwd_C_wood_recalc'] / dwd_cohort['dwd_N_wood_recalc'] if dwd_cohort['dwd_N_wood_recalc'] > 0 else None
            dwd_cohort['dwd_CN_bark_labile'] = dwd_cohort['dwd_C_bark_labile'] / dwd_cohort['dwd_N_bark_labile'] if dwd_cohort['dwd_N_bark_labile'] > 0 else None
            dwd_cohort['dwd_CN_bark_recalc'] = dwd_cohort['dwd_C_bark_recalc'] / dwd_cohort['dwd_N_bark_recalc'] if dwd_cohort['dwd_N_bark_recalc'] > 0 else None



            ### Accumulate Nitrogen for Summary Tracking
            total_dwd_N_wood_labile += dwd_cohort['dwd_N_wood_labile']
            total_dwd_N_wood_recalc += dwd_cohort['dwd_N_wood_recalc']
            total_dwd_N_bark_labile += dwd_cohort['dwd_N_bark_labile']
            total_dwd_N_bark_recalc += dwd_cohort['dwd_N_bark_recalc']
            total_dwd_N_wood_total = total_dwd_N_wood_labile + total_dwd_N_wood_recalc
            total_dwd_N_bark_total = total_dwd_N_bark_labile + total_dwd_N_bark_recalc
            total_dwd_N_total = total_dwd_N_wood_total + total_dwd_N_bark_total
            
            # === STORE COHORT-LEVEL DATA ===
            dwd_cohort_data['run_number'].append(run + 1)
            dwd_cohort_data['year'].append(year)
            dwd_cohort_data['cohort_number'].append(dwd_cohort['cohort_number'])
            dwd_cohort_data['cohort_age'].append(dwd_cohort['cohort_age'])
            
            # Store Carbon & Nitrogen pools
            dwd_cohort_data['dwd_C_wood_labile'].append(dwd_cohort['dwd_C_wood_labile'])
            dwd_cohort_data['dwd_C_wood_recalc'].append(dwd_cohort['dwd_C_wood_recalc'])
            dwd_cohort_data['dwd_C_bark_labile'].append(dwd_cohort['dwd_C_bark_labile'])
            dwd_cohort_data['dwd_C_bark_recalc'].append(dwd_cohort['dwd_C_bark_recalc'])
            
            dwd_cohort_data['dwd_N_wood_labile'].append(dwd_cohort['dwd_N_wood_labile'])
            dwd_cohort_data['dwd_N_wood_recalc'].append(dwd_cohort['dwd_N_wood_recalc'])
            dwd_cohort_data['dwd_N_bark_labile'].append(dwd_cohort['dwd_N_bark_labile'])
            dwd_cohort_data['dwd_N_bark_recalc'].append(dwd_cohort['dwd_N_bark_recalc'])


            dwd_cohort_data['initial_dwd_C_wood_labile'].append(dwd_cohort['initial_dwd_C_wood_labile'])
            dwd_cohort_data['initial_dwd_C_wood_recalc'].append(dwd_cohort['initial_dwd_C_wood_recalc'])
            dwd_cohort_data['initial_dwd_C_bark_labile'].append(dwd_cohort['initial_dwd_C_bark_labile'])
            dwd_cohort_data['initial_dwd_C_bark_recalc'].append(dwd_cohort['initial_dwd_C_bark_recalc'])
            
            dwd_cohort_data['initial_dwd_N_wood_labile'].append(dwd_cohort['initial_dwd_N_wood_labile'])
            dwd_cohort_data['initial_dwd_N_wood_recalc'].append(dwd_cohort['initial_dwd_N_wood_recalc'])
            dwd_cohort_data['initial_dwd_N_bark_labile'].append(dwd_cohort['initial_dwd_N_bark_labile'])
            dwd_cohort_data['initial_dwd_N_bark_recalc'].append(dwd_cohort['initial_dwd_N_bark_recalc'])


            
            # Store %N
            dwd_cohort_data['dwd_N_perc_wood_labile'].append(dwd_cohort['dwd_N_perc_wood_labile'])
            dwd_cohort_data['dwd_N_perc_wood_recalc'].append(dwd_cohort['dwd_N_perc_wood_recalc'])
            dwd_cohort_data['dwd_N_perc_bark_labile'].append(dwd_cohort['dwd_N_perc_bark_labile'])
            dwd_cohort_data['dwd_N_perc_bark_recalc'].append(dwd_cohort['dwd_N_perc_bark_recalc'])
            
            # Store C:N Ratio
            dwd_cohort_data['dwd_CN_wood_labile'].append(dwd_cohort['dwd_CN_wood_labile'])
            dwd_cohort_data['dwd_CN_wood_recalc'].append(dwd_cohort['dwd_CN_wood_recalc'])
            dwd_cohort_data['dwd_CN_bark_labile'].append(dwd_cohort['dwd_CN_bark_labile'])
            dwd_cohort_data['dwd_CN_bark_recalc'].append(dwd_cohort['dwd_CN_bark_recalc'])
                    
        
        total_dwd_decay_C_total_running_sum += total_dwd_decay_C_total
        total_dwd_C_to_microbial_biomass_total_running_sum += total_dwd_C_to_microbial_biomass_total                                                 
            

        # === Store Summary Data (AFTER Processing dwd Decay Each Year) ===
            
        # Standing dwd Carbon (After Decay) across all cohorts
        summary_data['dwd_C_wood_labile'].append(sum(dwd_cohort['dwd_C_wood_labile'] for dwd_cohort in existing_dwd_cohorts))
        summary_data['dwd_C_wood_recalc'].append(sum(dwd_cohort['dwd_C_wood_recalc'] for dwd_cohort in existing_dwd_cohorts))
        summary_data['dwd_C_bark_labile'].append(sum(dwd_cohort['dwd_C_bark_labile'] for dwd_cohort in existing_dwd_cohorts))
        summary_data['dwd_C_bark_recalc'].append(sum(dwd_cohort['dwd_C_bark_recalc'] for dwd_cohort in existing_dwd_cohorts))
        summary_data['dwd_C_wood_total'].append(summary_data['dwd_C_wood_labile'][i] + summary_data['dwd_C_wood_recalc'][i])
        summary_data['dwd_C_bark_total'].append(summary_data['dwd_C_bark_labile'][i] + summary_data['dwd_C_bark_recalc'][i])
        summary_data['dwd_C_total'].append(summary_data['dwd_C_wood_total'][i] + summary_data['dwd_C_bark_total'][i])

        # dwd Decay (CO₂ Loss) across all cohorts
        summary_data['dwd_C_decay_wood_labile'].append(total_dwd_decay_C_wood_labile)
        summary_data['dwd_C_decay_wood_recalc'].append(total_dwd_decay_C_wood_recalc)
        summary_data['dwd_C_decay_bark_labile'].append(total_dwd_decay_C_bark_labile)
        summary_data['dwd_C_decay_bark_recalc'].append(total_dwd_decay_C_bark_recalc)
        summary_data['dwd_C_decay_wood_total'].append(total_dwd_decay_C_wood_total)
        summary_data['dwd_C_decay_bark_total'].append(total_dwd_decay_C_bark_total)
        summary_data['dwd_C_decay_total'].append(total_dwd_decay_C_total)
        summary_data['dwd_C_decay_total_running_sum'].append(total_dwd_decay_C_total_running_sum)

        # === Dwd Carbon Lost to Microbial Biomass ===
        summary_data['dwd_C_to_microbial_biomass_wood_labile'].append(total_dwd_C_to_microbial_biomass_wood_labile)
        summary_data['dwd_C_to_microbial_biomass_wood_recalc'].append(total_dwd_C_to_microbial_biomass_wood_recalc)
        summary_data['dwd_C_to_microbial_biomass_bark_labile'].append(total_dwd_C_to_microbial_biomass_bark_labile)
        summary_data['dwd_C_to_microbial_biomass_bark_recalc'].append(total_dwd_C_to_microbial_biomass_bark_recalc)
        summary_data['dwd_C_to_microbial_biomass_wood_total'].append(total_dwd_C_to_microbial_biomass_wood_total)
        summary_data['dwd_C_to_microbial_biomass_bark_total'].append(total_dwd_C_to_microbial_biomass_bark_total)
        summary_data['dwd_C_to_microbial_biomass_total'].append(total_dwd_C_to_microbial_biomass_total)
        summary_data['dwd_C_to_microbial_biomass_total_running_sum'].append(total_dwd_C_to_microbial_biomass_total_running_sum)

        # Standing Dwd Nitrogen (After Decay) across all cohorts
        summary_data['dwd_N_wood_labile'].append(total_dwd_N_wood_labile)
        summary_data['dwd_N_wood_recalc'].append(total_dwd_N_wood_recalc)
        summary_data['dwd_N_bark_labile'].append(total_dwd_N_bark_labile)
        summary_data['dwd_N_bark_recalc'].append(total_dwd_N_bark_recalc)
        summary_data['dwd_N_wood_total'].append(total_dwd_N_wood_total)
        summary_data['dwd_N_bark_total'].append(total_dwd_N_bark_total)
        summary_data['dwd_N_total'].append(total_dwd_N_total)      
        
        # N Immobilization Totals across all cohorts
        summary_data['dwd_N_min_imm_wood_labile'].append(total_dwd_N_min_imm_wood_labile)
        summary_data['dwd_N_min_imm_wood_recalc'].append(total_dwd_N_min_imm_wood_recalc)
        summary_data['dwd_N_min_imm_bark_labile'].append(total_dwd_N_min_imm_bark_labile)
        summary_data['dwd_N_min_imm_bark_recalc'].append(total_dwd_N_min_imm_bark_recalc)
        summary_data['dwd_N_min_imm_wood_total'].append(summary_data['dwd_N_min_imm_wood_labile'][i] + summary_data['dwd_N_min_imm_wood_recalc'][i])
        summary_data['dwd_N_min_imm_bark_total'].append(summary_data['dwd_N_min_imm_bark_labile'][i] + summary_data['dwd_N_min_imm_bark_recalc'][i])
        summary_data['dwd_N_min_imm_total'].append(summary_data['dwd_N_min_imm_wood_total'][i] + summary_data['dwd_N_min_imm_bark_total'][i])



        # Compute total dwd nitrogen and carbon (sum of all pools)
        total_dwd_N = sum(dwd_cohort['dwd_N_wood_labile'] + dwd_cohort['dwd_N_wood_recalc'] + dwd_cohort['dwd_N_bark_labile'] + dwd_cohort['dwd_N_bark_recalc']
            for dwd_cohort in existing_dwd_cohorts)
        
        total_dwd_C = sum(dwd_cohort['dwd_C_wood_labile'] + dwd_cohort['dwd_C_wood_recalc'] + dwd_cohort['dwd_C_bark_labile'] + dwd_cohort['dwd_C_bark_recalc']
            for dwd_cohort in existing_dwd_cohorts)
        

        # Compute %N for Each Pool (Ensure No Division by Zero)
        summary_data['dwd_N_perc_wood_labile'].append((summary_data['dwd_N_wood_labile'][i] / summary_data['dwd_C_wood_labile'][i]) * 2
            if summary_data['dwd_C_wood_labile'][i] > 0 else 0)
        
        summary_data['dwd_N_perc_wood_recalc'].append((summary_data['dwd_N_wood_recalc'][i] / summary_data['dwd_C_wood_recalc'][i]) * 2
            if summary_data['dwd_C_wood_recalc'][i] > 0 else 0)
        
        summary_data['dwd_N_perc_bark_labile'].append((summary_data['dwd_N_bark_labile'][i] / summary_data['dwd_C_bark_labile'][i]) * 2
            if summary_data['dwd_C_bark_labile'][i] > 0 else 0)
        
        summary_data['dwd_N_perc_bark_recalc'].append((summary_data['dwd_N_bark_recalc'][i] / summary_data['dwd_C_bark_recalc'][i]) * 2
            if summary_data['dwd_C_bark_recalc'][i] > 0 else 0)
        
        summary_data['dwd_N_perc_wood_total'].append((summary_data['dwd_N_wood_total'][i] / summary_data['dwd_C_wood_total'][i]) * 2
            if summary_data['dwd_C_wood_total'][i] > 0 else 0)
        
        summary_data['dwd_N_perc_bark_total'].append((summary_data['dwd_N_bark_total'][i] / summary_data['dwd_C_bark_total'][i]) * 2
            if summary_data['dwd_C_bark_total'][i] > 0 else 0)
        
        summary_data['dwd_N_perc_total'].append((summary_data['dwd_N_total'][i] / summary_data['dwd_C_total'][i]) * 2
            if summary_data['dwd_C_total'][i] > 0 else 0)
        
        
        # === Compute C:N Ratio for Each Pool (Ensure No Division by Zero) ===
        summary_data['dwd_CN_wood_labile'].append(summary_data['dwd_C_wood_labile'][i] / summary_data['dwd_N_wood_labile'][i]
            if summary_data['dwd_N_wood_labile'][i] > 0 else 0)
        
        summary_data['dwd_CN_wood_recalc'].append(summary_data['dwd_C_wood_recalc'][i] / summary_data['dwd_N_wood_recalc'][i]
            if summary_data['dwd_N_wood_recalc'][i] > 0 else 0)
        
        summary_data['dwd_CN_bark_labile'].append(summary_data['dwd_C_bark_labile'][i] / summary_data['dwd_N_bark_labile'][i]
            if summary_data['dwd_N_bark_labile'][i] > 0 else 0)
        
        summary_data['dwd_CN_bark_recalc'].append(summary_data['dwd_C_bark_recalc'][i] / summary_data['dwd_N_bark_recalc'][i]
            if summary_data['dwd_N_bark_recalc'][i] > 0 else 0)
        
        summary_data['dwd_CN_wood_total'].append(summary_data['dwd_C_wood_total'][i] / summary_data['dwd_N_wood_total'][i]
            if summary_data['dwd_N_wood_total'][i] > 0 else 0)
        
        summary_data['dwd_CN_bark_total'].append(summary_data['dwd_C_bark_total'][i] / summary_data['dwd_N_bark_total'][i]
            if summary_data['dwd_N_bark_total'][i] > 0 else 0)

        summary_data['dwd_CN_total'].append(summary_data['dwd_C_total'][i] / summary_data['dwd_N_total'][i]
            if summary_data['dwd_N_total'][i] > 0 else 0)

               
             
    if params['write_cohort_results'].lower() == "yes":
        # Snag Cohorts
        snag_cohort_df = pd.DataFrame(snag_cohort_data)
        all_snag_cohorts.append(snag_cohort_df)
    
        # DWD Cohorts
        dwd_cohort_df = pd.DataFrame(dwd_cohort_data)
        all_dwd_cohorts.append(dwd_cohort_df)





    # Convert summary data dictionary to DataFrame for this run
    summary_df = pd.DataFrame(summary_data)
    
    summary_df['Total_cumulative_C_lost_to_decay'] = (summary_df['snag_C_decay_total_running_sum'] + summary_df['dwd_C_decay_total_running_sum'])
    summary_df['Total_cumulative_C_to_microbial_biomass'] = (summary_df['snag_C_to_microbial_biomass_total_running_sum'] + summary_df['dwd_C_to_microbial_biomass_total_running_sum'])
  
    # Append this run's summary data to the list
    all_summaries.append(summary_df)


# === Write all cohort data once all runs are complete ===
if params['write_cohort_results'].lower() == "yes":
    final_snag_cohort_df = pd.concat(all_snag_cohorts, ignore_index=True)
    final_dwd_cohort_df = pd.concat(all_dwd_cohorts, ignore_index=True)

    final_snag_cohort_df.to_csv(output_path / "Snag_Cohort_Results.csv", index=False)
    final_dwd_cohort_df.to_csv(output_path / "DWD_Cohort_Results.csv", index=False)

    print("All cohort results saved.")



# Combine all summary DataFrames across runs
final_summary_df = pd.concat(all_summaries, ignore_index=True)


# Load all run results
final_results = pd.concat(results, ignore_index=True)


# Ensure merge keys exist in both dataframes
if 'run_number' not in final_results.columns or 'year' not in final_results.columns:
    print("Warning: Missing merge keys in results data!")
if 'run_number' not in final_summary_df.columns or 'year' not in final_summary_df.columns:
    print("Warning: Missing merge keys in summary data!")

# Merge results with summary data
final_merged_results = final_results.merge(final_summary_df, on=["run_number", "year"], how="left")

final_merged_results["Net_Immobilization_Mineralization_kgN_ha-1_yr-1"] = ((final_merged_results["snag_N_min_imm_total"] + final_merged_results["dwd_N_min_imm_total"]) * 10)
final_merged_results["Total_Mortality_N_Inputs_kgN_ha-1_yr-1"] = ((final_merged_results["total_mort_wood"] * 2 * final_merged_results["wood_perc_N"] * 10) + (final_merged_results["total_mort_bark"] * 2 * final_merged_results["bark_perc_N"] * 10))
final_merged_results["Net_N_Flux_ToFrom_Dead_Wood_kgN_ha-1_yr-1"] = final_merged_results["Net_Immobilization_Mineralization_kgN_ha-1_yr-1"] + final_merged_results["Total_Mortality_N_Inputs_kgN_ha-1_yr-1"] 
final_merged_results["Total_N_in_Snags_and_DWD_gN_m-2"] = final_merged_results['dwd_N_total'] + final_merged_results['snag_N_total'] 
final_merged_results["Total_C_in_Snags_and_DWD_gC_m-2"] = final_merged_results['dwd_C_total'] + final_merged_results['snag_C_total'] 
final_merged_results['Net_N_Increment_in_Dead_Wood_Pools_kgN_ha-1_yr-1'] = (final_merged_results.groupby('run_number')['Total_N_in_Snags_and_DWD_gN_m-2'].diff().fillna(0) * 10)
final_merged_results['Microbial_CUE'] = (10/final_merged_results['critical_CN_wood'] * (1 - final_merged_results['bark_frac_wood'])) + (
    10/final_merged_results['critical_CN_bark'] * (final_merged_results['bark_frac_wood']))


# Define the final output file path
final_output_file = output_path / "CWD_Model_Final_Results.csv"

# Save the merged results to CSV
final_merged_results.to_csv(final_output_file, index=False)

print("Final merged results saved.")

et = time.perf_counter() - st
print(f'Total Execution Time: {et:.2f} seconds')






'''########################################################################################
                                  PLOTS
#########################################################################################'''

df = pd.read_csv(r"C:\Users\Andrew Ouimette\Documents\pnet_cwd\outputs\HBEF-W6_Case_Study\CWD_Model_Final_Results_Hubbard_Brook_5000.csv")
final_merged_results = df


'''#############################################################################
                              BIOMASS
##############################################################################'''

# Observed data (Mg ha⁻¹)
observed_biomass = {
    1965: 125, 1977: 173, 1982: 188, 1987: 186, 1992: 188,
    1997: 187, 2002: 185, 2007: 174, 2012: 170, 2017: 175, 2022: 171
}
campbell_observed_biomass = {
    1927.5: 1.5, 1931.75: 7.875, 1937: 18.75, 1940.5: 38.625,
    1944.25: 57.375, 1949.15: 71.25, 1955.77: 94.125
}

# --- Process and pivot ---
final_merged_results_sorted = final_merged_results.sort_values(['run_number', 'year'])
pivot_df = final_merged_results_sorted.pivot(index='year', columns='run_number', values='total_biomass')

# Convert gC/m² to Mg/ha (multiply by 2, divide by 100)
pivot_df_scaled = pivot_df * 2 / 100

# --- Trim top and bottom 2.5% per year ---
pivot_df_trimmed = pivot_df_scaled.apply(
    lambda row: row[(row >= row.quantile(0.025)) & (row <= row.quantile(0.975))],
    axis=1
)

# --- Median and year filter ---
median_biomass_filtered = pivot_df_trimmed.median(axis=1)
pivot_df_filtered = pivot_df_trimmed.loc[(pivot_df_trimmed.index >= 1900) & (pivot_df_trimmed.index <= 2100)]
median_biomass_filtered = median_biomass_filtered.loc[(median_biomass_filtered.index >= 1900) & (median_biomass_filtered.index <= 2100)]

# --- Plotting ---
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 18})
fig, ax = plt.subplots(figsize=(8, 5))

# Plot each filtered line in gray
for col in pivot_df_filtered.columns:
    ax.plot(pivot_df_filtered.index, pivot_df_filtered[col], color='gray', alpha=0.2, linewidth=2)

# Plot the median simulation
ax.plot(median_biomass_filtered.index, median_biomass_filtered.values,
        linestyle="--", color='black', linewidth=2, label='Median simulation')

# Plot observed HBEF-W6 values
for year, value in observed_biomass.items():
    if 1900 <= year <= 2100:
        ax.scatter(year, value, color='black', s=80, zorder=5)

# Plot Campbell open circles
for year, value in campbell_observed_biomass.items():
    ax.scatter(year, value, marker='o', facecolors='none', edgecolors='black', s=80, zorder=5,
               label='HBEF Other watersheds' if year == 1927.5 else "")

# --- Styling ---
ax.set_xlabel("Year")
ax.set_ylabel("Total Wood Biomass (Mg ha$^{-1}$)")
ax.set_xlim(1900, 2100)
ax.set_ylim(0, 300)
ax.grid(False)
plt.tight_layout()

# --- Legend ---
legend_elements = [
    Line2D([0], [0], color='gray', lw=2, alpha=1, label='Simulated runs'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='Median simulation'),
    Line2D([0], [0], marker='o', color='black', linestyle='None', label='Observed HBEF-W6'),
    Line2D([0], [0], marker='o', markerfacecolor='none', markeredgecolor='black', linestyle='None', label='HBEF Other watersheds')
]
legend = plt.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=15, labelspacing=0.2)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(1.0)

# --- Save and Show ---
plt.savefig("C:/Users/Andrew Ouimette/Documents/pnet_cwd/outputs/HBEF_Biomass_Trajectory_Trimmed.tiff",
            dpi=600, format='tiff', bbox_inches='tight')
plt.show()



'''#############################################################################
                             PRODUCTIVITY
##############################################################################'''

# Sort for consistency
final_merged_results_sorted = final_merged_results.sort_values(['run_number', 'year'])

# Pivot to get wood production: years as index, run_numbers as columns
pivot_prod = final_merged_results_sorted.pivot(index='year', columns='run_number', values='wood_production')

# Compute median across runs
median_prod = pivot_prod.median(axis=1)

# Filter to the same year range
pivot_prod_filtered = pivot_prod.loc[(pivot_prod.index >= 1900) & (pivot_prod.index <= 2100)]
median_prod_filtered = median_prod.loc[(median_prod.index >= 1900) & (median_prod.index <= 2100)]

# Set global font
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 18})

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 5))

# Plot all simulations in light gray
for col in pivot_prod_filtered.columns:
    ax.plot(pivot_prod_filtered.index, pivot_prod_filtered[col], color='gray', alpha=0.2, linewidth=2)

# Plot median
ax.plot(pivot_prod_filtered.index, median_prod_filtered, linestyle="--", color='black', linewidth=2, label='Median simulation')


# Observed productivity from Whittaker 1974
observed_productivity = {
    1958: 304,
    1963: 248}


for year, prod in observed_productivity.items():
    if 1900 <= year <= 2100:
        ax.scatter(year, prod, color='black', marker='o', s=80, zorder=5, label='Observed' if year == 1958 else "")


# Whittaker data
whittaker_proportional_NPP = {
    1928: 90,
    1933: 117,
    1938: 148,
    1943: 212,
    1948: 278,
    1953: 291,
    1958: 304,
    1963: 248}

for year, prod in whittaker_proportional_NPP.items():
    if 1900 <= year <= 2100:
        ax.scatter(year, prod, marker='o', facecolors='none', edgecolors='black', s=80, zorder=5,
                   label='Whittaker (tree rings)' if year == 1948 else "")

# Custom legend line for simulated runs
gray_line = Line2D([0], [0], color='gray', alpha=1, linewidth=2, label='Simulated runs')

# Collect handles and labels, and prepend the gray line
handles, labels = ax.get_legend_handles_labels()
handles = [gray_line] + handles

legend = ax.legend(
    handles=handles,
    loc="upper right",
    frameon=True,
    fontsize=16,        # Adjust font size here
    labelspacing=0.25    # Tighten vertical spacing
)

legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(1.0)


# Labels
ax.set_xlabel("Year")
ax.set_ylabel("Wood Production (gC m$^{-2}$ yr$^{-1}$)")
ax.grid(False)

# Save and show
plt.tight_layout()
plt.savefig("C:/Users/Andrew Ouimette/Documents/pnet_cwd/outputs/HBEF_W6_Wood_Production.tiff",
            dpi=600, format='tiff', bbox_inches='tight')
plt.show()


'''############################################################################
        Biomass and productivity model-observation stats
#############################################################################'''

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Observed biomass (Mg/ha)
observed_biomass = {
    1965: 125, 1977: 173, 1982: 188, 1987: 186, 1992: 188,
    1997: 187, 2002: 185, 2007: 174, 2012: 170, 2017: 175, 2022: 171
}
obs_biomass_series = pd.Series(observed_biomass)
sim_biomass_series = median_biomass_filtered.loc[obs_biomass_series.index]

# Biomass metrics
r2_biomass = r2_score(obs_biomass_series, sim_biomass_series)
rmse_biomass = mean_squared_error(obs_biomass_series, sim_biomass_series, squared=False)
mae_biomass = mean_absolute_error(obs_biomass_series, sim_biomass_series)
bias_biomass = (sim_biomass_series - obs_biomass_series).mean()

# Observed productivity (gC/m2/yr)
observed_productivity = {1958: 304, 1963: 248}
obs_prod_series = pd.Series(observed_productivity)
sim_prod_series = median_prod_filtered.loc[obs_prod_series.index]

# Productivity metrics
r2_prod = r2_score(obs_prod_series, sim_prod_series)
rmse_prod = mean_squared_error(obs_prod_series, sim_prod_series, squared=False)
mae_prod = mean_absolute_error(obs_prod_series, sim_prod_series)
bias_prod = (sim_prod_series - obs_prod_series).mean()

# Print results
print("\nBiomass Comparison:")
print(f"  R²:   {r2_biomass:.3f}")
print(f"  RMSE: {rmse_biomass:.1f} Mg ha⁻¹")
print(f"  MAE:  {mae_biomass:.1f} Mg ha⁻¹")
print(f"  Bias: {bias_biomass:.1f} Mg ha⁻¹")

print("\nProductivity Comparison:")
print(f"  R²:   {r2_prod:.3f}")
print(f"  RMSE: {rmse_prod:.1f} g C m⁻² yr⁻¹")
print(f"  MAE:  {mae_prod:.1f} g C m⁻² yr⁻¹")
print(f"  Bias: {bias_prod:.1f} g C m⁻² yr⁻¹")


'''#############################################################################
                             SNAGS
##############################################################################'''

# Apply 2.5–97.5 percentile filter per year for snags
snag_df = final_merged_results[['year', 'run_number', 'snag_C_total']].copy()
filtered_snag = snag_df.groupby('year').apply(
    lambda g: g[
        g['snag_C_total'].between(
            g['snag_C_total'].quantile(0.01),
            g['snag_C_total'].quantile(0.99)
        )
    ]
).reset_index(drop=True)

pivot_snag = filtered_snag.pivot(index='year', columns='run_number', values='snag_C_total')
median_snag = pivot_snag.median(axis=1)


plt.rcParams.update({'font.family': 'serif','font.serif': ['Times New Roman'],'font.size': 18})
light_gray = to_rgba("#bebebe")
plt.figure(figsize=(8, 5))
for col in pivot_snag.columns:
    plt.plot(pivot_snag.index, pivot_snag[col], color=light_gray, alpha=0.2)
plt.plot(pivot_snag.index, median_snag, color='black', linewidth=2, linestyle='-', label='Median snag C')


observed_standing_dead = {
    1977:	859 *.5,
    1982:	947 *.5,
    1987:	1179 *.5,
    1992:	1331 *.5,
    1997:	1326 *.5,
    2002:	2061 *.5,
    2007:	2544 *.5,
    2012:	2170 *.5,
    2017:	1605 *.5}

# Plot observed standing dead as black circles
# for year, value in observed_standing_dead.items():
#     plt.scatter(year, value, color='black', s=80, zorder=5, label='Observed' if year == 1977 else "")


plt.xlabel("Year")
plt.ylabel("Standing Dead (g C m$^{-2}$)")
#plt.title("Simulated Snag Carbon Over Time")

# Calculate observed range
obs_min = min(observed_standing_dead.values())
obs_max = max(observed_standing_dead.values())

# Add horizontal dashed blue lines at min and max of observed values
plt.hlines(y=obs_min, xmin=1975, xmax=2025, colors='blue', linestyles='--', linewidth=1.5, label='Obs. range')
plt.hlines(y=obs_max, xmin=1975, xmax=2025, colors='blue', linestyles='--', linewidth=1.5)


# Custom gray line for legend
gray_line = Line2D([0], [0], color=light_gray, alpha=1, linewidth=2, label='Simulated runs')

# Get existing handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

# Prepend the gray line
handles = [gray_line] + handles

# Apply legend with updated handles
legend = plt.legend(
    handles=handles,
    loc='upper right',  # or 'upper right' for snags
    fontsize=16,
    labelspacing=0.2,    # Tighten vertical spacing
    frameon=True
)
legend.get_frame().set_edgecolor('black')

plt.xlim(1900, 2100)   # Set x-axis range
plt.ylim(0, 3000)     # Set y-axis range (example for gC/m² biomass)
plt.grid(False)
plt.tight_layout()
plt.savefig("C:/Users/Andrew Ouimette/Documents/pnet_cwd/outputs/HBEF_W6_Snags.tiff",
            dpi=600, format='tiff', bbox_inches='tight')
plt.show()

'''############################################################################
            Standing Dead Statistics and Validation
############################################################################'''

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

# Convert observed dictionary to Series
obs_snag_series = pd.Series(observed_standing_dead)

# Ensure years align between observed and simulated median
common_years = obs_snag_series.index.intersection(median_snag.index)

# Extract observed and simulated values for those years
obs_vals = obs_snag_series.loc[common_years].values
sim_vals = median_snag.loc[common_years].values

# Calculate performance metrics
r2 = r2_score(obs_vals, sim_vals)
rmse = mean_squared_error(obs_vals, sim_vals, squared=False)
mae = mean_absolute_error(obs_vals, sim_vals)
bias = np.mean(sim_vals - obs_vals)

print("Quantitative Evaluation of Standing Dead Carbon:")
print(f"  R²     = {r2:.3f}")
print(f"  RMSE   = {rmse:.1f} g C m⁻²")
print(f"  MAE    = {mae:.1f} g C m⁻²")
print(f"  Bias   = {bias:.1f} g C m⁻²")

# Get percentiles from full simulation for uncertainty bounds
p25 = pivot_snag.quantile(0.25, axis=1)
p75 = pivot_snag.quantile(0.75, axis=1)
p05 = pivot_snag.quantile(0.05, axis=1)
p95 = pivot_snag.quantile(0.95, axis=1)

# Evaluate coverage
within_iqr = 0
within_90 = 0

for year in common_years:
    obs_val = obs_snag_series[year]
    if p25[year] <= obs_val <= p75[year]:
        within_iqr += 1
    if p05[year] <= obs_val <= p95[year]:
        within_90 += 1

total_points = len(common_years)
print(f"\nCoverage of Observed Points:")
print(f"  Within IQR (25–75%): {within_iqr} of {total_points} ({100 * within_iqr / total_points:.1f}%)")
print(f"  Within 90% CI (5–95%): {within_90} of {total_points} ({100 * within_90 / total_points:.1f}%)")



import pandas as pd
import numpy as np

# Observed snag carbon values
observed_standing_dead = {
    1977: 859 * 0.5,
    1982: 947 * 0.5,
    1987: 1179 * 0.5,
    1992: 1331 * 0.5,
    1997: 1326 * 0.5,
    2002: 2061 * 0.5,
    2007: 2544 * 0.5,
    2012: 2170 * 0.5,
    2017: 1605 * 0.5
}

# Step 1: Observed decadal summary
obs_df = pd.DataFrame(list(observed_standing_dead.items()), columns=['year', 'obs_snag_C'])
obs_df['decade'] = (obs_df['year'] // 10) * 10
obs_decade_summary = obs_df.groupby('decade')['obs_snag_C'].agg(['mean', 'min', 'max'])
obs_decade_means = obs_decade_summary['mean']
observed_years = obs_df['year'].unique()

# Step 2: Simulated median decadal mean
median_df = median_snag.reset_index()
median_df.columns = ['year', 'median_snag_C']
median_df['decade'] = (median_df['year'] // 10) * 10
sim_decade_summary = median_df[median_df['year'].isin(observed_years)].groupby('decade')['median_snag_C'].mean()

# Step 3: IQR for observed years
iqr_obs_years = pivot_snag.loc[observed_years]
iqr_obs_25 = iqr_obs_years.quantile(0.25, axis=1)
iqr_obs_75 = iqr_obs_years.quantile(0.75, axis=1)

# Step 4: Calculate IQR range across the observed period
iqr_25_mean = iqr_obs_25.mean()
iqr_75_mean = iqr_obs_75.mean()

# Step 5: Final summary stats
obs_mean = obs_decade_means.mean()
obs_range = (obs_decade_means.min(), obs_decade_means.max())
sim_mean = sim_decade_summary.mean()

# Step 6: Print results
print(f"Observed mean standing dead C (decadal): {obs_mean:.0f} g C m⁻²")
print(f"Observed range: {obs_range[0]:.0f}–{obs_range[1]:.0f} g C m⁻²")
print(f"Simulated mean (median trajectory): {sim_mean:.0f} g C m⁻²")
print(f"Simulated IQR range over observed years: {iqr_25_mean:.0f}–{iqr_75_mean:.0f} g C m⁻²")




''' #################################################
                Downed Woody Debris
##################################################'''


# Apply 2.5–97.5 percentile filter per year for DWD
dwd_df = final_merged_results[['year', 'run_number', 'dwd_C_total']].copy()
filtered_dwd = dwd_df.groupby('year').apply(
    lambda g: g[
        g['dwd_C_total'].between(
            g['dwd_C_total'].quantile(0.01),
            g['dwd_C_total'].quantile(0.99)
        )
    ]
).reset_index(drop=True)

pivot_dwd = filtered_dwd.pivot(index='year', columns='run_number', values='dwd_C_total')
median_dwd = pivot_dwd.median(axis=1)

plt.rcParams.update({'font.family': 'serif','font.serif': ['Times New Roman'],'font.size': 18})
light_gray = to_rgba("#bebebe")
plt.figure(figsize=(8, 5))

for col in pivot_dwd.columns:
    plt.plot(pivot_dwd.index, pivot_dwd[col], color=light_gray, alpha=0.2, linewidth=2.5)

plt.plot(pivot_dwd.index, median_dwd, color='black', linewidth=2, linestyle='-', label='Median DWD C')

# Add two horizontal lines to show observed range
plt.hlines(800, xmin=1975, xmax=2020, colors='blue', linestyles='--', linewidth=1.5, label='Observed range')
plt.hlines(1600, xmin=1975, xmax=2020, colors='blue', linestyles='--', linewidth=1.5)

plt.xlabel("Year")
plt.ylabel("DWD (g C m$^{-2}$)")
#plt.title("Simulated Downed Woody Debris Carbon Over Time")

# Custom gray line for legend
gray_line = Line2D([0], [0], color=light_gray, alpha=1, linewidth=2, label='Simulated runs')

# Get existing handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

# Prepend the gray line
handles = [gray_line] + handles

# Apply legend with updated handles
legend = plt.legend(
    handles=handles,
    loc='upper right',  # or 'upper right' for snags
    fontsize=16,
    labelspacing=0.2,    # Tighten vertical spacing
    frameon=True
)
legend.get_frame().set_edgecolor('black')

plt.xlim(1900, 2100)   # Set x-axis range
plt.ylim(0, 3000)     # Set y-axis range (example for gC/m² biomass)
plt.grid(False)
plt.tight_layout()
plt.savefig("C:/Users/Andrew Ouimette/Documents/pnet_cwd/outputs/HBEF_W6_DWD.tiff",
            dpi=600, format='tiff', bbox_inches='tight')
plt.show()



'''####################################################################
             DWD statics for model-observation comparison
###################################################################'''

# Filter to observed period (1975–2020) to match figure range
observed_window = median_dwd.loc[1975:2020]

# Simulated summary stats (median trajectory)
dwd_sim_mean = observed_window.mean()

# IQR across all 5000 simulations for the same period
iqr_25 = pivot_dwd.loc[1975:2020].quantile(0.25, axis=1).mean()
iqr_75 = pivot_dwd.loc[1975:2020].quantile(0.75, axis=1).mean()

# Observed range from figure
obs_dwd_min = 816
obs_dwd_max = 1800

# Print summary
print(f"Observed DWD range: {obs_dwd_min}–{obs_dwd_max} g C m⁻²")
print(f"Simulated median DWD mean (1975–2020): {dwd_sim_mean:.0f} g C m⁻²")
print(f"Simulated IQR over same period: {iqr_25:.0f}–{iqr_75:.0f} g C m⁻²")





''' #################################################
                Nitrogen in Dead Wood
##################################################'''


# Apply 2.5–97.5 percentile filter per year for DWD
dwd_N_df = final_merged_results[['year', 'run_number', 'Total_N_in_Snags_and_DWD_gN_m-2']].copy()
filtered_dwd_N = dwd_N_df.groupby('year').apply(
    lambda g: g[
        g['Total_N_in_Snags_and_DWD_gN_m-2'].between(
            g['Total_N_in_Snags_and_DWD_gN_m-2'].quantile(0.01),
            g['Total_N_in_Snags_and_DWD_gN_m-2'].quantile(0.99)
        )
    ]
).reset_index(drop=True)

pivot_dwd_N = filtered_dwd_N.pivot(index='year', columns='run_number', values='Total_N_in_Snags_and_DWD_gN_m-2')
median_dwd_N = pivot_dwd_N.median(axis=1)

plt.rcParams.update({'font.family': 'serif','font.serif': ['Times New Roman'],'font.size': 18})
plt.figure(figsize=(8, 5))

for col in pivot_dwd_N.columns:
    plt.plot(pivot_dwd_N.index, pivot_dwd_N[col], color='gray', alpha=0.2, linewidth=2.5)

plt.plot(pivot_dwd_N.index, median_dwd_N, color='black', linewidth=2, linestyle='--', label='Median Dead Wood N')

# Add two horizontal lines to show observed range
#plt.hlines(816, xmin=1975, xmax=2020, colors='blue', linestyles='--', linewidth=1.5, label='Observed range')
#plt.hlines(1800, xmin=1975, xmax=2020, colors='blue', linestyles='--', linewidth=1.5)

plt.xlabel("Year")
plt.ylabel("Dead Wood N (g N m$^{-2}$)")
#plt.title("Simulated Downed Woody Debris Carbon Over Time")

# Custom gray line for legend
gray_line = Line2D([0], [0], color='gray', alpha=1, linewidth=2, label='Simulated runs')

# Get existing handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

# Prepend the gray line
handles = [gray_line] + handles

# Apply legend with updated handles
legend = plt.legend(
    handles=handles,
    loc='upper right',  # or 'upper right' for snags
    fontsize=16,
    labelspacing=0.2,    # Tighten vertical spacing
    frameon=True
)
legend.get_frame().set_edgecolor('black')

plt.xlim(1900, 2100)   # Set x-axis range
plt.ylim(0, 100)     # Set y-axis range (example for gC/m² biomass)
plt.grid(False)
plt.tight_layout()
plt.savefig("C:/Users/Andrew Ouimette/Documents/pnet_cwd/outputs/HBEF_W6_DeadWoodN.tiff",
            dpi=600, format='tiff', bbox_inches='tight')
plt.show()

''' #################################################
                Net N Flux
##################################################'''

# Load Hubbard Brook Deposition-Export data
hubbard_csv_path = "C:/Users/Andrew Ouimette/Documents/pnet_cwd/outputs/Hubbard_Brook_Deposition_Export.csv"
hubbard_df = pd.read_csv(hubbard_csv_path)

# Step 1: Keep all simulation years; just filter extreme values
df_ranked = final_merged_results[
    final_merged_results["Net_N_Increment_in_Dead_Wood_Pools_kgN_ha-1_yr-1"].between(-1500, 1500)
].copy()

# Step 2: Compute quantile rank by year (for coloring only)
df_ranked["Quantile_Rank"] = df_ranked.groupby("year")[
    "Net_N_Increment_in_Dead_Wood_Pools_kgN_ha-1_yr-1"
].rank(pct=True)

# Step 3: 

green_gray = to_rgba("#bebebe")  #a1bfa1
light_gray = to_rgba("#bebebe")
dark_green = to_rgba("green") #dark_green = to_rgba("#006837")

colors = [
    (0.0, light_gray),
    (0.2, green_gray),
    (0.5, dark_green),
    (0.8, green_gray),
    (1.0, light_gray)
]

custom_cmap = LinearSegmentedColormap.from_list("green_fade_center", colors)
norm = mcolors.Normalize(vmin=0, vmax=1)

# Step 4: Sort and prepare arrays
df_sorted = df_ranked.sort_values(["run_number", "year"]).reset_index(drop=True)
runs = df_sorted["run_number"].astype(int).values
years = df_sorted["year"].values
flux = df_sorted["Net_N_Increment_in_Dead_Wood_Pools_kgN_ha-1_yr-1"].values
ranks = df_sorted["Quantile_Rank"].values

# Step 5: Create segment mask where run doesn't change and year is consecutive
year_diff = years[1:] - years[:-1]
valid = (runs[1:] == runs[:-1]) & (year_diff == 1)

# Step 6: Vectorized segment creation
x0 = years[:-1][valid]
x1 = years[1:][valid]
y0 = flux[:-1][valid]
y1 = flux[1:][valid]
c  = ranks[:-1][valid]

segments = np.stack([np.stack([x0, y0], axis=1), np.stack([x1, y1], axis=1)], axis=1)

# Step 7: Assign white color to extreme quantiles, otherwise use colormap
rgba_colors = custom_cmap(norm(c))
outlier_mask = (c < 0.05) | (c > 0.95)
rgba_colors[outlier_mask] = [1, 1, 1, 1]  # White in RGBA
segment_colors = rgba_colors



# Step 8: Plot
plt.rcParams.update({'font.family': 'serif','font.serif': ['Times New Roman'],'font.size': 18})
fig, ax = plt.subplots(figsize=(8, 5))
lc = LineCollection(segments, colors=segment_colors, linewidth=0.3, alpha=0.3)
ax.add_collection(lc)

# Step 9: Colorbar
sm = cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Quantile Rank of Simulations")

# Step 10: Median line
median_vals = df_sorted.groupby("year")[
    "Net_N_Increment_in_Dead_Wood_Pools_kgN_ha-1_yr-1"
].median()
ax.plot(median_vals.index, median_vals.values, color="black", linewidth=1.5, label="Median")

# Step 11: Hubbard Brook line
ax.plot(
    hubbard_df["Year"],
    hubbard_df["Deposition - Export kgN ha-1 yr-1"],
    color="black", linestyle="--", linewidth=1.25, label="Dep. - Export"
)

# Step 12: Reference lines
#ax.hlines(y=-14.2, xmin=1956, xmax=1976, colors='blue', linestyles='--', linewidth=1.25)
#ax.hlines(y=-0.2, xmin=1977, xmax=1992, colors='blue', linestyles='--', linewidth=1.25)
ax.hlines(y=8.4, xmin=1992, xmax=2007, colors='blue', linestyles='--', linewidth=1.25, label='N imbalance')

# Final polish
ax.axhline(0, color="black", linestyle='-', linewidth=0.5)
ax.set_xlabel("Year")
ax.set_ylabel("Net N Flux to/from dead wood pool (kg ha⁻¹ y⁻¹)")
ax.set_xlim(1900, 2050)
ax.set_ylim(-15, 15)
legend = ax.legend(loc="lower right", frameon=True)
legend.get_frame().set_edgecolor('black')
plt.tight_layout()

# Save
plt.savefig(
    "C:/Users/Andrew Ouimette/Documents/pnet_cwd/outputs/Monte_Carlo_Net_N_Flux.tiff",
    dpi=600, format='tiff', bbox_inches='tight')

plt.show()

# Calculate median net N flux across all runs for selected time periods

# 1. 1992–2007 period
median_1992_2007 = (
    df_sorted[df_sorted["year"].between(1992, 2007)]
    .groupby("year")["Net_N_Increment_in_Dead_Wood_Pools_kgN_ha-1_yr-1"]
    .median()
)
mean_median_1992_2007 = median_1992_2007.mean()
print(f"Mean of annual medians (1992–2007): {mean_median_1992_2007:.2f} kg N ha⁻¹ yr⁻¹")

# 2. 1975–2025 period
median_1975_2025 = (
    df_sorted[df_sorted["year"].between(1975, 2025)]
    .groupby("year")["Net_N_Increment_in_Dead_Wood_Pools_kgN_ha-1_yr-1"]
    .median()
)
mean_median_1975_2025 = median_1975_2025.mean()
print(f"Mean of annual medians (1975–2025): {mean_median_1975_2025:.2f} kg N ha⁻¹ yr⁻¹")


# Exclude 1998 from the average
exclude_years = [1998]

median_1992_2007_no1998 = (
    df_sorted[df_sorted["year"].between(1992, 2007) & ~df_sorted["year"].isin(exclude_years)]
    .groupby("year")["Net_N_Increment_in_Dead_Wood_Pools_kgN_ha-1_yr-1"]
    .median()
)

# Mean of annual medians for 1975–2025, excluding 1998
mean_median_1975_2025_no1998 = (
    median_vals.loc[[y for y in range(1975, 2026) if y != 1998]].mean()
)
print(f"Mean of annual medians (1975–2025, excluding 1998): {mean_median_1975_2025_no1998:.2f} kg N ha⁻¹ yr⁻¹")


mean_median_1992_2007_no1998 = median_1992_2007_no1998.mean()
print(f"Mean of annual medians (1992–2007, excluding 1998): {mean_median_1992_2007_no1998:.2f} kg N ha⁻¹ yr⁻¹")

# Median of the median line from 1992–2007
median_of_medians_1992_2007 = median_vals.loc[1992:2007].median()
print(f"Median of the median line (1992–2007): {median_of_medians_1992_2007:.2f} kg N ha⁻¹ yr⁻¹")

# Optionally exclude 1998
median_of_medians_1992_2007_no1998 = median_vals.loc[[y for y in range(1992, 2008) if y != 1998]].median()
print(f"Median of the median line (1992–2007, excluding 1998): {median_of_medians_1992_2007_no1998:.2f} kg N ha⁻¹ yr⁻¹")

# Median of the median line from 2026–2100
median_of_medians_2075_2100 = median_vals.loc[2075:2100].median()
mean_of_medians_2075_2100 = median_vals.loc[2075:2100].mean()

print(f"Median of median-modeled fluxes (2075–2100): {median_of_medians_2075_2100:.2f} kg N ha⁻¹ yr⁻¹")
print(f"Mean of median-modeled fluxes (2075–2100): {mean_of_medians_2075_2100:.2f} kg N ha⁻¹ yr⁻¹")




# Find annual net flux after 1917
annual_post = df_sorted[df_sorted["year"] > 1917].copy()

# Mark years where annual net flux is positive (net immobilization)
annual_post["immobilizing"] = annual_post["Net_N_Increment_in_Dead_Wood_Pools_kgN_ha-1_yr-1"] > 0

# Count how many years each run had net immobilization after 1917
immobilization_counts = (
    annual_post[annual_post["immobilizing"]]
    .groupby("run_number")["year"]
    .count()
)

# Optional: Count immobilization years between 1970–2000 only
immobilization_window = (
    annual_post[annual_post["year"].between(1970, 2000) & annual_post["immobilizing"]]
    .groupby("run_number")["year"]
    .count()
)

print(f"Number of runs with net immobilization (any year post-1917): {(immobilization_counts > 0).sum()}")
print(f"Median number of net immobilization years per run (1918–2100): {immobilization_counts.median()}")

print(f"\nNumber of runs with net immobilization during 1970–2000: {(immobilization_window > 0).sum()}")
print(f"Median number of immobilization years per run in 1970–2000: {immobilization_window.median()}")




'''#########################################################################################
                                Overlay of N Inputs and Outputs
#########################################################################################'''



# Group by year and calculate medians across simulations
median_df = final_merged_results.groupby("year")[
    ["Total_Mortality_N_Inputs_kgN_ha-1_yr-1",
     "Net_Immobilization_Mineralization_kgN_ha-1_yr-1",
     "Net_N_Flux_ToFrom_Dead_Wood_kgN_ha-1_yr-1"]
].median().sort_values("year").reset_index()

# Rename for clarity
median_df.rename(columns={
    "Total_Mortality_N_Inputs_kgN_ha-1_yr-1": "N Inputs (Mortality)",
    "Net_Immobilization_Mineralization_kgN_ha-1_yr-1": "Immobilization/Mineralization",
    "Net_N_Flux_ToFrom_Dead_Wood_kgN_ha-1_yr-1": "Net N Flux"
}, inplace=True)

# Plot
plt.rcParams.update({'font.family': 'serif','font.serif': ['Times New Roman'],'font.size': 20})
fig, ax = plt.subplots(figsize=(8, 5))

# Plot each line in black with different styles
ax.plot(median_df["year"], median_df["N Inputs (Mortality)"],
        color="black", linestyle="-.", linewidth=2)
ax.plot(median_df["year"], median_df["Immobilization/Mineralization"],
        color="black", linestyle="--", linewidth=2)
ax.plot(median_df["year"], median_df["Net N Flux"],
        color="black", linestyle="-", linewidth=2)

# Label positions for each line
label_year_mortality = 2005
label_year_immobil = 1980
label_year_net_flux = 2015

# Vertical offsets
offset_mortality = 1
offset_immobil = -3.5
offset_net_flux = 1

# Y-values for each label year
y_mortality = median_df.loc[median_df["year"] == label_year_mortality, "N Inputs (Mortality)"].values[0]
y_immobilization = median_df.loc[median_df["year"] == label_year_immobil, "Immobilization/Mineralization"].values[0]
y_net_flux = median_df.loc[median_df["year"] == label_year_net_flux, "Net N Flux"].values[0]

# Text annotations with full control
ax.text(label_year_mortality, y_mortality + offset_mortality,
        "N Inputs (Mortality)", color="black", fontsize=16, ha="left")

ax.text(label_year_immobil, y_immobilization + offset_immobil,
        "Immobilization / Mineralization", color="black", fontsize=16, ha="left")

ax.text(label_year_net_flux, y_net_flux + offset_net_flux,
        "Net N Flux", color="black", fontsize=16, ha="left")

# Format plot
ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
ax.set_xlabel("Year")
ax.set_ylabel("kg N ha⁻¹ yr⁻¹")
ax.set_ylim(-15, 15)
ax.set_xlim(1900, 2050)
ax.set_xticks(range(1900, 2051, 25))

ax.tick_params(axis='both', which='major')
plt.tight_layout()

# Show or save

plt.savefig("C:/Users/Andrew Ouimette/Documents/pnet_cwd/outputs/N_Inputs_Outputs_Overlay.tiff",
    dpi=600, format='tiff', bbox_inches='tight')
plt.show()

