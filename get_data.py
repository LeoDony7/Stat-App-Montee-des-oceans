# Import modules
import copernicusmarine
from pprint import pprint

# Define parameters to run the extraction
dataset_id = "cmems-IFREMER-ATL-SST-L4-REP-OBS_FULL_TIME_SERIE"


# Define output storage parameters
output_directory = "./my_data_folder/eursst_folder"

# Call the get function to save data
get_result = copernicusmarine.get(
    dataset_id=dataset_id,
    filter = "*0701*",
    output_directory=output_directory,
    no_directories=True)

pprint(f"List of saved files: {get_result}")


