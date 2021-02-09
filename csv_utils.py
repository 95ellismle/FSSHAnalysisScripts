import numpy as np
import csv
import os


def _write_csv_helper_(folder, filename, data, names, timesteps=False):
    """
    Will write a csv file.

    This is a private function used by write_MSD and write_IPR.

    Inputs:
        * folder <str> => The folder to save in (will be created)
        * filename <str> => The filename
        * data <np.ndarray> => The data to write.
        * name <tuple<str>> => The column headers
    """
    # Sort out some file admin
    if not os.path.isdir(folder): os.makedirs(folder)
    filepath = f"{folder}/{filename}"
    if filepath.split(".")[-1] != "csv": filepath += ".csv"
    if type(names) == str: names = (names,)

    # First construct the data
    if len(np.shape(data)) == 1: # if we have a 1D array
        write_data = [[i] for i in data]
    else:
        write_data = [i.flatten() for i in data]

    # Add timestep data if required
    if timesteps is not False:
        write_data = [[t, *d] for t, d in zip(timesteps, write_data)]
        names = ('time', *names)

    # Write the file
    with open(filepath, 'w') as f:
        csvWriter = csv.writer(f)
        csvWriter.writerow(names)
        csvWriter.writerows(write_data)
