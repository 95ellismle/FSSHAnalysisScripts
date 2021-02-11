import numpy as np
import time
import os

import xyz_utils as xyz_ut
import coeff_utils as coeff_ut
import csv_utils as csv_ut

# Params
root_folder = ".."   # Where to check for data
all_dir = [f"{root_folder}/{i}" for i in os.listdir(root_folder) if 'run-fssh-' in i] # The folders to calculate for
save_dir = ".." # Where to save the data
natoms = 36     # num atoms per molecule
reCalc_COM_eachTraj = False

#all_dir = all_dir[:3]

print(f"Found {len(all_dir)} folders in '{root_folder}'")
print(f"Saving analysis data in '{save_dir}'")


def calc_and_save(folder, COM=False):
    """
    Can be used with the multIPRocessing library to parallelise if necessary.
    """
    FolderData = coeff_ut.FSSHRun(folder, COM)
    FolderData.calc_3D_MSD_Elstner()
    FolderData.calc_IPR()

    filename = folder.split("/")[-1].replace('run-', '').replace("-", "_")

    FolderData.write_MSD(f"{save_dir}/MSD_Data", f"MSD_{filename}")
    FolderData.write_IPR(f"{save_dir}/IPR_Data", f"IPR_{filename}")
    return FolderData


COM = False
if reCalc_COM_eachTraj is False:
    Pos = xyz_ut.XYZ_File(f"{all_dir[0]}/pos-init.xyz")
    COM = xyz_ut.calc_COM(Pos, natoms)

# loop over all folders and calc MSD and IPR
allFolderData = []
len_folds = len(all_dir)
all_t = []
for count, folder in enumerate(all_dir, 1):
    t1 = time.time()

    # Do the calculations
    if os.path.isfile(f"{folder}/run-coeff-1.xyz"):
        allFolderData.append(calc_and_save(folder, COM))
    else: continue

    # Print things in a pretty way and give some useful info
    len_prog_bar = 30
    numH = int(len_prog_bar * count / len_folds)
    t2 = time.time()

    all_t.append(t2 - t1)
    avg_t = np.mean(all_t)
    t_left = (len_folds - count + 1) * avg_t

    print("|" + "#"*numH + " "*(len_prog_bar - numH)
           + f"| ({100.*count/len_folds:.0f}% complete"
           + f", {t_left:.0f}s remaining.)    ",
          end="\r")


# Calc and write averaged data
t1 = time.time()

avg_IPR, all_IPR, timesteps_IPR = coeff_ut.average_quantity(allFolderData, 'IPR')
avg_MSD, all_MSD, timesteps_MSD = coeff_ut.average_quantity(allFolderData, 'MSD')

csv_ut._write_csv_helper_(f"{save_dir}/averaged_data", "ipr.csv", avg_IPR, ["ipr"],
                   timesteps_IPR)
csv_ut._write_csv_helper_(f"{save_dir}/averaged_data", "msd.csv", avg_MSD,
                   ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"],
                   timesteps_MSD)
t2 = time.time()

print(f"All done. Time Taken: {sum(all_t) + t2 - t1:.0f}s                                                                                               ")
