import numpy as np
import glob
import os
import re

import xyz_utils as xyz_ut
import csv_utils as csv_ut

#this class reads normal hamiltonian and coefficients and gives all post-process info
class FSSHRun(object):
    def __init__(self, folder):
        self.folder = folder
        self.init = False
        self.Coeff = xyz_ut.XYZ_File(f"{folder}/run-coeff-1.xyz")
        self.timesteps = self.Coeff.timesteps
        self.Pops = self.Coeff[:, :, 0]**2 + self.Coeff[:, :, 1]**2

    def read_DECOMP(self):
        """
        If the DECOMP.inp file exists then use it to splice the COM.

        Will save the mol_nums as a variable named mol_nums.
        """
        all_files = os.listdir(self.folder)
        decompFilename=""
        fCount=0
        for f in all_files:
            if 'DECOMP.' in f:
                decompFilename = f
                fCount += 1

        if fCount > 1:
            raise SystemExit(f"Too many DECOMP files found. Please check the folder: '{self.folder}'")

        if fCount == 0: return

        decompFilepath = f"{self.folder}/{decompFilename}"
        with open(decompFilepath, 'r') as f:
            ltxt = f.read().split('\n')

        self.mol_nums = [int(i)-1 for i in ltxt[1].split()[1:]]

    def apply_mol_nums_to_COM(self, COM):
        """
        Will read the DECOMP file and splice the COM using it.

        Inputs:
            * COM <np.ndarray> => The centers of mass (nmol, 3)
        """
        self.read_DECOMP()
        if hasattr(self, "mol_nums"):
            return COM[self.mol_nums, :]
        return COM


    def calc_3D_MSD_Elstner(self):
        """
        Will read the initial positions file -pos-init.xyz and
        use the coeff populations to calculate Elstner MSD.
        """
        Pos = xyz_ut.XYZ_File(f"{self.folder}/pos-init.xyz")
        COM = xyz_ut.calc_COM(Pos)
        COM = self.apply_mol_nums_to_COM(COM)
        nstep = len(self.Pops)
        nmol = len(COM)

        # Initial population weighted center of mass
        initVec = np.sum(self.Pops[0][:,None]* COM, axis=0)

        istep = 0
        self.MSD = np.zeros((nstep, 3, 3))

        MSD_R = np.zeros((nmol, 3, 3))
        for imol in range(nmol):
            R = COM[imol, :] - initVec
            MSD_R[imol] = np.outer(R, R)

        for istep in range(nstep):
            for imol in range(nmol):
                u = self.Pops[istep, imol]
                self.MSD[istep] += u * MSD_R[imol]

        return self.MSD

    def calc_IPR(self):
        """
        Will calculate the IPR.
        """
        self.IPR = np.array([1/np.sum([i**2 for i in j]) for j in self.Pops])
        return self.IPR

    def write_MSD(self, folder, filename):
        """
        Will write the MSD data to a file.

        Inputs:
            * folder <str> => The folder to save in (will be created)
            * filename <str> => The filename
        """
        csv_ut._write_csv_helper_(folder,
                           filename,
                           self.MSD,
                           ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"),
                           self.timesteps)

    def write_IPR(self, folder, filename):
        """
        Will write the IPR to a file.

        Inputs:
            * folder <str> => The folder to save in (will be created)
            * filename <str> => The filename
        """
        csv_ut._write_csv_helper_(folder,
                           filename,
                           self.IPR,
                           ("ipr"),
                           self.timesteps)


def average_quantity(allFolderData, quantity_name):
    """
    Will return all occurances of a quantity from a list of the FolderData classes.

    The available quantities to average over are the attributes from the FolderData
    class. Use dir() to check what these are or read the code above.

    Inputs:
        * allFolderData <list<FolderData>> => A list of all the FolderData instances
                                              you want to average over.
        * quantity_name <str> => The name of the attribute within the FolderData
                                 class to average (e.g: 'MSD' or 'IPR'...)
    """
    # Some input checking
    if len(allFolderData) == 0:
        raise SystemExit("\n\n\nError in average_quantity function.\n\nCan't average over an empy array.")

    if not hasattr(allFolderData[0], quantity_name):
        qnts = [i for i in dir(allFolderData[0])
                if isinstance(getattr(allFolderData[0], i), (list, tuple, np.ndarray))]
        qnts = '\n\t* '.join(qnts)
        raise SystemExit("\n\n\nCan't find the quantity specified please choose from the list below:\n\t* "+qnts)

    # Actually do the averaging
    all_data = []
    min_len = float("inf")
    for FolderData in allFolderData:
        qnt = getattr(FolderData, quantity_name)
        all_data.append(qnt)
        min_len = len(qnt) if len(qnt) < min_len else min_len

    all_data = np.array([i[:min_len] for i in all_data])
    avg_data = np.mean(all_data, axis=0)
    timesteps = np.array(allFolderData[0].timesteps[:min_len])

    return avg_data, all_data, timesteps



