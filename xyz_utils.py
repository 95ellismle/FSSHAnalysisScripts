#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:32:22 2019

A module containing methods relevant to xyz files.

The class XYZ_File at the top will store the data read from an xyz file and overload operators
in order to make the data manipulatable.

To read an xyz file use the function `read_xyz_file(filepath <str>)`

To write an xyz file use the function `write_xyz_file(xyz_data <np.array | list>, filepath <str>)`
"""

import re
import copy
import os
from collections import OrderedDict
import numpy as np

def cut_mols(xyz, slice_instruct=False, at_inds=False, ats_per_mol=36):
    """
    Will slice a molecular system by center of mass.

    Parameters
    ----------
    xyz : XYZ_File
        The xyz file data.
    slice_instruct : dict
        Instructions on how to slice.
        Can have keys'xmax', 'xmin', 'ymax', 'ymin', 'zmax', 'zmin'.
        max is inclusive of the boundary
        min is inclusive of the boundary

    Returns
    -------
    array:
        xyz_data of sliced system.

    """
    if slice_instruct is not False:
        COMs = calc_COM(xyz, ats_per_mol)
        mol_inds = np.arange(len(COMs))
        dim_map = {'x': 0, 'y': 1, 'z': 2}

        # Loop over possible slices and apply them
        for dim in 'xyz':
            for lim in ('min', 'max',):

                key = f"{dim}{lim}"
                if (key in slice_instruct):
                    dim_ind = dim_map[dim]

                    if 'max' in key:
                        mask = COMs[:, dim_ind] <= slice_instruct[key]
                        mol_inds = mol_inds[mask]
                        COMs = COMs[mask]
                    else:
                        mask = COMs[:, dim_ind] >= slice_instruct[key]
                        mol_inds = mol_inds[mask]
                        COMs = COMs[mask]

        at_inds = [j for i in mol_inds for j in range(i*ats_per_mol, (i+1)*ats_per_mol)]

    else:  mol_inds = np.arange(len(at_inds)//ats_per_mol)

    # Select the cut atoms
    cut_xyz = xyz.copy()
    cut_xyz.xyz_data, cut_xyz.cols = cut_xyz.index_atoms(at_inds)

    return cut_xyz, at_inds, mol_inds


def calc_COM(XYZ, ats_per_mol=36):
    """
    Will calculate the center of mass of a molecule

    Inputs:
        * XYZ => The xyz file class (see below)
        * ats_per_mol => num ats per molecule
    Outputs:
        <array> Center of masses
    """
    mass_dict = {'C': 12.011, 'H': 1.008}
    masses = copy.deepcopy(XYZ.cols)
    for i in set(XYZ.cols[0]):
        masses[masses == i] = mass_dict[i]

    XYZ_shape = XYZ.shape
    nmol = float(XYZ_shape[1]) / ats_per_mol
    if int(nmol) != nmol:
        raise SystemExit("Incorrect number of atoms or ats_per_mol.")
    nmol = int(nmol)

    masses = np.reshape(masses, (nmol, ats_per_mol)).astype(float)
    mol_pos = np.reshape(XYZ, (nmol, ats_per_mol, 3))

    for i in range(3):
        mol_pos[:, :, i] *= masses
    COMs = np.sum(mol_pos, axis=1) / sum(masses[0])

    return COMs


def eval_type(String):
	"""
	Will convert a string to a number if possible. If it isn't return a string.

	Inputs:
		* String  =>  Any string

	Outpus:
		* If the string can be converted to a number it will be with ints being
		  preferable to floats. Else will return the same string
	"""
	if is_float(String):
		return float(String)
	elif is_num(String):
		return int(String)
	elif String == "True":
		return True
	elif String == "False":
		return False
	else:
		return String


def is_num(Str):
	"""
	Check whether a string can be represented as a number

	Inputs:
	  * Str <str> => A string to check

	Outputs:
	  <bool> Whether the string can be represented as a number or not.
	"""
	try:
		float(Str)
		return True
	except ValueError:
		return False


def is_float(Str):
    """
    Check whether a string can be represented as a non-integer float
    Inputs:
        * Str <str> => A string to check

    Outputs:
      <bool> Whether the string can be represented as a non-integer float or not.
    """
    if isinstance(Str, (str, )):
        if is_num(Str) and '.' in Str:
            return True
        return False


def is_int(num, msg=False):
	"""
	Will check whether a parameter is a float or not.

	If a msg is supplied a TypeError will be thrown if the float isn't an int.

	Inputs:
		* num <float> => Number to check
		* msg <str> => The msg to report if it isn't an int
	Outputs:
		<bool>
	"""
	if not num.is_integer():
		if msg is not False:
			raise TypeError(msg)
		else:
			return False
	else:
		return True


def rm_quotation_marks(String):
    """
    Will remove any quotation marks from a string.

    Inputs:
        * String <str> => A string that needs quotation marks removed.
    Outputs:
        <str> A string with quotation marks removed.
    """
    String = String.strip()
    if len(String) < 2: return String
    if len(String) == 2:
        if String == "''" or String == '""':
            return ""

    for i in ('"', "'"):
        while String[0] == i and String[-1] == i:
            String = String[1:-1]

    return String


def get_str_between_delims(string, start_delim='"', end_delim=False):
    """
    Will get the string between 2 delimeters.

    E.g. if a string = 'bob "alice"' this function would return
    ('bob ', 'alice')

    Inputs:
        * string <str> => The txt to search through
        * delim <str> => The delimeter
    Outputs:
        (<str>, <str>) The line without the text within the delimeter and the text within
    """
    if end_delim is False:  end_delim = start_delim

    start_ind = string.find(start_delim)
    if start_ind == -1:
        return "", string

    end_ind = get_bracket_close(string[start_ind:], start_delim=start_delim,
                                end_delim=end_delim)
    end_ind += start_ind

    txt_within_delim = string[start_ind+1: end_ind]
    txt_without_delim = string[:start_ind+1] + r"??!%s!?" + string[end_ind:]
    return txt_within_delim, txt_without_delim

def split_str_by_multiple_splitters(string, splitters):
    """
    Will split a string by multiple values.

    For example if a string was 'a,b.c-d' and the splitters were ',.-' then this
    would return ['a', 'b', 'c']

    Inputs:
        * splitters <iterable> => The values to split the string by.
    Outputs:
        <list<str>> The string split by many splitters
    """
    split_parts = []
    build_up = ""
    for i in string:
        if i in splitters:
            if build_up:
                split_parts.append(build_up)
            build_up = ""
        else:
            build_up += i
    if build_up: split_parts.append(build_up)
    return split_parts

def rm_comment_from_line(line, comment_str='#'):
    """
    Will remove any comments in a line for parsing.

    Inputs:
        * line   =>  line from input file

    Ouputs:
        The line with comments removed
    """
    # Split the line by the comment_str and join the bit after the comment delim
    words = line.split(comment_str)
    if len(words) >= 1:
        line = words[0]
        comment = comment_str.join(words[1:])
    else:
        line = ''.join(words[:-1])
        comment = ""

    return line, comment

def get_bracket_close(txt, start_delim='(', end_delim=')'):
    """
    Get the close of the bracket of a delimeter in a string.

    This will work for nested and non-nested delimeters e.g. "(1 - (n+1))" or
    "(1 - n)" would return the end index of the string.

    Inputs:
        * txt <str> => A string with a bracket to be closed including the opening
                       bracket.
        * delim <str> OPTIONAL => A delimeter (by default it is an open bracket)
    Outputs:
        <int> The index of the corresponding end_delim
    """
    start_ind = txt.find(start_delim)
    brack_num = 1
    for ichar in range(start_ind+1, len(txt)):
        char = txt[ichar]
        if char == end_delim: brack_num -= 1
        elif char == start_delim: brack_num += 1

        if brack_num == 0:
            return ichar + start_ind

    else:
        if txt[-1] == end_delim:
            return len(txt) - 1
        return -1



def get_nums_in_str(string, blank_is_0=False):
    """
    Will get only the numbers from a string

    Inputs:
        * string <str> => The string to get numbers from
    Outputs:
        <list<float>> The numbers from a string
    """
    all_nums = re.findall("[0-9.]+", string)
    if blank_is_0 and len(all_nums) == 0:
        return [0.0]
    else:
        return [float(i) for i in all_nums]


def remove_num_from_str(string):
    """
    Will remove any numbers from a string object.

    Inputs:
        * string <str> => The string to remove numbers from
    Outputs:
        <str> A string without numbers
    """
    all_nums = re.findall("[0-9]", string)
    for i in all_nums:
        string = string.replace(i, "")

    return string

class DataFileStorage(object):
    """
    A class to act as a template for other classes that load and store data from files to use.

    The class takes the filepath as an input and calls a method named parse(). Any data stored
    under the name numeric_data will be manipulated via the operator overload functions.

    Inputs:
        * filepath <str> => The path to the file to be loaded.
    """
    numeric_data = 0
    _poss_num_types_ = ('numeric_data', 'xyz_data', "csv_data")
    _numeric_data_types = []
    _defaults = {}
    metadata = {'file_type': 'txt'}
    def __init__(self, filepath):
        # Copy all dictionaries to make them unique for each instance
        all_vars = [i for i in dir(self) if i[0] != '_']
        for i in all_vars:
            if i[0] != '_':
                var = getattr(self, i)
                if not callable(var) and isinstance(var, (dict, list, tuple)):
                    setattr(self, i, copy.deepcopy(var))

        self.filepath = filepath
        self.file_txt = open_read(self.filepath)

        # Set the default parameters
        for key in self._defaults:
            if key not in self.metadata:
                self.metadata[key] = self._defaults[key]

        self.parse()
        for var in dir(self):
            if var in self._poss_num_types_:
                self._numeric_data_types.append(var)


    # Dummy method to hold the place of an actual parser later
    def parse(self):
        """
        Should be overridden in any child classes.
        """
        pass

    def set_data(self):
        """
        Should be overwritten, is a dummy function for deciding which data to calc with.
        """
        pass

    # Overload adding
    def __add__(self, val):
        for i in self._numeric_data_types:
            att = getattr(self, i)
            att += float(val)
            setattr(self, i, att)
        return self

    # Overload multiplying
    def __mul__(self, val):
        for i in self._numeric_data_types:
            att = getattr(self, i)
            att *= float(val)
            setattr(self, i, att)
        return self

    # Overload subtracting
    def __sub__(self, val):
        for i in self._numeric_data_types:
            att = getattr(self, i)
            att -= float(val)
            setattr(self, i, att)
        return self

    # Overload division operator i.e. a / b
    def __truediv__(self, val):
        for i in self._numeric_data_types:
            att = getattr(self, i)
            att /= float(val)
            setattr(self, i, att)
        return self

    # Overload floor division operator i.e. a // b
    def __floordiv__(self, val):
        for i in self._numeric_data_types:
            att = getattr(self, i)
            att //= float(val)
            setattr(self, i, att)
        return self

    # Overload the power operator
    def __pow__(self, val):
        for i in self._numeric_data_types:
            att = getattr(self, i)
            att **= float(val)
            setattr(self, i, att)
        return self

    # str() would return filetxt by default
    def __str__(self):
        return self.file_txt


class Write_File(object):
    """
    A parent class for other file writing classes.

    The general procedure for writing a file is first create the file text
    (saved as self.file_txt) and then write it as a file to the filepath given
    as an argument.

    The function create_file_str must be set and must return the file text.

    This class can also be used to write general files

    Inputs:
       * Data_Class <class> => The class containing all the data to be written
       * filepath <str>     => The path to the file to be written.
       * extension <str> OPTIONAL   => The file extension. Default is False.
    """
    def __init__(self, Data_Class, filepath, ext=False):
        self.filepath = filepath
        self.Data = Data_Class
        self.extension = ext

        # Get the str to write to a file
        self.file_txt = self.create_file_str()

        if type(self.file_txt) == list:
            if len(self.file_txt) == 1:
                self.write_single_file(self.file_txt[0], filepath, ext)

            else:
                for i, s in enumerate(self.file_txt):
                    self.write_single_file(s, filepath, ext, i)

        else:
            self.write_single_file(self.file_txt, filepath, ext)


    def write_single_file(self, s, filepath, ext, file_num=False):
        """
        Will write a single file from a string.

        Inputs:
            * s <str> => The text to write
        """
        # Correct the extension
        if self.extension:
            filepath, user_ext = remove_file_extension(filepath)
            if ext is False: ext = user_ext

            if file_num:
                count = 0
                filepath_with_num = f"{filepath}_{count}.{ext}"

                while os.path.isfile(filepath_with_num):
                    filepath_with_num = f"{filepath}_{count}.{ext}"
                    count += 1
                filepath = filepath_with_num

            else:
                filepath = f"{filepath}.{ext}"

        # Write the file with the correct extension
        with open(filepath, 'w') as f:
            f.write(s)


    def create_file_str(self):
      """
      To be overwritten by children to create the str to be written to a file.

      By default we write the str(self.Data)
      """
      return str(self.Data)


# Reads a file and closes it
def open_read(filename, throw_error=True):
    """
    A slightly unnecessary function to open a file and read it safely, with the choice to raise an error.

    Inputs:
        * filename <str> => The path to the file that needs opening
        * throw_error <bool> OPTIONAL => Choose whether to throw an error if the file can't be found. Default: True

    Outputs:
        <str> The file text.
    """
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            txt = f.read()
        return txt
    else:
        if throw_error:
            raise SystemExit("The %s file doesn't exist!" % filename)
        return False


# Get folder from filepath
def get_folder_from_filepath(filepath):
   """
   Will get the folderpath of the directory containing the file given in filepath.

   This will not raise an error if the filepath doesn't exist. If the filepath given
   is actually a folderpath the input will be returned.

   Inputs:
      * filepath <str> => A str pointing towards a file.
   Outputs:
      <str> folderpath
   """
   if os.path.isdir(filepath) or '/' not in filepath:
      return ""

   folder = filepath[:filepath.rfind('/')]  # Splice up to the last '/'
   return folder

# Get filename from filepath
def get_filename_from_filepath(filepath):
   """
   Will get the name of the file being pointed to in the filepath.

   This will not raise an error if the filepath doesn't exist. If the filepath given
   is actually a folderpath the top directory in that folderpath will be returned. If
   the filepath given doesn't point to any files it will simply be returned.

   Inputs:
      * filepath <str> => A str pointing towards a file.
   Outputs:
      <str> filename
   """
   if '/' not in filepath:
      return filepath

   filename = filepath[filepath.rfind('/')+1:]  # Splice up to the last '/'
   return filename


def remove_file_extension(filepath):
    """
    Will remove the file extension from a filepath.

    Inputs:
        * filepath <str> => The filepath which should have the extension removed.
    Outputs:
        (<str>, <str>) The filepath with it's extension removed.
    """
    if '.' not in filepath:
        return filepath

    # The index of the extension
    ext_ind = filepath.rfind('.')

    # Splice the filepath into extension and filepath
    new_filepath = filepath[:ext_ind]
    ext = filepath[ext_ind+1:]
    return new_filepath, ext


def get_abs_path(filepath):
   """
   Will check if a file/folder path exists and if it does return it's absolute path.

   If the filepath doesn't exists an error IOError will be raised.

   Inputs:
      * filepath <str> => A str pointing towards a file.
   Outputs:
      <str> filepath
   """
   filepath = os.path.expanduser(filepath)
   filepath = os.path.abspath(filepath)

   if not os.path.isfile(filepath) and not os.path.isdir(filepath):
      raise IOError("Can't find file '%s'" % filepath)
   return filepath


class XYZ_File(DataFileStorage):
    """
    A container to store xyz data in.

    This class will store data from an xyz file and also overload the in_built
    mathematical operators e.g. adding, subtracting, multiplying etc... to allow
    easy manipulation of the data without loosing any metadata.

    Inputs/Attributes:
        * xyz_data <numpy.array> => The parsed xyz data from an xyz file.
        * cols <numpy.array> => The parsed column data from the xyz file.
        * timesteps <numpy.array> => The parsed timesteps from the xyz file.
    """
    metadata = {'file_type': 'xyz'}
    # write_precision = 5
    def __init__(self, filepath=False, xyz_data=False, cols=False,
                       timesteps=False):
        if filepath is not False:
            super().__init__(filepath)
        else:
            self.cols = np.array(cols)
            self.xyz_data = np.array(xyz_data)
            self.timesteps = np.array(timesteps)
            self.__parse2()

    def __parse2(self):
        """
        Will handle the 2nd half of the parsing of the data.
        """
        # Make sure the data are in numpy arrays
        self.cols, self.xyz_data = np.array(self.cols), np.array(self.xyz_data)
        self.timesteps = np.array(self.timesteps)

        # Get some metadata
        self.nstep = len(self.xyz_data)
        self.natom = self.xyz_data.shape[1]
        self.ncol = self.xyz_data.shape[2]

        # Set the metadata
        self.metadata['number_steps'] = self.nstep
        self.metadata['number_atoms'] = self.natom

        # Get numpy attributes and set them in this class (a horrible hack)
        # This makes the XYZ_File object behave like a numpy array -thouugh
        # there are problems.
        #  If shape is called on a changed array then the previous shape will
        #  be returned!
        for i in dir(self.xyz_data):
            if not callable(getattr(self.xyz_data, i)) and '_' != i[:1]:
                setattr(self, i, getattr(self.xyz_data, i))

    def parse(self):
        """
        Will call the 'read_xyz_file' function to parse an xyz file.

        For more info on the specifics of reading the xyz file see 'read_xyz_file'
        below.
        """
        self.xyz_data, self.cols, self.timesteps = read_xyz_file(self.filepath)

        self.__parse2()

    def write(self, filepath):
        s = self._create_single_file_str_(self.xyz_data, self.cols, self.timesteps)
        with open(filepath, "w") as f:
            f.write(s)

    def _create_file_str_(self):
          """
          Will create the string that contains an xyz file, this is save as self.file_txt.
          """
          all_lists = (self.cols, self.xyz_data, self.timesteps)
          if all(type(j) == list for j in all_lists):

              if len(self.cols) != len(self.xyz_data) != self.timesteps:
                raise SystemError("\n\nThe length of the cols, xyz_data and timesteps arrays are different.\n\n"
                                  + "These arrays should all be the same length and should contain info for each file to write.")

              all_file_strings = [self.create_single_file_str(xyz_data, cols, timesteps)
                                  for cols, xyz_data, timesteps in zip(self.cols, self.xyz_data, self.timesteps)]

          # Create the str
          else:
              all_file_strings = self.create_single_file_str(self.xyz_data, self.cols, self.timesteps)


          return all_file_strings

    def _create_single_file_str_(self, xyz_data, cols, timesteps):
        """Will create the xyz file string for a single file."""

        # Create an array of spaces/newlines to add between data columns in str
        natom = len(cols[0])
        space = "    "

        # Convert floats to strings (the curvy brackets are important for performance here)
        xyz = xyz_data#.astype(str)
        xyz = ([np.array2string(line, separator=space, formatter={'float_kind': lambda x: f"{x: .10f}"})[1:-1]
                for line in step_data] for step_data in xyz)

        if len(cols.shape) == 3 and cols.shape[-1] != 1:
            cols = cols[0]
            axis = 1
        else:
            axis = 0

        cols = np.apply_along_axis(lambda x: space.join(map(lambda i: i.ljust(7), x)), axis, cols)

        head_str = '%i\ntime = ' % natom


        s = (head_str + ("%.3f\n" % t) + '\n'.join(np.char.add(cols, step_data)) + "\n"
             for step_data, t in zip(xyz, timesteps))

        return ''.join(s)

    # Overload the str function (useful for displaying data).
    def __str__(self):
        # Create an array of spaces/newlines to add between data columns in str
        space = ["    "] * self.natom

        # Convert floats to strings (the curvy brackets are important for performance here)
        xyz = self.xyz_data.astype(str)
        xyz = (['    '.join(line) for line in step_data] for step_data in xyz)
        if np.shape(self.cols)[-1] > 1:
            cols = [['\t'.join(i) for i in step_data] for step_data in self.cols]
            cols = np.char.add(cols[0], space)
        else:
            cols = np.char.add(self.cols[0], space)
        head_str = '%i\ntime = ' % self.natom
        s = (head_str + ("%.3f\n" % t) + '\n'.join(np.char.add(cols, step_data)) + "\n"
             for step_data, t in zip(xyz, self.timesteps))

        # Create the str
        return ''.join(s)

    def copy(self):
        """
        Return a copy of this class
        """
        return XYZ_File(xyz_data=self.xyz_data, cols=self.cols,
                        timesteps=self.timesteps)

    def index_atoms(self, ats):
        """
        Will index certain atoms depending on those give in the ats list.

        Parameters
        ----------
        ats : list or int
            Get the atoms depending on the index.

        Returns
        -------
        2x array:
            Atom and cols indexed.

        """
        return self.xyz_data[:, ats], self.cols[:, ats]

    def __repr__(self):
        return repr(self.xyz_data)

    def __add__(self, val):
        return self.xyz_data + val

    def __mul__(self, val):
        return self.xyz_data * val

    def __sub__(self, val):
        return self.xyz_data - val

    def __truediv__(self, val):
        return self.xyz_data / val

    def __floordiv__(self, val):
        return self.xyz_data // val

    def __mod__(self, val):
        return self.xyz_data % val

    def __pow__(self, val):
        return self.xyz_data ** val

    def __radd__(self, val):
        return val + self.xyz_data

    def __rmul__(self, val):
        return val + self.xyz_data

    def __rsub__(self, val):
        return val - self.xyz_data

    def __rtruediv__(self, val):
        return val - self.xyz_data

    def __rfloordiv__(self, val):
        return val // self.xyz_data

    def __rmod__(self, val):
        return val % self.xyz_data

    def __rpow__(self, val):
        return val **elf.xyz_data

    def __iadd__(self, val):
        self.xyz_data += val
        return self

    def __imul__(self, val):
        self.xyz_data *= val
        return self

    def __isub__(self, val):
        self.xyz_data -= val
        return self

    def __itruediv__(self, val):
        self.xyz_data /= val
        return self

    def __ifloordiv__(self, val):
        self.xyz_data //= val
        return self

    def __imod__(self, val):
        self.xyz_data %= val
        return self

    def __ipow__(self, val):
        self.xyz_data **= val
        return self

    def __iand__(self, val):
        self.xyz_data &= val
        return self

    def __ior__(self, val):
        self.xyz_data |= val
        return self

    def __idiv__(self, val):
        self.xyz_data /= val
        return self

    def __setitem__(self, ind, val):
        self.xyz_data[ind] = val

    def __getitem__(self, ind):
        return self.xyz_data[ind]

    def __len__(self):
        return len(self.xyz_data)

class Write_XYZ_File(Write_File):
      """
      Will handle the writing of xyz files.

      The parent class Write_File will handle the actual writing and
      this just creates an xyz file string.

     Inputs:
        * Data_Class <class> => The class containing all the data to be written
        * filepath <str>     => The path to the file to be written.
      """
      def __init__(self, Data_Class, filepath):
          # If we can set the xyz variables in the data class then set them
          required_data_fncs = ('get_xyz_data', 'get_xyz_cols', 'get_xyz_timesteps',)

          if all(j in dir(Data_Class) for j in required_data_fncs):
              self.xyz_data = Data_Class.get_xyz_data()
              self.cols = Data_Class.get_xyz_cols()
              self.timesteps = Data_Class.get_xyz_timesteps()
          else:
              non_implemented = ""
              for i in required_data_fncs:
                  if i not in dir(Data_Class): non_implemented += f"'{i}' "
              raise SystemError(f"\n\n\nPlease implement the methods {non_implemented} in {type(Data_Class)}\n\n")

          # Run standard file writing procedure
          super().__init__(Data_Class, filepath)

      def create_file_str(self):
          """
          Will create the string that contains an xyz file, this is save as self.file_txt.
          """
          all_lists = (self.cols, self.xyz_data, self.timesteps)
          if all(type(j) == list for j in all_lists):

              if len(self.cols) != len(self.xyz_data) != self.timesteps:
                raise SystemError("\n\nThe length of the cols, xyz_data and timesteps arrays are different.\n\n"
                                  + "These arrays should all be the same length and should contain info for each file to write.")

              all_file_strings = [self.create_single_file_str(xyz_data, cols, timesteps)
                                  for cols, xyz_data, timesteps in zip(self.cols, self.xyz_data, self.timesteps)]

          # Create the str
          else:
              all_file_strings = self.create_single_file_str(self.xyz_data, self.cols, self.timesteps)


          return all_file_strings

      def create_single_file_str(self, xyz_data, cols, timesteps):
          """Will create the xyz file string for a single file."""

          # Create an array of spaces/newlines to add between data columns in str
          natom = len(cols[0])
          space = ["    "] * natom

          # Convert floats to strings (the curvy brackets are important for performance here)
          xyz = xyz_data.astype(str)
          xyz = (['    '.join(line) for line in step_data] for step_data in xyz)
          cols = np.char.add(cols[0], space)
          head_str = '%i\ntime = ' % natom

          s = (head_str + ("%.3f\n" % t) + '\n'.join(np.char.add(cols, step_data)) + "\n"
               for step_data, t in zip(xyz, timesteps))

          return ''.join(s)

def string_between(Str, substr1, substr2):
    """
    Returns the string between 2 substrings within a string e.g. the string between A and C in 'AbobC' is bob.

    Inputs:
      * line <str> => A line from the xyz file

    Outputs:
      <str> The string between 2 substrings within a string.
    """
    Str = Str[Str.find(substr1)+len(substr1):]
    Str = Str[:Str.find(substr2)]
    return Str

def is_atom_line(line):
    """
    Checks if a line of text is an atom line in a xyz file

    Inputs:
      * line <str> => A line from the xyz file

    Outputs:
      <bool> Whether the line contains atom data
    """
    words = line.split()
    # If the line is very short it isn't a data line
    if not len(words):
        return False

    # Check to see how many float there are compared to other words
    fline = [len(re.findall("[0-9]\.[0-9]", i)) > 0 for i in words]
    percentFloats = fline.count(True) / float(len(fline))
    if percentFloats < 0.5:
        return False
    else:
        return True


def atom_find_more_rigorous(ltxt):
    """
    Will try to find where the atom section starts and ends using some patterns in the file.

    The idea behind this function is we split the first 100 lines into words and find their
    length. The most common length is then assumed to be the length of the atom line and any
    line with this num of words is assumed to be an atom line.

    Inputs:
      * ltxt <list<str>> => The file_txt split by lines
    Outputs:
      * <int>, <int> Where the atom section starts and ends
    """
    # Get the most common length of line -this will be the length of atom line.
    first_100_line_counts = [len(line.split()) for line in ltxt[:100]]
    unique_vals = set(first_100_line_counts)
    modal_val = max(unique_vals, key=first_100_line_counts.count)

    # This means either we have 1 title line, 2 title lines but 1 has the same num words as the atom lines
    #    or 2 title lines and they both have the same length.
    # If this function needs to be more rigorous this can be modified.
    if len(unique_vals) == 2:
      pass

    # This means we haven't found any difference in any lines.
    if len(unique_vals) == 1:
       raise SystemError("Can't find the atom section in this file!")

    start = False
    for line_num, line in enumerate(ltxt):
        len_words = len(line.split())

        # Have started and len words changed so will end
        if len_words != modal_val and start is True:
            atom_end = line_num
            break

        # Haven't started and len words changed so will start
        if len_words == modal_val and start is False:
            start = True
            prev_line = False
            atom_start = line_num
    else:
      atom_end = len(ltxt)

    return atom_start, atom_end - atom_start



# Finds the number of title lines and number of atoms with a step
def find_num_title_lines(step): # should be the text in a step split by line
    """
    Finds the number of title lines and number of atoms with a step

    Inputs:
       * step <list<str>> => data of 1 step in an xyz file.

    Outputs:
       <int> The number of lines preceeding the data section in an xyz file.
    """
    num_title_lines = 0
    for line in step:
        if is_atom_line(line):
            break
        num_title_lines += 1

    return num_title_lines


# Finds the delimeter for the time-step in the xyz_file title
def find_time_delimeter(step, filename):
    """
    Will find the delimeter for the timestep in the xyz file title.

    Inputs:
       * step <list<str>> => data of 1 step in an xyz file.
       * filename <str> => the name of the file (only used for error messages)

    Outputs:
       <str> The time delimeter used to signify the timestep in the xyz file.
    """
    # Check to see if we can find the word time in the title lines.
    for linenum, txt in enumerate(step):
        txt = txt.lower()
        if 'time' in txt:
            break
    else:
        return [False] * 2

    # find the character before the first number after the word time.
    prev_char, count = False, 0
    txt = txt[txt.lower().find("time"):]
    for char in txt.replace(" ",""):
        isnum = (char.isdigit() or char == '.')
        if isnum != prev_char:
            count += 1
        prev_char = isnum
        if count == 2:
            break
    if char.isdigit(): return '\n', linenum
    else: return char, linenum
    return [False] * 2


def get_num_data_cols(ltxt, filename, num_title_lines, lines_in_step):
    """
    Will get the number of columns in the xyz file that contain data. This isn't a foolproof method
    so if there are odd results maybe this is to blame.

    Inputs:
        * ltxt <list<str>> => A list with every line of the input file as a different element.
        * filename <str> => The path to the file that needs opening
        * num_title_lines <int> => The number of non-data lines
        * lines_in_step <int> => How many lines in 1 step of the xyz file
    Outputs:
        <int> The number of columns in the data section.
    """
    dataTxt = [ltxt[num_title_lines + (i*lines_in_step) : (i+1)*lines_in_step]
               for i in range(len(ltxt) // lines_in_step)]

    num_data_cols_all = []
    for step in dataTxt[:20]:
       for line in step:
          splitter = line.split()
          count = 0
          for item in splitter[-1::-1]:
              if not is_float(item):
                  num_data_cols_all.append(count)
                  break
              count += 1

    num_data_cols = max(set(num_data_cols_all), key=num_data_cols_all.count)
    return num_data_cols


# Will get necessary metadata from an xyz file such as time step_delim, lines_in_step etc...
# This will also create the step_data dictionary with the data of each step in
def get_xyz_metadata(filename, ltxt=False):
    """
    Get metadata from an xyz file.

    This function is used in the reading of xyz files and will retrieve data necessary for reading
    xyz files.

    Inputs:
       * filename <str> => the path to the file that needs parsing
       * ltxt <list<str>> OPTIONAL => the parsed txt of the file with each line in a separate element.

    Outputs:
       <dict> A dictionary containing useful parameters such as how many atom lines, how to get the timestep, num of frames.
    """
    if ltxt == False:
        ltxt = open_read(filename).split('\n')
    # Check whether to use the very stable but slow parser or quick slightly unstable one
    most_stable = False
    if any('**' in i for i in ltxt[:300]):
        most_stable = True

    if not most_stable:
        num_title_lines, num_atoms = atom_find_more_rigorous(ltxt)
        lines_in_step = num_title_lines + num_atoms
        if len(ltxt) > lines_in_step+1: # take lines from the second step instead of first as it is more reliable
           step_data = {i: ltxt[i*lines_in_step:(i+1)*lines_in_step] for i in range(1,2)}
        else: #If there is no second step take lines from the first
           step_data = {1:ltxt[:lines_in_step]}
    else:
        lines_in_step = find_num_title_lines(ltxt, filename)
        step_data = {i: ltxt[i*lines_in_step:(i+1)*lines_in_step] for i in range(1,2)}
        num_title_lines = find_num_title_lines(step_data[1])

    nsteps = int(len(ltxt)/lines_in_step)
    time_delim, time_ind = find_time_delimeter(step_data[1][:num_title_lines],
                                               filename)
    num_data_cols = get_num_data_cols(ltxt, filename, num_title_lines, lines_in_step)
    return {'time_delim': time_delim,
            'time_ind': time_ind,
            'lines_in_step': lines_in_step,
            'num_title_lines': num_title_lines,
            'num_data_cols': num_data_cols,
            'nsteps': nsteps}


def splitter(i):
    """
    Splits each string in a list of strings.
    """
    return [j.split() for j in i]


def read_xyz_file(filename, num_data_cols=False,
                  min_time=0, max_time='all', stride=1,
                  ignore_steps=[], metadata=False):
    """
    Will read 1 xyz file with a given number of data columns.

    Inputs:
        * filename => the path to the file to be read
        * num_data_cols => the number of columns which have data (not metadata)
        * min_time => time to start reading from
        * max_time => time to stop reading at
        * stride => what stride to take when reading
        * ignore_steps => a list of any step numbers to ignore.
        * metadata => optional dictionary containing the metadata

    Outputs:
        * data, cols, timesteps = the data, metadata and timesteps respectively
    """
    # Quick type check of param fed into the func
    if type(stride) != int and isinstance(min_time, (int, float)):
            print("min_time = ", min_time, " type = ", type(min_time))
            print("max_time = ", max_time, " type = ", type(max_time))
            print("stride = ", stride, " type = ", type(stride))
            raise SystemExit("Input parameters are the wrong type!")

    # Get bits of metadata
    if num_data_cols is not False:
       num_data_cols = -num_data_cols
    ltxt = [i for i in open_read(filename).split('\n') if i]
    if metadata is False:
        metadata = get_xyz_metadata(filename, ltxt)
        if num_data_cols is False:
           num_data_cols = -metadata['num_data_cols']
    lines_in_step = metadata['lines_in_step']
    num_title_lines = metadata['num_title_lines']
    get_timestep = metadata['time_ind'] is not False
    if get_timestep:
       time_ind = metadata['time_ind']
       time_delim = metadata['time_delim']
    num_steps = metadata['nsteps']

    # Ignore any badly written steps at the end
    badEndSteps = 0
    for i in range(num_steps, 0, -1):
      stepData = ltxt[(i-1)*lines_in_step:(i)*lines_in_step][num_title_lines:]
      badStep = False
      for line in stepData:
         if '*********************' in line:
            badEndSteps += 1
            badStep = True
            break
      if badStep is False:
         break


    # The OrderedDict is actually faster than a list here.
    #   (time speedup at the expense of space)
    step_data = OrderedDict()  # keeps order of frames -Important
    all_steps = [i for i in range(0, num_steps-badEndSteps, stride)
                 if i not in ignore_steps]

    # Get the timesteps
    if get_timestep:
       timelines = (ltxt[time_ind+(i*lines_in_step)] for i in all_steps)
       timesteps = (string_between(line, "time = ", time_delim)
                    for line in timelines)
       timesteps = (get_nums_in_str(i, True) for i in timesteps)
       timesteps = np.array([i[0] if len(i) == 1 else 0.0 for i in timesteps])
       timesteps = timesteps.astype(float)

    else:
       print("***********WARNING***************\n\nThe timesteps could not be extracted\n\n***********WARNING***************")
       timesteps = [0] * len(all_steps)

    # Get the correct steps (from min_time and max_time)
    all_steps = np.array(all_steps)
    if get_timestep:
       mask = timesteps >= min_time
       if type(max_time) == str:
         if 'all' not in max_time.lower():
            msg = "You inputted max_time = `%s`\n" % max_time
            msg += "Only the following are recognised as max_time parameters:\n\t*%s" % '\n\t*'.join(['all'])
            print(msg)
            raise SystemExit("Unknown parameter for max_time.\n\n"+msg)
       else:
         mask = mask & (timesteps <= max_time)
       all_steps = all_steps[mask]
       timesteps = timesteps[mask]

    # Get the actual data (but only up to the max step)
    min_step, max_step = min(all_steps), max(all_steps)+1
    all_data = np.array(ltxt)[min_step*metadata['lines_in_step']:max_step*metadata['lines_in_step']]

    # get the data from each step in a more usable format
    step_data = np.reshape(all_data, (len(all_steps), lines_in_step))
    step_data = step_data[:, num_title_lines:]

    # This bit is the slowest atm and would benefit the most from optimisation
    # tmp = np.array([[len(i.split()) for i in j] for j in step_data])
    # print(step_data[tmp != 3])
    step_data = np.apply_along_axis(splitter, 1, step_data)
    data = step_data[:, :, num_data_cols:].astype(float)

    # If there is only one column in the cols then don't create another list!
    if (len(step_data[0, 0]) + num_data_cols) == 1:
        cols = step_data[:, :, 0]
    else:
        cols = step_data[:, :, :num_data_cols]

    return data, cols, timesteps


class Psuedo_Ham:
    """
    Will read the psuedo-hamiltonian file (which is itself a psuedo xyz file).

    Inputs:
        * filename <str> = The location of the file

    Important Parameters:
        * self.data <list<dict>> => The data straight from the file.
    """
    timesteps = []

    def __init__(self, filename):
        self.filename = filename

        with open(self.filename, 'r') as f:
            self.ftxt = f.read()

        self._parse()

    def _parse(self):
        """
        Will do the parsing and store the data in a structure named self.data.
        """
        steps = self._get_steps()
        self.data = [self._parse_step(step) for step in steps]


    def _get_steps(self):
        """
        Will return the steps as a generator object.
        """
        steps = self.ftxt.split("H (hartree);")[1:]
        return map(lambda x: ('H (hartree)' + x).split("\n"), steps)

    def _parse_step(self, step_ltxt):
        """
        Will parse 1 step.

        Inputs:
            * step_ltxt <list<str>> => The data for each step in a 2D list.

        Outputs:
            dict:
                Will return 1 dict with the reduced hamiltonian in it.
        """
        titles = step_ltxt[:2]
        if not hasattr(self, 'nstate'):
            self.nstate = int(string_between(titles[0], "nadiab =", ";"))
        self.timesteps.append(float(string_between(titles[1], "time =", "(fs)")))

        txt_data = step_ltxt[2:]
        data = {}
        for line in txt_data:
            if line:
                i, j, val = line.split()
                data.setdefault(int(i)-1, {})[int(j)-1] = float(val)

        return data

    def create_H(self, data_step):
        """
        Will create a Hamiltonian from a data dictionary

        Inputs:
            * data_step <dict | int> => If int: will use that index in self.data and return the Hamiltonian corresponding to that.
                                        If dict: will use that dictionary as the data and return the H corresponding to that.

        Outputs:
            2D Array:
                Hamiltonian data in full representation
        """
        if type(data_step) is int:
            if data_step > len(self.data) or data_step < 0:
                raise SystemExit("Data index out of bounds!")
            data = self.data[data_step]

        elif type(data_step) == dict:
            data = data_step

        else:
            raise SystemExit("Unknown argument type!")

        H = np.zeros((self.nstate, self.nstate))
        inds = []
        for i in data:
            for j in data[i]:
                H[i, j] = data[i][j]
                H[j, i] = data[i][j]

        return H


def calc_IPR(mol_pops):
    """
    Will calculate the IPR from populations

    Inputs:
        * mol_pops <array: float> => The molecular populations
                                     shape(nstep, nmol)
    """
    return np.array([1/np.sum([i**2 for i in j]) for j in mol_pops])




