import numpy as np
import tools
import tqdm

def read_cbin(filename, bits=32, order='C', dimensions=3, records=False):

        """
        Read a binary file with three inital integers (a cbin file).
        
        Parameters:
                * filename (string): the filename to read from
                * bits = 32 (integer): the number of bits in the file
                * order = 'C' (string): the ordering of the data. Can be 'C'
                          for C style ordering, or 'F' for fortran style.
                * dimensions (int): the number of dimensions of the data (default:3)
                * records (boolean): does the file contain record separators?
                        
        Returns:
                The data as a N dimensional numpy array.
        
        Example:
                > data = read_cbin(filename='nsrc_z6.981.dat')

        """

        assert(bits==32 or bits==64)

        f = open(filename)
        
        print('Reading cbin file: %s' % filename)

        # read in the header
        counter = dimensions+3 if records else dimensions 
        header = np.fromfile(f, count=counter, dtype='int32')
        print('header:', header)

        # array shape for the reshaping of the data
        if records: temp_mesh=header[1:4]
        else: temp_mesh=header[0:3]

        # extract and reshape data
        datatype = np.float32 if bits==32 else np.float64
        data = np.fromfile(f, dtype=datatype, count=np.prod(temp_mesh))
        data = data.reshape(temp_mesh, order=order)
        f.close()

        return data


def convert_dat_to_npy(filename, bits=32, order='C', dimensions=3, records=False):

        """
        Convert .dat to .npy
        """

        assert(bits==32 or bits==64)

        f = open(filename)
        
        print('Reading cbin file: %s' % filename)

        # get array shape for the reshaping of the data
        counter = dimensions+3 if records else dimensions
        header = np.fromfile(f, count=counter, dtype='int32')
        if records: temp_mesh=header[1:4]
        else: temp_mesh=header[0:3]

        # extract, reshape and save the data
        datatype = np.float32 if bits==32 else np.float64
        data = np.fromfile(f, dtype=datatype, count=np.prod(temp_mesh))
        new_filename = filename[:-3] + 'npy'
        data = data.reshape(temp_mesh, order=order)
        print('new_filename, shape:', new_filename, data.shape)
        np.save(new_filename, data)
        
        f.close()
        


def convert_npy_to_dat(filename, bits=32, order='C', dimensions=3, records=False):

        """
        Convert .npz to .dat
        """

        assert(bits==32 or bits==64)

        data = np.load(filename, mmap_mode='r')
        
        print('filename, shape:', filename, shape)

        new_filename = filename[:-3] + 'dat'
        f = open(new_filename, 'wb')
        
        # header contains the array shape of the data
        mesh = np.array(data.shape).astype('int32')
        mesh.tofile(f)

        # flatten the data and write it to the file
        datatype = (np.float32 if bits==32 else np.float64)
        data.flatten(order=order).astype(datatype).tofile(f)
        
        f.close()


def save_cbin(filename, data, bits=32, order='C'):

        """
        Save a binary file with three inital integers (a cbin file).
        
        Parameters:
                * filename (string): the filename to save to
                * data (numpy array): the data to save
                * bits = 32 (integer): the number of bits in the file
                * order = 'C' (string): the ordering of the data. Can be 'C'
                        for C style ordering, or 'F' for fortran style.
                        
        Returns:
                Nothing
        """
        
        print('Saving cbin file: %s' % filename)
        assert(bits==32 or bits==64)
        f = open(filename, 'wb')
        mesh = np.array(data.shape).astype('int32')
        mesh.tofile(f)
        datatype = (np.float32 if bits==32 else np.float64)
        data.flatten(order=order).astype(datatype).tofile(f)
        f.close()


# --------------------------------------------------------------------

# irate = read_cbin('irate_z10.110.dat')
# print('irate[0,0,0]:',irate[0,0,0])

# msrc = read_cbin('msrc_z10.110.dat')
# print('msrc[0,0,0]:',msrc[0,0,0])

datatypes = ['msrc', 'irate', 'overd', 'xHII']
filepath = "/cosma8/data/dp004/dc-seey1/ml_reion/data/AI4EoR_dataset/"
for datatype in datatypes:
        z_str, files = tools._get_files(filepath, datatype, extension='dat')
        for (z, filename) in files.items():
                convert_dat_to_npy(filename)

