from pathlib import Path
import numpy as np


class readsim:
    def __init__(self, path):
        self.path = path

    def __repr__(self):
        return self.path

    def readall(self):
        """
        Read the configuration file
        Read the lookup table
        Read the index array
        """
        p = Path(self.path)

        dirContents = ['config.txt', 'configheaders.txt',
                       'swc.bin', 'lut.bin', 'index.bin',
                       'pairs.bin', 'bounds.bin', 'dims.txt',
                       'dimheaders.txt']

        for fp in dirContents:
            if (not (p.joinpath(fp).exists())):
                raise RuntimeError(
                    "Incomplete Dir: Does not contain {}".format(fp))

        print("Found all files \n")

        configdata = p.joinpath(dirContents[0]).read_text().split("\t\n")
        headerdata = p.joinpath(dirContents[1]).read_text().split("\n")

        swcdata = np.fromfile(p.joinpath(
            dirContents[2]), dtype=np.double, count=-1, sep='', offset=0, like=None)

        lutdata = np.fromfile(p.joinpath(
            dirContents[3]), dtype=np.uint64, count=-1, sep='', offset=0, like=None)

        indexdata = np.fromfile(p.joinpath(
            dirContents[4]), dtype=np.uint64, count=-1, sep='', offset=0, like=None)

        pairsdata = np.fromfile(p.joinpath(
            dirContents[5]), dtype=np.uint64, count=-1, sep='', offset=0, like=None)

        boundsdata = np.fromfile(p.joinpath(
            dirContents[6]), dtype=np.uint64, count=-1, sep='', offset=0, like=None)

        dimdata = p.joinpath(dirContents[7]).read_text().split("\n")
        dimheaderdata = p.joinpath(dirContents[8]).read_text().split("\n")

        dimensions = dict()

        """
        FIXME: this is broken. Dimensions are not being interpreted right.
        """

        for x, y in zip(dimdata, dimheaderdata):
            if len(x) > 0:
                val = np.uint(x.split('\t'))
                dimensions[y] = val.transpose()

        swcdata = swcdata.reshape(dimensions["swc_dims"], order="F")
        lutdata = lutdata.reshape(dimensions["lut_dims"], order="F")
        indexdata = indexdata.reshape(dimensions["index_dims"], order="F")
        pairsdata = pairsdata.reshape(dimensions["pairs_dims"], order="F")
        boundsdata = boundsdata.reshape(dimensions["bounds_dims"], order="F")

        boundsdata.ndim
        d = dict()

        for x, y in zip(headerdata, configdata):
            if len(x) > 0:
                d[x] = np.double(y)

        params = dict()
        for key in d.keys():
            k = d[key]
            params[key] = k

        params["swc"] = swcdata
        params["lut"] = lutdata
        params["index"] = indexdata
        params["pairs"] = pairsdata
        params["bounds"] = boundsdata
        return params
