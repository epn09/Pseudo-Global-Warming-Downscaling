# Copy from https://github.com/coecms/nccompress/blob/master/nccompress/nc2nc.py

import operator
from functools import reduce

import numpy as np
import numpy.ma as ma


def numVals(shape):
    """Return number of values in chunk of specified shape, given by a list of dimension lengths.

    shape -- list of variable dimension sizes"""
    if len(shape) == 0:
        return 1
    return reduce(operator.mul, shape)


def cascadeRounding(array):
    """Implement cascase rounding
    http://stackoverflow.com/questions/792460/how-to-round-floats-to-integers-while-preserving-their-sum
    """

    sort_index = np.argsort(array)
    integer_array = []

    total_float = 0
    total_int = 0

    # We place a hard limit on the total of the array, which keeps
    # the rounded values from exceeding the total of the array
    limit = np.floor(sum(array))

    for idx in sort_index:
        total_float += array[idx]
        integer_array.append(min(round(total_float), limit) - total_int)
        total_int += integer_array[-1]

    rounded_array = np.zeros(len(array))

    # Should make this a comprehension, but I couldn't comprehend it
    for i in range(len(sort_index)):
        rounded_array[sort_index[i]] = integer_array[i]

    return rounded_array


def calcChunkShape(chunkVol, varShape):
    """
    Calculate a chunk shape for a given volume/area for the dimensions in varShape.

    chunkVol   -- volume/area of the chunk
    chunkVol   -- array of dimensions for the whole dataset
    """

    return np.array(
        cascadeRounding(
            np.asarray(varShape)
            * (chunkVol / float(numVals(varShape))) ** (1.0 / len(varShape))
        ),
        dtype="int",
    )


def chunk_shape_nD(varShape, valSize=4, chunkSize=4096, minDim=1):
    """
    Return a 'good shape' for an nD variable, assuming balanced 1D, 2D access

    varShape  -- list of variable dimension sizes
    chunkSize -- minimum chunksize desired, in bytes (default 4096)
    valSize   -- size of each data value, in bytes (default 4)
    minDim    -- mimimum chunk dimension (if var dimension larger
                 than this value, otherwise it is just var dimension)

    Returns integer chunk lengths of a chunk shape that provides
    balanced access of 1D subsets and 2D subsets of a netCDF or HDF5
    variable var. 'Good shape' for chunks means that the number of
    chunks accessed to read any kind of 1D or 2D subset is approximately
    equal, and the size of each chunk (uncompressed) is at least
    chunkSize, which is often a disk block size.
    """

    varShapema = ma.array(varShape)

    chunkVals = min(
        chunkSize / float(valSize), numVals(varShapema)
    )  # ideal number of values in a chunk

    # Make an ideal chunk shape array
    chunkShape = ma.array(calcChunkShape(chunkVals, varShapema), dtype=int)

    # Short circuit for 1D arrays. Logic below unecessary & can have divide by zero
    if len(varShapema) == 1:
        return chunkShape.filled(fill_value=1)

    # And a copy where we'll store our final values
    chunkShapeFinal = ma.masked_all(chunkShape.shape, dtype=int)

    if chunkVals < numVals(np.minimum(varShapema, minDim)):
        while chunkVals < numVals(np.minimum(varShapema, minDim)):
            minDim -= 1
        sys.stderr.write("Mindim too large for variable, reduced to : %d\n" % minDim)

    lastChunkCount = -1

    while True:
        # Loop over the axes in chunkShape, making sure they are at
        # least minDim in length.
        for i in range(len(chunkShape)):
            if ma.is_masked(chunkShape[i]):
                continue
            if chunkShape[i] < minDim:
                # Set the final chunk shape for this dimension
                chunkShapeFinal[i] = min(minDim, varShapema[i])
                # mask it out of the array of possible chunkShapes
                chunkShape[i] = ma.masked

        # Have we fixed any dimensions and filled them in chunkShapeFinal?
        if chunkShapeFinal.count() > 0:
            chunkCount = numVals(chunkShapeFinal[~chunkShapeFinal.mask])
        else:
            if lastChunkCount == -1:
                # Haven't modified initial guess, break out of
                # this loop and accept chunkShape
                break

        if chunkCount != lastChunkCount and len(varShapema[~chunkShape.mask]) > 0:
            # Recalculate chunkShape array, with reduced dimensions
            chunkShape[~chunkShape.mask] = calcChunkShape(
                chunkVals / chunkCount, varShapema[~chunkShape.mask]
            )
            lastChunkCount = chunkCount
        else:
            break

    # This doesn't work when chunkShape has no masked values. Weird.
    # chunkShapeFinal[chunkShapeFinal.mask] = chunkShape[~chunkShape.mask]
    for i in range(len(chunkShapeFinal)):
        if ma.is_masked(chunkShapeFinal[i]):
            chunkShapeFinal[i] = chunkShape[i]

    return chunkShapeFinal.filled(fill_value=1)
