#ifndef TBG_FILEHANDLING_H
#define TBG_FILEHANDLING_H

#include <vector>
#include <string>

#include "Constants.h"


void ReadIn(vector<dvec> &MAT, const string& filename);
/**
 *	Read in real valued matrix
 */


void writeReal1DArrayToHdf5(const vector<double> data, const string filename);
/**
 * Output a double array
 */


/**
 *
 * @param data - implicit 2D array to be outputted
 * @param filename - filename for written file - should end in hdf5 or h5
 * @param dimension1 - first dimension of data
 * @param dimension2 - second dimension of data
 */
void writeReal2DArrayToHdf5(const vector<double> data,
                            const string filename,
                            const unsigned long dimension1,
                            const unsigned long dimension2);

/**
 *
 * @param data - complex input data
 * @param filename - name of output file - should end in hdf5 or h5
 * @param dimension1 - first dimension of inputdata
 * @param dimension2 - second dimension of inputdata
 */
void writeComplex2DArrayToHdf5(const vector<std::complex<double>> data,
                               const string filename,
                               const unsigned long dimension1,
                               const unsigned long dimension2);

/**
 *
 * @param data - data to be written into the file
 * @param filename - filename of output file - should end with hdf5 or h5
 * @param dimension1 - first dimension of output data
 * @param dimension2 - second dimension of output data
 * @param dimension3 - third dimension of output data
 */
void writeComplex3DArrayToHdf5(const std::vector<std::complex<double>> data,
                               const string filename,
                               const unsigned long dimension1,
                               const unsigned long dimension2,
                               const unsigned long dimension3);

/**
 * create naming for output
 * @param baseName base name to identify what should be outputted
 * @return stuck together name
 */
std::string createOutputString(const std::string &baseName);

/**
 *
 * @param readInArray array which is filled with to be read data
 * @param fileName of to be read data
 */
void readInComplex3DArray(std::vector<std::complex<double>> &readInArray, const std::string &fileName);

/**
 *
 * @param readInArray array which is filled with to be read data
 * @param fileName of to be read data
 */
void readInComplex2DArray(std::vector<std::complex<double>> &readInArray, const std::string &fileName);


#endif //TBG_FILEHANDLING_H
