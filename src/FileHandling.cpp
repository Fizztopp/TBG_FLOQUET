#include <iostream>
#include <vector>
#include <fstream>
#include <iterator>
#include <string>
#include <H5public.h>
#include <algorithm>
#include <cassert>

#include "H5Cpp.h"
#include "Constants.h"

void ReadIn(vector<dvec> &MAT, const string &filename) {
/**
 *	Read in real valued matrix
 */
    ifstream in(filename);
    string record;
    if (in.fail()) {
        cout << "file" << filename << "could not be found!" << endl;
    }
    while (getline(in, record)) {
        istringstream is(record);
        dvec row((istream_iterator<double>(is)),
                 istream_iterator<double>());
        MAT.push_back(row);
    }
    in.close();
}

/**
 * Output a double array
 */
void writeReal1DArrayToHdf5(const vector<double> data, const string filename) {

    H5::H5File file(filename, H5F_ACC_TRUNC);

    const hsize_t dataSize = data.size();
    H5::DataSpace dataSpace(1, &dataSize);
    H5::FloatType datatype(H5::PredType::NATIVE_DOUBLE);
    datatype.setOrder(H5T_ORDER_LE);

    H5::DataSet dataset = file.createDataSet("Data1D", datatype, dataSpace);
    dataset.write(&data[0], datatype);
}

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
                            const unsigned long dimension2) {

    assert(data.size() == dimension1 * dimension2);

    H5::H5File file(filename, H5F_ACC_TRUNC);

    const hsize_t dataShape[2] = {dimension1, dimension2};
    H5::DataSpace dataSpace(2, dataShape);
    H5::FloatType datatype(H5::PredType::NATIVE_DOUBLE);
    datatype.setOrder(H5T_ORDER_LE);

    H5::DataSet dataset = file.createDataSet("Real", datatype, dataSpace);
    dataset.write(&data[0], datatype);
}

/**
 *
 * @param data - complex input data
 * @param filename - name of output file - should end in hdf5 or h5
 * @param dimension1 - first dimension of inputdata
 * @param dimension2 - second dimension of inputdata
 */
void writeComplex2DArrayToHdf5(const std::vector<std::complex<double>> data,
                               const string filename,
                               const unsigned long dimension1,
                               const unsigned long dimension2) {

    assert(data.size() == dimension1 * dimension2);

    std::vector<double> realData(dimension1 * dimension2, 0.0);
    std::vector<double> imagData(dimension1 * dimension2, 0.0);

    std::transform(data.begin(),
                   data.end(),
                   realData.begin(),
                   [](const std::complex<double> entry) -> double {
                       return entry.real();
                   });
    std::transform(data.begin(),
                   data.end(),
                   imagData.begin(),
                   [](const std::complex<double> entry) -> double {
                       return entry.imag();
                   });

    H5::H5File file(filename, H5F_ACC_TRUNC);

    const hsize_t dataShape[2] = {dimension1, dimension2};
    H5::DataSpace dataSpace(2, dataShape);
    H5::FloatType datatype(H5::PredType::NATIVE_DOUBLE);
    datatype.setOrder(H5T_ORDER_LE);

    H5::DataSet datasetReal = file.createDataSet("Real", datatype, dataSpace);
    H5::DataSet datasetImag = file.createDataSet("Imag", datatype, dataSpace);
    datasetReal.write(&realData[0], datatype);
    datasetImag.write(&imagData[0], datatype);

}

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
                               const unsigned long dimension3) {

    assert(data.size() == dimension1 * dimension2 * dimension3);

    std::vector<double> realData(dimension1 * dimension2 * dimension3, 0.0);
    std::vector<double> imagData(dimension1 * dimension2 * dimension3, 0.0);

    std::transform(data.begin(),
                   data.end(),
                   realData.begin(),
                   [](const std::complex<double> entry) -> double {
                       return entry.real();
                   });
    std::transform(data.begin(),
                   data.end(),
                   imagData.begin(),
                   [](const std::complex<double> entry) -> double {
                       return entry.imag();
                   });

    H5::H5File file(filename, H5F_ACC_TRUNC);

    const hsize_t dataShape[3] = {dimension1, dimension2, dimension3};
    H5::DataSpace dataSpace(3, dataShape);
    H5::FloatType datatype(H5::PredType::NATIVE_DOUBLE);
    datatype.setOrder(H5T_ORDER_LE);

    H5::DataSet datasetReal = file.createDataSet("Real", datatype, dataSpace);
    H5::DataSet datasetImag = file.createDataSet("Imag", datatype, dataSpace);
    datasetReal.write(&realData[0], datatype);
    datasetImag.write(&imagData[0], datatype);

}

/**
 *
 * @param readInArray array which is filled with to be read data
 * @param fileName of to be read data
 */
void readInComplex2DArray(std::vector<std::complex<double>> &readInArray, const std::string &fileName) {
    H5::H5File file(fileName, H5F_ACC_RDONLY);

    H5::DataSet realDataset = file.openDataSet("Real");
    H5::DataSet imagDataset = file.openDataSet("Imag");

    H5T_class_t typeClass = realDataset.getTypeClass();
    assert(typeClass == H5T_FLOAT);
    typeClass = imagDataset.getTypeClass();
    assert(typeClass == H5T_FLOAT);

    H5::DataSpace realDataSpace = realDataset.getSpace();
    H5::DataSpace imagDataSpace = imagDataset.getSpace();

    int rank = realDataSpace.getSimpleExtentNdims();
    assert(rank == 2);
    rank = imagDataSpace.getSimpleExtentNdims();
    assert(rank == 2);

    hsize_t dimsOut[2];
    int ndims = realDataSpace.getSimpleExtentDims(dimsOut, NULL);
    assert(dimsOut[0] * dimsOut[1] == readInArray.size());
    ndims = imagDataSpace.getSimpleExtentDims(dimsOut, NULL);
    assert(dimsOut[0] * dimsOut[1] == readInArray.size());

    std::vector<double> realInput(readInArray.size(), 0.0);
    std::vector<double> imagInput(readInArray.size(), 0.0);

    const hsize_t inDimension[1] = {readInArray.size()};
    H5::DataSpace memspace(1, inDimension);

    realDataset.read(&realInput[0], H5::PredType::NATIVE_DOUBLE, memspace, realDataSpace);
    imagDataset.read(&imagInput[0], H5::PredType::NATIVE_DOUBLE, memspace, imagDataSpace);

    for(auto ind = 0ul; ind < readInArray.size(); ++ind){
        readInArray[ind] = std::complex<double> (realInput[ind], imagInput[ind]);
    }
}


/**
 * create naming for output
 * @param baseName base name to identify what should be outputted
 * @return stuck together name
 */
std::string createOutputString(const std::string &baseName) {
    std::string strN = to_string(SC);
    string returnName = baseName + "_" + strN;
    return returnName;
}
