#include <iostream>
#include <vector>
#include <fstream>
#include <iterator>

#include "Constants.h"

void ReadIn(vector<dvec> &MAT, const string& filename)
{
/**
 *	Read in real valued matrix
 */
    ifstream in(filename);
    string record;
    if(in.fail()){
        cout << "file" << filename << "could not be found!" << endl;
    }
    while (getline(in, record))
    {
        istringstream is( record );
        dvec row((istream_iterator<double>(is)),
                 istream_iterator<double>());
        MAT.push_back(row);
    }
    in.close();
}

