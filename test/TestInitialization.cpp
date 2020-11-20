#include <vector>
#include <complex>
#include <cassert>

#include "Constants.h"
#include "FileHandling.h"
#include "HkA.h"
#include "TestInitialization.h"


void testInitialization(std::vector<double> &lvec, std::vector<std::vector<double>> &UNIT_CELL) {

    const int a = SC + 1;
    const int b = SC;

    assert(NATOM == 4 * (SC * SC + (SC + 1) * SC + (SC + 1) * (SC + 1)));

    const double angle1 = atan2(double(b) * sqrt(3.) / 2., double(a) + double(b) / 2.);
    const double angle2 = angle1 + PI / 3.;
    // side length of super cell
    const double d = sqrt(double(b * b) * 3. / 4. + pow(double(a) + double(b) / 2., 2.));

    // superlattice bravis translational vectors
    lvec = {d * cos(angle1), d * sin(angle1), d * sin(PI / 6. - angle1), d * cos(PI / 6. - angle1)};

    //Read in atomic positions
    ReadIn(UNIT_CELL, "testInputFiles/Unit_Cell.dat");

}
