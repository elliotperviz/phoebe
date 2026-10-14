#include "dos_app.h"

#include <array>
#include <utility>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

#include "constants.h"
#include "delta_function.h"
#include "points.h"
#include "mpiHelper.h"
#include "parser.h"
#include <nlohmann/json.hpp>

// Forward declare this helper function so that
// app::run functions can remain at the top
std::tuple<std::vector<double>, std::vector<double>> calcDOS(
        Context& context, FullBandStructure& fullBandStructure);

void outputMeshToJSON(FullBandStructure &fullBandStructure, Context &context,
        Points &meshPoints, const std::string &outFileName);

void outputDOSToJSON(std::vector<double> energies, std::vector<double> dos,
        Particle& particle, const std::string &outFileName);

/* ------------------- PhononDoSApp --------------------*/
// Compute the DOS with the tetrahedron method
void PhononDosApp::run(Context &context) {
  if (mpi->mpiHead()) {
    std::cout << "Starting phonon DoS calculation" << std::endl;
  }

  // Read the necessary input files
  auto tup = Parser::parsePhHarmonic(context);
  auto crystal = std::get<0>(tup);
  auto phononH0 = std::get<1>(tup);

  // first we make compute the band structure on the fine grid
  Points fullPoints(crystal, context.getQMesh());
  fullPoints.setIrreduciblePoints();
  bool withVelocities = context.getOutputGroupVelocities();
  bool withEigenvectors = context.getOutputEigendisplacements();
  FullBandStructure fullBandStructure =
      phononH0.populate(fullPoints, withVelocities, withEigenvectors);
  auto particle = fullBandStructure.getParticle();

  // calculate DOS via the tetrahedron method
  auto tup1 = calcDOS(context, fullBandStructure);
  std::vector<double> energies = std::get<0>(tup1);
  std::vector<double> dos = std::get<1>(tup1);

  // save dos to an output file
  // arguments are: energies, dos, particleType, context outFileName
  outputDOSToJSON(energies, dos, particle, "phonon_dos.json");
  outputMeshToJSON(fullBandStructure, context, fullPoints, "mesh.json");

  if (mpi->mpiHead()) {
    std::cout << "Phonon DoS computed." << std::endl;
  }
}

/* ---------------- ElectronWannierDos --------------------*/
// Compute the Electron DOS with tetrahedron method and Wannier interpolation
void ElectronWannierDosApp::run(Context &context) {
  if (mpi->mpiHead())
    std::cout << "Starting electronic (Wannier) DoS calculation" << std::endl;

  // Read the necessary input files
  auto tup = Parser::parseElHarmonicWannier(context);
  auto crystal = std::get<0>(tup);
  auto h0 = std::get<1>(tup);

  // first we make compute the band structure on the fine grid
  Points fullPoints(crystal, context.getKMesh());
  fullPoints.setIrreduciblePoints();
  bool withVelocities = false;
  bool withEigenvectors = false;
  FullBandStructure fullBandStructure =
      h0.populate(fullPoints, withVelocities, withEigenvectors);
  auto particle = fullBandStructure.getParticle();

  // calculate DOS via the tetrahedron method
  auto tup1 = calcDOS(context, fullBandStructure);
  std::vector<double> energies = std::get<0>(tup1);
  std::vector<double> dos = std::get<1>(tup1);

  // save dos to an output file
  // arguments are: energies, dos, particleType, context outFileName
  outputDOSToJSON(energies, dos, particle, "electron_dos.json");

  if (mpi->mpiHead()) {
    std::cout << "Electronic (Wannier) DoS computed" << std::endl;
  }
}

/* ---------------- ElectronFourierDos --------------------*/
// Compute the Electron DOS with tetrahedron method and Fourier interpolation
void ElectronFourierDosApp::run(Context &context) {
  if (mpi->mpiHead())
    std::cout << "Starting electronic (Fourier) DoS calculation" << std::endl;

  // Read the necessary input files
  auto tup = Parser::parseElHarmonicFourier(context);
  auto crystal = std::get<0>(tup);
  auto h0 = std::get<1>(tup);

  // first we make compute the band structure on the fine grid
  Points fullPoints(crystal, context.getKMesh());
  fullPoints.setIrreduciblePoints();
  bool withVelocities = false;
  bool withEigenvectors = false;
  FullBandStructure fullBandStructure =
      h0.populate(fullPoints, withVelocities, withEigenvectors);
  auto particle = fullBandStructure.getParticle();

  // calculate DOS via the tetrahedron method
  auto tup1 = calcDOS(context, fullBandStructure);
  std::vector<double> energies = std::get<0>(tup1);
  std::vector<double> dos = std::get<1>(tup1);

  // save dos to an output file
  // arguments are: energies, dos, particleType, context outFileName
  outputDOSToJSON(energies, dos, particle, "electron_dos.json");

  if (mpi->mpiHead()) {
    std::cout << "Electronic (Fourier) DoS computed" << std::endl;
  }
}

/*
 * Generic helper function to calculate DoS via tetrahedron method
 */
std::tuple<std::vector<double>, std::vector<double>> calcDOS(
        Context& context, FullBandStructure& fullBandStructure) {

  // Form tetrahedra and fill them with eigenvalues
  TetrahedronDeltaFunction tetrahedra(fullBandStructure);

  double minEnergy = context.getDosMinEnergy();
  double maxEnergy = context.getDosMaxEnergy();
  double deltaEnergy = context.getDosDeltaEnergy();
  int numEnergies = int((maxEnergy - minEnergy) / deltaEnergy) + 1;

  // create instructions about how to divide up the work
  auto divs = mpi->divideWork(numEnergies);
  // the amount of values this process has to take care of
  int start = divs[0];
  int stop = divs[1];
  int workFraction = stop - start;

  std::vector<double> energies(workFraction);
#pragma omp parallel for default(none) shared(start,stop,energies,minEnergy,deltaEnergy)
  for (int i = start; i < stop; i++) {
    energies[i - start] = double(i) * deltaEnergy + minEnergy;
  }

  // Calculate phonon density of states (DOS) [1/Ry]
  std::vector<double> dos(workFraction, 0.);  // DOS initialized to zero
#pragma omp parallel for default(none) shared(workFraction,tetrahedra,dos,energies)
  for (int i = 0; i < workFraction; i++) {
    dos[i] += tetrahedra.getDOS(energies[i]);
  }

  std::vector<double> dosTotal;
  std::vector<double> eneTotal;
  // root allocates memory to receive the gathered data
  if (mpi->mpiHead()) {
    dosTotal.resize(numEnergies);
    eneTotal.resize(numEnergies);
  }
  // all processes send their data to be gathered into dos/eneTotal
  mpi->gatherv(&energies, &eneTotal);
  mpi->gatherv(&dos, &dosTotal);
  return std::make_tuple(eneTotal, dosTotal);

}

/*
 * helper function to output dos to a json file
 */
void outputDOSToJSON(std::vector<double> energies, std::vector<double> dos,
          Particle& particle, const std::string &outFileName) {

  if ( !mpi->mpiHead()) return;

  // convert energies
  double energyConversion =
        particle.isPhonon() ? energyRyToEv*1000 : energyRyToEv;

  for (unsigned int i = 0; i < energies.size(); i++) {
    energies[i] *= energyConversion;
    dos[i] /= energyConversion;
  }
  // output to json
  nlohmann::json output;
  output["energies"] = energies;
  output["dos"] = dos;
  output["particleType"] = particle.isPhonon() ? "phonon" : "electron";
  output["energyUnit"] = particle.isPhonon() ? "meV" : "eV";
  output["dosUnit"] = particle.isPhonon() ? "1/meV" : "1/eV";
  std::ofstream o(outFileName);
  o << std::setw(3) << output << std::endl;
  o.close();
}

/* Output the phonon dispersion on the DOS mesh. Wavevectors are in crystal
 * reciprocal coordinates, folded into the Wigner-Seitz Brillouin zone. */
void outputMeshToJSON(FullBandStructure &fullBandStructure, Context &context,
                      Points &meshPoints, const std::string &outFileName) {
  if (!mpi->mpiHead()) return;

  auto particle = fullBandStructure.getParticle();

  const int numBands = fullBandStructure.getNumBands();
  const bool writeEigendisplacements = context.getOutputEigendisplacements();
  const double energyConversion = particle.isPhonon() ? 1000*energyRyToEv : energyRyToEv;

  std::vector<int> wavevectorIndices;
  std::vector<std::vector<double>> wavevectorCoordinates;
  std::vector<std::vector<double>> outEnergies;
  std::vector<std::vector<std::vector<double>>> outGroupVelocities;

  // [wavevector][band][atom][Cartesian direction][real, imaginary]
  using ComplexPair = std::array<double, 2>;
  using AtomDisplacement = std::vector<ComplexPair>;
  using BandDisplacement = std::vector<AtomDisplacement>;
  using WavevectorDisplacement = std::vector<BandDisplacement>;
  std::vector<WavevectorDisplacement> eigendisplacements;

  for (int ik = 0; ik < meshPoints.getNumPoints(); ++ik) {
    WavevectorIndex ikIndex(ik);
    wavevectorIndices.push_back(ik);

    auto point = meshPoints.getPoint(ik);
    auto coord = point.getCoordinates(Points::crystalCoordinates, true);
    wavevectorCoordinates.push_back({coord[0], coord[1], coord[2]});

    Eigen::VectorXd energies = fullBandStructure.getEnergies(ikIndex);
    std::vector<double> bandEnergies;
    bandEnergies.reserve(numBands);
    for (int ib = 0; ib < numBands; ++ib) {
      bandEnergies.push_back(energies(ib) * energyConversion);
    }
    outEnergies.push_back(std::move(bandEnergies));

    // Phonon group velocities in Cartesian coordinates, m/s
    if (particle.isPhonon() && context.getOutputGroupVelocities()) {
      Eigen::MatrixXd velocities =
          fullBandStructure.getGroupVelocities(ikIndex);

      std::vector<std::vector<double>> bandVelocities;

      for (int ib = 0; ib < numBands; ib++) {
        bandVelocities.push_back({
            velocities(ib, 0) * velocityRyToSi,
            velocities(ib, 1) * velocityRyToSi,
            velocities(ib, 2) * velocityRyToSi
        });
      }

      outGroupVelocities.push_back(bandVelocities);
    }

    if (writeEigendisplacements && particle.isPhonon()) {
      // getPhEigenvectors() returns mass-scaled phonon displacements,
      // with indices (Cartesian direction, atom, band).

      const int numAtoms = numBands / 3;
      auto eigenvectors = fullBandStructure.getPhEigenvectors(ikIndex);

      WavevectorDisplacement modeDisplacements;
      modeDisplacements.reserve(numBands);

      for (int ib = 0; ib < numBands; ++ib) {
        BandDisplacement atomDisplacements;
        atomDisplacements.reserve(numAtoms);
        for (int ia = 0; ia < numAtoms; ++ia) {
          AtomDisplacement displacement;
          displacement.reserve(3);
          for (int idir = 0; idir < 3; ++idir) {
            auto value = eigenvectors(idir, ia, ib) * distanceBohrToAng;
            displacement.push_back({value.real(), value.imag()});
          }
          atomDisplacements.push_back(std::move(displacement));
        }
        modeDisplacements.push_back(std::move(atomDisplacements));
      }
      eigendisplacements.push_back(std::move(modeDisplacements));
    }
  }

  nlohmann::json output;
  output["wavevectorIndices"] = wavevectorIndices;
  output["wavevectorCoordinates"] = wavevectorCoordinates;
  output["numBands"] = numBands;
  output["energies"] = outEnergies;
  output["particleType"] = particle.isPhonon() ? "phonon" : "electron";
  output["energyUnit"] = particle.isPhonon() ? "meV" : "eV";
  output["coordsType"] = "crystal";

  if (particle.isPhonon() && context.getOutputGroupVelocities()) {
    output["groupVelocities"] = outGroupVelocities;
    output["velocityUnit"] = "m/s";
    output["velocityCoordinatesType"] = "cartesian";
  }

  if (writeEigendisplacements && particle.isPhonon()) {
    Crystal crystal = meshPoints.getCrystal();

    const int numAtoms = numBands / 3;
    auto atomPositions = crystal.getAtomicPositions();

    std::vector<std::vector<double>> latticeVectors;
    std::vector<std::vector<double>> atomPositionsCartesian;

    for (int i = 0; i < 3; ++i) {
      latticeVectors.push_back({
          crystal.getDirectUnitCell()(i, 0) * distanceBohrToAng,
          crystal.getDirectUnitCell()(i, 1) * distanceBohrToAng,
          crystal.getDirectUnitCell()(i, 2) * distanceBohrToAng});
    }
    for (int ia = 0; ia < numAtoms; ++ia) {
      atomPositionsCartesian.push_back({
          atomPositions(ia, 0) * distanceBohrToAng,
          atomPositions(ia, 1) * distanceBohrToAng,
          atomPositions(ia, 2) * distanceBohrToAng});
    }

    output["numAtoms"] = numAtoms;
    output["phononEigendisplacements"] = eigendisplacements;
    output["coordsTypePhononEigendisplacements"] = "cartesian";
    output["latticeVectors"] = latticeVectors;
    output["distanceUnit"] = "Angstrom";
    output["atomPositionsCartesian"] = atomPositionsCartesian;
    output["atomSpecies"] = crystal.getAtomicNames();
  }

  std::ofstream out(outFileName);
  out << std::setw(3) << output << std::endl;
}

void PhononDosApp::checkRequirements(Context &context) {
  throwErrorIfUnset(context.getPhFC2FileName(), "PhFC2FileName");
  throwErrorIfUnset(context.getQMesh(), "qMesh");
  throwErrorIfUnset(context.getDosMinEnergy(), "dosMinEnergy");
  throwErrorIfUnset(context.getDosMaxEnergy(), "dosMaxEnergy");
  throwErrorIfUnset(context.getDosDeltaEnergy(), "dosDeltaEnergy");
  throwWarningIfUnset(context.getSumRuleFC2(), "sumRuleFC2");
}

void ElectronWannierDosApp::checkRequirements(Context &context) {
  throwErrorIfUnset(context.getElectronH0Name(), "electronH0Name");
  throwErrorIfUnset(context.getKMesh(), "kMesh");
  throwErrorIfUnset(context.getDosMinEnergy(), "dosMinEnergy");
  throwErrorIfUnset(context.getDosMaxEnergy(), "dosMaxEnergy");
  throwErrorIfUnset(context.getDosDeltaEnergy(), "dosDeltaEnergy");
  std::string crystalMsg = "crystal structure";
  if(context.getJDFTxScfOutFile().empty()) { // Wannier90 doesn't give us this, user must set it
    throwErrorIfUnset(context.getInputAtomicPositions(), crystalMsg);
    throwErrorIfUnset(context.getInputSpeciesNames(), crystalMsg);
    throwErrorIfUnset(context.getInputAtomicSpecies(), crystalMsg);
  }
}

void ElectronFourierDosApp::checkRequirements(Context &context) {
  throwErrorIfUnset(context.getElectronH0Name(), "electronH0Name");
  throwErrorIfUnset(context.getKMesh(), "kMesh");
  throwErrorIfUnset(context.getDosMinEnergy(), "dosMinEnergy");
  throwErrorIfUnset(context.getDosMaxEnergy(), "dosMaxEnergy");
  throwErrorIfUnset(context.getDosDeltaEnergy(), "dosDeltaEnergy");
  throwErrorIfUnset(context.getElectronFourierCutoff(),
                    "electronFourierCutoff");
}
