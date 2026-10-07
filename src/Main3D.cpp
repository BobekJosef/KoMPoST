// KoMPoST3D: runs KoMPoST independently on every eta_s slice of a 3D T^{mu nu}(tIn), treating each
// slice as a boost-invariant system, and writes the Landau-matched hydro fields at tOut for vHLLE.
//
//   KoMPoST3D.exe setup.ini tmunu_in.bin kompost_out.bin [-v]
//
// setup.ini: [KoMPoSTInputs] tOut and the [KoMPoSTParameters] section, as for KoMPoST.exe. tIn and the
// grid come from the input file ([KoMPoSTInputs] tIn and [EventInput] are ignored): KoMPoST's lattice
// is Ns = nxy, afm = 2 xy_max / (nxy - 1).
//
// Input, TMUNU3D (written by MC-EKRT/ekrt_to_kompost):
//   char magic[8] = "TMUNU3D", int32 version = 1, ncomp = 10, nxy, neta,
//   double tau [fm], xy_max [fm], eta_max, then nxy*nxy*neta*ncomp doubles ordered
//   ieta (slowest), iy, ix, component (fastest); x, y on nxy points in [-xy_max, xy_max], eta_s on
//   neta points in [-eta_max, eta_max]. Components: T^{mu nu} [GeV/fm^3], upper Milne indices with
//   the eta index multiplied by tau (z = tau eta), order tt, tx, ty, tz, xx, xy, xz, yy, yz, zz.
//
// Output, KMPST3D: same header (magic "KMPST3D", ncomp = 16, tau = tOut) and ordering, components
//   e [GeV/fm^3], u^t, u^x, u^y, u^z,
//   pi^{tt}, pi^{tx}, pi^{ty}, pi^{tz}, pi^{xx}, pi^{xy}, pi^{xz}, pi^{yy}, pi^{yz}, pi^{zz} [GeV/fm^3],
//   Pi [GeV/fm^3]
// with the same z = tau eta convention, i.e. what vHLLE's Cell stores. e and u^mu are the Landau
// matched (eigenvector) energy density and flow, pi^{mu nu} and Pi follow KoMPoST's conformal
// decomposition (get_shear_and_bulk_from_Tmunu): T^{mu nu} = e u u - (e/3 + Pi) Delta + pi.
// A hydro code with another EoS keeps T^{mu nu} by using Pi + e/3 - p(e) as its bulk pressure.
//
// Each slice only sees its own T^{mu nu}(x, y): no flow or gradients along eta_s are propagated, which
// is exact for the eta_s = y free streaming of the minijets but not for the eta smearing.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>
#include <omp.h>

#include "INIReader.h"
#include "EnergyMomentumTensor.h"
#include "KineticEvolution.h"
#include "GreensFunctions.h"
#include "EnergyMomentumTensorIO_music.inc"

// Same value as ENERGY_CUTOFF in KineticEvolution.cpp [GeV^4]: cells whose smeared input energy is
// below it get KoMPoST's artificial background, which is written as vacuum instead.
static const double ENERGY_CUTOFF = 1E-5;

struct Header {
  char magic[8];
  int32_t version, ncomp, nxy, neta;
  double tau, xy_max, eta_max;
};

static void read_header(std::ifstream &in, Header &h) {
  in.read(h.magic, sizeof h.magic);
  in.read(reinterpret_cast<char *>(&h.version), 4 * sizeof(int32_t));
  in.read(reinterpret_cast<char *>(&h.tau), 3 * sizeof(double));
}

static void write_header(std::ofstream &out, const Header &h) {
  out.write(h.magic, sizeof h.magic);
  out.write(reinterpret_cast<const char *>(&h.version), 4 * sizeof(int32_t));
  out.write(reinterpret_cast<const char *>(&h.tau), 3 * sizeof(double));
}

// Point KoMPoST at the EKT/ response tables next to the executable unless KoMPoSTDATADIR is set.
static void set_data_dir() {
  if (std::getenv("KoMPoSTDATADIR")) return;
  char buf[PATH_MAX];
  ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
  if (n <= 0) return;
  buf[n] = '\0';
  std::string dir(buf);
  dir = dir.substr(0, dir.find_last_of('/'));
  setenv("KoMPoSTDATADIR", dir.c_str(), 1);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " setup.ini tmunu_in.bin kompost_out.bin [-v]\n"
              << "Runs KoMPoST on every eta_s slice of a TMUNU3D file (see src/Main3D.cpp).\n";
    return 1;
  }
  const std::string ini = argv[1], in_file = argv[2], out_file = argv[3];
  const bool verbose = argc > 4 && std::string(argv[4]) == "-v";

  INIReader reader(ini);
  if (reader.ParseError()) {
    std::cerr << "Error: cannot parse " << ini << std::endl;
    return 1;
  }

  // ---- input header and grid
  std::ifstream in(in_file, std::ios::binary);
  if (!in) {
    std::cerr << "Error: cannot open " << in_file << std::endl;
    return 1;
  }
  Header hin;
  read_header(in, hin);
  if (!in || std::strncmp(hin.magic, "TMUNU3D", 8) != 0 || hin.version != 1 || hin.ncomp != 10 ||
      hin.nxy < 2 || hin.neta < 1) {
    std::cerr << "Error: " << in_file << " is not a TMUNU3D v1 file with 10 components" << std::endl;
    return 1;
  }
  const double tIn = hin.tau;
  const double tOut = reader.GetReal("KoMPoSTInputs", "tOut", -1.0);
  if (tOut <= tIn) {
    std::cerr << "Error: [KoMPoSTInputs] tOut = " << tOut << " must be larger than tIn = " << tIn
              << " fm from " << in_file << std::endl;
    return 1;
  }
  const int Ns = hin.nxy, neta = hin.neta;
  const double afm = 2 * hin.xy_max / (Ns - 1);
  const double deta = neta > 1 ? 2 * hin.eta_max / (neta - 1) : 0.0;
  EventInput::afm = afm;
  EventInput::Ns = Ns;
  EventInput::xSTART = EventInput::ySTART = 0;
  EventInput::xEND = EventInput::yEND = Ns - 1;

  KoMPoSTParameters::Sigma = 0.1 / (tOut - tIn); // hard-coded in KoMPoST's Main.cpp as well
  KoMPoSTParameters::Setup(reader);
  std::cerr << "#KoMPoST3D: " << Ns << " x " << Ns << " x " << neta << " cells, afm = " << afm
            << " fm, eta_s in [-" << hin.eta_max << ", " << hin.eta_max << "], tIn = " << tIn
            << " fm -> tOut = " << tOut << " fm, " << omp_get_max_threads() << " threads" << std::endl;

  std::streambuf *cerr_buf = std::cerr.rdbuf();
  std::ostringstream sink;

  set_data_dir();
  if (!verbose) std::cerr.rdbuf(sink.rdbuf());
  GreensFunctions::Setup(KoMPoSTParameters::Sigma, 256, KoMPoSTParameters::ENERGY_PERTURBATIONS,
                         KoMPoSTParameters::MOMENTUM_PERTURBATIONS);
  std::cerr.rdbuf(cerr_buf);
  sink.str("");

  // ---- output
  std::ofstream out(out_file, std::ios::binary);
  if (!out) {
    std::cerr << "Error: cannot write " << out_file << std::endl;
    return 1;
  }
  Header hout = hin;
  std::memcpy(hout.magic, "KMPST3D", 8);
  hout.ncomp = 16;
  hout.tau = tOut;
  write_header(out, hout);

  const double hbarc3 = M_HBARC * M_HBARC * M_HBARC;
  const double tauOutGeV = tOut / M_HBARC;
  const double small_num = 1000 * GSL_DBL_MIN; // as in EnergyMomentumTensorMapLoad

  EnergyMomentumTensorMap TIn(tIn), TOutBG(tOut), TOutFull(tOut);
  std::vector<double> slice(static_cast<size_t>(Ns) * Ns * 10);
  std::vector<double> res(static_cast<size_t>(Ns) * Ns * 16);

  // bookkeeping: E = tau int (T^{tt} cosh eta + T^{tz} sinh eta) dx dy deta
  double E_in = 0, E_out = 0, E_out_matched = 0, E_failed = 0;
  long n_failed = 0, n_run = 0;
  const double cell = afm * afm * (neta > 1 ? deta : 1.0);

  for (int ie = 0; ie < neta; ie++) {
    const double eta = -hin.eta_max + ie * deta;
    in.read(reinterpret_cast<char *>(slice.data()), slice.size() * sizeof(double));
    if (!in) {
      std::cerr << "Error: " << in_file << " ended in slice " << ie << std::endl;
      return 1;
    }

    double tmax = 0;
    for (int iy = 0; iy < Ns; iy++)
      for (int ix = 0; ix < Ns; ix++) {
        const double *t = &slice[(static_cast<size_t>(iy) * Ns + ix) * 10];
        tmax = std::max(tmax, t[0]);
        E_in += tIn * (t[0] * cosh(eta) + t[3] * sinh(eta)) * cell;
      }
    std::fill(res.begin(), res.end(), 0.0);

    if (tmax * hbarc3 < ENERGY_CUTOFF) {
      // nothing above KoMPoST's cut-off: vacuum
      out.write(reinterpret_cast<const char *>(res.data()), res.size() * sizeof(double));
      if (verbose) std::cerr << "#slice " << ie << " eta_s = " << eta << ": empty" << std::endl;
      continue;
    }

    // T^{mu nu} [GeV/fm^3] -> KoMPoST's T_{mu nu}-like storage [GeV^4] (see EnergyMomentumTensor.h):
    // diagonal = upper components (zz = tau^2 T^{eta eta}), off-diagonal = minus upper components
    for (int iy = 0; iy < Ns; iy++)
      for (int ix = 0; ix < Ns; ix++) {
        const double *t = &slice[(static_cast<size_t>(iy) * Ns + ix) * 10];
        if (t[0] * hbarc3 < small_num)
          TIn.Set(ix, iy, 10 * small_num, 5 * small_num, 5 * small_num, 0, 0, 0, 0, 0, 0, 0);
        else
          TIn.Set(ix, iy, hbarc3 * t[0], hbarc3 * t[4], hbarc3 * t[7], hbarc3 * t[9], -hbarc3 * t[1],
                  -hbarc3 * t[2], -hbarc3 * t[3], -hbarc3 * t[5], -hbarc3 * t[8], -hbarc3 * t[6]);
      }

    if (!verbose) std::cerr.rdbuf(sink.rdbuf());
    KoMPoST::Run(&TIn, &TOutBG, &TOutFull);
    std::cerr.rdbuf(cerr_buf);
    sink.str("");
    n_run++;

    long failed = 0;
    double e_out = 0, e_out_matched = 0, e_failed = 0, emax = 0;
#pragma omp parallel for reduction(+ : failed, e_out, e_out_matched, e_failed) reduction(max : emax)
    for (int iy = 0; iy < Ns; iy++)
      for (int ix = 0; ix < Ns; ix++) {
        double *r = &res[(static_cast<size_t>(iy) * Ns + ix) * 16];
        if (TOutBG.GetCellData(0, ix, iy) <= ENERGY_CUTOFF) continue; // vacuum

        double T[4][4]; // T^{mu nu} [GeV^4], eta components in powers of GeV
        get_Tmunu_raised_hyperbolic(&TOutFull, ix, iy, T);
        // scale factor of an upper eta index: z = tau eta
        const double sc[4] = {1., 1., 1., tauOutGeV};

        // energy of the cell from the full T^{mu nu}
        const double ttt = T[0][0] / hbarc3, ttz = T[0][3] * sc[3] / hbarc3;
        const double dE = tOut * (ttt * cosh(eta) + ttz * sinh(eta)) * cell;
        e_out += dE;

        double e, u[4];
        error_codes err;
        get_energy_flow_from_Tmunu(T, tauOutGeV, &e, u, &err);
        if (err != decomp_ok || !(e > 0)) {
          // No timelike eigenvector, which happens where the linearised response is far from a
          // fluid (dilute edges): keep T^{tautau} in the grid rest frame, T^{tau i} is lost.
          failed++;
          e_failed += dE;
          e = T[0][0];
          u[0] = 1;
          u[1] = u[2] = u[3] = 0;
        }
        double pi[4][4], bulk;
        get_shear_and_bulk_from_Tmunu(T, tauOutGeV, e, u, pi, bulk);

        r[0] = e / hbarc3;
        for (int mu = 0; mu < 4; mu++) r[1 + mu] = u[mu] * sc[mu];
        int k = 5;
        for (int mu = 0; mu < 4; mu++)
          for (int nu = mu; nu < 4; nu++) r[k++] = pi[mu][nu] * sc[mu] * sc[nu] / hbarc3;
        r[15] = bulk / hbarc3;

        // energy from the matched fields (equals dE)
        const double p = r[0] / 3 + r[15];
        const double mt = (r[0] + p) * r[1] * r[1] - p + r[5];
        const double mz = (r[0] + p) * r[1] * r[4] + r[8];
        e_out_matched += tOut * (mt * cosh(eta) + mz * sinh(eta)) * cell;
        emax = std::max(emax, r[0]);
      }
    out.write(reinterpret_cast<const char *>(res.data()), res.size() * sizeof(double));
    n_failed += failed;
    E_out += e_out;
    E_out_matched += e_out_matched;
    E_failed += e_failed;
    std::cerr << "#slice " << std::setw(3) << ie << "  eta_s = " << std::setw(6) << eta
              << "  max e(tIn) = " << std::setw(10) << tmax << "  max e(tOut) = " << std::setw(10) << emax
              << " GeV/fm^3" << (failed ? "  Landau matching failed in " + std::to_string(failed) + " cells" : "")
              << std::endl;
  }
  if (!out) {
    std::cerr << "Error: writing " << out_file << " failed" << std::endl;
    return 1;
  }
  std::cerr << "#KoMPoST3D done: " << n_run << " of " << neta << " slices evolved\n"
            << "#  fluid E at tIn  = " << E_in << " GeV\n"
            << "#  fluid E at tOut = " << E_out << " GeV (from T^{mu nu}), " << E_out_matched
            << " GeV (from e, u, pi, Pi)\n"
            << "#  Landau matching failed in " << n_failed << " cells holding " << E_failed << " GeV ("
            << 100 * E_failed / E_out << " %); they are written with u = 0, e = T^{tautau}\n"
            << "#  wrote " << out_file << std::endl;
  return 0;
}
