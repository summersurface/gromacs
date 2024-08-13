/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2023- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */

/*! \internal \file
 * \brief Tests for NBNxM pair kernel forces and energies.
 *
 * These tests covers all compiled flavors of the NBNxM kernels, not only
 * those used by default by mdrun.
 * The forces and energies are compared to common reference data for
 * kernels that are expected to produce the same output (i.e. only
 * different kernel layout or analytical vs tabulated Ewald LR
 * correction).
 *
 * The only thing currently not covered is LJ-PME with the Lorentz-Berthelot
 * combination rule, as this is only implemented in the plain-C reference kernel
 * and currently the reference data is generated by the SIMD kernels.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_nbnxm
 */

#include "gmxpre.h"

#include "config.h"

#include <cmath>
#include <cstdint>

#include <algorithm>
#include <array>
#include <filesystem>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/ewald/ewald_utils.h"
#include "gromacs/gpu_utils/hostallocator.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/vec.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdlib/forcerec.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdtypes/atominfo.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/locality.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/gridset.h"
#include "gromacs/nbnxm/kernel_common.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/nbnxm_simd.h"
#include "gromacs/nbnxm/pairlistparams.h"
#include "gromacs/nbnxm/pairlistset.h"
#include "gromacs/nbnxm/pairlistsets.h"
#include "gromacs/nbnxm/pairsearch.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/forcefieldparameters.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/listoflists.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/range.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/stringutil.h"

#include "testutils/refdata.h"
#include "testutils/testasserts.h"
#include "testutils/testinit.h"

#include "testsystem.h"

namespace gmx
{

namespace test
{

namespace
{

/*! \brief How the kernel should compute energies
 *
 * Note that the construction of the test system is currently
 * not general enough to handle more than one case with multiple
 * energy groups. */
enum class EnergyHandling : int
{
    NoEnergies,
    Energies,
    ThreeEnergyGroups,
    Count
};

//! Lookup table for the number of energy groups in use
const EnumerationArray<EnergyHandling, int> sc_numEnergyGroups = { 0, 1, 3 };

//! Names for the kinds of energy handling
const EnumerationArray<EnergyHandling, const char*> sc_energyGroupNames = { "NoEnergies",
                                                                            "Energies",
                                                                            "ThreeEnergyGroups" };

//! The options for the kernel
struct KernelOptions
{
    //! Whether to use a GPU, currently GPUs are not supported
    bool useGpu = false;
    //! The number of OpenMP threads to use
    int numThreads = 1;
    //! The kernel setup
    NbnxmKernelSetup kernelSetup;
    //! The modifier for the VdW interactions
    InteractionModifiers vdwModifier = InteractionModifiers::PotShift;
    //! The LJ combination rule
    LJCombinationRule ljCombinationRule = LJCombinationRule::None;
    //! Whether we are using PME for LJ
    bool useLJPme = false;
    //! Ewald relative tolerance for LJ
    real ewaldRTolLJ = 1e-4;
    //! LJ combination rule for the LJ PME mesh part
    LongRangeVdW ljPmeCombinationRule = LongRangeVdW::Geom;
    //! The pairlist and interaction cut-off
    real pairlistCutoff = 0.9;
    //! The Coulomb Ewald coefficient
    real ewaldRTol = 1e-6;
    //! The Coulomb interaction function
    CoulombKernelType coulombType = CoulombKernelType::Ewald;
    //! How to handle energy computations
    EnergyHandling energyHandling = EnergyHandling::NoEnergies;
};

//! Returns the enum value for initializing the LJ PME-grid combination rule for nbxnm_atomdata_t
LJCombinationRule chooseLJPmeCombinationRule(const KernelOptions& options)
{
    if (options.useLJPme)
    {
        // We need to generate LJ combination parameters using the rule for LJ-PME
        switch (options.ljPmeCombinationRule)
        {
            case LongRangeVdW::Geom: return LJCombinationRule::Geometric;
            case LongRangeVdW::LB: return LJCombinationRule::LorentzBerthelot;
            default: GMX_RELEASE_ASSERT(false, "Unhandled combination rule");
        }
    }

    return LJCombinationRule::None;
}

//! Sets up and returns a Nbnxm object for the given benchmark options and system
std::unique_ptr<nonbonded_verlet_t> setupNbnxmForBenchInstance(const KernelOptions& options,
                                                               const TestSystem&    system)
{
    real minBoxSize = norm(system.box[XX]);
    for (int dim = YY; dim < DIM; dim++)
    {
        minBoxSize = std::min(minBoxSize, norm(system.box[dim]));
    }
    if (options.pairlistCutoff > 0.5 * minBoxSize)
    {
        gmx_fatal(FARGS, "The cut-off should be shorter than half the box size");
    }

    // We don't want to call gmx_omp_nthreads_init(), so we init what we need
    gmx_omp_nthreads_set(ModuleMultiThread::Pairsearch, options.numThreads);
    gmx_omp_nthreads_set(ModuleMultiThread::Nonbonded, options.numThreads);

    const auto pinPolicy =
            (options.useGpu ? PinningPolicy::PinnedIfSupported : PinningPolicy::CannotBePinned);
    const int numThreads = options.numThreads;

    PairlistParams pairlistParams(options.kernelSetup.kernelType, false, options.pairlistCutoff, false);

    GridSet gridSet(
            PbcType::Xyz, false, nullptr, nullptr, pairlistParams.pairlistType, false, numThreads, pinPolicy);

    auto pairlistSets = std::make_unique<PairlistSets>(pairlistParams, false, 0);

    auto pairSearch = std::make_unique<PairSearch>(
            PbcType::Xyz, false, nullptr, nullptr, pairlistParams.pairlistType, false, numThreads, pinPolicy);

    auto atomData = std::make_unique<nbnxn_atomdata_t>(
            pinPolicy,
            MDLogger(),
            options.kernelSetup.kernelType,
            options.useLJPme ? LJCombinationRule::None : options.ljCombinationRule,
            chooseLJPmeCombinationRule(options),
            system.nonbondedParameters,
            true,
            sc_numEnergyGroups[options.energyHandling],
            numThreads);

    // Put everything together
    auto nbv = std::make_unique<nonbonded_verlet_t>(
            std::move(pairlistSets), std::move(pairSearch), std::move(atomData), options.kernelSetup, nullptr);

    GMX_RELEASE_ASSERT(!TRICLINIC(system.box), "Only rectangular unit-cells are supported here");
    const rvec lowerCorner = { 0, 0, 0 };
    const rvec upperCorner = { system.box[XX][XX], system.box[YY][YY], system.box[ZZ][ZZ] };

    const real atomDensity = system.coordinates.size() / det(system.box);

    nbv->putAtomsOnGrid(system.box,
                        0,
                        lowerCorner,
                        upperCorner,
                        nullptr,
                        { 0, int(system.coordinates.size()) },
                        atomDensity,
                        system.atomInfo,
                        system.coordinates,
                        0,
                        nullptr);

    nbv->constructPairlist(gmx::InteractionLocality::Local, system.excls, 0, nullptr);

    nbv->setAtomProperties(system.atomTypes, system.charges, system.atomInfo);

    return nbv;
}

//! Convenience typedef of the test input parameters
struct KernelInputParameters
{
    //! This type must match the layout of \c KernelInputParameters
    using TupleT = std::tuple<NbnxmKernelType, CoulombKernelType, int, EnergyHandling>;
    //! The kernel type and cluster pair layout
    NbnxmKernelType kernelType;
    //! The Coulomb kernel type
    CoulombKernelType coulombKernelType;
    //! The VdW interaction type
    int vdwKernelType;
    //! How to handle energy computations
    EnergyHandling energyHandling = EnergyHandling::NoEnergies;
    KernelInputParameters(TupleT t) :
        kernelType(std::get<0>(t)),
        coulombKernelType(std::get<1>(t)),
        vdwKernelType(std::get<2>(t)),
        energyHandling(std::get<3>(t))
    {
    }
};

//! Class that sets up and holds a set of N atoms and a full NxM pairlist
class NbnxmKernelTest : public ::testing::TestWithParam<KernelInputParameters>
{
};

//! Returns the coulomb interaction type given the Coulomb kernel type
CoulombInteractionType coulombInteractionType(CoulombKernelType coulombKernelType)
{
    switch (coulombKernelType)
    {
        case CoulombKernelType::Ewald:
        case CoulombKernelType::Table:
        case CoulombKernelType::EwaldTwin:
        case CoulombKernelType::TableTwin: return CoulombInteractionType::Pme;
        case CoulombKernelType::ReactionField: return CoulombInteractionType::RF;
        default:
            GMX_RELEASE_ASSERT(false, "Unsupported CoulombKernelType");
            return CoulombInteractionType::Count;
    }
}

//! Return an interaction constants struct with members used in the benchmark set appropriately
interaction_const_t setupInteractionConst(const KernelOptions& options)
{
    t_inputrec ir;

    // The kernel selection code only use Cut and Pme
    ir.vdwtype      = (options.useLJPme ? VanDerWaalsType::Pme : VanDerWaalsType::Cut);
    ir.vdw_modifier = options.vdwModifier;
    if (options.coulombType == CoulombKernelType::EwaldTwin
        || options.coulombType == CoulombKernelType::TableTwin)
    {
        ir.rvdw = options.pairlistCutoff - 0.2;
    }
    else
    {
        ir.rvdw = options.pairlistCutoff;
    }
    ir.rvdw_switch = ir.rvdw - 0.2;
    if (ir.vdwtype == VanDerWaalsType::Pme)
    {
        GMX_RELEASE_ASSERT(options.ljPmeCombinationRule == LongRangeVdW::Geom,
                           "The SIMD kernels, used to generate the reference data, only support "
                           "geometric LJ-PME");

        ir.ljpme_combination_rule = options.ljPmeCombinationRule;
        ir.ewald_rtol_lj          = options.ewaldRTolLJ;
    }

    ir.coulombtype      = coulombInteractionType(options.coulombType);
    ir.coulomb_modifier = InteractionModifiers::PotShift;
    ir.rcoulomb         = options.pairlistCutoff;
    ir.ewald_rtol       = options.ewaldRTol;
    ir.epsilon_r        = 1;
    ir.epsilon_rf       = 0;

    gmx_mtop_t mtop;
    // Only reppow and functype[0] are used from mtop in init_interaction_const()
    mtop.ffparams.reppow = 12;
    mtop.ffparams.functype.resize(1);
    mtop.ffparams.functype[0] = F_LJ;

    interaction_const_t ic = init_interaction_const(nullptr, ir, mtop, false);
    init_interaction_const_tables(nullptr, &ic, options.pairlistCutoff, 0);

    return ic;
}

const EnumerationArray<CoulombKernelType, const char*> coulombKernelTypeName = { "ReactionField",
                                                                                 "Table",
                                                                                 "TableTwin",
                                                                                 "Ewald",
                                                                                 "EwaldTwin" };

const std::array<const char*, vdwktNR> vdwKernelTypeName = { "CutCombGeom", "CutCombLB",
                                                             "CutCombNone", "ForceSwitch",
                                                             "PotSwitch",   "EwaldCombGeom" };

/*! \brief Help GoogleTest name our test cases
 *
 * If changes are needed here, consider making matching changes in
 * makeRefDataFileName(). */
std::string nameOfTest(const testing::TestParamInfo<KernelInputParameters>& info)
{
    // We give tabulated Ewald the same name as Ewald to use the same reference data
    CoulombKernelType coulombKernelType = info.param.coulombKernelType;
    switch (coulombKernelType)
    {
        case CoulombKernelType::Table: coulombKernelType = CoulombKernelType::Ewald; break;
        case CoulombKernelType::TableTwin: coulombKernelType = CoulombKernelType::EwaldTwin; break;
        default: break;
    }
    std::string testName =
            formatString("type_%s_Tab%s_%s_Coulomb%s_Vdw%s",
                         nbnxmKernelTypeToName(info.param.kernelType),
                         info.param.coulombKernelType == CoulombKernelType::Table
                                         || info.param.coulombKernelType == CoulombKernelType::TableTwin
                                 ? "Yes"
                                 : "No",
                         sc_energyGroupNames[info.param.energyHandling],
                         coulombKernelTypeName[coulombKernelType],
                         vdwKernelTypeName[info.param.vdwKernelType]);

    // Note that the returned names must be unique and may use only
    // alphanumeric ASCII characters. It's not supposed to contain
    // underscores (see the GoogleTest FAQ
    // why-should-test-suite-names-and-test-names-not-contain-underscore),
    // but doing so works for now, is likely to remain so, and makes
    // such test names much more readable.
    testName = replaceAll(testName, "-", "_");
    testName = replaceAll(testName, ".", "_");
    testName = replaceAll(testName, " ", "_");
    return testName;
}

bool isTabulated(const CoulombKernelType coulombKernelType)
{
    return coulombKernelType == CoulombKernelType::Table || coulombKernelType == CoulombKernelType::TableTwin;
}

/*! \brief Construct a refdata filename for this test
 *
 * We want the same reference data to apply to every kernel type
 * that we test. That means we need to store it in a file whose
 * name relates to the name of the test excluding the part related to
 * the kernel type. By default, the reference data filename is
 * set via a call to gmx::TestFileManager::getTestSpecificFileName()
 * that queries GoogleTest and gets a string that includes the return
 * value for nameOfTest(). This code works similarly, but removes the
 * part that relates to kernel type. This logic must match the
 * implementation of nameOfTest() so that it works as intended.
 *
 * In particular, the name must include a "Coulomb" substring that
 * follows the name of the kernel type, so that this can be
 * removed. */
std::string makeRefDataFileName()
{
    // Get the info about the test
    const ::testing::TestInfo* testInfo = ::testing::UnitTest::GetInstance()->current_test_info();

    // Get the test name and edit it to remove the kernel-type
    // part.
    std::string testName(testInfo->name());
    auto        separatorPos = testName.find("Coulomb");
    testName                 = testName.substr(separatorPos);

    // Build the complete filename like getTestSpecificFilename() does
    // it.
    std::string testSuiteName(testInfo->test_suite_name());
    std::string refDataFileName = testSuiteName + "_" + testName + ".xml";
    std::replace(refDataFileName.begin(), refDataFileName.end(), '/', '_');

    // Use the check that the name isn't too long
    checkTestNameLength(refDataFileName);
    return refDataFileName;
}

} // namespace

//! Test case that checks that the NBNxM kernel produces correct output.
TEST_P(NbnxmKernelTest, WorksWith)
{
    // The test parameters with which the test case was instantiated
    // TODO rename these in a follow-up change to conform to style
    KernelInputParameters               parameters_ = GetParam();
    KernelOptions                       options_;
    std::unique_ptr<nonbonded_verlet_t> nbv_;

    // TODO remove this indentation in a follow-up change
    {
        options_.kernelSetup.kernelType = parameters_.kernelType;

        // Coulomb settings
        options_.kernelSetup.ewaldExclusionType = isTabulated(parameters_.coulombKernelType)
                                                          ? EwaldExclusionType::Table
                                                          : EwaldExclusionType::Analytical;
        options_.coulombType                    = parameters_.coulombKernelType;

        // Van der Waals settings
        switch (parameters_.vdwKernelType)
        {
            case vdwktLJCUT_COMBGEOM:
                options_.ljCombinationRule = LJCombinationRule::Geometric;
                break;
            case vdwktLJCUT_COMBLB:
                options_.ljCombinationRule = LJCombinationRule::LorentzBerthelot;
                break;
            default: options_.ljCombinationRule = LJCombinationRule::None; break;
        }
        switch (parameters_.vdwKernelType)
        {
            case vdwktLJFORCESWITCH:
                options_.vdwModifier = InteractionModifiers::ForceSwitch;
                break;
            case vdwktLJPOTSWITCH: options_.vdwModifier = InteractionModifiers::PotSwitch; break;
            default: options_.vdwModifier = InteractionModifiers::PotShift; break;
        }
        options_.useLJPme = (parameters_.vdwKernelType == vdwktLJEWALDCOMBGEOM);

        if (referenceDataMode() != ReferenceDataMode::Compare)
        {
            // Note that (for simplicity) runs in modes
            // ReferenceDataMode::CreateMissing or
            // ReferenceDataMode::UpdateChanged also skips
            // testing unchanged values that could have been compared.
            if (!GMX_DOUBLE)
            {
                ADD_FAILURE() << "Reference data can only be created or updated from a "
                                 "double-precision build of GROMACS";
            }

            if (options_.kernelSetup.kernelType == NbnxmKernelType::Cpu4x4_PlainC)
            {
                GTEST_SKIP() << "Plain-C kernels are never used to generate reference data";
            }

            if (options_.coulombType == CoulombKernelType::Table
                || options_.coulombType == CoulombKernelType::TableTwin)
            {
                GTEST_SKIP() << "Tabulated kernels are never used to generate reference data";
            }
        }

        if (!sc_haveNbnxmSimd4xmKernels && parameters_.kernelType == NbnxmKernelType::Cpu4xN_Simd_4xN)
        {
            GTEST_SKIP()
                    << "Cannot test or generate data for 4xN kernels without suitable SIMD support";
        }

        if (!sc_haveNbnxmSimd2xmmKernels && parameters_.kernelType == NbnxmKernelType::Cpu4xN_Simd_2xNN)
        {
            GTEST_SKIP() << "Cannot test or generate data for 2xNN kernels without suitable SIMD "
                            "support";
        }

        if (options_.kernelSetup.kernelType == NbnxmKernelType::Cpu4x4_PlainC
            && (options_.coulombType == CoulombKernelType::Ewald
                || options_.coulombType == CoulombKernelType::EwaldTwin))
        {
            GTEST_SKIP()
                    << "Analytical Ewald is not implemented for the plain-C kernel, skip this test";
        }

        if (options_.kernelSetup.kernelType == NbnxmKernelType::Cpu4x4_PlainC
            && (parameters_.vdwKernelType == vdwktLJCUT_COMBGEOM
                || parameters_.vdwKernelType == vdwktLJCUT_COMBLB))
        {
            GTEST_SKIP() << "There are no combination rule versions of the plain-C kernel";
        }

        GMX_ASSERT(*std::max_element(sc_numEnergyGroups.begin(), sc_numEnergyGroups.end())
                           == TestSystem::sc_numEnergyGroups,
                   "The test system should have a sufficient number of energy groups");

        // TODO rename this in a follow-up change to conform to style
        TestSystem system_(parameters_.vdwKernelType == vdwktLJCUT_COMBGEOM
                                   ? LJCombinationRule::Geometric
                                   : LJCombinationRule::LorentzBerthelot);

        const interaction_const_t ic = setupInteractionConst(options_);

        // Set up test checkers with suitable tolerances
        //
        // The reference data for double is generated with 44 accuracy bits,
        // so we should not compare with more than that accuracy
        const int  simdAccuracyBits = (GMX_DOUBLE ? std::min(GMX_SIMD_ACCURACY_BITS_DOUBLE, 44)
                                                  : std::min(GMX_SIMD_ACCURACY_BITS_SINGLE, 22));
        const real simdRealEps      = std::pow(0.5_real, simdAccuracyBits);

        TestReferenceData    refData(makeRefDataFileName());
        TestReferenceChecker forceChecker(refData.rootChecker());
        const real           forceMagnitude = 1000;
        const real           ulpTolerance   = 50;
        real                 tolerance      = forceMagnitude * simdRealEps * ulpTolerance;
        if (usingPmeOrEwald(ic.eeltype))
        {
            real ewaldRelError;
            if (isTabulated(options_.coulombType))
            {
                // The relative energy error for tables is 0.1 times the value at the cut-off.
                // We assume that for the force this factor is 1.
                ewaldRelError = options_.ewaldRTol;
            }
            else
            {
                ewaldRelError = GMX_DOUBLE ? 1e-11 : 1e-6;
            }
            const real maxEwaldPairForceError =
                    ic.epsfac * ewaldRelError * gmx::square(system_.maxCharge() / ic.rcoulomb);
            // We assume that the total force error is at max 20 times that of one pair
            tolerance = std::max(tolerance, 20 * maxEwaldPairForceError);
        }
        if (ic.vdwtype == VanDerWaalsType::Pme)
        {
            const real ulpToleranceExp = 400;
            tolerance = std::max(tolerance, forceMagnitude * simdRealEps * ulpToleranceExp);
        }
        forceChecker.setDefaultTolerance(absoluteTolerance(tolerance));

        TestReferenceChecker ljEnergyChecker(refData.rootChecker());
        // Energies per atom are more accurate than forces, but there is loss
        // of precision due to summation over all atoms. The tolerance on
        // the energy turns out to be the same as on the forces.
        ljEnergyChecker.setDefaultTolerance(absoluteTolerance(tolerance));
        TestReferenceChecker coulombEnergyChecker(refData.rootChecker());
        // Coulomb energy errors are higher
        coulombEnergyChecker.setDefaultTolerance(absoluteTolerance(10 * tolerance));

        // Finish setting up data structures
        nbv_ = setupNbnxmForBenchInstance(options_, system_);
        nbv_->constructPairlist(InteractionLocality::Local, system_.excls, 0, nullptr);

        std::vector<RVec> shiftVecs(c_numShiftVectors);
        calc_shifts(system_.box, shiftVecs);

        StepWorkload stepWork;
        stepWork.computeForces = true;
        stepWork.computeEnergy = options_.energyHandling != EnergyHandling::NoEnergies;

        std::vector<real> vVdw(square(sc_numEnergyGroups[options_.energyHandling]));
        std::vector<real> vCoulomb(square(sc_numEnergyGroups[options_.energyHandling]));

        // Call the kernel to test
        nbv_->dispatchNonbondedKernel(
                InteractionLocality::Local, ic, stepWork, enbvClearFYes, shiftVecs, vVdw, vCoulomb, nullptr);

        // Get and check the forces
        std::vector<RVec> forces(system_.coordinates.size(), { 0.0_real, 0.0_real, 0.0_real });
        nbv_->atomdata_add_nbat_f_to_f(AtomLocality::All, forces);
        forceChecker.checkSequence(forces.begin(), forces.end(), "Forces");

        // Check the energies, as applicable
        if (options_.energyHandling == EnergyHandling::NoEnergies)
        {
            // The force-only kernels can't compare with the reference
            // data for energies.
            ljEnergyChecker.disableUnusedEntriesCheck();
            coulombEnergyChecker.disableUnusedEntriesCheck();
        }
        else if (options_.energyHandling == EnergyHandling::Energies)
        {
            ljEnergyChecker.checkReal(vVdw[0], "VdW energy");
            coulombEnergyChecker.checkReal(vCoulomb[0], "Coulomb energy");
            // The energy kernels can't compare with the reference data
            // for energy groups.
            ljEnergyChecker.disableUnusedEntriesCheck();
            coulombEnergyChecker.disableUnusedEntriesCheck();
        }
        else if (options_.energyHandling == EnergyHandling::ThreeEnergyGroups)
        {
            // Cross check the sum of group energies with the total energies
            real vVdwGroupsSum     = std::accumulate(vVdw.begin(), vVdw.end(), 0.0_real);
            real vCoulombGroupsSum = std::accumulate(vCoulomb.begin(), vCoulomb.end(), 0.0_real);
            ljEnergyChecker.checkReal(vVdwGroupsSum, "VdW energy");
            coulombEnergyChecker.checkReal(vCoulombGroupsSum, "Coulomb energy");

            ljEnergyChecker.checkSequence(vVdw.begin(), vVdw.end(), "VdW group pair energy");
            coulombEnergyChecker.checkSequence(
                    vCoulomb.begin(), vCoulomb.end(), "Coulomb group pair energy");
        }
    }
};

INSTANTIATE_TEST_SUITE_P(Combinations,
                         NbnxmKernelTest,
                         ::testing::ConvertGenerator<KernelInputParameters::TupleT>(::testing::Combine(
                                 ::testing::Values(NbnxmKernelType::Cpu4x4_PlainC,
                                                   NbnxmKernelType::Cpu4xN_Simd_4xN,
                                                   NbnxmKernelType::Cpu4xN_Simd_2xNN),
                                 ::testing::Values(CoulombKernelType::ReactionField,
                                                   CoulombKernelType::Ewald,
                                                   CoulombKernelType::EwaldTwin,
                                                   CoulombKernelType::Table,
                                                   CoulombKernelType::TableTwin),
                                 ::testing::Values(int(vdwktLJCUT_COMBGEOM),
                                                   int(vdwktLJCUT_COMBLB),
                                                   int(vdwktLJCUT_COMBNONE),
                                                   int(vdwktLJFORCESWITCH),
                                                   int(vdwktLJPOTSWITCH),
                                                   int(vdwktLJEWALDCOMBGEOM)),
                                 ::testing::Values(EnergyHandling::NoEnergies,
                                                   EnergyHandling::Energies,
                                                   EnergyHandling::ThreeEnergyGroups))),
                         nameOfTest);

} // namespace test

} // namespace gmx
