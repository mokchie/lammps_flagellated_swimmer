# lammps flagellated swimmer

0. Installation

This package uses an external library in lib/linalg which must be
compiled before making LAMMPS.  See the lib/linalg/README file and the
LAMMPS manual for information on building LAMMPS with external
libraries.
Use the following scripts to include necessary packages:

`make yes-molecule`

`make yes-user-sdpd-ve`

`make yes-user-sdpdc`

`make yes-user-statistic`

=========================================================================

1. About the pair styles

pair_sdpdve.h

pair_sdpdve.cpp

pair_sdpdve_2d.h

pair_sdpdve_2d.cpp

Viscoelastic sdpd pair style using eigendynamics

——————————————————————————————————————————————————

pair_sdpdve_no_edyn.h

pair_sdpdve_no_edyn.cpp

pair_sdpdve_no_edyn_2d.h

pair_sdpdve_no_edyn_2d.cpp

Viscoelastic sdpd pair style not using the eigendynamics, a bit faster than using the eigendynamics.

——————————————————————————————————————————————————————————

pair_sdpdve_no_fluc_2d.h

pair_sdpdve_no_fluc_2d.cpp

Viscoelastic sdpd pair style without fluctuation in the conformation tensor

——————————————————————————————————————————————————————————

For pair styles using the eigendynamics, the corresponding atom style is `sdpdve` defined in files atom_vec_sdpdve.h and atom_vec_sdpdve.cpp; the corresponding fix style is `nve/sdpdve` defined in fix_nve_sdpdve.h and fix_nve_sdpdve.cpp.
For pair styles not using the eigendynamics, the corresponding atom style is `sdpd/no/edyn` defined in files atom_vec_sdpdve_no_edyn.h and atom_vec_sdpdve_no_edyn.cpp; the corresponding fix style is `nve/sdpdve/no/edyn` defined in fix_nve_sdpdve_no_edyn.h and fix_nve_sdpdve_no_edyn.cpp

Usage:

`atom_style 1 sdpdve/no/edyn`

`pair_style  sdpdve/no/fluc/2d ${P0} ${Pb} ${kappa} $T all ${rc} ${seed} empty ${rho0} ${tau} ${Np}`

`pair_coeff * * sdpdve/no/fluc/2d ${rho0} ${mu} 0 ${rc}`

`fix 1 all nve/sdpdve/no/edyn`

=========================================================================

2. About the angle styles

angle_harmonic_stochastic_omega.h

angle_harmonic_stochastic_omega.cpp

Angles are harmonic with their equilibrium angle changing sinusoidally.  And the frequency also changes stochastically
angle_style harmonic/stochastic/omega

Usage: `angle_coeff * ${Kb} ${b} ${omega_mu} ${omega_sigma} ${theta0} ${skew} ${t0}`

The parameter “Individual“  for the atom_style should be >=1 so that the initial phases are read from the data file.
If Ind=2 the amplitude b is also read from data file, it can be used to model ant-posteriorly asymmetric flagellum.
Kb: energy constant
b: amplitude of the angle oscillation
The varying frequency follows the lognormal distribution. It change its value only after one period which is determined by its current frequency.

omega_mu: mean value of the normal distribution

omega_sigma^2: variance of the normal distribution

theta0: a systematic offset on the equilibrium angle, used to model asymmetric flagellum

skew: rather than following a perfect sine wave the varying equilibrium angle can also follow a skewed sine wave

t0:  to avoid too fast movement, at the beginning of the simulation the actual oscillation amplitude increases smoothly from 0 to the specified value of b. t0 is the relaxation time of this smooth transition.

The unit of omega is radian/s, that of other angle parameters is angle.

——————————————————————————————————————————————————————————

angle_harmonic_eqvar.h

angle_harmonic_eqvar.cpp

Almost the same as harmonic_stochastic_omega except now the frequency omega is fixed.
Usage example:

`angle_style harmonic/eqvar`

`angle_coeff 1 ${Kb} ${b} ${omega} ${theta0} ${skew} ${t0}`

——————————————————————————————————————————————————————————

angle_harmonic_reciprocal.h

angle_harmonic_reciprocal.cpp

Almost the same as harmonic_eqvar except that now the formula for the angle is:
$theta = theta0 + b*sin(k*x)*sin(omega*t)$. 

Usage example:

`angle_style harmonic/eqvar`

`angle_coeff 1 ${Kb} ${b} ${omega} ${theta0} ${skew} ${t0}`


——————————————————————————————————————————————————————————


angle_harmonic_powconst.h

angle_harmonic_powconst.cpp

Usage:

`angle_style harmonic/powconst`

`angle_coeff 1 ${Kb} ${b} ${omega} ${theta0} ${skew} ${pw} ${start} c_ID ${pr} ${frac} ${t0}`

This angle style adjust the beating frequency of a flagellum to match its output power with the specified one: “pw”.
c_ID is the compute ID that computes the output power of the flagellum. See below for the compute group/group/fv
pr and frac affect how the omega is adjusted every step.

If pr>=1, every step the increment or decrement of the frequency is $|omega*frac|$.

If pr<1, the increment or decrement of the frequency is $|frac|$.

===========================================================================

3. Compute and fix 

compute_group_goup_fv.h

compute_group_goup_fv.cpp

Usage:

`compute ps1 flagellum1 group/group/fv sol vector yes velocity yes conseronly no`

It computes the output power of flagellum 1: the total output power of a flagellum is $P = -\sum_{i\le N} \sum_{j\le M} \mathbf{F}_{ij}\mathbf{v}_i$, where $N$ is the number of particles consisting of the flagellum, $M$ is the number of particles of the fluid.

The arg of velocity must be yes (default) otherwise only forces are accumulated.
The argument of vector determines whether to consider the forces that are not along the $r_{ij}$ direction (i.e. the dissipative forces). It must be yes (default) to calculate the correct output power. We can also set conseronly to be yes to consider only the conservative part of the inter-particle forces. The default is no. For SDPD particles setting “conseronly yes” overwrites “vector yes”. For SDPDVE particles the two will not be in conflict because the conservative forces are not along the $r_{ij}$ direction.

IMPORTANT: the function single_vec() is only defined in pair_sdpd.cpp and pair_sdpdve_no_fluc_2d.cpp, so only in these two pair_style can we use this compute with “vector yes” to get the output power. To add single_vec function to other pair_style is straightforward, do it yourself if you really need it.

——————————————————————————————————————————————————————————

compute_angle_local_phase.h

compute_angle_local_phase.cpp

Usage:

`compute phase1 flagellum1 angle/local/phase phase`

It compute the phases of flagellum 1

——————————————————————————————————————————————————————————

fix_spring_orientation.h

fix_spring_orientation.cpp

Usage:

`fix 4 sheet1 spring/orientation ${Km} ${phi1} xy 0`

`fix the orientation of the flagellum, it need to call some functions we newly defined in group.cpp (lr() and r2cm())`

Km: k_spring

phi1: specified orientation, radian, relative to the horizontal axis

xy: the plane, it can also be ‘yz’ or ‘xz’

ind_cal_ori: whether to use the initial orientation to be the target orientation
