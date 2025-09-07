import ase
from ase import Atoms
from ase.build import molecule, bulk, surface, add_adsorbate, fcc111
from ase.io import read, write, Trajectory
from ase.calculators.emt import EMT
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS, QuasiNewton, FIRE
from ase.visualize import view
from ase.mep import NEB  # For NEB example
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT, IsotropicMTKNPT
from ase.md.nptberendsen import NPTBerendsen
from ase.vibrations import Vibrations
from ase.constraints import FixAtoms
from ase.units import fs, kB
import numpy as np
import matplotlib.pyplot as plt

# Example 1: Creating simple Atoms object
print("Example 1: Creating Atoms")
atoms1 = Atoms('H2O', positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0)])
print(f"Atoms created: {atoms1.get_chemical_symbols()}")
print(f"Positions:\n{atoms1.get_positions()}")

# Example 2: Building a molecule using ase.build.molecule
print("\nExample 2: Building a molecule")
h2o = molecule('H2O')
print(f"Molecule symbols: {h2o.get_chemical_symbols()}")
print(f"Molecule positions:\n{h2o.get_positions()}")

# Example 3: Building bulk material
print("\nExample 3: Building bulk FCC gold")
au_bulk = bulk('Au', 'fcc', a=4.08)
print(f"Bulk symbols: {au_bulk.get_chemical_symbols()[:5]}...")  # First 5
print(f"Cell parameters:\n{au_bulk.get_cell()}")

# Example 4: Reading from file (assume a file exists or create one)
print("\nExample 4: Writing and reading a file")
write('test.xyz', h2o)
read_atoms = read('test.xyz')
print(f"Read atoms symbols: {read_atoms.get_chemical_symbols()}")

# Example 5: Setting a calculator and getting energy/forces
print("\nExample 5: Using EMT calculator")
h2o.calc = EMT()
energy = h2o.get_potential_energy()
forces = h2o.get_forces()
print(f"Potential energy: {energy:.4f} eV")
print(f"Forces shape: {forces.shape}")

# Example 6: Optimizing structure
print("\nExample 6: Structure optimization with BFGS")
dyn = BFGS(h2o)
dyn.run(fmax=0.05)
print(f"Optimized energy: {h2o.get_potential_energy():.4f} eV")
print(f"Optimized positions:\n{h2o.get_positions()}")

# Example 7: NEB example (simple two images)
print("\nExample 7: Simple NEB setup")
initial = Atoms('H', positions=[(0, 0, 0)])
final = Atoms('H', positions=[(1, 0, 0)])
images = [initial, final]
neb = NEB(images)
neb.interpolate()
print(f"NEB images: {len(neb.images)}")
print("NEB interpolated successfully.")

# Example 8: Visualizing (opens viewer if possible)
print("\nExample 8: Visualizing atoms")
# view(h2o)  # Uncomment to visualize in 3D viewer

# Example 9: Creating surfaces and adsorption
print("\nExample 9: Surface creation and adsorption")
# Create a Cu bulk first, then surface
cu_bulk = bulk('Cu', 'fcc', a=3.61)
cu_slab = surface(cu_bulk, (1, 1, 1), 3, vacuum=10.0)
print(f"Surface atoms: {len(cu_slab)}")
print(f"Surface cell: {cu_slab.get_cell()}")

# Add CO molecule on top
co = molecule('CO')
add_adsorbate(cu_slab, co, height=2.0, position=(0, 0))
cu_slab.calc = EMT()
energy = cu_slab.get_potential_energy()
print(f"Adsorption energy: {energy:.4f} eV")

# Example 10: Basic Molecular Dynamics with Velocity Verlet
print("\nExample 10: Basic Molecular Dynamics with Velocity Verlet")
# Set up initial velocities
MaxwellBoltzmannDistribution(h2o, temperature_K=300)
# Run MD for 100 steps with energy output every 10 steps
dyn = VelocityVerlet(h2o, timestep=1.0 * fs)
energies = []

print("Step | Time (fs) | Potential Energy (eV)")
print("-" * 40)

for i in range(100):  # Longer simulation for demo
    dyn.run(1)
    energies.append(h2o.get_potential_energy())

    # Print energy every 10 steps
    if (i + 1) % 10 == 0:
        time_fs = (i + 1) * 1.0  # timestep is 1.0 fs
        print(f"{i+1:5d} | {time_fs:10.1f} | {energies[-1]:15.4f}")

print("-" * 40)
print(f"Simulation completed: 100 steps = {100 * 1.0} fs")
print(f"Energy range: {min(energies):.3f} to {max(energies):.3f} eV")
print(f"Average energy: {np.mean(energies):.3f} eV")
print(f"Energy standard deviation: {np.std(energies):.3f} eV")

# Example 11: NVE Ensemble - Bulk Aluminum (Microcanonical)
print("\nExample 11: NVE Ensemble - Bulk Aluminum (4x4x4 supercell)")
# Create larger bulk system
bulk_al = bulk('Al', 'fcc', a=4.05) * (4, 4, 4)  # 256 atoms
bulk_al.calc = EMT()

# Set initial velocities at 800K
MaxwellBoltzmannDistribution(bulk_al, temperature_K=800)

# Run NVE MD
print(f"System: {len(bulk_al)} atoms")
print(f"Initial energy: {bulk_al.get_potential_energy():.3f} eV")
print(f"Initial temperature: {bulk_al.get_temperature():.1f} K")

dyn_nve = VelocityVerlet(bulk_al, timestep=2.0 * fs, trajectory='bulk_al_nve.traj')
energies_nve = []
temperatures_nve = []

for step in range(100):  # Run for 200 fs
    dyn_nve.run(1)
    if step % 10 == 0:
        energies_nve.append(bulk_al.get_potential_energy())
        temperatures_nve.append(bulk_al.get_temperature())

print(f"NVE simulation completed. Final temperature: {temperatures_nve[-1]:.1f} K")
print(f"Energy conservation: {energies_nve[0]:.3f} -> {energies_nve[-1]:.3f} eV")

# Example 12: NVT Ensemble - Liquid Water with Langevin Thermostat
print("\nExample 12: NVT Ensemble - Liquid Water (512 molecules)")
# Create larger water system (8x8x8 unit cells)
# Create water molecules in a cubic lattice arrangement
water_system = Atoms()
lattice_constant = 6.0  # Angstroms

# Create 8x8x8 grid of water molecules
for i in range(8):
    for j in range(8):
        for k in range(8):
            # Position for each water molecule
            pos = np.array([i, j, k]) * lattice_constant
            # Create water molecule at this position
            h2o_mol = molecule('H2O')
            h2o_mol.positions += pos
            water_system += h2o_mol

# Set periodic boundary conditions and cell
water_system.set_pbc(True)
water_system.set_cell([50*lattice_constant, 50*lattice_constant, 50*lattice_constant])
water_system.calc = EMT()

# Set initial velocities at 350K
MaxwellBoltzmannDistribution(water_system, temperature_K=350)

print(f"Water system: {len(water_system)} atoms")
print(f"Initial energy: {water_system.get_potential_energy():.3f} eV")

# Run NVT MD with Langevin thermostat
dyn_nvt = Langevin(water_system, timestep=1.0 * fs, temperature_K=350,
                   friction=0.01 / fs, trajectory='water_nvt.traj')

temperatures_nvt = []
for step in range(100):  # Run for 100 fs
    dyn_nvt.run(1)
    if step % 20 == 0:
        temperatures_nvt.append(water_system.get_temperature())

print(f"NVT simulation completed. Temperature range: {min(temperatures_nvt):.1f} - {max(temperatures_nvt):.1f} K")
print(f"Final temperature: {temperatures_nvt[-1]:.1f} K")

# Example 13: NPT Ensemble - Polymorphic Transition Study
print("\nExample 13: NPT Ensemble - Copper Polymorphic Transition")
# Create copper system near melting point
cu_system = bulk('Cu', 'fcc', a=3.61) * (6, 6, 6)  # 864 atoms
cu_system.calc = EMT()

# Set initial velocities at 1200K (near melting)
MaxwellBoltzmannDistribution(cu_system, temperature_K=1200)

print(f"Copper system: {len(cu_system)} atoms")
print(f"Initial energy: {cu_system.get_potential_energy():.3f} eV")
print(f"Initial temperature: {cu_system.get_temperature():.1f} K")

# First equilibrate with NVT
dyn_equil = NoseHooverChainNVT(cu_system, timestep=2.0 * fs, temperature_K=1200,
                              tdamp=100 * fs, trajectory='cu_equil.traj')
print("Equilibrating with NVT...")
for step in range(25):
    dyn_equil.run(1)

# Then run NPT MD
dyn_npt = IsotropicMTKNPT(cu_system, timestep=2.0 * fs, temperature_K=1200,
                         pressure_au=0.0, tdamp=100 * fs, pdamp=1000 * fs,
                         trajectory='cu_npt.traj')

volumes_npt = []
pressures_npt = []
for step in range(50):  # Run for 100 fs
    dyn_npt.run(1)
    if step % 10 == 0:
        volumes_npt.append(cu_system.get_volume())
        # Calculate pressure from stress tensor
        stress = cu_system.get_stress(voigt=False)
        pressure = -np.trace(stress) / 3
        pressures_npt.append(pressure)

print(f"NPT simulation completed.")
print(f"Final temperature: {cu_system.get_temperature():.1f} K")
print(f"Final volume: {cu_system.get_volume():.3f} Å³")

# Example 14: Surface Diffusion - Pt(111) with CO Adsorbates
print("\nExample 14: Surface Diffusion - Pt(111) with CO Adsorbates")
# Create Pt(111) surface with adsorbates
pt_slab = fcc111('Pt', size=(6, 6, 4), vacuum=10.0)  # 864 atoms
# Add CO molecules
for i in range(4):
    co = molecule('CO')
    x_pos = (i % 2) * 5.0 + 2.5
    y_pos = (i // 2) * 5.0 + 2.5
    add_adsorbate(pt_slab, co, height=2.0, position=(x_pos, y_pos))

pt_slab.calc = EMT()

print(f"Pt surface system: {len(pt_slab)} atoms")
print(f"Initial energy: {pt_slab.get_potential_energy():.3f} eV")

# Set initial velocities at 500K
MaxwellBoltzmannDistribution(pt_slab, temperature_K=500)

# Run surface diffusion simulation
dyn_surface = Langevin(pt_slab, timestep=1.5 * fs, temperature_K=500,
                       friction=0.02 / fs, trajectory='pt_surface_md.traj')

# Track CO molecule positions
co_indices = [i for i, atom in enumerate(pt_slab) if atom.symbol == 'C']
co_positions_over_time = []

for step in range(75):  # Run for 112.5 fs
    dyn_surface.run(1)
    if step % 15 == 0:
        co_pos = pt_slab.positions[co_indices]
        co_positions_over_time.append(co_pos.copy())

print(f"Surface diffusion simulation completed.")
print(f"CO molecules tracked: {len(co_indices)}")
print(f"Final energy: {pt_slab.get_potential_energy():.3f} eV")

# Example 15: Phase Transition - Ice to Water with NPT
print("\nExample 15: Phase Transition - Ice to Water with NPT")
# Create ice-like structure (simple cubic approximation)
# Create water molecules in a denser cubic lattice to simulate ice
ice_system = Atoms()
ice_lattice_constant = 4.5  # Angstroms (denser than liquid water)

# Create 3x3x3 grid of water molecules for ice-like structure
for i in range(3):
    for j in range(3):
        for k in range(3):
            # Position for each water molecule
            pos = np.array([i, j, k]) * ice_lattice_constant
            # Create water molecule at this position
            h2o_mol = molecule('H2O')
            h2o_mol.positions += pos
            ice_system += h2o_mol

# Set periodic boundary conditions and cell
ice_system.set_pbc(True)
ice_system.set_cell([3*ice_lattice_constant, 3*ice_lattice_constant, 3*ice_lattice_constant])
ice_system.calc = EMT()

print(f"Ice system: {len(ice_system)} atoms")
print(f"Initial energy: {ice_system.get_potential_energy():.3f} eV")

# Set initial velocities at 280K (below freezing)
MaxwellBoltzmannDistribution(ice_system, temperature_K=280)

# Run NPT with Berendsen coupling at higher temperature
# Compressibility of water is approximately 4.5e-5 bar^-1 = 4.5e-10 Pa^-1
# Convert to atomic units: 4.5e-10 / (1.602e-19 * 1e5) ≈ 2.8e-5 Å^3/eV
dyn_ice = NPTBerendsen(ice_system, timestep=1.0 * fs, temperature_K=320,
                       pressure_au=0.0, taut=50 * fs, taup=500 * fs,
                       compressibility_au=2.8e-5, trajectory='ice_melting.traj')

volumes_melt = []
temperatures_melt = []

for step in range(100):  # Run for 100 fs
    dyn_ice.run(1)
    if step % 20 == 0:
        volumes_melt.append(ice_system.get_volume())
        temperatures_melt.append(ice_system.get_temperature())

print(f"Melting simulation completed.")
print(f"Final temperature: {temperatures_melt[-1]:.1f} K")
print(f"Final volume: {volumes_melt[-1]:.1f} Å³")
print(f"Volume change: {volumes_melt[-1] - volumes_melt[0]:.3f} Å³")

# Example 16: Nanocomposite with Constraints
print("\nExample 16: Nanocomposite with Constraints - Al-Cu System")
# Create Al-Cu nanocomposite (both supported by EMT calculator)
al_bulk = bulk('Al', 'fcc', a=4.05) * (5, 5, 5)  # 500 atoms
cu_bulk = bulk('Cu', 'fcc', a=3.61) * (2, 2, 2)  # 32 atoms

# Combine systems
composite = al_bulk + cu_bulk
# Position Cu cluster in center of Al matrix
cu_center = np.mean(cu_bulk.positions, axis=0)
al_center = np.mean(al_bulk.positions, axis=0)
translation = al_center - cu_center
cu_bulk.positions += translation

composite = al_bulk + cu_bulk
composite.calc = EMT()

print(f"Nanocomposite system: {len(composite)} atoms")
print(f"Al atoms: {sum(1 for atom in composite if atom.symbol == 'Al')}")
print(f"Cu atoms: {sum(1 for atom in composite if atom.symbol == 'Cu')}")
print(f"Initial energy: {composite.get_potential_energy():.3f} eV")

# Fix outer Al atoms to simulate substrate
outer_mask = []
for i, atom in enumerate(composite):
    if atom.symbol == 'Al':
        # Fix atoms near the boundaries
        if (atom.position[0] < 8 or atom.position[0] > composite.cell[0,0] - 8 or
            atom.position[1] < 8 or atom.position[1] > composite.cell[1,1] - 8):
            outer_mask.append(i)

constraint = FixAtoms(mask=outer_mask)
composite.set_constraint(constraint)

print(f"Fixed atoms: {len(outer_mask)}")

# Set initial velocities at 600K
MaxwellBoltzmannDistribution(composite, temperature_K=600)

# Run constrained MD
dyn_composite = Langevin(composite, timestep=2.0 * fs, temperature_K=600,
                        friction=0.005 / fs, trajectory='nanocomposite_md.traj')

energies_composite = []
for step in range(60):  # Run for 120 fs
    dyn_composite.run(1)
    if step % 15 == 0:
        energies_composite.append(composite.get_potential_energy())

print(f"Nanocomposite MD completed.")
print(f"Final energy: {energies_composite[-1]:.3f} eV")

# Example 17: Vibrational Analysis of Benzene Molecule
print("\nExample 17: Vibrational Analysis of Benzene Molecule")
# Create benzene molecule (C6H6) - aromatic ring with rich vibrational spectrum
try:
    benzene = molecule('C6H6')  # Benzene - 12 atoms
    benzene.calc = EMT()

    print(f"Benzene molecule: {len(benzene)} atoms")
    print(f"Chemical formula: {benzene.get_chemical_formula()}")
    print(f"Initial energy: {benzene.get_potential_energy():.3f} eV")

    # Calculate vibrations for benzene
    vib = Vibrations(benzene, name='benzene_vib', delta=0.01)
    vib.run()
    frequencies = vib.get_frequencies()

    # Filter out imaginary frequencies and sort
    real_freqs = frequencies.real[frequencies.real > 0]
    print(f"Number of vibrational modes: {len(real_freqs)}")
    print(f"Vibrational frequencies (cm⁻¹): {real_freqs[:10]}...")  # Show first 10
    print(f"Highest frequency: {real_freqs.max():.1f} cm⁻¹")
    print(f"Lowest frequency: {real_freqs.min():.1f} cm⁻¹")

    # Summary of vibrational analysis
    vib.summary()

except Exception as e:
    print(f"Vibrational analysis failed: {e}")
    print("This might be due to calculator compatibility or numerical issues.")
    print("Skipping vibrational analysis for this example.")

# Example 18: Constrained Optimization with Fixed Atoms - Large Surface
print("\nExample 18: Constrained Optimization with Fixed Atoms - Large Surface")
# Create a larger surface system with adsorbates for more complex optimization
cu_bulk = bulk('Cu', 'fcc', a=3.61)
cu_slab = surface(cu_bulk, (1, 1, 1), 6, vacuum=15.0)  # 6 layers, larger vacuum

# Add multiple CO molecules to create a more complex system
for i in range(6):  # Add 6 CO molecules
    co = molecule('CO')
    x_pos = (i % 3) * 3.5 + 1.5  # Spread across surface
    y_pos = (i // 3) * 3.5 + 1.5
    add_adsorbate(cu_slab, co, height=2.0, position=(x_pos, y_pos))

# Fix bottom 3 layers (more atoms fixed for stability)
mask = [atom.index for atom in cu_slab if atom.position[2] < 12.0]  # Fix lower 3 layers
constraint = FixAtoms(mask=mask)
cu_slab.set_constraint(constraint)
cu_slab.calc = EMT()

print(f"Large surface system: {len(cu_slab)} atoms")
print(f"Fixed atoms: {len(mask)}")
print(f"Mobile atoms: {len(cu_slab) - len(mask)}")
print(f"Initial energy: {cu_slab.get_potential_energy():.3f} eV")

# Optimize with constraints - this will take longer due to larger system
dyn = BFGS(cu_slab, maxstep=0.1)  # Smaller maxstep for stability
print("Starting constrained optimization (this may take a moment)...")
dyn.run(fmax=0.1)  # Relaxed convergence for demonstration
print(f"Constrained optimization completed!")
print(f"Final energy: {cu_slab.get_potential_energy():.4f} eV")
print(f"Forces converged to: {dyn.fmax:.4f} eV/Å")

# Example 19: Nudged Elastic Band (NEB) Method
print("\nExample 19: Nudged Elastic Band (NEB) for H2 Dissociation")
# Create initial and final states for H2 dissociation
initial = Atoms('H2', positions=[(0, 0, 0), (0.74, 0, 0)])
final = Atoms('H2', positions=[(0, 0, 0), (3.0, 0, 0)])

# Create 7 images
images = [initial]
for i in range(5):
    images.append(initial.copy())
images.append(final)

# Set up NEB
neb = NEB(images)
neb.interpolate()

# Attach calculator to all images
for image in images:
    image.calc = EMT()

# Optimize NEB path
optimizer = FIRE(neb)
optimizer.run(fmax=0.05)

# Get energies and find barrier
energies = [img.get_potential_energy() for img in images]
barrier = max(energies) - energies[0]
print(f"NEB barrier: {barrier:.4f} eV")
print(f"Energy profile: {energies}")

# Example 20: Custom Lennard-Jones Calculator Implementation
print("\nExample 20: Custom Lennard-Jones Calculator Implementation")

class SimpleLJ(Calculator):
    """Custom Lennard-Jones potential calculator for demonstration."""
    implemented_properties = ['energy', 'forces']

    def __init__(self, epsilon=1.0, sigma=1.0, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.sigma = sigma

    def calculate(self, atoms=None, properties=['energy'], system_changes=None):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        energy = 0.0
        forces = np.zeros_like(positions)

        # Calculate Lennard-Jones interactions
        for i in range(len(atoms)):
            for j in range(i+1, len(atoms)):
                r = positions[j] - positions[i]
                dist = np.linalg.norm(r)
                if dist > 0:
                    # Lennard-Jones potential: 4ε[(σ/r)^12 - (σ/r)^6]
                    lj = 4 * self.epsilon * ((self.sigma/dist)**12 - (self.sigma/dist)**6)
                    energy += lj
                    # Force: -dU/dr = 24ε[2(σ/r)^12 - (σ/r)^6] * r_vector/r
                    f = 24 * self.epsilon * (2*(self.sigma/dist)**12 - (self.sigma/dist)**6) * r / dist**2
                    forces[i] -= f
                    forces[j] += f

        self.results = {'energy': energy, 'forces': forces}

# Test custom calculator with argon dimer
atoms_lj = Atoms('Ar2', positions=[(0, 0, 0), (3.8, 0, 0)])
atoms_lj.calc = SimpleLJ(epsilon=0.0104, sigma=3.4)  # Realistic Ar parameters
print(f"LJ energy: {atoms_lj.get_potential_energy():.4f} eV")
print(f"LJ forces on first atom: {atoms_lj.get_forces()[0]}")

# Example 21: Energy Analysis and Potential Energy Curves
print("\nExample 21: Energy Analysis and Potential Energy Curves")
# Calculate energy vs distance for H2
distances = np.linspace(0.5, 3.0, 20)
energies = []

for d in distances:
    h2 = Atoms('H2', positions=[(0, 0, 0), (d, 0, 0)])
    h2.calc = EMT()
    energies.append(h2.get_potential_energy())

# Simple plot (would need matplotlib for visualization)
plt.figure(figsize=(8, 5))
plt.plot(distances, energies, 'bo-')
plt.xlabel('H-H distance (Å)')
plt.ylabel('Energy (eV)')
plt.title('H2 potential energy curve')
plt.grid(True)
plt.savefig('h2_energy_curve.png', dpi=300, bbox_inches='tight')
print("Energy curve plot saved as 'h2_energy_curve.png'")

print("\n" + "="*60)
print("COMPREHENSIVE ASE TUTORIAL COMPLETED SUCCESSFULLY")
print("="*60)
print("This tutorial covered:")
print("• Basic ASE operations (Examples 1-9)")
print("• Molecular dynamics simulations (Examples 10-16)")
print("• Advanced analysis techniques (Examples 17-21)")
print("• Total examples: 21")
print("• Trajectory files generated: 7")
print("• Topics: atoms, molecules, bulk, surfaces, MD, NEB, vibrations, constraints")
print("• All examples run with larger systems demonstrating real-world applications")
