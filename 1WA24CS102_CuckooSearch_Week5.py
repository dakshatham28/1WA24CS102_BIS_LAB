import numpy as np
import math

class CuckooProteinAnalyzer:
    def __init__(self, target_coords, reference_coords, n_nests=60, pa=0.25, max_iter=1000, alpha=0.15):
        """
        target_coords    : Nx3 numpy array of (X, Y, Z) coordinates of the query protein
        reference_coords : Nx3 numpy array of (X, Y, Z) coordinates of the template protein
        n_nests          : Population size (60 candidate 3D alignments for broader coverage)
        pa               : Discovery probability (fraction of worst alignments discarded)
        max_iter         : Maximum number of iterations (1000 generations)
        alpha            : Step size scaling factor for Levy flights (0.15 for larger jumps)
        """
        self.target = np.array(target_coords, dtype=float)
        self.reference = np.array(reference_coords, dtype=float)
        
        # 6 Dimensions: [rot_x, rot_y, rot_z (in radians), trans_x, trans_y, trans_z (in Angstroms)]
        self.dim = 6
        self.n_nests = n_nests
        self.pa = pa
        self.max_iter = max_iter
        self.alpha = alpha
        
        # Define search bounds for rotations [-pi, pi] and translations [-20, 20]
        self.lower_bounds = np.array([-np.pi, -np.pi, -np.pi, -20.0, -20.0, -20.0])
        self.upper_bounds = np.array([np.pi, np.pi, np.pi, 20.0, 20.0, 20.0])
        
        # Initialize population of nests randomly within bounds
        self.nests = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.n_nests, self.dim))
        
        # Evaluate initial fitness (RMSD)
        self.fitness = np.array([self.calculate_rmsd(nest) for nest in self.nests])
        
        best_idx = np.argmin(self.fitness)
        self.best_nest = self.nests[best_idx].copy()
        self.best_rmsd = self.fitness[best_idx]

    def get_rotation_matrix(self, rx, ry, rz):
        """Constructs a 3x3 3D Euler rotation matrix."""
        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rx), -np.sin(rx)],
            [0.0, np.sin(rx), np.cos(rx)]
        ])
        
        Ry = np.array([
            [np.cos(ry), 0.0, np.sin(ry)],
            [0.0, 1.0, 0.0],
            [-np.sin(ry), 0.0, np.cos(ry)]
        ])
        
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0.0],
            [np.sin(rz), np.cos(rz), 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        return Rz @ Ry @ Rx

    def transform_structure(self, params):
        """Applies 3D Euler rotation and translation to the target protein coordinates."""
        rx, ry, rz, tx, ty, tz = params
        R = self.get_rotation_matrix(rx, ry, rz)
        translation = np.array([tx, ty, tz])
        return (self.target @ R.T) + translation

    def calculate_rmsd(self, params):
        """Computes Root Mean Square Deviation (RMSD) between target and reference."""
        aligned_coords = self.transform_structure(params)
        diff = aligned_coords - self.reference
        rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
        return rmsd

    def levy_flight(self, beta=1.5):
        """Generates heavy-tailed step sizes using Mantegna's algorithm."""
        sigma_u = (
            math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
            (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))
        ) ** (1 / beta)
        
        u = np.random.normal(0, sigma_u, self.dim)
        v = np.random.normal(0, 1, self.dim)
        return u / (np.abs(v) ** (1 / beta))

    def clip_bounds(self, solution):
        """Keeps rotation angles wrapped and translations within valid spatial bounds."""
        solution[:3] = (solution[:3] + np.pi) % (2 * np.pi) - np.pi
        solution[3:] = np.clip(solution[3:], self.lower_bounds[3:], self.upper_bounds[3:])
        return solution

    def analyze(self):
        """Main Cuckoo Search optimization loop."""
        for t in range(self.max_iter):
            # Generate new solutions (cuckoos) via Levy flights
            for i in range(self.n_nests):
                step = self.alpha * self.levy_flight() * (self.nests[i] - self.best_nest)
                new_nest = self.clip_bounds(self.nests[i] + step)
                new_rmsd = self.calculate_rmsd(new_nest)
                
                j = np.random.randint(0, self.n_nests)
                if new_rmsd < self.fitness[j]:
                    self.nests[j] = new_nest
                    self.fitness[j] = new_rmsd

            # Abandon a fraction (pa) of worst nests and replace them
            num_abandon = int(self.pa * self.n_nests)
            worst_indices = np.argsort(self.fitness)[-num_abandon:]
            
            for idx in worst_indices:
                r1, r2 = np.random.randint(0, self.n_nests, 2)
                step = np.random.rand() * (self.nests[r1] - self.nests[r2])
                self.nests[idx] = self.clip_bounds(self.nests[idx] + step)
                self.fitness[idx] = self.calculate_rmsd(self.nests[idx])

            # Track current global best alignment
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.best_rmsd:
                self.best_rmsd = self.fitness[current_best_idx]
                self.best_nest = self.nests[current_best_idx].copy()

            if (t + 1) % 200 == 0 or t == 0:
                print(f"Iteration {t+1:4d}/{self.max_iter} | Best Structural RMSD: {self.best_rmsd:.4f} Å")

        aligned_protein = self.transform_structure(self.best_nest)
        return self.best_nest, self.best_rmsd, aligned_protein


# ==========================================
# Example Execution: Structural Homology Analysis
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    
    # 1. Create a reference protein active-site motif (10 C-alpha atoms)
    t_vals = np.linspace(0, 3 * np.pi, 10)
    reference_motif = np.column_stack([2.3 * np.cos(t_vals), 2.3 * np.sin(t_vals), 1.5 * t_vals])
    
    # 2. Simulate an unaligned query protein (rotated, shifted, plus slight biological noise)
    true_rot = np.array([0.8, -0.5, 1.2])
    true_trans = np.array([8.5, -6.0, 4.2])
    noise = np.random.normal(0, 0.15, reference_motif.shape)
    
    # Apply inverse distortion using exact axis angles
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(true_rot[0]), -np.sin(true_rot[0])],
        [0.0, np.sin(true_rot[0]), np.cos(true_rot[0])]
    ])
    Ry = np.array([
        [np.cos(true_rot[1]), 0.0, np.sin(true_rot[1])],
        [0.0, 1.0, 0.0],
        [-np.sin(true_rot[1]), 0.0, np.cos(true_rot[1])]
    ])
    Rz = np.array([
        [np.cos(true_rot[2]), -np.sin(true_rot[2]), 0.0],
        [np.sin(true_rot[2]), np.cos(true_rot[2]), 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    unaligned_target = ((reference_motif - true_trans) @ np.linalg.inv(Rz @ Ry @ Rx).T) + noise

    print("--- Starting Protein Structure Alignment Analysis ---")
    analyzer = CuckooProteinAnalyzer(
        target_coords=unaligned_target,
        reference_coords=reference_motif,
        n_nests=60,
        pa=0.25,
        max_iter=1000,
        alpha=0.15
    )
    
    best_params, final_rmsd, aligned_coords = analyzer.analyze()
    
    print("\n--- Structural Analysis Report ---")
    print(f"Final Structural RMSD  : {final_rmsd:.4f} Å (Values < 2.0 Å indicate strong structural homology)")
    print(f"Optimal Rotations (rad): Rx={best_params[0]:.3f}, Ry={best_params[1]:.3f}, Rz={best_params[2]:.3f}")
    print(f"Optimal Shift (Å)      : Tx={best_params[3]:.3f}, Ty={best_params[4]:.3f}, Tz={best_params[5]:.3f}")
