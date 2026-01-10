"""Result caching and checkpointing for susceptibility computations.

This module provides efficient storage and retrieval of n_infinity values
using HDF5 format. The cache enables rapid recomputation of susceptibility
χₙ(γ,L) by storing the expensive n_infinity calculations and reusing them
for different dg values.

Classes
-------
ResultCache :
    HDF5-based cache for n_infinity values with hierarchical organization

Functions
---------
create_cache :
    Factory function to create a new cache file
load_cache :
    Load an existing cache file

Cache Structure
---------------
The HDF5 file is organized hierarchically:

    /metadata
        - version : str
        - created : timestamp
        - git_commit : str (optional)
    /L{L}/gamma_{gamma:.6e}
        - n_infinity : float
        - converged : bool
        - steps : int
        - convergence_step : int
        - t_sat : float
        - final_norm : float
        - final_hermiticity_error : float
        - runtime_sec : float
        - timestamp : str

Examples
--------
>>> # Create a new cache
>>> cache = ResultCache('results/n_inf_cache.h5', mode='w')
>>> 
>>> # Store a result
>>> from quantum_measurement.analysis import compute_n_inf
>>> result = compute_n_inf(gamma=4.0, L=17)
>>> cache.store_n_inf(gamma=4.0, L=17, result)
>>> 
>>> # Retrieve a result
>>> if cache.has_key(gamma=4.0, L=17):
...     n_inf, diagnostics = cache.get_n_inf(gamma=4.0, L=17)
...     print(f"n_inf = {n_inf}")
>>> 
>>> # Close cache
>>> cache.close()
"""

from __future__ import annotations

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import warnings


class ResultCache:
    """HDF5-based cache for n_infinity computation results.
    
    This class provides efficient storage and retrieval of steady-state
    occupation values with their associated diagnostics. The cache uses
    a hierarchical structure organized by L and gamma values.
    
    Parameters
    ----------
    filename : str or Path
        Path to HDF5 cache file
    mode : {'r', 'r+', 'w', 'a'}, optional
        File mode:
        - 'r': Read-only (file must exist)
        - 'r+': Read/write (file must exist)
        - 'w': Create file, truncate if exists
        - 'a': Read/write if exists, create otherwise
        Default is 'a'.
    git_commit : str, optional
        Git commit hash for version tracking
        
    Attributes
    ----------
    filename : Path
        Cache file path
    file : h5py.File
        HDF5 file handle
    mode : str
        File access mode
        
    Examples
    --------
    >>> cache = ResultCache('cache.h5', mode='a')
    >>> cache.store_n_inf(4.0, 17, {'n_infinity': 0.25, 'diagnostics': {...}})
    >>> n_inf, diag = cache.get_n_inf(4.0, 17)
    >>> cache.close()
    """
    
    def __init__(
        self,
        filename: Union[str, Path],
        mode: str = 'a',
        git_commit: Optional[str] = None
    ):
        self.filename = Path(filename)
        self.mode = mode
        
        # Create parent directory if needed
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        
        # Open HDF5 file
        self.file = h5py.File(self.filename, mode)
        
        # Initialize metadata if creating new file
        if mode in ['w', 'a'] and 'metadata' not in self.file:
            self._init_metadata(git_commit)
    
    def _init_metadata(self, git_commit: Optional[str] = None):
        """Initialize cache metadata."""
        meta = self.file.create_group('metadata')
        meta.attrs['version'] = '1.0'
        meta.attrs['created'] = datetime.now().isoformat()
        if git_commit:
            meta.attrs['git_commit'] = git_commit
    
    def _get_dataset_path(self, gamma: float, L: int) -> str:
        """Get HDF5 path for a given (gamma, L) pair."""
        return f"/L{L}/gamma_{gamma:.6e}"
    
    def has_key(self, gamma: float, L: int) -> bool:
        """Check if cache contains data for (gamma, L).
        
        Parameters
        ----------
        gamma : float
            Measurement rate
        L : int
            System size
            
        Returns
        -------
        bool
            True if cache contains this key
        """
        path = self._get_dataset_path(gamma, L)
        return path in self.file
    
    def store_n_inf(
        self,
        gamma: float,
        L: int,
        result: Dict
    ):
        """Store n_infinity computation result in cache.
        
        Parameters
        ----------
        gamma : float
            Measurement rate
        L : int
            System size
        result : dict
            Result from compute_n_inf containing:
            - 'n_infinity' : float
            - 'diagnostics' : dict with convergence info
            
        Examples
        --------
        >>> from quantum_measurement.analysis import compute_n_inf
        >>> result = compute_n_inf(gamma=4.0, L=17)
        >>> cache.store_n_inf(4.0, 17, result)
        """
        if self.mode == 'r':
            raise ValueError("Cannot write to cache opened in read-only mode")
        
        # Create group for this L if it doesn't exist
        L_group = f"/L{L}"
        if L_group not in self.file:
            self.file.create_group(L_group)
        
        # Create dataset path
        path = self._get_dataset_path(gamma, L)
        
        # Delete if exists (overwrite)
        if path in self.file:
            del self.file[path]
        
        # Create dataset for n_infinity
        dset = self.file.create_dataset(path, data=result['n_infinity'])
        
        # Store diagnostics as attributes
        diag = result['diagnostics']
        dset.attrs['converged'] = diag['converged']
        dset.attrs['steps'] = diag['steps']
        dset.attrs['convergence_step'] = diag['convergence_step']
        dset.attrs['t_sat'] = diag['t_sat']
        dset.attrs['final_norm'] = diag['final_norm']
        dset.attrs['final_hermiticity_error'] = diag['final_hermiticity_error']
        dset.attrs['max_steps_allocated'] = diag['max_steps_allocated']
        dset.attrs['runtime_sec'] = diag['runtime_sec']
        dset.attrs['timestamp'] = datetime.now().isoformat()
        
        # Flush to disk
        self.file.flush()
    
    def get_n_inf(
        self,
        gamma: float,
        L: int
    ) -> Tuple[float, Dict]:
        """Retrieve n_infinity and diagnostics from cache.
        
        Parameters
        ----------
        gamma : float
            Measurement rate
        L : int
            System size
            
        Returns
        -------
        n_infinity : float
            Steady-state occupation
        diagnostics : dict
            Convergence diagnostics
            
        Raises
        ------
        KeyError
            If (gamma, L) not found in cache
            
        Examples
        --------
        >>> if cache.has_key(4.0, 17):
        ...     n_inf, diag = cache.get_n_inf(4.0, 17)
        ...     print(f"Converged: {diag['converged']}")
        """
        path = self._get_dataset_path(gamma, L)
        
        if path not in self.file:
            raise KeyError(f"No cached result for gamma={gamma}, L={L}")
        
        dset = self.file[path]
        n_infinity = float(dset[()])
        
        # Extract diagnostics from attributes
        diagnostics = {
            'converged': bool(dset.attrs['converged']),
            'steps': int(dset.attrs['steps']),
            'convergence_step': int(dset.attrs['convergence_step']),
            't_sat': float(dset.attrs['t_sat']),
            'final_norm': float(dset.attrs['final_norm']),
            'final_hermiticity_error': float(dset.attrs['final_hermiticity_error']),
            'max_steps_allocated': int(dset.attrs['max_steps_allocated']),
            'runtime_sec': float(dset.attrs['runtime_sec'])
        }
        
        return n_infinity, diagnostics
    
    def get_all_keys(self) -> List[Tuple[int, float]]:
        """Get list of all (L, gamma) pairs in cache.
        
        Returns
        -------
        list of (int, float)
            List of (L, gamma) tuples
        """
        keys = []
        
        for L_group_name in self.file.keys():
            if L_group_name == 'metadata':
                continue
            
            L = int(L_group_name[1:])  # Remove 'L' prefix
            L_group = self.file[L_group_name]
            
            for gamma_name in L_group.keys():
                # Parse gamma from name: "gamma_1.234000e+00"
                gamma_str = gamma_name.split('_')[1]
                gamma = float(gamma_str)
                keys.append((L, gamma))
        
        return keys
    
    def get_L_values(self) -> List[int]:
        """Get list of all L values in cache.
        
        Returns
        -------
        list of int
            Sorted list of system sizes
        """
        L_values = []
        for key in self.file.keys():
            if key.startswith('L') and key != 'metadata':
                L = int(key[1:])
                L_values.append(L)
        return sorted(L_values)
    
    def get_gamma_values(self, L: int) -> List[float]:
        """Get list of all gamma values for a given L.
        
        Parameters
        ----------
        L : int
            System size
            
        Returns
        -------
        list of float
            Sorted list of gamma values
        """
        L_group = f"/L{L}"
        if L_group not in self.file:
            return []
        
        gamma_values = []
        for gamma_name in self.file[L_group].keys():
            gamma_str = gamma_name.split('_')[1]
            gamma = float(gamma_str)
            gamma_values.append(gamma)
        
        return sorted(gamma_values)
    
    def validate(self) -> Dict:
        """Validate cache integrity.
        
        Returns
        -------
        dict
            Validation results with:
            - 'valid' : bool
            - 'num_entries' : int
            - 'L_values' : list
            - 'warnings' : list of str
        """
        warnings_list = []
        
        # Check metadata
        if 'metadata' not in self.file:
            warnings_list.append("Missing metadata group")
        
        # Count entries
        num_entries = len(self.get_all_keys())
        L_values = self.get_L_values()
        
        # Check for NaN values
        for L, gamma in self.get_all_keys():
            n_inf, _ = self.get_n_inf(gamma, L)
            if np.isnan(n_inf) or np.isinf(n_inf):
                warnings_list.append(f"Invalid n_inf at (L={L}, gamma={gamma})")
        
        return {
            'valid': len(warnings_list) == 0,
            'num_entries': num_entries,
            'L_values': L_values,
            'warnings': warnings_list
        }
    
    def close(self):
        """Close the cache file."""
        self.file.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __repr__(self):
        """String representation."""
        num_entries = len(self.get_all_keys())
        return f"ResultCache('{self.filename}', mode='{self.mode}', entries={num_entries})"


def create_cache(
    filename: Union[str, Path],
    git_commit: Optional[str] = None
) -> ResultCache:
    """Create a new cache file (truncates if exists).
    
    Parameters
    ----------
    filename : str or Path
        Path to cache file
    git_commit : str, optional
        Git commit hash
        
    Returns
    -------
    ResultCache
        New cache instance
    """
    return ResultCache(filename, mode='w', git_commit=git_commit)


def load_cache(filename: Union[str, Path]) -> ResultCache:
    """Load an existing cache file.
    
    Parameters
    ----------
    filename : str or Path
        Path to cache file
        
    Returns
    -------
    ResultCache
        Cache instance
        
    Raises
    ------
    FileNotFoundError
        If cache file doesn't exist
    """
    if not Path(filename).exists():
        raise FileNotFoundError(f"Cache file not found: {filename}")
    
    return ResultCache(filename, mode='r+')
