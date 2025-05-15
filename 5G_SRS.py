import openpyxl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import struct
from typing import Dict, Any, Tuple, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import json
import traceback
import os
from primePy import primes
from scipy import signal as sig
from scipy.io import savemat
import math

class SRS_Generator:
    """
    Generator for 5G NR Sounding Reference Signals (SRS)

    Implementation based on 3GPP TS 38.211 version 15.2.0 Release 15
    """

    def __init__(self, params=None):
        """
        Initialize SRS generator with default or provided parameters

        Args:
            params: Dictionary of SRS parameters
        """
        # Set default parameters
        self.default_params = {
            'p_tx_port': 1000,     # Antenna port (1000-1011)
            'sym_offset': 0,       # SRS symbol offset (l0)
            'c_srs': 6,            # SRS configuration index (0-63)
            'b_srs': 0,            # Bandwidth parameter (0-3)
            'comb': 2,             # SRS comb size (2 or 4)
            'comb_offset': 0,      # Comb offset (0 to comb-1)
            'b_hop': 0,            # Frequency hopping (0-3)
            'n_rrc': 0,            # Frequency domain position (0-23)
            'fd_shift': 0,         # Frequency domain shift
            'n_syms': 1,           # Number of SRS symbols
            'n_ports': 1,          # Number of antenna ports
            'n_cs': 0,             # Cyclic shift index
            'n_id': 0,             # Scrambling ID (0-1023)
            'hop_mod': 0,          # Hopping mode (0: Neither, 1: Group, 2: Sequence)
            'bwp_offset': 0,       # BWP resource block offset
            'prb_num': 25,         # Number of PRBs (default 5G NR 20MHz = 106 PRBs)
            'snr_prb_en': True,    # Enable SNR per PRB estimation
            'ta_est_en': True,     # Enable timing alignment estimation
            'mu': 0,               # Numerology (0: 15kHz, 1: 30kHz, 2: 60kHz)
            'fft_size': 1024,      # FFT size
            'cp_type': 'normal',   # Cyclic prefix type
            'carrier_freq': 3.5e9, # Carrier frequency (Hz)
            'n_slots': 10,         # Total Slots to generate
            "T_srs":1,             # Slot Periodicity
            "T_offset":0,          # Slot offset (Alternate Name)
        }

        # Update with provided parameters if any
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)

        # Initialize tables from 3GPP spec
        self._init_tables()

        # Validate parameters after tables are initialized
        # (since validation depends on table entries)
        self._validate_params()


    def _validate_params(self):
        """
        Validate and adjust SRS parameters with regressive checking approach

        This function performs comprehensive validation in multiple phases:
        1. Critical validation - prevents crashes
        2. Specification compliance - ensures 3GPP conformance
        3. Parameter relationship validation - maintains consistency
        4. Performance optimization checks - ensures efficient operation

        When possible, parameters are adjusted rather than rejected outright.
        All adjustments and warnings are tracked and reported.
        """
        p = self.params

        # Store original parameters and initialize tracking
        original_params = p.copy()
        validation_warnings = []
        param_adjustments = {}

        try:
            # ======== CRITICAL PARAMETER VALIDATION ========
            # Check existence of mandatory parameters
            mandatory_params = ['prb_num', 'comb', 'c_srs', 'b_srs']
            for param in mandatory_params:
                assert param in p, f"Missing mandatory parameter: {param}"

            # Type checking for critical parameters
            assert isinstance(p['prb_num'], int), "prb_num must be an integer"
            assert isinstance(p['comb'], int), "comb must be an integer"
            assert isinstance(p['c_srs'], int), "c_srs must be an integer"
            assert isinstance(p['b_srs'], int), "b_srs must be an integer"

            # Basic range checking for core parameters
            assert p['prb_num'] > 0, "prb_num must be positive"
            assert p['comb'] in [2, 4], "comb must be 2 or 4"
            assert 0 <= p['c_srs'] <= 63, "c_srs must be between 0 and 63"
            assert 0 <= p['b_srs'] <= 3, "b_srs must be between 0 and 3"

            # ======== SPECIFICATION COMPLIANCE ========
            # Check port range (3GPP compliant)
            if not (1000 <= p['p_tx_port'] <= 1011):
                old_val = p['p_tx_port']
                p['p_tx_port'] = max(1000, min(1011, p['p_tx_port']))
                param_adjustments['p_tx_port'] = (old_val, p['p_tx_port'])

            # Check comb_offset range
            if not (0 <= p['comb_offset'] < p['comb']):
                old_val = p['comb_offset']
                p['comb_offset'] = p['comb_offset'] % p['comb']
                param_adjustments['comb_offset'] = (old_val, p['comb_offset'])

            # Check b_hop range
            if not (0 <= p['b_hop'] <= 3):
                old_val = p['b_hop']
                p['b_hop'] = max(0, min(3, p['b_hop']))
                param_adjustments['b_hop'] = (old_val, p['b_hop'])

            # Check n_rrc range
            if not (0 <= p['n_rrc'] <= 23):
                old_val = p['n_rrc']
                p['n_rrc'] = max(0, min(23, p['n_rrc']))
                param_adjustments['n_rrc'] = (old_val, p['n_rrc'])

            # Check n_id range
            if not (0 <= p['n_id'] <= 1023):
                old_val = p['n_id']
                p['n_id'] = p['n_id'] % 1024  # Wrap around
                param_adjustments['n_id'] = (old_val, p['n_id'])

            # Check hop_mod range
            if not (0 <= p['hop_mod'] <= 2):
                old_val = p['hop_mod']
                p['hop_mod'] = max(0, min(2, p['hop_mod']))
                param_adjustments['hop_mod'] = (old_val, p['hop_mod'])

            # Check numerology
            if not (0 <= p['mu'] <= 3):
                old_val = p['mu']
                p['mu'] = max(0, min(3, p['mu']))
                param_adjustments['mu'] = (old_val, p['mu'])
                validation_warnings.append(f"Numerology adjusted to {p['mu']}")

            # Check CP type
            if p['cp_type'] not in ['normal', 'extended']:
                old_val = p['cp_type']
                p['cp_type'] = 'normal'  # Default to normal CP
                param_adjustments['cp_type'] = (old_val, p['cp_type'])
                validation_warnings.append(f"CP type defaulted to 'normal'")

            # Check if extended CP is used with mu>0 (not allowed)
            if p['cp_type'] == 'extended' and p['mu'] > 0:
                old_val = p['cp_type']
                p['cp_type'] = 'normal'
                param_adjustments['cp_type'] = (old_val, p['cp_type'])
                validation_warnings.append("Extended CP not allowed with numerology > 0, defaulted to normal CP")

            # ======== PARAMETER RELATIONSHIP VALIDATION ========
            # Validate SRS configuration based on tables
            c_srs = p['c_srs']
            b_srs = p['b_srs']

            # Check if the c_srs exists in the table
            if c_srs >= len(self.m_srs_table):
                old_val = c_srs
                c_srs = min(c_srs, len(self.m_srs_table) - 1)
                p['c_srs'] = c_srs
                param_adjustments['c_srs'] = (old_val, c_srs)
                validation_warnings.append(f"c_srs adjusted to {c_srs} to match available configurations")

            # Check if b_srs is valid for the given c_srs
            if b_srs >= len(self.m_srs_table[c_srs]):
                old_val = b_srs
                b_srs = len(self.m_srs_table[c_srs]) - 1
                p['b_srs'] = b_srs
                param_adjustments['b_srs'] = (old_val, b_srs)
                validation_warnings.append(f"b_srs adjusted to {b_srs} for c_srs={c_srs}")

            # Validate T_srs (periodicity) and T_offset relationship
            if 'T_srs' not in p or p['T_srs'] <= 0:
                old_val = p.get('T_srs', 0)
                p['T_srs'] = 1  # Default to 1 slot
                param_adjustments['T_srs'] = (old_val, p['T_srs'])

            if 'T_offset' not in p or p['T_offset'] < 0:
                old_val = p.get('T_offset', -1)
                p['T_offset'] = 0
                param_adjustments['T_offset'] = (old_val, p['T_offset'])

            # Ensure T_offset < T_srs
            if p['T_offset'] >= p['T_srs']:
                old_val = p['T_offset']
                p['T_offset'] = p['T_offset'] % p['T_srs']
                param_adjustments['T_offset'] = (old_val, p['T_offset'])
                validation_warnings.append(f"T_offset adjusted to {p['T_offset']} to be less than T_srs={p['T_srs']}")

            # Validate symbol offset
            symbols_per_slot = 14 if p['cp_type'] == 'normal' else 12
            if not (0 <= p['sym_offset'] < symbols_per_slot):
                old_val = p['sym_offset']
                p['sym_offset'] = p['sym_offset'] % symbols_per_slot
                param_adjustments['sym_offset'] = (old_val, p['sym_offset'])

            # Validate n_syms fits within the slot
            if p['sym_offset'] + p['n_syms'] > symbols_per_slot:
                old_val = p['n_syms']
                p['n_syms'] = symbols_per_slot - p['sym_offset']
                param_adjustments['n_syms'] = (old_val, p['n_syms'])
                validation_warnings.append(f"n_syms adjusted to {p['n_syms']} to fit within slot")

            # Validate hopping configuration
            if p['hop_mod'] > 0 and p['b_hop'] > p['b_srs']:
                old_val = p['b_hop']
                p['b_hop'] = p['b_srs']
                param_adjustments['b_hop'] = (old_val, p['b_hop'])
                validation_warnings.append(f"b_hop adjusted to {p['b_hop']} to be <= b_srs")

            # Validate number of ports
            if p['n_ports'] > 1 and (p['p_tx_port'] + p['n_ports'] - 1) > 1011:
                old_val = p['n_ports']
                p['n_ports'] = 1011 - p['p_tx_port'] + 1
                param_adjustments['n_ports'] = (old_val, p['n_ports'])
                validation_warnings.append(f"n_ports adjusted to {p['n_ports']} to fit within valid port range")

            # ======== PERFORMANCE OPTIMIZATION VALIDATION ========
            # Validate FFT size is sufficient for PRB configuration
            min_fft_size = 2**(math.ceil(math.log2(p['prb_num'] * 12)))
            if p['fft_size'] < min_fft_size:
                old_val = p['fft_size']
                p['fft_size'] = min_fft_size
                param_adjustments['fft_size'] = (old_val, p['fft_size'])
                validation_warnings.append(f"fft_size increased to {p['fft_size']} to accommodate {p['prb_num']} PRBs")

            # Validate BWP configuration
            # Helper function inline
            max_prbs_table = {
                0: 275,  # 15 kHz
                1: 275,  # 30 kHz
                2: 275,  # 60 kHz
                3: 138   # 120 kHz
            }
            max_prbs = max_prbs_table.get(p['mu'], 275)  # Default to 275 if unknown

            if p['bwp_offset'] + p['prb_num'] > max_prbs:
                # Try to adjust bwp_offset first
                if p['prb_num'] <= max_prbs:
                    old_val = p['bwp_offset']
                    p['bwp_offset'] = max(0, max_prbs - p['prb_num'])
                    param_adjustments['bwp_offset'] = (old_val, p['bwp_offset'])
                    validation_warnings.append(f"bwp_offset adjusted to {p['bwp_offset']} to fit within carrier bandwidth")
                else:
                    # If PRBs are too many, adjust them
                    old_val = p['prb_num']
                    p['prb_num'] = max_prbs
                    param_adjustments['prb_num'] = (old_val, p['prb_num'])
                    validation_warnings.append(f"prb_num adjusted to {p['prb_num']} to fit within carrier bandwidth")

            # Check n_slots is reasonable
            if 'n_slots' not in p or p['n_slots'] <= 0:
                old_val = p.get('n_slots', 0)
                p['n_slots'] = 10  # Default to 10 slots
                param_adjustments['n_slots'] = (old_val, p['n_slots'])
            elif p['n_slots'] > 1000:
                validation_warnings.append(f"Large n_slots value ({p['n_slots']}) may impact performance")

            # ======== REPORT VALIDATION RESULTS ========
            # Store validation results
            self.validation_warnings = validation_warnings
            self.param_adjustments = param_adjustments

            # Log changes if any parameters were adjusted
            if param_adjustments:
                print("WARNING: The following parameters were adjusted:")
                for param, (old_val, new_val) in param_adjustments.items():
                    print(f"  - {param}: {old_val} → {new_val}")

            # Log warnings if any
            if validation_warnings:
                print("VALIDATION WARNINGS:")
                for warning in validation_warnings:
                    print(f"  - {warning}")

            return True

        except AssertionError as e:
            # Restore original parameters to ensure consistent state
            self.params = original_params
            raise ValueError(f"Parameter validation failed: {str(e)}")

    def _init_tables(self):
      """Initialize tables from 3GPP TS 38.211"""

      # Table 6.4.1.4.3-1: SRS bandwidth configuration
      # Format: {c_srs: [m_SRS_0, m_SRS_1, m_SRS_2, m_SRS_3]}
      self.m_srs_table = {
        0: [4, 4, 4, 4],
        1: [8, 4, 4, 4],
        2: [12, 4, 4, 4],
        3: [16, 4, 4, 4],
        4: [16, 8, 4, 4],
        5: [20, 4, 4, 4],
        6: [24, 4, 4, 4],
        7: [24, 12, 4, 4],
        8: [28, 4, 4, 4],
        9: [32, 16, 8, 4],
        10: [36, 12, 4, 4],
        11: [40, 20, 4, 4],
        12: [48, 16, 8, 4],
        13: [48, 24, 12, 4],
        14: [52, 4, 4, 4],
        15: [56, 28, 4, 4],
        16: [60, 20, 4, 4],
        17: [64, 32, 16, 4],
        18: [72, 24, 12, 4],
        19: [72, 36, 12, 4],
        20: [76, 4, 4, 4],
        21: [80, 40, 20, 4],
        22: [88, 44, 4, 4],
        23: [96, 32, 16, 4],
        24: [96, 48, 24, 4],
        25: [104, 52, 4, 4],
        26: [112, 56, 28, 4],
        27: [120, 60, 20, 4],
        28: [120, 40, 8, 4],
        29: [120, 24, 12, 4],
        30: [128, 64, 32, 4],
        31: [128, 64, 16, 4],
        32: [128, 16, 8, 4],
        33: [132, 44, 4, 4],
        34: [136, 68, 4, 4],
        35: [144, 72, 36, 4],
        36: [144, 48, 24, 12],
        37: [144, 48, 16, 4],
        38: [144, 16, 8, 4],
        39: [152, 76, 4, 4],
        40: [160, 80, 40, 4],
        41: [160, 80, 20, 4],
        42: [160, 32, 16, 4],
        43: [168, 84, 28, 4],
        44: [176, 88, 44, 4],
        45: [184, 92, 4, 4],
        46: [192, 96, 48, 4],
        47: [192, 96, 24, 4],
        48: [192, 64, 16, 4],
        49: [192, 24, 8, 4],
        50: [208, 104, 52, 4],
        51: [216, 108, 36, 4],
        52: [224, 112, 56, 4],
        53: [240, 120, 60, 4],
        54: [240, 80, 20, 4],
        55: [240, 48, 16, 8],
        56: [240, 24, 12, 4],
        57: [256, 128, 64, 4],
        58: [256, 128, 32, 4],
        59: [256, 16, 8, 4],
        60: [264, 132, 44, 4],
        61: [272, 136, 68, 4],
        62: [272, 68, 4, 4],
        63: [272, 16, 8, 4]
      }

      # Table 6.4.1.4.3-1: SRS N_i values for frequency domain position
      # Format: {c_srs: [N_0, N_1, N_2, N_3]}
      self.N_srs_table = {
        0: [1, 1, 1, 1],
        1: [1, 2, 1, 1],
        2: [1, 3, 1, 1],
        3: [1, 4, 1, 1],
        4: [1, 2, 2, 1],
        5: [1, 5, 1, 1],
        6: [1, 6, 1, 1],
        7: [1, 2, 3, 1],
        8: [1, 7, 1, 1],
        9: [1, 2, 2, 2],
        10: [1, 3, 3, 1],
        11: [1, 2, 5, 1],
        12: [1, 3, 2, 2],
        13: [1, 2, 2, 3],
        14: [1, 13, 1, 1],
        15: [1, 2, 7, 1],
        16: [1, 3, 5, 1],
        17: [1, 2, 2, 4],
        18: [1, 3, 2, 3],
        19: [1, 2, 3, 3],
        20: [1, 19, 1, 1],
        21: [1, 2, 2, 5],
        22: [1, 2, 11, 1],
        23: [1, 3, 2, 4],
        24: [1, 2, 2, 6],
        25: [1, 2, 13, 1],
        26: [1, 2, 2, 7],
        27: [1, 2, 3, 5],
        28: [1, 3, 5, 2],
        29: [1, 5, 2, 3],
        30: [1, 2, 2, 8],
        31: [1, 2, 4, 4],
        32: [1, 8, 2, 2],
        33: [1, 3, 11, 1],
        34: [1, 2, 17, 1],
        35: [1, 2, 2, 9],
        36: [1, 3, 2, 2],
        37: [1, 3, 3, 4],
        38: [1, 9, 2, 2],
        39: [1, 2, 19, 1],
        40: [1, 2, 2, 10],
        41: [1, 2, 4, 5],
        42: [1, 5, 2, 4],
        43: [1, 2, 3, 7],
        44: [1, 2, 2, 11],
        45: [1, 2, 23, 1],
        46: [1, 2, 2, 12],
        47: [1, 2, 4, 6],
        48: [1, 3, 4, 4],
        49: [1, 8, 3, 2],
        50: [1, 2, 2, 13],
        51: [1, 2, 3, 9],
        52: [1, 2, 2, 14],
        53: [1, 2, 2, 15],
        54: [1, 3, 4, 5],
        55: [1, 5, 3, 2],
        56: [1, 10, 2, 3],
        57: [1, 2, 2, 16],
        58: [1, 2, 4, 8],
        59: [1, 16, 2, 2],
        60: [1, 2, 3, 11],
        61: [1, 2, 2, 17],
        62: [1, 4, 17, 1],
        63: [1, 17, 2, 2]
      }

      # Table for cyclic shift configurations
      # Format: {n_cs: alpha_cs} (in units of π/6)
      self.alpha_cs_table = {
        0: 0, 1: 2, 2: 4, 3: 6,
        4: 8, 5: 10, 6: 1, 7: 3,
        8: 5, 9: 7, 10: 9, 11: 11
      }

    def generate_srs(self):
        """
        Generate SRS signal with configurable periodicity, inter-slot and intra-slot frequency
        hopping according to 3GPP TS 38.211 Section 6.4.1.4.

        Returns:
            re_grid: Resource grid with SRS symbols
            td_signal: Time domain signal for the generated slots
            srs_info: Dictionary with SRS generation metadata
        """
        p = self.params

        # Extract key parameters
        N_ap = p.get("n_ports", self.default_params.get("n_ports", 1))
        p_i = p.get("tx_port", p.get("p_tx_port", self.default_params.get("p_tx_port", 1000)))
        N_symb_SRS = p.get("n_syms", self.default_params.get("n_syms", 1))
        l_offset = p.get("start_pos", p.get("sym_offset", self.default_params.get("sym_offset", 0)))
        K_TC = p.get("comb", self.default_params.get("comb", 2))
        k_bar_TC = p.get("comb_offset", self.default_params.get("comb_offset", 0))
        n_cs = p.get("comb_cs", self.default_params.get("comb_cs", 0))
        C_SRS = p.get("fh_c", p.get("c_srs", self.default_params.get("c_srs", 0)))
        B_SRS = p.get("fh_b", p.get("b_srs", self.default_params.get("b_srs", 0)))
        n_RRC = p.get("fd_pos", p.get("n_rrc", self.default_params.get("n_rrc", 0)))
        n_shift = p.get("fd_shift", p.get("fd_shift", 0))
        b_hop = p.get("fh_b_hop", p.get("b_hop", 0))
        bwp_offset = p.get("bwp_offset", self.default_params.get("bwp_offset", 0))
        N_RB = p.get("prb_num", self.default_params.get("prb_num", 52))
        n_id = p.get("n_id", self.default_params.get("n_id", 0))
        n_slots = p.get("n_slots", self.default_params.get("n_slots", 10))
        T_SRS = p.get("periodicity", p.get("T_srs", self.default_params.get("T_srs", 1)))
        T_offset = p.get("offset", p.get("T_offset", self.default_params.get("T_offset", 0)))
        seq_hopping_mode = p.get("seq_hopping_mode", "neither")

        # Read SRS table or use default if not available
        m_row = self.m_srs_table[C_SRS]
        N_row = self.N_srs_table[C_SRS]

        # Initialize resource grid
        n_sc = N_RB * 12
        re_grid = np.zeros((n_sc, n_slots * 14), dtype=np.complex128)

        # Calculate SRS symbol positions in slot
        l0 = 14 - 1 - l_offset
        l_sym = np.array([l0 - i for i in range(N_symb_SRS)])

        # Determine maximum shifts based on comb size
        max_shifts = 8 if K_TC == 2 else 12

        # Calculate cyclic shift
        shift_idx = (n_cs + (max_shifts * (p_i - 1000) / N_ap)) % max_shifts
        alpha = 2 * np.pi * shift_idx / max_shifts

        # Handle group/sequence hopping
        if seq_hopping_mode == "neither":
            f_gh = 0
            v = 0
        else:
            # For future implementation of group/sequence hopping
            f_gh = 0  # Placeholder
            v = 0     # Placeholder

        # Calculate sequence group
        u = (f_gh + n_id) % 30

        # Determine slots with SRS transmission based on periodicity
        srs_slots = [slot_idx for slot_idx in range(n_slots) if (slot_idx - T_offset) % T_SRS == 0]

        # For tracking allocations
        srs_allocation = {}

        for slot_idx in srs_slots:
            srs_allocation[slot_idx] = {"symbols": [], "freq_positions": []}

            # Calculate SRS occasion counter
            n_slot_SRS = (slot_idx - T_offset) // T_SRS

            for n_SRS in range(N_symb_SRS):
                sym_idx = slot_idx * 14 + l_sym[n_SRS]

                # Determine the bandwidth configuration
                B_SRS_max = min(B_SRS, len(m_row) - 1)
                m_SRS = m_row[B_SRS_max]
                N_i = N_row[B_SRS_max]
                M_SRS_b = m_SRS // N_i

                # Calculate frequency position index n_b
                if b_hop >= B_SRS:
                    # No frequency hopping
                    n_b = n_RRC
                else:
                    # Apply frequency hopping according to TS 38.211 Section 6.4.1.4.3
                    n_b = n_RRC % N_i

                    if b_hop == 1:  # Type 1 hopping
                        n_b = (n_slot_SRS % N_i)
                    elif b_hop == 2:  # Type 2 hopping
                        n_b = ((n_slot_SRS // 2) + (n_slot_SRS % 2) * (N_i // 2)) % N_i

                    # Apply intra-slot hopping if multiple symbols
                    if N_symb_SRS > 1:
                        F_b = self.read_srs_F_b(n_SRS, N_i, b_hop)
                        n_b = (n_b + F_b) % N_i

                # Calculate k0 - starting subcarrier position
                k_0 = bwp_offset + n_b * M_SRS_b + n_shift

                # Track for verification
                srs_allocation[slot_idx]["symbols"].append(int(l_sym[n_SRS]))
                srs_allocation[slot_idx]["freq_positions"].append(int(k_0))

                # Calculate sequence length
                seq_length = M_SRS_b * 12 // K_TC

                # Generate the appropriate sequence
                if seq_length <= 36:
                    # For short sequences, use standard low PAPR sequence
                    srs_seq = self._generate_low_papr_sequence(u, v, alpha, seq_length)
                else:
                    srs_seq = self._generate_ZC_sequence(u, v, alpha, seq_length)
                pd.DataFrame({'real': srs_seq.real.flatten(), 'imag': srs_seq.imag.flatten()}).to_excel("srs_seq.xlsx", index=False)

                # Map sequence to RE grid using comb pattern
                seq_idx = 0
                mapped_positions = []

                # Determine k_TC (subcarrier offset within a comb)
                if N_ap == 4 and (p_i == 1001 or p_i == 1003) and ((K_TC == 2 and n_cs > 3) or (K_TC == 4 and n_cs > 5)):
                    k_TC = (k_bar_TC + K_TC // 2) % K_TC
                else:
                    k_TC = k_bar_TC

                # Calculate base subcarrier position
                k0_bar = n_shift * 12 + k_TC

                # Map sequence to grid
                for k in range(int(k_0) * 12, int(k_0 + M_SRS_b) * 12, K_TC):
                    if k + k_TC < n_sc and seq_idx < len(srs_seq):
                        re_grid[k + k_TC, sym_idx] = srs_seq[seq_idx]
                        mapped_positions.append(k + k_TC)
                        seq_idx += 1

                # Store mapped positions for verification
                srs_allocation[slot_idx]["mapped_sc"] = mapped_positions

        # Generate time-domain signal
        td_signal = self._generate_periodic_signal(re_grid, n_slots, p['T_srs'], p['T_offset'], l_sym)

        # Prepare metadata
        mu = p.get("mu", 0)
        buffer_length_ms = n_slots * (1.0 / (2 ** mu))

        # Create list of all SRS subcarrier indices
        all_sc_indices = []
        for slot in srs_allocation:
            if "mapped_sc" in srs_allocation[slot]:
                all_sc_indices.extend(srs_allocation[slot]["mapped_sc"])
        all_sc_indices = sorted(list(set(all_sc_indices)))

        # Compile SRS information
        srs_info = {
            "C_SRS": C_SRS,
            "B_SRS": B_SRS,
            "m_SRS": m_row[B_SRS_max] if 'B_SRS_max' in locals() else None,
            "N_i": N_row[B_SRS_max] if 'B_SRS_max' in locals() else None,
            "K_TC": K_TC,
            "k_bar_TC": k_bar_TC,
            "N_symb_SRS": N_symb_SRS,
            "n_slots": n_slots,
            "buffer_length_ms": buffer_length_ms,
            "T_SRS": T_SRS,
            "T_offset": T_offset,
            "srs_slots": srs_slots,
            "fh_type": b_hop,
            "fh_enabled": b_hop > 0,
            "allocation": srs_allocation,
            "l_sym": l_sym.tolist(),
            "srs_sc_indices": all_sc_indices,
            "n_shift": n_shift,
            "u": u,
            "v": v,
            "alpha": alpha,
            "cyclic_shift_idx": shift_idx
        }

        return re_grid, td_signal, srs_info

    def read_srs_F_b(self, n_SRS, N_i, b_hop):
        """
        Calculate the frequency hopping pattern F_b(n_SRS) for SRS as defined in 3GPP TS 38.211 Section 6.4.1.4.3

        Args:
            n_SRS: Symbol index for SRS (0-based)
            N_i: Number of bandwidth parts from the table for the selected B_SRS
            b_hop: Frequency hopping parameter (0-3)

        Returns:
            F_b: Frequency hopping offset value
        """
        # If b_hop is 0, no frequency hopping
        if b_hop == 0:
            return 0

        # Get the SRS configuration parameters
        C_SRS = self.params.get('fh_c', 0)
        B_SRS = self.params.get('fh_b', 0)
        B_SRS_max = min(B_SRS, 3)

        # Calculate F_b according to 38.211 Section 6.4.1.4.3
        if b_hop > 0 and b_hop <= B_SRS_max:
            # Calculate N_i^b_hop (N_i to the power of b_hop)
            N_i_power_b_hop = N_i ** b_hop

            # Calculate N_i^(B_SRS - b_hop)
            N_i_power_diff = N_i ** (B_SRS_max - b_hop)

            # Calculate F_b(n_SRS) = (n_SRS mod N_i^b_hop) * N_i^(B_SRS - b_hop)
            F_b = (n_SRS % N_i_power_b_hop) * N_i_power_diff

            return F_b

        # Alternative implementation from the spec that handles special cases
        else:
            # Table of predefined patterns for various configurations
            if B_SRS_max == 1:
                if b_hop == 1:
                    return 0 if n_SRS % 2 == 0 else N_i
            elif B_SRS_max == 2:
                if b_hop == 1:
                    return 0 if n_SRS % 2 == 0 else N_i**1
                elif b_hop == 2:
                    return (n_SRS % 4) * N_i**0
            elif B_SRS_max == 3:
                if b_hop == 1:
                    return 0 if n_SRS % 2 == 0 else N_i**2
                elif b_hop == 2:
                    return (n_SRS % 4) // 2 * N_i**1
                elif b_hop == 3:
                    return (n_SRS % 8) * N_i**0

            # Default case - no hopping
            return 0

    def _generate_low_papr_sequence(self, u, v, alpha, m):
        """
        Generate low PAPR sequence

        Parameters:
        u : int or array-like
            Group number
        v : int or array-like
            Not used in this implementation
        alpha : array-like
            Alpha values
        m : int
            Sequence length

        Returns:
        seq : ndarray
            Low PAPR sequence
        """
        # Reshape alpha to (1, N)
        alpha = np.reshape(alpha, (1, -1))

        # Get base sequence
        base_seq = self._get_base_seq(u, m)

        # Generate low-PAPR sequence from the base sequence
        n = np.arange(m).reshape(-1, 1)
        seq = np.exp(1j * n * alpha) * np.tile(base_seq, (1, alpha.shape[1]))

        return seq

    def _get_base_seq(self,u, m):
        """
        Get base sequence based on the group number u and length m

        Parameters:
        u : int or array-like
            Group number(s)
        m : int
            Sequence length

        Returns:
        base_seq : ndarray
            Base sequence
        """
        # Define phi tables based on sequence length m
        if m == 6:
            phi_table = np.array([
                [-3, -1, 3, 3, -1, -3],
                [-3, 3, -1, -1, 3, -3],
                [-3, -3, -3, 3, 1, -3],
                [1, 1, 1, 3, -1, -3],
                [1, 1, 1, -3, -1, 3],
                [-3, 1, -1, -3, -3, -3],
                [-3, 1, 3, -3, -3, -3],
                [-3, -1, 1, -3, 1, -1],
                [-3, -1, -3, 1, -3, -3],
                [-3, -3, 1, -3, 3, -3],
                [-3, 1, 3, 1, -3, -3],
                [-3, -1, -3, 1, 1, -3],
                [1, 1, 3, -1, -3, 3],
                [1, 1, 3, 3, -1, 3],
                [1, 1, 1, -3, 3, -1],
                [1, 1, 1, -1, 3, -3],
                [-3, -1, -1, -1, 3, -1],
                [-3, -3, -1, 1, -1, -3],
                [-3, -3, -3, 1, -3, -1],
                [-3, 1, 1, -3, -1, -3],
                [-3, 3, -3, 1, 1, -3],
                [-3, 1, -3, -3, -3, -1],
                [1, 1, -3, 3, 1, 3],
                [1, 1, -3, -3, 1, -3],
                [1, 1, 3, -1, 3, 3],
                [1, 1, -3, 1, 3, 3],
                [1, 1, -1, -1, 3, -1],
                [1, 1, -1, 3, -1, -1],
                [1, 1, -1, 3, -3, -1],
                [1, 1, -3, 1, -1, -1]
            ])
        elif m == 12:
            phi_table = np.array([
                [-3, 1, -3, -3, -3, 3, -3, -1, 1, 1, 1, -3],
                [-3, 3, 1, -3, 1, 3, -1, -1, 1, 3, 3, 3],
                [-3, 3, 3, 1, -3, 3, -1, 1, 3, -3, 3, -3],
                [-3, -3, -1, 3, 3, 3, -3, 3, -3, 1, -1, -3],
                [-3, -1, -1, 1, 3, 1, 1, -1, 1, -1, -3, 1],
                [-3, -3, 3, 1, -3, -3, -3, -1, 3, -1, 1, 3],
                [1, -1, 3, -1, -1, -1, -3, -1, 1, 1, 1, -3],
                [-1, -3, 3, -1, -3, -3, -3, -1, 1, -1, 1, -3],
                [-3, -1, 3, 1, -3, -1, -3, 3, 1, 3, 3, 1],
                [-3, -1, -1, -3, -3, -1, -3, 3, 1, 3, -1, -3],
                [-3, 3, -3, 3, 3, -3, -1, -1, 3, 3, 1, -3],
                [-3, -1, -3, -1, -1, -3, 3, 3, -1, -1, 1, -3],
                [-3, -1, 3, -3, -3, -1, -3, 1, -1, -3, 3, 3],
                [-3, 1, -1, -1, 3, 3, -3, -1, -1, -3, -1, -3],
                [1, 3, -3, 1, 3, 3, 3, 1, -1, 1, -1, 3],
                [-3, 1, 3, -1, -1, -3, -3, -1, -1, 3, 1, -3],
                [-1, -1, -1, -1, 1, -3, -1, 3, 3, -1, -3, 1],
                [-1, 1, 1, -1, 1, 3, 3, -1, -1, -3, 1, -3],
                [-3, 1, 3, 3, -1, -1, -3, 3, 3, -3, 3, -3],
                [-3, -3, 3, -3, -1, 3, 3, 3, -1, -3, 1, -3],
                [3, 1, 3, 1, 3, -3, -1, 1, 3, 1, -1, -3],
                [-3, 3, 1, 3, -3, 1, 1, 1, 1, 3, -3, 3],
                [-3, 3, 3, 3, -1, -3, -3, -1, -3, 1, 3, -3],
                [3, -1, -3, 3, -3, -1, 3, 3, 3, -3, -1, -3],
                [-3, -1, 1, -3, 1, 3, 3, 3, -1, -3, 3, 3],
                [-3, 3, 1, -1, 3, 3, -3, 1, -1, 1, -1, 1],
                [-1, 1, 3, -3, 1, -1, 1, -1, -1, -3, 1, -1],
                [-3, -3, 3, 3, 3, -3, -1, 1, -3, 3, 1, -3],
                [1, -1, 3, 1, 1, -1, -1, -1, 1, 3, -3, 1],
                [-3, 3, -3, 3, -3, -3, 3, -1, -1, 1, 3, -3]
            ])
        elif m == 18:
            phi_table = np.array([
                [-1, 3, -1, -3, 3, 1, -3, -1, 3, -3, -1, -1, 1, 1, 1, -1, -1, -1],
                [3, -3, 3, -1, 1, 3, -3, -1, -3, -3, -1, -3, 3, 1, -1, 3, -3, 3],
                [-3, 3, 1, -1, -1, 3, -3, -1, 1, 1, 1, 1, 1, -1, 3, -1, -3, -1],
                [-3, -3, 3, 3, 3, 1, -3, 1, 3, 3, 1, -3, -3, 3, -1, -3, -1, 1],
                [1, 1, -1, -1, -3, -1, 1, -3, -3, -3, 1, -3, -1, -1, 1, -1, 3, 1],
                [3, -3, 1, 1, 3, -1, 1, -1, -1, -3, 1, 1, -1, 3, 3, -3, 3, -1],
                [-3, 3, -1, 1, 3, 1, -3, -1, 1, 1, -3, 1, 3, 3, -1, -3, -3, -3],
                [1, 1, -3, 3, 3, 1, 3, -3, 3, -1, 1, 1, -1, 1, -3, -3, -1, 3],
                [-3, 1, -3, -3, 1, -3, -3, 3, 1, -3, -1, -3, -3, -3, -1, 1, 1, 3],
                [3, -1, 3, 1, -3, -3, -1, 1, -3, -3, 3, 3, 3, 1, 3, -3, 3, -3],
                [-3, -3, -3, 1, -3, 3, 1, 1, 3, -3, -3, 1, 3, -1, 3, -3, -3, 3],
                [-3, -3, 3, 3, 3, -1, -1, -3, -1, -1, -1, 3, 1, -3, -3, -1, 3, -1],
                [-3, -1, -3, -3, 1, 1, -1, -3, -1, -3, -1, -1, 3, 3, -1, 3, 1, 3],
                [1, 1, -3, -3, -3, -3, 1, 3, -3, 3, 3, 1, -3, -1, 3, -1, -3, 1],
                [-3, 3, -1, -3, -1, -3, 1, 1, -3, -3, -1, -1, 3, -3, 1, 3, 1, 1],
                [3, 1, -3, 1, -3, 3, 3, -1, -3, -3, -1, -3, -3, 3, -3, -1, 1, 3],
                [-3, -1, -3, -1, -3, 1, 3, -3, -1, 3, 3, 3, 1, -1, -3, 3, -1, -3],
                [-3, -1, 3, 3, -1, 3, -1, -3, -1, 1, -1, -3, -1, -1, -1, 3, 3, 1],
                [-3, 1, -3, -1, -1, 3, 1, -3, -3, -3, -1, -3, -3, 1, 1, 1, -1, -1],
                [3, 3, 3, -3, -1, -3, -1, 3, -1, 1, -1, -3, 1, -3, -3, -1, 3, 3],
                [-3, 1, 1, -3, 1, 1, 3, -3, -1, -3, -1, 3, -3, 3, -1, -1, -1, -3],
                [1, -3, -1, -3, 3, 3, -1, -3, 1, -3, -3, -1, -3, -1, 1, 3, 3, 3],
                [-3, -3, 1, -1, -1, 1, 1, -3, -1, 3, 3, 3, 3, -1, 3, 1, 3, 1],
                [3, -1, -3, 1, -3, -3, -3, 3, 3, -1, 1, -3, -1, 3, 1, 1, 3, 3],
                [3, -1, -1, 1, -3, -1, -3, -1, -3, -3, -1, -3, 1, 1, 1, -3, -3, 3],
                [-3, -3, 1, -3, 3, 3, 3, -1, 3, 1, 1, -3, -3, -3, 3, -3, -1, -1],
                [-3, -1, -1, -3, 1, -3, 3, -1, -1, -3, 3, 3, -3, -1, 3, -1, -1, -1],
                [-3, -3, 3, 3, -3, 1, 3, -1, -3, 1, -1, -3, 3, -3, -1, -1, -1, 3],
                [-1, -3, 1, -3, -3, -3, 1, 1, 3, 3, -3, 3, 3, -3, -1, 3, -3, 1],
                [-3, 3, 1, -1, -1, -1, -1, 1, -1, 3, 3, -3, -1, 1, 3, -1, 3, -1]
            ])
        elif m == 24:
            phi_table = np.array([
                [-1, -3, 3, -1, 3, 1, 3, -1, 1, -3, -1, -3, -1, 1, 3, -3, -1, -3, 3, 3, 3, -3, -3, -3],
                [-1, -3, 3, 1, 1, -3, 1, -3, -3, 1, -3, -1, -1, 3, -3, 3, 3, 3, -3, 1, 3, 3, -3, -3],
                [-1, -3, -3, 1, -1, -1, -3, 1, 3, -1, -3, -1, -1, -3, 1, 1, 3, 1, -3, -1, -1, 3, -3, -3],
                [1, -3, 3, -1, -3, -1, 3, 3, 1, -1, 1, 1, 3, -3, -1, -3, -3, -3, -1, 3, -3, -1, -3, -3],
                [-1, 3, -3, -3, -1, 3, -1, -1, 1, 3, 1, 3, -1, -1, -3, 1, 3, 1, -1, -3, 1, -1, -3, -3],
                [-3, -1, 1, -3, -3, 1, 1, -3, 3, -1, -1, -3, 1, 3, 1, -1, -3, -1, -3, 1, -3, -3, -3, -3],
                [-3, 3, 1, 3, -1, 1, -3, 1, -3, 1, -1, -3, -1, -3, -3, -3, -3, -1, -1, -1, 1, 1, -3, -3],
                [-3, 1, 3, -1, 1, -1, 3, -3, 3, -1, -3, -1, -3, 3, -1, -1, -1, -3, -1, -1, -3, 3, 3, -3],
                [-3, 1, -3, 3, -1, -1, -1, -3, 3, 1, -1, -3, -1, 1, 3, -1, 1, -1, 1, -3, -3, -3, -3, -3],
                [1, 1, -1, -3, -1, 1, 1, -3, 1, -1, 1, -3, 3, -3, -3, 3, -1, -3, 1, 3, -3, 1, -3, -3],
                [-3, -3, -3, -1, 3, -3, 3, 1, 3, 1, -3, -1, -1, -3, 1, 1, 3, 1, -1, -3, 3, 1, 3, -3],
                [-3, 3, -1, 3, 1, -1, -1, -1, 3, 3, 1, 1, 1, 3, 3, 1, -3, -3, -1, 1, -3, 1, 3, -3],
                [3, -3, 3, -1, -3, 1, 3, 1, -1, -1, -3, -1, 3, -3, 3, -1, -1, 3, 3, -3, -3, 3, -3, -3],
                [-3, 3, -1, 3, -1, 3, 3, 1, 1, -3, 1, 3, -3, 3, -3, -3, -1, 1, 3, -3, -1, -1, -3, -3],
                [-3, 1, -3, -1, -1, 3, 1, 3, -3, 1, -1, 3, 3, -1, -3, 3, -3, -1, -1, -3, -3, -3, 3, -3],
                [-3, -1, -1, -3, 1, -3, -3, -1, -1, 3, -1, 1, -1, 3, 1, -3, -1, 3, 1, 1, -1, -1, -3, -3],
                [-3, -3, 1, -1, 3, 3, -3, -1, 1, -1, -1, 1, 1, -1, -1, 3, -3, 1, -3, 1, -1, -1, -1, -3],
                [3, -1, 3, -1, 1, -3, 1, 1, -3, -3, 3, -3, -1, -1, -1, -1, -1, -3, -3, -1, 1, 1, -3, -3],
                [-3, 1, -3, 1, -3, -3, 1, -3, 1, -3, -3, -3, -3, -3, 1, -3, -3, 1, 1, -3, 1, 1, -3, -3],
                [-3, -3, 3, 3, 1, -1, -1, -1, 1, -3, -1, 1, -1, 3, -3, -1, -3, -1, -1, 1, -3, 3, -1, -3],
                [-3, -3, -1, -1, -1, -3, 1, -1, -3, -1, 3, -3, 1, -3, 3, -3, 3, 3, 1, -1, -1, 1, -3, -3],
                [3, -1, 1, -1, 3, -3, 1, 1, 3, -1, -3, 3, 1, -3, 3, -1, -1, -1, -1, 1, -3, -3, -3, -3],
                [-3, 1, -3, 3, -3, 1, -3, 3, 1, -1, -3, -1, -3, -3, -3, -3, 1, 3, -1, 1, 3, 3, 3, -3],
                [-3, -1, 1, -3, -1, -1, 1, 1, 1, 3, 3, -1, 1, -1, 1, -1, -1, -3, -3, -3, 3, 1, -1, -3],
                [-3, 3, -1, -3, -1, -1, -1, 3, -1, -1, 3, -3, -1, 3, -3, 3, -3, -1, 3, 1, 1, -1, -3, -3],
                [-3, 1, -1, -3, -3, -1, 1, -3, -1, -3, 1, 1, -1, 1, 1, 3, 3, 3, -1, 1, -1, 1, -1, -3],
                [-1, 3, -1, -1, 3, 3, -1, -1, -1, 3, -1, -3, 1, 3, 1, 1, -3, -3, -3, -1, -3, -1, -3, -3],
                [3, -3, -3, -1, 3, 3, -3, -1, 3, 1, 1, 1, 3, -1, 3, -3, -1, 3, -1, 3, 1, -1, -3, -3],
                [-3, 1, -3, 1, -3, 1, 1, 3, 1, -3, -3, -1, 1, 3, -1, -3, 3, 1, -1, -3, -3, -3, -3, -3],
                [3, -3, -1, 1, 3, -1, -1, -3, -1, 3, -1, -3, -1, -3, 3, -1, 3, 1, 1, -3, 3, -3, -3, -3]
            ])
        else:
            raise ValueError("ERROR: Sequence length is not supported")

        # Initialize the base sequence
        base_seq = np.zeros((m, len(u) if hasattr(u, "__len__") else 1), dtype=complex)

        # Phase values based on group number u from the phi table
        for i in range(len(u) if hasattr(u, "__len__") else 1):
            idx = u[i] if hasattr(u, "__len__") else u
            phi = phi_table[idx, :].reshape(-1, 1)
            base_seq[:, i] = np.exp(1j * phi * np.pi / 4).flatten()

        return base_seq

    def _generate_ZC_sequence(self, u, v, alpha, m_zc):
        """
        Generate SRS based on Zadoff-Chu sequence

        Args:
            u: Base sequence group number (0-29)
            v: Base sequence number (0-1)
            alpha: Phase rotation value
            m_zc: Length of sequence

        Returns:
            r_seq: Low PAPR sequence
        """
        # Find largest prime smaller than m_zc
        prime_list = list(primes.upto(m_zc))
        n_zc = prime_list[-1] if prime_list else 1

        # Determine q and q_bar
        q_bar = n_zc * (u + 1) / 31
        q = int(np.floor(q_bar + 0.5) + v * (-1)**np.floor(2 * q_bar))

        # Compute x_q (base Zadoff-Chu sequence)
        m = np.arange(n_zc)
        x_q = np.exp((-1) * 1j * np.pi * q / n_zc * (m * (m + 1)))

        # Extend x_q to obtain r_seq
        r_seq = np.zeros(m_zc, dtype=complex)
        r_seq[:n_zc] = x_q

        if m_zc > n_zc:
            r_seq[n_zc:m_zc] = x_q[:(m_zc - n_zc)]

        # Apply phase rotation for low PAPR
        n = np.arange(m_zc)
        r_seq = np.exp(1j * n * alpha) * r_seq

        return r_seq

    def _generate_full_slot_signal(self, re_grid, n_slots, l_sym=None):
        """
        Generate time-domain signal from the resource grid according to 3GPP TS 38.211 Section 5.3.1.
        This improved implementation handles proper slot and symbol positions with correct CP insertion.

        Args:
            re_grid: Resource element grid [n_sc × n_symbols]
            n_slots: Number of slots
            l_sym: Optional symbol positions to use within each slot

        Returns:
            Time domain signal
        """
        p = self.params
        n_sc, total_symbols = re_grid.shape

        # Get numerology and CP type from parameters
        mu = p.get('mu', 0)
        cp_type = p.get('cp_type', 'normal')
        fft_size = p.get('fft_size', int(2**np.ceil(np.log2(n_sc))))

        # Calculate symbols per slot (always 14 for normal CP, 12 for extended CP)
        symbols_per_slot = 14 if cp_type == 'normal' else 12

        # If l_sym is provided, it specifies which symbol positions within each slot to use
        # If not provided, assume symbols are sequential starting from the beginning of each slot
        if l_sym is None:
            # Default: use all symbols in slot
            l_sym = list(range(min(symbols_per_slot, total_symbols // n_slots)))

        # Ensure l_sym is sorted
        l_sym = sorted(l_sym)

        # Calculate CP lengths according to 3GPP TS 38.211 Section 5.3.1
        if cp_type == 'normal':
            # First (0th) and middle (7th) symbols in each slot have longer CP
            cp_length_long = int((160 * (2**(-mu))) * (fft_size / 2048))
            cp_length_short = int((144 * (2**(-mu))) * (fft_size / 2048))
        else:  # extended CP
            # All symbols have the same CP length for extended CP
            cp_length_long = cp_length_short = int((512 * (2**(-mu))) * (fft_size / 2048))

        # Calculate symbol lengths
        sym_len_long = fft_size + cp_length_long
        sym_len_short = fft_size + cp_length_short

        # Calculate total length of one slot
        slot_length = 0
        for sym_idx in range(symbols_per_slot):
            uses_long_cp = (sym_idx == 0 or sym_idx == 7)
            slot_length += sym_len_long if uses_long_cp else sym_len_short

        # Total signal length for all slots
        total_length = n_slots * slot_length
        td_signal = np.zeros(total_length, dtype=np.complex128)

        # Calculate positions of each symbol in re_grid
        symbols_per_output_slot = len(l_sym)
        total_output_symbols = n_slots * symbols_per_output_slot

        # Check if we have enough symbols in re_grid
        if total_output_symbols > total_symbols:
            # Adjust the actual number of slots we'll generate
            n_slots = total_symbols // symbols_per_output_slot
            if n_slots == 0:
                n_slots = 1
                # Warn if insufficient symbols
                warnings.warn(f"re_grid has only {total_symbols} symbols, fewer than requested for {n_slots} slots with {len(l_sym)} symbols each.")

        # Process each slot
        current_pos = 0
        grid_symbol_idx = 0

        for slot_idx in range(n_slots):
            # Calculate slot start position
            slot_start = slot_idx * slot_length

            # For each symbol position specified in l_sym
            for sym_rel_idx, sym_pos in enumerate(l_sym):
                # Ensure sym_pos is valid
                if sym_pos < 0 or sym_pos >= symbols_per_slot:
                    warnings.warn(f"Symbol position {sym_pos} is out of range (0-{symbols_per_slot-1}). Skipping.")
                    continue

                # Check if we've exhausted symbols in re_grid
                if grid_symbol_idx >= total_symbols:
                    break

                # Calculate absolute position in the slot
                sym_offset = 0
                for prev_sym in range(sym_pos):
                    uses_long_cp = (prev_sym == 0 or prev_sym == 7)
                    sym_offset += sym_len_long if uses_long_cp else sym_len_short

                # Get the appropriate frequency domain symbol from re_grid
                symbol_fd = np.zeros(fft_size, dtype=np.complex128)

                # Place the data in the center (DC in the middle)
                dc_idx = fft_size // 2
                start_idx = dc_idx - n_sc // 2
                symbol_fd[start_idx:start_idx + n_sc] = re_grid[:, grid_symbol_idx]

                # IFFT shift to standard format (DC at index 0), then IFFT, and normalize
                symbol_fd_shifted = np.fft.ifftshift(symbol_fd)
                symbol_td = np.fft.ifft(symbol_fd_shifted) * np.sqrt(fft_size)

                # Determine CP length for this symbol
                uses_long_cp = (sym_pos == 0 or sym_pos == 7)
                cp_length = cp_length_long if uses_long_cp else cp_length_short

                # Create CP by copying the end of the time domain symbol
                cp_samples = symbol_td[-cp_length:]
                symbol_with_cp = np.concatenate([cp_samples, symbol_td])

                # Calculate position in output buffer and place the symbol
                position = slot_start + sym_offset
                td_signal[position:position + len(symbol_with_cp)] = symbol_with_cp

                # Move to next symbol in the grid
                grid_symbol_idx += 1

        # Return the time domain signal
        return td_signal

    def _generate_periodic_signal(self, re_grid, n_slots, periodicity=1, slot_offset=0, l_sym=None):
        """
        Generate time-domain signal with proper periodicity and slot offset handling.

        Args:
            re_grid: Resource element grid [n_sc × n_symbols]
            n_slots: Total number of slots to generate
            periodicity: SRS periodicity in slots (default: 1 = every slot)
            slot_offset: Slot offset for SRS positioning (default: 0)
            l_sym: Optional symbol positions to use within each slot

        Returns:
            Time domain signal
        """
        p = self.params
        n_sc, total_symbols = re_grid.shape

        # Calculate symbols per slot
        symbols_per_slot = 14 if p.get('cp_type', 'normal') == 'normal' else 12

        # Calculate CP lengths
        mu = p.get('mu', 0)
        fft_size = p.get('fft_size', int(2**np.ceil(np.log2(n_sc))))

        if p.get('cp_type', 'normal') == 'normal':
            cp_length_long = int((160 * (2**(-mu))) * (fft_size / 2048))
            cp_length_short = int((144 * (2**(-mu))) * (fft_size / 2048))
        else:  # extended CP
            cp_length_long = cp_length_short = int((512 * (2**(-mu))) * (fft_size / 2048))

        # Calculate symbol lengths
        sym_len_long = fft_size + cp_length_long
        sym_len_short = fft_size + cp_length_short

        # Calculate total length of one slot
        slot_length = 0
        for sym_idx in range(symbols_per_slot):
            uses_long_cp = (sym_idx == 0 or sym_idx == 7)
            slot_length += sym_len_long if uses_long_cp else sym_len_short

        # Total signal length for all slots
        total_length = n_slots * slot_length
        td_signal = np.zeros(total_length, dtype=np.complex128)

        # If l_sym is provided, it specifies which symbol positions within each slot to use
        if l_sym is None:
            # Default: use SRS symbols based on configuration
            l_offset = p.get('sym_offset', 0)
            n_symbols = p.get('n_syms', 1)
            l0 = symbols_per_slot - 1 - l_offset
            l_sym = [l0 - i for i in range(n_symbols)]
            # Ensure l_sym is sorted
            l_sym = sorted(l_sym)

        # Print diagnostic information
        print(f"Processing RE grid of shape {re_grid.shape}")
        print(f"Symbols per slot: {symbols_per_slot}, Total slots: {n_slots}")
        print(f"Symbol positions to use: {l_sym}")
        print(f"Periodicity: {periodicity}, Slot offset: {slot_offset}")

        # Calculate active slots (those that will contain signals based on periodicity)
        active_slots = [(i, True) for i in range(n_slots)
                      if (i - slot_offset) % periodicity == 0 and i >= slot_offset]

        print(f"Active slots: {[s[0] for s in active_slots if s[1]]}")

        # Precalculate all symbol offsets within a slot
        sym_offsets = {}
        for sym_pos in range(symbols_per_slot):
            offset = 0
            for prev_sym in range(sym_pos):
                uses_long_cp = (prev_sym == 0 or prev_sym == 7)
                offset += sym_len_long if uses_long_cp else sym_len_short
            sym_offsets[sym_pos] = offset

        # For each slot in the frame
        for slot_idx in range(n_slots):
            # Check if this slot should be active based on periodicity
            is_active_slot = slot_idx >= slot_offset and (slot_idx - slot_offset) % periodicity == 0

            if not is_active_slot:
                continue

            print(f"Processing active slot {slot_idx}")

            # Calculate slot start position in the output signal
            slot_start = slot_idx * slot_length

            # For each symbol position specified in l_sym
            for sym_idx, sym_pos in enumerate(l_sym):
                # Skip invalid symbol positions
                if sym_pos < 0 or sym_pos >= symbols_per_slot:
                    print(f"Skipping invalid symbol position {sym_pos}")
                    continue

                # Check if we still have symbols left in the RE grid
                if sym_pos >= total_symbols:
                    print(f"Ran out of symbols in RE grid at index {sym_pos}")
                    break

                print(f"  Processing symbol at position {sym_pos} (grid index {sym_pos})")

                # Get the symbol offset within the slot
                sym_offset = sym_offsets[sym_pos]

                # Extract the frequency domain data for this symbol
                symbol_data = re_grid[:, ((slot_idx * symbols_per_slot) + sym_pos)]
                # Create padded frequency domain vector (centered around DC)
                padded_fd = np.zeros(fft_size, dtype=np.complex128)

                # Place RE grid data in the center of the FFT input
                # For odd n_sc, position carefully around DC
                start_idx = (fft_size - n_sc) // 2
                padded_fd[start_idx:start_idx + n_sc] = symbol_data

                # Shift for IFFT (moving DC to index 0)
                shifted_fd = np.fft.ifftshift(padded_fd)

                # IFFT to get time domain and normalize
                time_domain = np.fft.ifft(shifted_fd) * np.sqrt(fft_size)

                # Determine CP length for this symbol
                uses_long_cp = (sym_pos == 0 or sym_pos == 7)
                cp_length = cp_length_long if uses_long_cp else cp_length_short

                # Create CP by copying the end of the time domain symbol
                cp = time_domain[-cp_length:]

                # Create complete symbol with CP
                symbol_with_cp = np.concatenate([cp, time_domain])

                # Calculate position in output signal
                position = slot_start + sym_offset
                end_position = position + len(symbol_with_cp)

                # Ensure we don't exceed buffer bounds
                if end_position <= len(td_signal):
                    # Place the symbol in the output signal
                    td_signal[position:end_position] = symbol_with_cp

                    # Log placement
                    print(f"    Placed symbol at positions {position}:{end_position}")
                    print(f"    Symbol energy: {np.sum(np.abs(symbol_with_cp)**2)}")
                else:
                    print(f"    WARNING: Symbol placement would exceed buffer bounds")


        # Remove debug prints in production
        print(f"Final signal energy: {np.sum(np.abs(td_signal)**2)}")

        return td_signal

    def export_binary(self, data, filename, scale=2**12, format='int16', endian='big', metadata=None):
        """
        Export the complete SRS signal (e.g., td_signal) for 10ms or 20ms duration
        with metadata as a separate JSON file or embedded header.

        Args:
            data (ndarray): Complex 1D array (e.g., time-domain signal `td_signal`) to export.
            filename (str): Output filename for binary file.
            scale (float): Scaling factor for I/Q samples to boost amplitude.
            format (str): Output binary data format ('int16', 'int8', 'float32').
            endian (str): Endianness for output binary file ('little' or 'big').
            metadata (dict, optional): Dictionary containing metadata to associate with the binary.

        Returns:
            str: Path to the exported binary file.
        """
        # Ensure data is a 1D array
        if data.ndim != 1:
            raise ValueError("Data must be a 1D array. Export is for time-domain signals only!")

        if metadata is None:
            metadata = {}

        # Export dat in matlab format
        savemat('td_signal.mat', {'array_name': data})

        # Scale signal and split into real (I) and imaginary (Q) parts
        i_samples = np.real(data) * scale
        q_samples = np.imag(data) * scale

        pd.DataFrame({'Real':i_samples, 'Imaginary': q_samples}).to_excel("td_signal.xlsx", index=False)

        # Interleave I and Q samples
        iq_samples = np.zeros(2 * len(data), dtype=np.float32)
        iq_samples[0::2] = i_samples
        iq_samples[1::2] = q_samples

        # Convert to specified format
        if format == 'int16':
            iq_samples = np.clip(iq_samples, -32768, 32767).astype(np.int16)
            dtype_size = 2  # 2 bytes per sample (int16)
            struct_fmt = '>h' if endian == 'big' else '<h'  # '>' for big-endian, '<' for little-endian
        elif format == 'int8':
            iq_samples = np.clip(iq_samples / 256, -128, 127).astype(np.int8)
            dtype_size = 1  # 1 byte per sample (int8)
            struct_fmt = '>b' if endian == 'big' else '<b'  # '>' for big-endian, '<' for little-endian
        elif format == 'float32':
            iq_samples = iq_samples.astype(np.float32)
            dtype_size = 4  # 4 bytes per sample (float32)
            struct_fmt = '>f' if endian == 'big' else '<f'  # '>' for big-endian, '<' for little-endian
        else:
            raise ValueError("Unsupported format. Use 'int16', 'int8', or 'float32'.")

        # Export metadata (JSON file)
        metadata_filename = f"{filename}_metadata.json"
        metadata.update({
            "scale": scale,
            "format": format,
            "endianness": endian,
            "data_length": len(iq_samples),
            "samples_per_frame": len(data) // dtype_size,
        })

        with open(metadata_filename, 'w') as meta_file:
            json.dump(metadata, meta_file, indent=4)

        print(f"Metadata saved to: {metadata_filename}")

        # Write the interleaved I/Q data to the binary file
        with open(filename, 'wb') as f:
            for sample in iq_samples:
                f.write(struct.pack(struct_fmt, sample))

        # Log details about the export
        file_size = os.path.getsize(filename)
        samples = file_size // dtype_size
        subcarrier_spacing = 15000 * (2 ** self.params['mu'])  # Subcarrier spacing (Hz)
        sampling_rate = self.params['fft_size'] * subcarrier_spacing  # Sampling rate (Hz)

        # Calculate the duration (ms) of the file
        duration_ms = samples / (2 * sampling_rate) * 1000  # 2 accounts for interleaved I/Q samples

        print(f"Exported {samples // 2} IQ samples ({file_size} bytes, {endian}-endian) to {filename}")
        print(f"Signal duration: {duration_ms:.2f} ms (Expected: 10ms or 20ms)")

        return filename

    def export_resource_grid_to_excel(self, resource_grid, excel_path, scale_factor=1):
        """
        Export a resource grid directly to Excel in a simple grid format
        with subcarrier indices as rows and symbol indices as columns.
        Real and imaginary parts of complex values are placed in separate columns.

        Args:
            resource_grid: Complex matrix representing the resource grid
            excel_path: Path to save the Excel file
            scale_factor: Optional scaling factor to apply before export (default: 1)

        Returns:
            True if export succeeds, otherwise raises an error.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(excel_path)), exist_ok=True)

            # Get dimensions of the grid
            num_subcarriers, num_symbols = resource_grid.shape

            # Apply scaling if needed
            if scale_factor != 1:
                grid_to_export = resource_grid * scale_factor
            else:
                grid_to_export = resource_grid.copy()

            # Create column headers for real and imaginary parts
            columns = []
            for j in range(num_symbols):
                columns.append(f"Symbol_{j}_Real")
                columns.append(f"Symbol_{j}_Imag")

            # Create index for rows
            index = [f"Subcarrier_{i}" for i in range(num_subcarriers)]

            # Initialize data array with twice as many columns (real and imaginary for each symbol)
            data = np.zeros((num_subcarriers, num_symbols * 2))

            # Fill data array with real and imaginary values
            for i in range(num_subcarriers):
                for j in range(num_symbols):
                    data[i, j*2] = np.real(grid_to_export[i, j])        # Real part
                    data[i, j*2 + 1] = np.imag(grid_to_export[i, j])    # Imaginary part

            # Create DataFrame with proper indices
            df = pd.DataFrame(
                data=data,
                index=index,
                columns=columns
            )

            # Write to Excel
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Resource_Grid')

                # Apply formatting to improve readability
                workbook = writer.book
                worksheet = writer.sheets['Resource_Grid']
                worksheet.freeze_panes = 'B2'  # Freeze headers

                # Optional: Add formatting to distinguish real/imag columns
                for col_idx in range(1, num_symbols * 2 + 1, 2):  # Every odd column (1-indexed)
                    for row in range(1, num_subcarriers + 2):  # +2 for header and 1-indexing
                      cell = worksheet.cell(row=row, column=col_idx + 1)  # +1 for index column
                      cell.font = openpyxl.styles.Font(color="0000FF")  # Blue for imaginary

            return True

        except Exception as e:
            raise RuntimeError(f"Failed to export resource grid to Excel: {e}")

class SRS_Analyzer:
    """
    Analyzer for 5G NR Sounding Reference Signals (SRS)
    with support for multi-slot processing
    """
    def __init__(self, params=None):
        """
        Initialize SRS analyzer with default or provided parameters.
        Args:
            params: Dictionary of SRS parameters
        """
        # Set default parameters (aligned with SRS_Generator)
        self.default_params = {
            'p_tx_port': 1000,     # Antenna port (1000-1011)
            'sym_offset': 0,       # SRS symbol offset (l0)
            'cfg_idx': 0,          # SRS configuration index C_SRS
            'b_idx': 0,            # Bandwidth index
            'comb': 2,             # SRS comb size (2 or 4)
            'comb_offset': 0,      # Comb offset (0 to comb-1)
            'b_hop': 0,            # Frequency hopping (0-3)
            'n_rrc': 0,            # Frequency domain position (0-23)
            'fd_shift': 0,         # Frequency domain shift
            'n_syms': 1,           # Number of SRS symbols
            'n_ports': 1,          # Number of antenna ports
            'n_cs': 0,             # Cyclic shift index
            'n_id': 0,             # Scrambling ID (0-1023)
            'hop_mod': 0,          # Hopping mode (0: None, 1: Group, 2: Sequence)
            'bwp_offset': 0,       # BWP resource block offset
            'prb_num': 106,        # Number of PRBs (default 5G NR 20MHz = 106 PRBs)
            'snr_prb_en': True,    # Enable SNR per PRB estimation
            'ta_est_en': True,     # Enable timing alignment estimation
            'mu': 1,               # Numerology (0: 15kHz, 1: 30kHz, 2: 60kHz)
            'fft_size': 2048,      # FFT size
            'cp_type': 'normal',   # Cyclic prefix type
            'carrier_freq': 3.5e9, # Carrier frequency (Hz)
            'n_slots': 10          # Total Slots to generate
        }
        # Update with user-provided parameters
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)
        # Create SRS generator for reference signal
        self.srs_gen = SRS_Generator(self.params)

    def demodulate_srs(self, td_signal, n_slots=1, noise_power=0):
        """
        Demodulate SRS signal and perform measurements with
        proper handling of periodicity and slot alignment.
        Args:
            td_signal: Time domain signal
            n_slots: Number of slots in the signal
            noise_power: Noise power for simulation
        Returns:
            re_grid: Demodulated resource grid
            measurements: Dictionary with SRS measurements
        """
        p = self.params
        # Generate reference SRS for comparison
        ref_grid, ref_td_signal, srs_info = self.srs_gen.generate_srs()
        # Extract SRS parameters
        n_rb = p['prb_num']
        n_syms = p.get('n_syms', 1)  # Number of SRS symbols per slot
        symbols_per_slot = 14 if p.get('cp_type', 'normal') == 'normal' else 12
        # Get SRS periodicity and offset
        periodicity = p.get('T_srs', 1)  # SRS periodicity in slots
        slot_offset = p.get('T_offset', 0)  # SRS offset in slots
        # Extract the actual SRS symbol positions from srs_info
        if 'l_sym' in srs_info:
            srs_symbol_positions = np.array(srs_info['l_sym'])
        else:
            # Fallback if l_sym is not in srs_info - derive from parameters
            l_offset = p.get('sym_offset', 0)
            l0 = symbols_per_slot - 1 - l_offset
            srs_symbol_positions = np.array([l0 - i for i in range(n_syms)])

        # Ensure symbol positions are properly ordered
        srs_symbol_positions = np.sort(srs_symbol_positions)

        # Perform time alignment estimation
        timing_alignment = self._estimate_timing_alignment(td_signal, ref_td_signal)
        print(f"Estimated timing alignment: {timing_alignment} samples")
        # Adjust signal based on timing alignment
        if timing_alignment > 0:
            # Shift signal to align with reference
            aligned_signal = td_signal[timing_alignment:]
            # Pad with zeros if needed
            if len(aligned_signal) < len(ref_td_signal):
                aligned_signal = np.pad(aligned_signal, (0, len(ref_td_signal) - len(aligned_signal)))
        else:
            # Prepend zeros if timing is earlier than expected
            aligned_signal = np.pad(td_signal, (-timing_alignment, 0))[:len(ref_td_signal)]
        # Add simulated noise if specified
        if noise_power > 0:
            noisy_signal = aligned_signal + np.sqrt(noise_power / 2) * (
                np.random.normal(0, 1, len(aligned_signal)) +
                1j * np.random.normal(0, 1, len(aligned_signal))
            )
        else:
            noisy_signal = aligned_signal
        # Perform OFDM demodulation for the entire signal
        re_grid = self._ofdm_demodulation(noisy_signal, n_slots, srs_symbol_positions)
        # Extract SRS subcarrier indices
        srs_sc_indices = self._get_srs_subcarrier_indices(srs_info)

        print(f"SRS symbol positions: {srs_symbol_positions}")
        print(f"Number of SRS subcarriers: {len(srs_sc_indices)}")
        # Initialize measurement results
        snr_estimates = []
        snr_prb_estimates = []
        # Collect SNR only for slots that contain SRS based on periodicity
        active_slots = []
        for slot_idx in range(n_slots):
            # Check if this slot should contain SRS
            if (slot_idx - slot_offset) % periodicity == 0 and slot_idx >= slot_offset:
                active_slots.append(slot_idx)

        print(f"Active SRS slots: {active_slots}")
        for slot_idx in active_slots:
            # For each active slot, extract the symbols where SRS is present
            for sym_idx, sym_pos in enumerate(srs_symbol_positions):
                # Calculate absolute symbol index in the RE grid
                abs_sym_idx = slot_idx * symbols_per_slot + sym_pos

                # Extract the symbol column from the received grid
                rx_sym = re_grid[:, slot_idx * n_syms + sym_idx]

                # Extract the reference symbol
                ref_sym = ref_grid[:, abs_sym_idx]

                # Use only SRS subcarriers for SNR estimation
                rx_srs = rx_sym[srs_sc_indices]
                ref_srs = ref_sym[srs_sc_indices]

                # Estimate SNR for this symbol
                sym_snr = self._estimate_symbol_snr(rx_srs, ref_srs)
                snr_estimates.append(sym_snr)

                # Optional: SNR per PRB if enabled
                if p.get('snr_prb_en', False):
                    sym_snr_prb = self._estimate_snr_per_prb_symbol(rx_srs, ref_srs, srs_sc_indices, n_rb)
                    snr_prb_estimates.append(sym_snr_prb)
        # Compute average SNR and compile results
        measurements = {
            'snr_est': np.mean(snr_estimates) if snr_estimates else 0,
            'snr_per_symbol': np.array(snr_estimates),
            'snr_prb': np.mean(snr_prb_estimates, axis=0) if p.get('snr_prb_en', False) and snr_prb_estimates else None,
            'timing_est': timing_alignment
        }
        return re_grid, measurements

    def _ofdm_demodulation(self, td_signal, n_slots=1, srs_symbol_positions=None):
        """
        Perform OFDM demodulation on time-domain signal with proper slot alignment
        and support for SRS periodicity.
        Args:
            td_signal: Time-domain signal
            n_slots: Number of slots to process
            srs_symbol_positions: List of symbol positions within a slot for SRS
        Returns:
            re_grid: Demodulated resource grid containing SRS symbols
        """
        # Extract configuration from params
        p = self.params
        fft_size = p['fft_size']
        N_RB = p['prb_num']
        n_subcarriers = N_RB * 12
        n_symbols = p['n_syms']  # SRS symbols per slot
        mu = p['mu']
        # If SRS symbol positions not provided, calculate them
        if srs_symbol_positions is None:
            l_offset = p.get('sym_offset', 0)
            l0 = 14 - 1 - l_offset
            srs_symbol_positions = [l0 - i for i in range(n_symbols)]
            # Ensure symbols are sorted in ascending order within slot
            srs_symbol_positions.sort()
        # Calculate CP lengths based on numerology
        if p.get('cp_type', 'normal') == 'normal':
            # Normal CP length calculation (3GPP specifications)
            cp_length_long = int(np.ceil((160 * (2**(-mu))) * (fft_size / 2048)))
            cp_length_short = int(np.ceil((144 * (2**(-mu))) * (fft_size / 2048)))
        else:  # Extended CP
            # Extended CP length calculation
            cp_length_long = cp_length_short = int(np.ceil((512 * (2**(-mu))) * (fft_size / 2048)))
        # Pre-calculate symbol lengths for all positions in a slot (14 symbols)
        symbol_lengths = np.zeros(14, dtype=int)
        for i in range(14):
            symbol_lengths[i] = fft_size + (cp_length_long if i == 0 or i == 7 else cp_length_short)
        # Calculate cumulative symbol offsets within a slot
        symbol_offsets = np.zeros(14, dtype=int)
        for i in range(1, 14):
            symbol_offsets[i] = symbol_offsets[i-1] + symbol_lengths[i-1]
        # Calculate complete slot length in samples
        slot_length = symbol_offsets[-1] + symbol_lengths[-1]
        # Initialize resource grid for SRS symbols from all slots
        re_grid = np.zeros((n_subcarriers, n_slots * n_symbols), dtype=np.complex128)
        # Get SRS periodicity and offset (if specified)
        periodicity = p.get('periodicity', 1)  # Default: every slot
        slot_offset = p.get('slot_offset', 0)  # Default: no offset
        # Process each slot
        for slot_idx in range(n_slots):
            # Skip slots that don't contain SRS based on periodicity and offset
            if periodicity > 1 and (slot_idx - slot_offset) % periodicity != 0:
                continue
            # For each SRS symbol in the slot
            for sym_idx, sym_pos in enumerate(srs_symbol_positions):
                if sym_pos < 0 or sym_pos >= 14:
                    warnings.warn(f"Invalid symbol position {sym_pos}. Must be between 0-13.")
                    continue
                # Calculate exact position in time domain signal
                symbol_start = slot_idx * slot_length + symbol_offsets[sym_pos]

                # Determine CP length for this symbol
                cp_length = cp_length_long if (sym_pos == 0 or sym_pos == 7) else cp_length_short
                # Skip if insufficient samples
                if symbol_start + cp_length + fft_size > len(td_signal):
                    warnings.warn(f"Insufficient samples for symbol {sym_pos} in slot {slot_idx}")
                    continue
                # Extract symbol without CP
                symbol_without_cp = td_signal[symbol_start + cp_length:symbol_start + cp_length + fft_size]
                # Perform FFT to transform to frequency domain
                fft_data = np.fft.fft(symbol_without_cp) / np.sqrt(fft_size)
                # Shift FFT to center DC subcarrier
                fft_data = np.fft.fftshift(fft_data)
                # Extract the subcarriers for the resource grid
                dc_idx = fft_size // 2
                start_idx = dc_idx - n_subcarriers // 2
                end_idx = start_idx + n_subcarriers
                # Calculate correct output column index in re_grid
                out_idx = slot_idx * n_symbols + sym_idx

                # Ensure indices are within bounds
                if 0 <= start_idx < end_idx <= fft_size and out_idx < re_grid.shape[1]:
                    re_grid[:, out_idx] = fft_data[start_idx:end_idx]
        return re_grid

    def _get_srs_subcarrier_indices(self, srs_info=None):
        """
        Get indices of subcarriers containing SRS

        Args:
            srs_info: Optional SRS info dictionary from generator

        Returns:
            srs_indices: Array of subcarrier indices
        """
        p = self.params

        # If SRS info provided, use it
        if srs_info is not None and 'srs_sc_indices' in srs_info:
            return srs_info['srs_sc_indices']

        # Map parameter names to handle different naming conventions
        c_srs = p.get('fh_c', p.get('c_srs', p.get('cfg_idx', 0)))
        b_srs = p.get('fh_b', p.get('b_srs', p.get('b_idx', 0)))
        n_rrc = p.get('fd_pos', p.get('n_rrc', 0))
        fd_shift = p.get('fd_shift', 0)
        bwp_offset = p.get('bwp_offset', 0)
        N_RB = p.get('prb_num', 52)
        comb_size = p.get('comb', 2)
        comb_offset = p.get('comb_offset', 0)

        # Generate reference SRS to get tables and info
        if not hasattr(self.srs_gen, 'm_srs_table') or not hasattr(self.srs_gen, 'N_srs_table'):
            # Get these from the SRS generator
            _, _, srs_info = self.srs_gen.generate_srs()
            m_srs = srs_info['m_SRS']
            N_i = srs_info['N_i']
            M_SRS_b = srs_info['M_SRS_b']
            n_b = srs_info['n_b']
        else:
            # Use generator's table data
            m_srs = self.srs_gen.m_srs_table[c_srs][min(b_srs, 3)]
            N_i = self.srs_gen.N_srs_table[c_srs][min(b_srs, 3)]
            M_SRS_b = m_srs // N_i
            n_b = n_rrc % N_i

        # Calculate k_0 (starting PRB index)
        k_0 = fd_shift + bwp_offset + n_b * M_SRS_b

        # Calculate indices
        indices = []

        for i in range(M_SRS_b):
            rb_idx = i + k_0
            if rb_idx < N_RB:
                for j in range(12 // comb_size):
                    k = rb_idx * 12 + j * comb_size + comb_offset
                    if 0 <= k < N_RB * 12:
                        indices.append(k)

        return np.array(indices)

    def _estimate_symbol_snr(self, rx_symbol, ref_symbol):
        """
        Estimate SNR for a single OFDM symbol.

        Args:
            rx_symbol: Received symbol data (srs subcarriers only)
            ref_symbol: Reference symbol data (srs subcarriers only)

        Returns:
            Estimated SNR in dB
        """
        # Apply channel estimation/equalization if needed
        signal_power = np.mean(np.abs(ref_symbol)**2)

        # Calculate error
        error = rx_symbol - ref_symbol
        noise_power = np.mean(np.abs(error)**2)

        # Avoid division by zero
        if noise_power < 1e-10:
            return 100.0  # Very high SNR as default

        # Calculate SNR and convert to dB
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)

        return snr_db

    def _estimate_snr_per_prb_symbol(self, rx_symbol, ref_symbol, srs_sc_indices, n_rb):
        """
        Estimate SNR per PRB for a single OFDM symbol.

        Args:
            rx_symbol: Received symbol data (srs subcarriers only)
            ref_symbol: Reference symbol data (srs subcarriers only)
            srs_sc_indices: Indices of SRS subcarriers
            n_rb: Number of resource blocks

        Returns:
            Array of SNR values per PRB in dB
        """
        # Number of subcarriers per RB
        sc_per_rb = 12

        # Initialize signal and noise power arrays separately
        signal_power_per_rb = np.zeros(n_rb)
        noise_power_per_rb = np.zeros(n_rb)

        # Track how many subcarriers contribute to each PRB
        sc_count_per_rb = np.zeros(n_rb)

        # Map SRS subcarriers to their PRBs
        for i, sc_idx in enumerate(srs_sc_indices):
            prb_idx = sc_idx // sc_per_rb
            if prb_idx < n_rb:
                # Calculate signal and noise power for this subcarrier
                signal_power = np.abs(ref_symbol[i])**2
                noise_power = np.abs(rx_symbol[i] - ref_symbol[i])**2

                # Accumulate per PRB
                signal_power_per_rb[prb_idx] += signal_power
                noise_power_per_rb[prb_idx] += noise_power
                sc_count_per_rb[prb_idx] += 1

        # Calculate SNR per PRB
        snr_per_prb = np.zeros(n_rb)
        for i in range(n_rb):
            if sc_count_per_rb[i] > 0:  # Only process PRBs that have SRS subcarriers
                if noise_power_per_rb[i] > 0:
                    snr_per_prb[i] = 10 * np.log10(signal_power_per_rb[i] / noise_power_per_rb[i])
                else:
                    snr_per_prb[i] = 100.0  # High SNR default
            else:
                snr_per_prb[i] = np.nan  # Mark PRBs without SRS as NaN

        return snr_per_prb

    def _estimate_timing_alignment(self, rx_signal, ref_signal):
        """
        Estimate timing alignment between received signal and reference
        using cross-correlation.

        Args:
            rx_signal: Received time-domain signal
            ref_signal: Reference time-domain signal

        Returns:
            timing_offset: Estimated timing offset in samples
        """
        # Limit signals to comparable length for correlation
        min_len = min(len(rx_signal), len(ref_signal))
        rx_segment = rx_signal[:min_len]
        ref_segment = ref_signal[:min_len]

        # Use magnitude for more robust correlation
        rx_mag = np.abs(rx_segment)
        ref_mag = np.abs(ref_segment)

        # Compute cross-correlation
        correlation = np.correlate(rx_mag, ref_mag, mode='full')

        # Find peak correlation position
        max_idx = np.argmax(correlation)

        # Calculate timing offset (center of correlation array is zero offset)
        timing_offset = max_idx - (min_len - 1)

        return timing_offset

    def analyze_frequency_response(self, rx_grid, ref_grid):
        """
        Analyze frequency response of the channel using SRS

        Args:
            rx_grid: Received resource grid
            ref_grid: Reference resource grid

        Returns:
            freq_response: Estimated frequency response
            metrics: Dictionary with channel metrics
        """
        # Get SRS subcarrier indices
        srs_indices = self._get_srs_subcarrier_indices()

        # Extract SRS from grids
        rx_srs = rx_grid[srs_indices, 0]
        ref_srs = ref_grid[srs_indices, 0]

        # Calculate frequency response (transfer function)
        freq_response = rx_srs / ref_srs
        freq_response = np.where(np.isfinite(freq_response), freq_response, 0)

        # Calculate metrics
        avg_gain = np.mean(np.abs(freq_response))
        phase_slope = np.unwrap(np.angle(freq_response))
        delay_spread = np.std(phase_slope) / (2 * np.pi * 15000 * (2**self.params['mu']))

        metrics = {
            'avg_gain_db': 20 * np.log10(avg_gain) if avg_gain > 0 else -100,
            'delay_spread': delay_spread,
            'coherence_bw': 1 / (5 * delay_spread) if delay_spread > 0 else float('inf')
        }

        return freq_response, metrics

def plot_srs_results(re_grid, td_signal, measurements, num_slots=None):
    """
    Plot SRS generation and analysis results using Plotly in a single interactive dashboard.

    Args:
        re_grid (ndarray): Complete resource grid for all slots.
        td_signal (ndarray): Time-domain signal.
        measurements (dict): Analysis results (e.g., SNR, timing).
        num_slots (int, optional): Number of slots. Defaults to automatic detection from numerology.
    """
    # Auto-detect numerology and slots if not specified
    if num_slots is None:
        signal_len = len(td_signal)
        if signal_len >= 300000:  # ~307,200 samples for 30kHz SCS
            num_slots = 20  # μ=1 (30kHz SCS)
        else:
            num_slots = 10  # μ=0 (15kHz SCS)
        print(f"Auto-detected {num_slots} slots in the signal ({signal_len} samples)")

    # Dimensions of the resource grid
    n_subcarriers, n_symbols_total = re_grid.shape
    n_symbols_per_slot = n_symbols_total // num_slots

    # Create a single figure with 4 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Resource Grid Magnitude",
            "Time Domain Signal",
            "SNR per PRB",
            "Frequency Spectrum"
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # 1. Resource Grid Visualization - properly formatted
    print("Generating SRS resource grid visualization...")

    # Create resource grid magnitude data
    re_grid_magnitude = np.abs(re_grid)

    # Create a transposed view for better visualization (subcarriers as y-axis)
    heatmap_data = re_grid_magnitude

    # Create custom x-axis labels for slot/symbol clarity
    x_labels = []
    for slot in range(num_slots):
        for sym in range(n_symbols_per_slot):
            if sym == 0:  # Only mark first symbol of each slot
                x_labels.append(f"S{slot}")
            else:
                x_labels.append("")

    # Add heatmap to the first subplot
    fig.add_trace(
        go.Heatmap(
            z=heatmap_data,
            colorscale='Viridis',
            colorbar=dict(
                title="Magnitude",
                len=0.4,
                y=0.85,
                thickness=15
            ),
            hovertemplate='Subcarrier: %{y}<br>Symbol: %{x}<br>Magnitude: %{z:.4f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Update axes for resource grid
    fig.update_xaxes(
        title_text="OFDM Symbols",
        tickvals=list(range(0, n_symbols_total, n_symbols_per_slot)),
        ticktext=[f"Slot {i}" for i in range(num_slots)],
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Subcarriers",
        row=1, col=1
    )

    # 2. Time-Domain Signal Plot
    print("Generating time-domain signal visualization...")

    # Create time axis in milliseconds
    time_ms = np.linspace(0, 10, len(td_signal))  # Assuming 10ms for frame

    # Add time domain signal trace
    fig.add_trace(
        go.Scatter(
            x=time_ms,
            y=np.real(td_signal),
            mode='lines',
            name='Real Part',
            line=dict(color='#1f77b4', width=1)
        ),
        row=1, col=2
    )

    # Add slot boundaries as vertical lines
    slot_boundaries = np.linspace(0, 10, num_slots + 1)
    for boundary in slot_boundaries:
        fig.add_shape(
            type="line",
            x0=boundary, y0=-max(np.abs(td_signal)),
            x1=boundary, y1=max(np.abs(td_signal)),
            line=dict(color="red", width=1, dash="dash"),
            row=1, col=2
        )

    fig.update_xaxes(title_text="Time (ms)", row=1, col=2)
    fig.update_yaxes(title_text="Amplitude", row=1, col=2)

    # 3. SNR per PRB Plot
    print("Generating SNR per PRB visualization...")
    snr_prb_data = measurements.get("snr_prb", measurements.get("snr_prb_est", None))

    if snr_prb_data is not None:
        snr_prb_data = np.array(snr_prb_data)

        # Check dimensions of SNR data
        if snr_prb_data.ndim > 1:  # Multi-slot SNR
            snr_prb_avg = np.mean(snr_prb_data, axis=0)

            # Add average SNR trace
            fig.add_trace(
                go.Bar(
                    x=np.arange(len(snr_prb_avg)),
                    y=snr_prb_avg,
                    name="Avg SNR per PRB",
                    marker_color='#2ca02c',
                    hovertemplate='PRB: %{x}<br>SNR: %{y:.2f} dB<extra></extra>'
                ),
                row=2, col=1
            )

            # Show first slot SNR as reference
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(snr_prb_data[0])),
                    y=snr_prb_data[0],
                    mode='lines+markers',
                    name="First Slot SNR",
                    marker=dict(size=4),
                    line=dict(color='#ff7f0e', width=1),
                    hovertemplate='PRB: %{x}<br>SNR: %{y:.2f} dB<extra></extra>'
                ),
                row=2, col=1
            )
        else:  # Single-slot SNR
            fig.add_trace(
                go.Bar(
                    x=np.arange(len(snr_prb_data)),
                    y=snr_prb_data,
                    name="SNR per PRB",
                    marker_color='#ff7f0e',
                    hovertemplate='PRB: %{x}<br>SNR: %{y:.2f} dB<extra></extra>'
                ),
                row=2, col=1
            )
    else:
        # Add placeholder message if no SNR data
        fig.add_annotation(
            text="No SNR data available",
            xref="x3", yref="y3",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14),
            row=2, col=1
        )

    fig.update_xaxes(title_text="PRB Index", row=2, col=1)
    fig.update_yaxes(title_text="SNR (dB)", row=2, col=1)

    # 4. Frequency Spectrum Plot
    print("Generating frequency spectrum visualization...")

    # Use Welch's method for better spectrum estimation
    f, Pxx = sig.welch(td_signal, fs=len(td_signal)/10e-3, nperseg=1024, scaling='spectrum')

    # Shift to center frequency
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)

    # Convert to dB scale
    Pxx_dB = 10 * np.log10(Pxx + 1e-10)

    # Add frequency spectrum trace
    fig.add_trace(
        go.Scatter(
            x=f / 1e6,  # Convert to MHz
            y=Pxx_dB,
            mode='lines',
            name='PSD',
            line=dict(color='#9467bd', width=1.5),
            hovertemplate='Frequency: %{x:.2f} MHz<br>Power: %{y:.2f} dB<extra></extra>'
        ),
        row=2, col=2
    )

    fig.update_xaxes(title_text="Frequency (MHz)", row=2, col=2)
    fig.update_yaxes(title_text="Power Spectral Density (dB)", row=2, col=2)

    # Update layout for the entire figure
    fig.update_layout(
        title_text="SRS Signal Analysis",
        height=800,  # Increase height for better visibility
        width=1200,  # Set width
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        template="plotly_white"  # Clean white template
    )

    fig.show()

    # To save as an HTML file (optional)
    fig.write_html("srs_results.html")

    return fig

def generate_srs_example():
    """Example function to demonstrate SRS generation and analysis"""
    # Define parameters with both naming conventions for compatibility
    params = {
        # Core parameters
        'prb_num': 25,         # Number of PRBs
        'n_id': 500,           # Scrambling ID
        'comb': 2,             # SRS comb size
        'comb_offset': 0,      # Comb offset

        # Parameters with dual naming for compatibility
        'p_tx_port': 1000,     # Antenna port (original naming)
        'tx_port': 1000,       # Antenna port (alternative naming)

        'cfg_idx': 6,          # Configuration index (original naming)
        'c_srs': 6,            # Configuration index (alternative naming)
        'fh_c': 6,             # Configuration index (another alternative)

        'b_idx': 0,            # Bandwidth index (original naming)
        'b_srs': 0,            # Bandwidth index (alternative naming)
        'fh_b': 0,             # Bandwidth index (another alternative)

        'n_rrc': 0,            # Frequency domain position (original naming)
        'fd_pos': 0,           # Frequency domain position (alternative naming)

        'b_hop': 0,            # Frequency hopping (original naming)
        'fh_b_hop': 0,         # Frequency hopping (alternative naming)

        'sym_offset': 3,       # Symbol offset (original naming)
        'start_pos': 3,        # Symbol offset (alternative naming)

        'n_cs': 0,             # Cyclic shift (original naming)
        'comb_cs': 0,          # Cyclic shift (alternative naming)

        # Additional required parameters
        'n_syms': 2,           # Number of symbols
        'n_ports': 1,          # Number of ports
        'fd_shift': 0,         # Frequency domain shift
        'bwp_offset': 0,       # BWP offset

        # Signal processing parameters
        'cp_type': 'normal',   # Cyclic prefix type
        'mu': 0,               # 15kHz subcarrier spacing
        'fft_size': 512,      # FFT size

        # Signal Generation
        "periodicity": 1,      # SRS periodicity in slots
        "T_srs":1,

        "offset":0,            # SRS offset
        "T_offset":0,          # SRS offset (Alternate Name)
        "slot_offset":0,       # SRS offset

        'n_slots': 10,          # Total Slots to generate

        # Analysis parameters
        'snr_prb_en': True,    # Enable SNR per PRB estimation
        'ta_est_en': True,     # Enable timing alignment estimation

    }

    # Create SRS generator
    srs_gen = SRS_Generator(params)

    # Generate SRS signal
    re_grid, td_signal, srs_info = srs_gen.generate_srs()

    # Export to binary file
    srs_gen.export_binary(td_signal, "srs_buffer.bin", scale=2**14, endian='big', metadata=srs_info)
    srs_gen.export_resource_grid_to_excel(re_grid, "grid_in.xlsx", scale_factor=2**14)

    # Create analyzer with same parameters
    srs_analyzer = SRS_Analyzer(params)

    # Add noise for testing
    noisy_signal = td_signal + 0.05 * (
        np.random.normal(0, 1, len(td_signal)) +
        1j * np.random.normal(0, 1, len(td_signal))
    )

    # Analyze SRS
    rx_grid, measurements = srs_analyzer.demodulate_srs(noisy_signal, params['n_slots'])

    # Print results - using the new key names from our updated demodulate_srs function
    print("SRS Analysis Results:")
    print(f"Estimated SNR: {measurements['snr_est']:.2f} dB")

    # Handle timing measurement with proper key name
    if 'timing_est' in measurements:
        print(f"Timing alignment: {measurements['timing_est']*1e9:.2f} ns")
    elif 'ta_est' in measurements:
        print(f"Timing alignment: {measurements['ta_est']*1e9:.2f} ns")
    else:
        print("Timing alignment: Not available")

    # Plot results if plotting function is available
    plot_srs_results(re_grid, td_signal, measurements, params['n_slots'])

if __name__ == "__main__":
    # For standalone execution, uncomment one of these:

    # Example 1: Generate and analyze SRS
    generate_srs_example()
    pass

