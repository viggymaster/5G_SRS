import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import struct
from scipy import signal
from typing import Dict, Any, Tuple, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json
import traceback
import os
import matplotlib.pyplot as plt
from primePy import primes

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
            'c_srs': 0,            # SRS configuration index (0-63)
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
            'prb_num': 106,        # Number of PRBs (default 5G NR 20MHz = 106 PRBs)
            'snr_prb_en': True,    # Enable SNR per PRB estimation
            'ta_est_en': True,     # Enable timing alignment estimation
            'mu': 1,               # Numerology (0: 15kHz, 1: 30kHz, 2: 60kHz)
            'fft_size': 2048,      # FFT size
            'cp_type': 'normal',   # Cyclic prefix type
            'carrier_freq': 3.5e9, # Carrier frequency (Hz)
            'n_slots': 10          # Total Slots to generate
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
        """Validate SRS parameters"""
        p = self.params

        # Basic validation
        assert 1000 <= p['p_tx_port'] <= 1011, "p_tx_port must be between 1000 and 1011"
        assert 0 <= p['c_srs'] <= 63, "c_srs must be between 0 and 63"
        assert 0 <= p['b_srs'] <= 3, "b_srs must be between 0 and 3"
        assert p['comb'] in [2, 4], "comb must be 2 or 4"
        assert 0 <= p['comb_offset'] < p['comb'], f"comb_offset must be between 0 and {p['comb']-1}"
        assert 0 <= p['b_hop'] <= 3, "b_hop must be between 0 and 3"
        assert 0 <= p['n_rrc'] <= 23, "n_rrc must be between 0 and 23"
        assert 0 <= p['n_id'] <= 1023, "n_id must be between 0 and 1023"
        assert 0 <= p['hop_mod'] <= 2, "hop_mod must be between 0 and 2"
        assert 0 <= p['mu'] <= 3, "mu must be between 0 and 3"
        assert p['cp_type'] in ['normal', 'extended'], "cp_type must be 'normal' or 'extended'"

        # Validate SRS configuration based on tables
        c_srs = p['c_srs']
        b_srs = p['b_srs']

        # Ensure the combination is valid (table may not have all b_srs values for every c_srs)
        assert b_srs < len(self.m_srs_table[c_srs]), f"b_srs={b_srs} invalid for c_srs={c_srs}. Max b_srs is {len(self.m_srs_table[c_srs])-1}"

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
        Generate SRS signal with configurable periodicity according to 3GPP TS 38.211 using low PAPR sequences.

        Args:
            n_slots: Number of slots to generate (default: 1)

        Returns:
            re_grid: Resource grid with SRS symbols.
            td_signal: Time domain signal for complete slots with SRS symbols.
            srs_info: Dictionary with SRS generation information.
        """
        p = self.params

        # Map the parameters using both naming conventions for compatibility
        N_ap = p.get('n_ports', self.default_params.get('n_ports', 1))  # Number of antenna ports
        p_i = p.get('tx_port', p.get('p_tx_port', self.default_params.get('p_tx_port', 0)))

        # Ensure p_i is in the correct range (1000-1003 as in spec)
        if p_i < 1000:
            p_i = 1000 + p_i  # Convert to proper port index format if given as offset

        N_symb_SRS = p.get('n_syms', self.default_params.get('n_syms', 1))  # Renamed for clarity
        l_offset = p.get('start_pos', p.get('sym_offset', self.default_params.get('sym_offset', 0)))
        K_TC = p.get('comb', self.default_params.get('comb', 2))  # Comb size
        k_bar_TC = p.get('comb_offset', self.default_params.get('comb_offset', 0))  # Comb offset
        n_cs = p.get('comb_cs', p.get('n_cs', self.default_params.get('n_cs', 0)))  # Cyclic shift
        C_SRS = p.get('fh_c', p.get('c_srs', self.default_params.get('c_srs', 0)))  # Frequency hopping configuration
        B_SRS = p.get('fh_b', p.get('b_srs', self.default_params.get('b_srs', 0)))  # Bandwidth configuration
        n_RRC = p.get('fd_pos', p.get('n_rrc', self.default_params.get('n_rrc', 0)))  # Frequency domain position
        n_shift = p.get('fd_shift', p.get('fd_shift', 0))  # Frequency shift
        b_hop = p.get('fh_b_hop', p.get('b_hop', self.default_params.get('b_hop', 0)))  # Frequency hopping bandwidth
        bwp_offset = p.get('bwp_offset', self.default_params.get('bwp_offset', 0))  # BWP offset in PRBs
        N_RB = p.get('prb_num', self.default_params.get('prb_num', 52))  # Default max BW for FR1
        n_id = p.get('n_id', self.default_params.get('n_id', 500))  # Scrambling ID
        n_slots = p.get('n_slots', self.default_params.get('n_slots', 10))  # Total Slots

        # -- Step 1: Generate a Resource Grid for SRS Symbols --
        if hasattr(self, 'm_srs_table') and hasattr(self, 'N_srs_table'):
            m_row = self.m_srs_table[C_SRS]
            N_row = self.N_srs_table[C_SRS]
        else:
            m_row, N_row = self.read_srs_table(C_SRS) if hasattr(self, 'read_srs_table') else self._default_srs_table(C_SRS)

        B_SRS_max = min(B_SRS, 3)
        m_SRS = m_row[B_SRS_max]
        N_i = N_row[B_SRS_max]

        # Calculate the actual symbol positions within the slot
        l0 = 14 - 1 - l_offset
        l_sym = np.array([l0 - i for i in range(N_symb_SRS)])

        n_b = n_RRC % N_i
        k_0 = n_shift + bwp_offset
        M_SRS_b = m_SRS // N_i
        n_sc = N_RB * 12
        re_grid = np.zeros((n_sc, N_symb_SRS), dtype=np.complex128)

        fh_enabled = (b_hop > 0) and (N_symb_SRS > 1)

        # Calculate n_cs_i based on OctNrGenSrs logic
        if K_TC == 2:
            n_cs_max = 8
        else:  # K_TC == 4
            n_cs_max = 12

        n_cs_i = (n_cs + (n_cs_max * (p_i - 1000) // N_ap)) % n_cs_max

        # Calculate alpha using n_cs_i
        # alpha = 2 * pi * n_cs_i / n_cs_max
        alpha = 2 * np.pi * n_cs_i / n_cs_max

        # Calculate sequence group using n_id
        u_seq_grp = n_id % 30  # As per OctNrGenSrs: u_seq_grp = (f_gh + n_id) % 30, where f_gh = 0
        v_seq_num = 0  # As per C code for non-hopping mode

        for n_SRS in range(N_symb_SRS):
            if fh_enabled:
                F_b = self.read_srs_F_b(n_SRS, N_i, b_hop)
                n_b_hop = (n_b + F_b) % N_i
            else:
                n_b_hop = n_b

            k_0_b = k_0 + n_b_hop * M_SRS_b
            sequence_length = M_SRS_b * 12 // K_TC

            # Generate the sequence using u_seq_grp derived from n_id
            if sequence_length <= 36:
                srs_seq = self._generate_low_papr_sequence(u_seq_grp, v_seq_num, np.array([alpha]), sequence_length)
            else:
                srs_seq = self._generate_ZC_sequence(u_seq_grp, v_seq_num, alpha, sequence_length)

            seq_idx = 0
            for k in range(k_0_b * 12, (k_0_b + M_SRS_b) * 12, K_TC):
                if k + k_bar_TC < n_sc:
                    re_grid[k + k_bar_TC, n_SRS] = srs_seq[seq_idx]
                    seq_idx += 1

        # Generate complete slot signal with the correct SRS symbol positions
        td_signal = self._generate_full_slot_signal(re_grid, n_slots, l_sym)

        # Calculate buffer length in ms
        # For numerology μ, we have 2^μ slots per subframe
        mu = p['mu']
        slots_per_subframe = 2**mu
        buffer_length_ms = (n_slots / slots_per_subframe) * 1.0  # Each subframe is 1ms

        srs_info = {
            'C_SRS': C_SRS,
            'B_SRS': B_SRS,
            'm_SRS': m_SRS,
            'N_i': N_i,
            'k_0': k_0,
            'M_SRS_b': M_SRS_b,
            'comb_size': K_TC,
            'comb_offset': k_bar_TC,
            'fh_enabled': fh_enabled,
            'n_b': n_b,
            'N_symb_SRS': N_symb_SRS,
            'l_sym': l_sym.tolist(),  # Symbol positions
            'n_slots': n_slots,
            'buffer_length_ms': buffer_length_ms,
            'n_cs_i': n_cs_i,
            'u_seq_grp': u_seq_grp,
            'v_seq_num': v_seq_num
        }

        return re_grid, td_signal, srs_info



    def read_srs_table(self, c_idx):
        """
        Returns SRS bandwidth configuration as per 3GPP 38.211 Table 6.4.1.4.3-1

        Args:
            c_idx: C_SRS value (0-63) for row selection

        Returns:
            m_row: Array of m_SRS values [m_SRS_0, m_SRS_1, m_SRS_2, m_SRS_3]
            N_row: Array of N_i values [N_0, N_1, N_2, N_3]
        """
        # Table 6.4.1.4.3-1 from 3GPP TS 38.211
        # Format: [C_SRS, m_SRS_0, N_0, m_SRS_1, N_1, m_SRS_2, N_2, m_SRS_3, N_3]
        srs_table = np.array([
            [0,   4, 1, 4, 1, 4, 1, 4, 1],
            [1,   8, 1, 4, 2, 4, 1, 4, 1],
            [2,  12, 1, 4, 3, 4, 1, 4, 1],
            [3,  16, 1, 4, 4, 4, 1, 4, 1],
            [4,  16, 1, 8, 2, 4, 2, 4, 1],
            [5,  20, 1, 4, 5, 4, 1, 4, 1],
            [6,  24, 1, 4, 6, 4, 1, 4, 1],
            [7,  24, 1, 12, 2, 4, 3, 4, 1],
            [8,  28, 1, 4, 7, 4, 1, 4, 1],
            [9,  32, 1, 16, 2, 8, 2, 4, 2],
            [10, 36, 1, 12, 3, 4, 3, 4, 1],
            [11, 40, 1, 20, 2, 4, 5, 4, 1],
            [12, 48, 1, 16, 3, 8, 2, 4, 2],
            [13, 48, 1, 24, 2, 12, 2, 4, 3],
            [14, 52, 1, 4, 13, 4, 1, 4, 1],
            [15, 56, 1, 28, 2, 4, 7, 4, 1],
            [16, 60, 1, 20, 3, 4, 5, 4, 1],
            [17, 64, 1, 32, 2, 16, 2, 4, 4],
            [18, 72, 1, 24, 3, 12, 2, 4, 3],
            [19, 72, 1, 36, 2, 12, 3, 4, 3],
            [20, 76, 1, 4, 19, 4, 1, 4, 1],
            [21, 80, 1, 40, 2, 20, 2, 4, 5],
            [22, 88, 1, 44, 2, 4, 11, 4, 1],
            [23, 96, 1, 32, 3, 16, 2, 4, 4],
            [24, 96, 1, 48, 2, 24, 2, 4, 6],
            [25, 104, 1, 52, 2, 4, 13, 4, 1],
            [26, 112, 1, 56, 2, 28, 2, 4, 7],
            [27, 120, 1, 60, 2, 20, 3, 4, 5],
            [28, 120, 1, 40, 3, 8, 5, 4, 2],
            [29, 120, 1, 24, 5, 12, 2, 4, 3],
            [30, 128, 1, 64, 2, 32, 2, 4, 8],
            [31, 128, 1, 64, 2, 16, 4, 4, 4],
            [32, 128, 1, 16, 8, 8, 2, 4, 2],
            [33, 132, 1, 44, 3, 4, 11, 4, 1],
            [34, 136, 1, 68, 2, 4, 17, 4, 1],
            [35, 144, 1, 72, 2, 36, 2, 4, 9],
            [36, 144, 1, 48, 3, 24, 2, 12, 2],
            [37, 144, 1, 48, 3, 16, 3, 4, 4],
            [38, 144, 1, 16, 9, 8, 2, 4, 2],
            [39, 152, 1, 76, 2, 4, 19, 4, 1],
            [40, 160, 1, 80, 2, 40, 2, 4, 10],
            [41, 160, 1, 80, 2, 20, 4, 4, 5],
            [42, 160, 1, 32, 5, 16, 2, 4, 4],
            [43, 168, 1, 84, 2, 28, 3, 4, 7],
            [44, 176, 1, 88, 2, 44, 2, 4, 11],
            [45, 184, 1, 92, 2, 4, 23, 4, 1],
            [46, 192, 1, 96, 2, 48, 2, 4, 12],
            [47, 192, 1, 96, 2, 24, 4, 4, 6],
            [48, 192, 1, 64, 3, 16, 4, 4, 4],
            [49, 192, 1, 24, 8, 8, 3, 4, 2],
            [50, 208, 1, 104, 2, 52, 2, 4, 13],
            [51, 216, 1, 108, 2, 36, 3, 4, 9],
            [52, 224, 1, 112, 2, 56, 2, 4, 14],
            [53, 240, 1, 120, 2, 60, 2, 4, 15],
            [54, 240, 1, 80, 3, 20, 4, 4, 5],
            [55, 240, 1, 48, 5, 16, 3, 8, 2],
            [56, 240, 1, 24, 10, 12, 2, 4, 3],
            [57, 256, 1, 128, 2, 64, 2, 4, 16],
            [58, 256, 1, 128, 2, 32, 4, 4, 8],
            [59, 256, 1, 16, 16, 8, 2, 4, 2],
            [60, 264, 1, 132, 2, 44, 3, 4, 11],
            [61, 272, 1, 136, 2, 68, 2, 4, 17],
            [62, 272, 1, 68, 4, 4, 17, 4, 1],
            [63, 272, 1, 16, 17, 8, 2, 4, 2]
        ])

        # Extract m_SRS values (columns 2, 4, 6, 8)
        m_row = srs_table[c_idx, [1, 3, 5, 7]]

        # Extract N_i values (columns 3, 5, 7, 9)
        N_row = srs_table[c_idx, [2, 4, 6, 8]]

        return m_row, N_row

    def _default_srs_table(self, C_SRS):
        """Default implementation if read_srs_table is not available"""
        # Simple implementation based on pattern in 38.211
        if C_SRS < 64:
            # This is a simplified implementation - actual table should be used
            m_base = 4 * (C_SRS + 1)
            m_row = np.array([m_base, m_base//2, m_base//4, m_base//8])
            N_row = np.array([1, max(1, C_SRS % 4 + 1), max(1, C_SRS % 3 + 1), max(1, C_SRS % 2 + 1)])
        else:
            # Invalid C_SRS, return default values
            m_row = np.array([4, 4, 4, 4])
            N_row = np.ones(4, dtype=int)

        return m_row, N_row

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

    def _calculate_k0(self, n_rrc, fd_shift, N_RB, N_srs_b):
        """
        Calculate frequency domain starting position k_0

        Args:
            n_rrc: Frequency domain position
            fd_shift: Frequency domain shift
            N_RB: Total number of resource blocks
            N_srs_b: SRS bandwidth in resource blocks

        Returns:
            k_0: Starting position in subcarriers
        """
        # Calculate number of SRS subbands
        N_sb = max(1, N_RB // N_srs_b)

        # Calculate frequency domain position according to 3GPP TS 38.211 Section 6.4.1.4.3
        F_b = self.params['bwp_offset'] + int(n_rrc * N_RB / 4) % N_sb
        k_0 = 12 * F_b  # Convert to subcarrier index

        # Apply frequency domain shift
        k_0 = (k_0 + fd_shift) % (N_RB * 12)

        return k_0

    def _calculate_frequency_hopping(self, sym_idx, b_hop, N_srs_b):
        """
        Calculate frequency hopping offset

        Args:
            sym_idx: Symbol index
            b_hop: Frequency hopping parameter
            N_srs_b: SRS bandwidth in resource blocks

        Returns:
            k_hop: Frequency hopping offset in subcarriers
        """
        if b_hop == 0:
            return 0

        # Simplified hopping pattern based on symbol index and b_hop
        hop_pattern = [0, 1, 0, 1]
        hop_index = sym_idx % len(hop_pattern)

        k_hop = hop_pattern[hop_index] * (b_hop * N_srs_b * 12 // 4)
        return k_hop

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

    def _generate_full_slot_signal(self, re_grid, n_slots=1, srs_symbol_positions=None):
        """
        Generate time domain signal for multiple complete slots with SRS symbols placed
        at specific positions according to 3GPP TS 38.211.

        Args:
            re_grid: Resource grid with SRS symbols
            n_slots: Number of slots to generate
            srs_symbol_positions: List of symbol positions within a slot for SRS
                                If None, uses positions from parameter settings

        Returns:
            td_signal: Time domain signal
        """
        p = self.params

        # Get dimensions
        n_subcarriers, n_symb_srs = re_grid.shape

        # Prepare time domain signal
        fft_size = p['fft_size']
        n_symb_slot = 14  # NR slots have 14 symbols

        # Calculate number of samples per slot based on numerology
        mu = p['mu']

        # Calculate CP length based on 3GPP TS 38.211 Section 5.3.1
        if p['cp_type'] == 'normal':
            # Normal CP length calculation
            # First (0th) and middle (7th) symbols in each slot: N_CP,l = 144·2^(-μ) + 16·2^(-μ) = 160·2^(-μ)
            # Other symbols: N_CP,l = 144·2^(-μ)
            cp_length_long = int((160 * (2**(-mu))) * (fft_size / 2048))
            cp_length_short = int((144 * (2**(-mu))) * (fft_size / 2048))
        else:  # extended CP
            # Extended CP length calculation
            # All symbols: N_CP,l = 512·2^(-μ)
            cp_length_long = cp_length_short = int((512 * (2**(-mu))) * (fft_size / 2048))

        # Calculate total symbol length
        sym_len_long = fft_size + cp_length_long
        sym_len_short = fft_size + cp_length_short

        # Calculate length for a full slot
        slot_length = 2 * sym_len_long + 12 * sym_len_short

        # If SRS symbol positions not provided, determine from parameters
        if srs_symbol_positions is None:
            l_offset = p.get('start_pos', p.get('sym_offset', self.default_params.get('sym_offset', 0)))
            l0 = 14 - 1 - l_offset
            srs_symbol_positions = [l0 - i for i in range(n_symb_srs)]

        # Make sure it's a list or array
        srs_symbol_positions = np.array(srs_symbol_positions)

        # Calculate total buffer size for all slots
        total_length = n_slots * slot_length

        # Create buffer for the entire signal
        td_signal = np.zeros(total_length, dtype=np.complex128)

        # For each slot
        for slot_idx in range(n_slots):
            slot_offset = slot_idx * slot_length

            # For each symbol in the slot
            for sym_idx in range(n_symb_slot):
                # Determine if this is an SRS symbol
                srs_match_indices = np.where(srs_symbol_positions == sym_idx)[0]
                is_srs_symbol = len(srs_match_indices) > 0

                # Calculate position within the slot
                sym_offset = 0
                for prev_sym in range(sym_idx):
                    uses_long_cp = (prev_sym == 0 or prev_sym == 7)
                    sym_offset += sym_len_long if uses_long_cp else sym_len_short

                # Determine if this symbol uses long CP
                uses_long_cp = (sym_idx == 0 or sym_idx == 7)
                cp_length = cp_length_long if uses_long_cp else cp_length_short

                # Generate OFDM symbol
                padded_data = np.zeros(fft_size, dtype=np.complex128)

                if is_srs_symbol:
                    # For SRS symbols, use the data from re_grid
                    srs_re_idx = srs_match_indices[0]
                    if srs_re_idx < n_symb_srs:
                        symbol_data = re_grid[:, srs_re_idx]

                        # Place data in the center of FFT (DC in the middle)
                        dc_idx = fft_size // 2
                        start_idx = dc_idx - n_subcarriers // 2
                        padded_data[start_idx:start_idx + n_subcarriers] = symbol_data

                # Shift to standard FFT format (DC at index 0)
                padded_data = np.fft.fftshift(padded_data)

                # IFFT to get time domain signal
                ifft_data = np.fft.ifft(padded_data) * np.sqrt(fft_size)

                # Add cyclic prefix
                cp = ifft_data[-cp_length:]

                # Append symbol with CP
                symbol_with_cp = np.concatenate((cp, ifft_data))

                # Calculate position in output buffer
                current_offset = slot_offset + sym_offset
                sym_length = len(symbol_with_cp)
                end_offset = current_offset + sym_length

                # Place in buffer
                if end_offset <= len(td_signal):
                    td_signal[current_offset:end_offset] = symbol_with_cp

        return td_signal

    def _generate_time_domain_signal(self, re_grid, symbol_positions=None):
        """
        Generate time domain signal from resource grid using IFFT according to 3GPP TS 38.211

        Args:
            re_grid: Resource grid with SRS symbols
            symbol_positions: List of symbol positions within a slot (e.g. [10, 9] for SRS)
                              If None, assumes sequential positions starting from 0

        Returns:
            td_signal: Time domain signal
        """
        p = self.params

        # Get dimensions
        n_subcarriers, n_symb_srs = re_grid.shape

        # Prepare time domain signal
        fft_size = p['fft_size']

        # Calculate number of samples per slot based on numerology
        mu = p['mu']
        n_symb_slot = 14  # NR slots have 14 symbols

        # Calculate CP length based on 3GPP TS 38.211 Section 5.3.1
        if p['cp_type'] == 'normal':
            # Normal CP length calculation
            # First (0th) and middle (7th) symbols in each slot: N_CP,l = 144·2^(-μ) + 16·2^(-μ) = 160·2^(-μ)
            # Other symbols: N_CP,l = 144·2^(-μ)
            cp_length_long = int((160 * (2**(-mu))) * (fft_size / 2048))
            cp_length_short = int((144 * (2**(-mu))) * (fft_size / 2048))
        else:  # extended CP
            # Extended CP length calculation
            # All symbols: N_CP,l = 512·2^(-μ)
            cp_length_long = cp_length_short = int((512 * (2**(-mu))) * (fft_size / 2048))

        # Calculate total symbol length for planning purposes
        sym_len_long = fft_size + cp_length_long
        sym_len_short = fft_size + cp_length_short

        # If symbol positions not provided, use sequential positions
        if symbol_positions is None:
            # Get the symbol positions from l_offset
            l_offset = p.get('start_pos', p.get('sym_offset', self.default_params.get('sym_offset', 0)))
            l0 = 14 - 1 - l_offset
            symbol_positions = [l0 - i for i in range(n_symb_srs)]

        # Convert to array for easier handling
        symbol_positions = np.array(symbol_positions)

        # Determine which slot the symbols are in (for multi-slot signals)
        slot_indices = symbol_positions // n_symb_slot
        unique_slots = np.unique(slot_indices)
        num_slots = len(unique_slots)

        # Calculate buffer size for the entire signal across all needed slots
        total_length = 0
        for slot_idx in unique_slots:
            # Calculate symbols in this slot
            symbols_in_slot = [pos % n_symb_slot for pos in symbol_positions if pos // n_symb_slot == slot_idx]

            # For a full slot
            if len(symbols_in_slot) == n_symb_slot:
                # A full slot has 2 long CP symbols and 12 short CP symbols
                total_length += 2 * sym_len_long + 12 * sym_len_short
            else:
                # For a partial slot, calculate exactly what we need
                for sym_pos in symbols_in_slot:
                    uses_long_cp = (sym_pos == 0 or sym_pos == 7)
                    sym_length = sym_len_long if uses_long_cp else sym_len_short
                    total_length += sym_length

        # Ensure we have enough space for all our symbols
        td_signal = np.zeros(total_length, dtype=np.complex128)

        # Process each symbol
        offset = 0
        for re_idx, slot_sym_idx in enumerate(symbol_positions):
            slot_idx = slot_sym_idx // n_symb_slot
            sym_pos = slot_sym_idx % n_symb_slot

            # Calculate the offset for this slot
            slot_offset = 0
            for prev_slot in range(slot_idx):
                slot_offset += 2 * sym_len_long + 12 * sym_len_short

            # Calculate the offset within this slot
            sym_offset = 0
            for prev_sym in range(sym_pos):
                uses_long_cp = (prev_sym == 0 or prev_sym == 7)
                sym_offset += sym_len_long if uses_long_cp else sym_len_short

            # Get symbol data and perform zero padding
            symbol_data = re_grid[:, re_idx]
            padded_data = np.zeros(fft_size, dtype=np.complex128)

            # Place data in the center of FFT (DC in the middle)
            dc_idx = fft_size // 2
            start_idx = dc_idx - n_subcarriers // 2
            padded_data[start_idx:start_idx + n_subcarriers] = symbol_data

            # Shift to standard FFT format (DC at index 0)
            padded_data = np.fft.fftshift(padded_data)

            # IFFT to get time domain signal
            ifft_data = np.fft.ifft(padded_data) * np.sqrt(fft_size)

            # Determine if this symbol should have a long CP (0th or 7th in slot)
            uses_long_cp = (sym_pos == 0 or sym_pos == 7)
            cp_length = cp_length_long if uses_long_cp else cp_length_short

            # Add cyclic prefix
            cp = ifft_data[-cp_length:]

            # Append symbol with CP
            symbol_with_cp = np.concatenate((cp, ifft_data))

            # Calculate position in output buffer
            current_offset = slot_offset + sym_offset
            sym_length = len(symbol_with_cp)
            end_offset = current_offset + sym_length

            # Ensure we don't exceed buffer length
            if end_offset <= len(td_signal):
                td_signal[current_offset:end_offset] = symbol_with_cp

        # Trim any unused buffer space
        nonzero_indices = np.nonzero(td_signal)[0]
        if len(nonzero_indices) > 0:
            last_nonzero = nonzero_indices[-1] + 1
            td_signal = td_signal[:last_nonzero]

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

    def export_binary_resource_grid(self, resource_grid, file_path, params, scale_factor=2**12, endian='big',
                                    excel_path=None, num_symbols=14):
        """
        Export a resource grid to a binary file in int16 format and to Excel with proper formatting,
        positioning the SRS according to the symbol offset parameter.

        Args:
            resource_grid: Complex array representing the SRS resource grid
            file_path: Path to save the binary file
            params: Dictionary of SRS parameters
            scale_factor: Scaling factor to apply before conversion to int16 (default: 2^12)
            endian: Endianness ('little' or 'big')
            excel_path: Path to save Excel file (if None, uses file_path with .xlsx extension)
            num_symbols: Number of symbols in a subframe (default: 14 for NR)

        Returns:
            True if export was successful
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        # Extract symbol offset from params (using both possible naming conventions)
        l_offset = 14 - 1 - params.get('sym_offset', params.get('start_pos', 0))
        n_syms = params.get('n_syms', 1)

        # Scale the grid if needed
        if scale_factor != 1:
            scaled_grid = resource_grid * scale_factor
        else:
            scaled_grid = resource_grid.copy()

        # Round values to integers
        scaled_grid = np.round(scaled_grid)

        # Get real and imaginary parts for binary export
        real_part = np.real(scaled_grid).flatten().astype(np.int16)
        imag_part = np.imag(scaled_grid).flatten().astype(np.int16)

        # Interleave real and imaginary parts
        interleaved = np.empty(real_part.size + imag_part.size, dtype=np.int16)
        interleaved[0::2] = real_part
        interleaved[1::2] = imag_part

        # Set default Excel path if not specified
        if excel_path is None:
            excel_path = os.path.splitext(file_path)[0] + ".xlsx"

        # Get dimensions of the resource grid
        num_subcarriers, num_available_symbols = scaled_grid.shape

        # Create a full-sized grid with 14 symbols (all zeros initially)
        full_grid = np.zeros((num_subcarriers, num_symbols), dtype=np.complex128)

        # Place the SRS grid at the correct symbol positions based on l_offset
        # Ensure we don't exceed grid boundaries
        if l_offset + num_available_symbols <= num_symbols:
            full_grid[:, l_offset:l_offset+num_available_symbols] = scaled_grid
        else:
            # If it would exceed, truncate or handle as needed
            available_symbols = num_symbols - l_offset
            full_grid[:, l_offset:] = scaled_grid[:, :available_symbols]
            print(f"Warning: SRS grid truncated - requested {num_available_symbols} symbols starting at position {l_offset}, but only {available_symbols} fit in the slot")

        # Create DataFrame with alternating columns for real and imaginary parts
        columns = []
        for i in range(num_symbols):
            columns.append(f'Symbol_{i}_Real')
            columns.append(f'Symbol_{i}_Imag')

        # Create data with alternating real and imaginary values
        data = np.zeros((num_subcarriers, num_symbols * 2), dtype=np.int16)
        for i in range(num_symbols):
            data[:, i*2] = np.real(full_grid[:, i]).astype(np.int16)      # Real part
            data[:, i*2+1] = np.imag(full_grid[:, i]).astype(np.int16)    # Imaginary part

        # Create Excel with real and imaginary values side by side
        df = pd.DataFrame(data, columns=columns)

        # Add parameter summary for reference
        param_df = pd.DataFrame([
            {'Parameter': 'PRB Number', 'Value': params.get('prb_num', 'N/A')},
            {'Parameter': 'Scrambling ID', 'Value': params.get('n_id', 'N/A')},
            {'Parameter': 'Comb Size', 'Value': params.get('comb', 'N/A')},
            {'Parameter': 'Comb Offset', 'Value': params.get('comb_offset', 'N/A')},
            {'Parameter': 'Symbol Offset', 'Value': l_offset},
            {'Parameter': 'Number of Symbols', 'Value': n_syms},
            {'Parameter': 'FFT Size', 'Value': params.get('fft_size', 'N/A')},
            {'Parameter': 'Subcarrier Spacing', 'Value': f"{15 * (2 ** params.get('mu', 0))} kHz"},
        ])

        # Create a multi-level Excel writer
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Write the main DataFrame with alternating real/imag columns
            df.to_excel(writer, sheet_name='Grid_Data', index_label='Subcarrier')

            # Write parameter summary
            param_df.to_excel(writer, sheet_name='Parameters', index=False)

            # Also export separate sheets for real and imaginary parts for clarity
            df_real = pd.DataFrame(np.real(full_grid).astype(np.int16),
                                  columns=[f'Symbol_{i}' for i in range(num_symbols)])
            df_imag = pd.DataFrame(np.imag(full_grid).astype(np.int16),
                                  columns=[f'Symbol_{i}' for i in range(num_symbols)])

            df_real.to_excel(writer, sheet_name='Real_Part', index_label='Subcarrier')
            df_imag.to_excel(writer, sheet_name='Imaginary_Part', index_label='Subcarrier')

            # Also export the original interleaved format
            pd.DataFrame({'Real': interleaved[0::2],
                        'Imaginary': interleaved[1::2]}).to_excel(writer,
                                                                  sheet_name='Interleaved',
                                                                  index=False)

        # Set endianness for binary export
        if endian == 'big':
            interleaved = interleaved.astype('>i2')
        else:  # little endian
            interleaved = interleaved.astype('<i2')

        # Write to binary file
        with open(file_path, 'wb') as f:
            f.write(interleaved.tobytes())

        return True



class SRS_Analyzer:
    """
    Analyzer for 5G NR Sounding Reference Signals (SRS)
    with support for multi-slot processing
    """

    def __init__(self, params=None):
        """
        Initialize SRS analyzer with default or provided parameters

        Args:
            params: Dictionary of SRS parameters
        """
        # Set default parameters
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
            'hop_mod': 0,          # Hopping mode (0: Neither, 1: Group, 2: Sequence)
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

        # Update with provided parameters if any
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)

        # Create SRS generator for reference signal
        self.srs_gen = SRS_Generator(self.params)

    def demodulate_srs(self, td_signal, n_slots=1, noise_power=0.01):
        """
        Demodulate SRS signal and perform measurements

        Args:
            td_signal: Time domain signal
            n_slots: Number of slots in the signal
            noise_power: Noise power for simulation

        Returns:
            re_grid: Demodulated resource grid
            measurements: Dictionary with SRS measurements
        """
        p = self.params

        # Generate reference SRS with the same parameters
        ref_grid, ref_td_signal, srs_info = self.srs_gen.generate_srs()

        # Extract SRS symbol positions within each slot
        srs_symbol_positions = srs_info['l_sym']

        # Add simulated noise if specified
        if noise_power > 0:
            noisy_signal = td_signal + np.sqrt(noise_power/2) * (
                np.random.normal(0, 1, len(td_signal)) +
                1j * np.random.normal(0, 1, len(td_signal))
            )
        else:
            noisy_signal = td_signal

        # OFDM demodulation - extract only SRS symbols
        re_grid = self._ofdm_demodulation(noisy_signal, n_slots, srs_symbol_positions)

        # Extract SRS subcarriers
        srs_sc_indices = self._get_srs_subcarrier_indices()

        # Initialize measurement results
        snr_est = []
        snr_prb_est = []
        timing_est = 0

        # Store PRB-level SNR as numpy arrays, not lists
        all_prb_snr = []

        # Process each slot
        for slot_idx in range(n_slots):
            # Extract the slot's resource elements
            slot_re = re_grid[:, slot_idx*p.get('n_syms', 1):(slot_idx+1)*p.get('n_syms', 1)]

            # Measure SNR for this slot
            slot_snr = self._estimate_snr(
                slot_re,
                ref_grid,
                srs_sc_indices
            )
            snr_est.append(slot_snr)

            # Measure SNR per PRB if enabled
            if p.get('snr_prb_en', False):
                slot_snr_prb = self._estimate_snr_per_prb(
                    slot_re,
                    ref_grid
                )
                # Convert to numpy array if it's a list
                if isinstance(slot_snr_prb, list):
                    slot_snr_prb = np.array(slot_snr_prb)
                all_prb_snr.append(slot_snr_prb)

        # Convert list of PRB SNR arrays to a single 2D numpy array if present
        if all_prb_snr:
            snr_prb_est = np.array(all_prb_snr)

        # Estimate timing alignment if enabled (using the entire signal)
        if p.get('ta_est_en', False):
            timing_est = self._estimate_timing_alignment(td_signal, ref_td_signal)

        # Pack measurements
        measurements = {
            'snr_est': np.mean(snr_est) if snr_est else 0,  # Average across slots
            'snr_per_slot': np.array(snr_est) if snr_est else np.array([]),
            'timing_est': timing_est,
            'snr_prb': snr_prb_est if all_prb_snr else None,
            # Add reference parameters from the updated generate_srs function
            'ref_params': {
                'n_ports': p.get('n_ports', 1),
                'tx_port': p.get('tx_port', p.get('p_tx_port', 1000)),
                'n_id': p.get('n_id', 500),
                'comb_size': p.get('comb', 2),
                'n_cs_i': srs_info.get('n_cs_i', 0),
                'u_seq_grp': srs_info.get('u_seq_grp', 0),
                'fh_enabled': srs_info.get('fh_enabled', False)
            }
        }

        return re_grid, measurements


    def _ofdm_demodulation(self, td_signal, n_slots=1, srs_symbol_positions=None):
        """
        Perform OFDM demodulation on time-domain signal with multi-slot support.

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

        # Calculate CP lengths based on numerology
        if p['cp_type'] == 'normal':
            # Normal CP length calculation
            cp_length_long = int((160 * (2**(-mu))) * (fft_size / 2048))
            cp_length_short = int((144 * (2**(-mu))) * (fft_size / 2048))
        else:  # Extended CP
            # Extended CP length calculation
            cp_length_long = cp_length_short = int((512 * (2**(-mu))) * (fft_size / 2048))

        # Calculate symbol lengths
        sym_len_long = fft_size + cp_length_long
        sym_len_short = fft_size + cp_length_short

        # Calculate slot length
        slot_length = 2 * sym_len_long + 12 * sym_len_short

        # Initialize resource grid for subcarriers and SRS symbols from all slots
        re_grid = np.zeros((n_subcarriers, n_slots * n_symbols), dtype=np.complex128)

        # Process each slot
        for slot_idx in range(n_slots):
            slot_offset = slot_idx * slot_length

            # Process only SRS symbols
            for sym_pos_idx, sym_pos in enumerate(srs_symbol_positions):
                # Calculate position within the slot for this symbol
                sym_offset = 0
                for prev_sym in range(sym_pos):
                    uses_long_cp = (prev_sym == 0 or prev_sym == 7)
                    sym_offset += sym_len_long if uses_long_cp else sym_len_short

                # Determine if this symbol uses long CP
                uses_long_cp = (sym_pos == 0 or sym_pos == 7)
                cp_length = cp_length_long if uses_long_cp else cp_length_short

                # Calculate exact position in time domain signal
                symbol_start = slot_offset + sym_offset

                # Extract symbol without CP if we have enough samples
                if symbol_start + cp_length + fft_size <= len(td_signal):
                    symbol_without_cp = td_signal[symbol_start + cp_length:symbol_start + cp_length + fft_size]

                    # Perform FFT to transform to frequency domain
                    fft_data = np.fft.fft(symbol_without_cp) / np.sqrt(fft_size)

                    # Shift FFT to center DC subcarrier
                    fft_data = np.fft.fftshift(fft_data)

                    # Extract the subcarriers for the resource grid
                    dc_idx = fft_size // 2
                    start_idx = dc_idx - n_subcarriers // 2
                    end_idx = start_idx + n_subcarriers

                    # Store in resource grid
                    re_grid_idx = slot_idx * n_symbols + sym_pos_idx
                    if start_idx >= 0 and end_idx <= fft_size and re_grid_idx < re_grid.shape[1]:
                        re_grid[:, re_grid_idx] = fft_data[start_idx:end_idx]
                else:
                    warnings.warn(f"Insufficient samples for symbol {sym_pos} in slot {slot_idx}")

        return re_grid

    def _get_srs_subcarrier_indices(self):
        """
        Get indices of subcarriers containing SRS

        Returns:
            srs_indices: Array of subcarrier indices
        """
        p = self.params

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

    def _estimate_snr(self, rx_grid, ref_grid, srs_indices):
        """
        Estimate SNR based on received and reference grids

        Args:
            rx_grid: Received resource grid
            ref_grid: Reference resource grid
            srs_indices: Indices of SRS subcarriers

        Returns:
            snr_est: Estimated SNR in dB
        """
        # Make sure we have a valid signal
        if rx_grid.size == 0 or ref_grid.size == 0:
            return 0

        # Extract SRS from grids - use first SRS symbol
        rx_srs = rx_grid[srs_indices, 0]
        ref_srs = ref_grid[srs_indices, 0]

        # Skip if reference is empty
        if np.sum(np.abs(ref_srs)) == 0:
            return 0

        # Calculate signal power (use reference signal power)
        signal_power = np.mean(np.abs(ref_srs)**2)

        # Scale received signal to compensate for channel effects
        scale_factor = np.sum(rx_srs * np.conj(ref_srs)) / np.sum(np.abs(ref_srs)**2)
        rx_srs_scaled = rx_srs / scale_factor

        # Calculate noise power
        noise = rx_srs_scaled - ref_srs
        noise_power = np.mean(np.abs(noise)**2)

        # Calculate SNR
        if noise_power > 0:
            snr = signal_power / noise_power
            snr_db = 10 * np.log10(snr)
        else:
            snr_db = 100  # Very high SNR

        return snr_db

    def _estimate_snr_per_prb(self, rx_grid, ref_grid):
        """
        Estimate SNR for each PRB

        Args:
            rx_grid: Received resource grid
            ref_grid: Reference resource grid

        Returns:
            snr_prb: Array of SNR per PRB in dB
        """
        p = self.params

        # Get parameters
        N_RB = p['prb_num']
        comb = p['comb']
        comb_offset = p['comb_offset']

        # Calculate SNR for each PRB
        snr_prb = np.zeros(N_RB)

        for rb in range(N_RB):
            # Get indices for this PRB
            indices = []
            for sc in range(12 // comb):
                k = rb * 12 + sc * comb + comb_offset
                if k < rx_grid.shape[0]:
                    indices.append(k)

            if indices:
                # Extract data from first symbol containing SRS
                rx_data = rx_grid[indices, 0]
                ref_data = ref_grid[indices, 0]

                # Skip if reference is zero
                if np.sum(np.abs(ref_data)) > 0:
                    # Calculate signal power
                    signal_power = np.mean(np.abs(ref_data)**2)

                    # Scale received signal
                    scale_factor = np.sum(rx_data * np.conj(ref_data)) / np.sum(np.abs(ref_data)**2)
                    rx_data_scaled = rx_data / scale_factor

                    # Calculate noise power
                    noise = rx_data_scaled - ref_data
                    noise_power = np.mean(np.abs(noise)**2)

                    # Calculate SNR
                    if noise_power > 0:
                        snr = signal_power / noise_power
                        snr_prb[rb] = 10 * np.log10(snr)
                    else:
                        snr_prb[rb] = 100  # Very high SNR

        return snr_prb

    def _estimate_timing_alignment(self, td_signal, ref_signal=None):
        """
        Estimate timing alignment using cross-correlation

        Args:
            td_signal: Time domain signal
            ref_signal: Reference time domain signal (if None, will generate one)

        Returns:
            ta_est: Estimated timing alignment in seconds
        """
        # Generate reference signal if not provided
        if ref_signal is None:
            _, ref_signal, _ = self.srs_gen.generate_srs()

        # Make sure signals are not empty
        if len(td_signal) == 0 or len(ref_signal) == 0:
            return 0

        # Limit reference signal length to received signal length if needed
        if len(ref_signal) > len(td_signal):
            ref_signal = ref_signal[:len(td_signal)]

        # Perform cross-correlation using scipy.signal
        correlation = signal.correlate(td_signal, ref_signal, mode='same')
        correlation_mag = np.abs(correlation)

        # Find peak
        peak_idx = np.argmax(correlation_mag)
        center_idx = len(correlation) // 2

        # Calculate timing offset in samples
        offset_samples = peak_idx - center_idx

        # Convert to seconds
        subcarrier_spacing = 15000 * (2 ** self.params['mu'])  # SCS in Hz
        sample_rate = self.params['fft_size'] * subcarrier_spacing  # Sample rate in Hz
        ta_est = offset_samples / sample_rate

        return ta_est

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
    Plot SRS generation and analysis results.

    Args:
        re_grid (ndarray): Resource grid with SRS mapping for 1 slot.
        td_signal (ndarray): Time domain signal for 10 ms.
        measurements (dict): Analysis measurements.
        num_slots (int, optional): Number of slots within the 10 ms signal.
                                  If None, will be automatically determined from numerology.
    """
    # Auto-detect numerology and slots if not specified
    if num_slots is None:
        # Estimate numerology based on signal length
        signal_len = len(td_signal)
        if signal_len >= 300000:  # ~307,200 samples for 30kHz SCS
            num_slots = 20  # μ=1 (30kHz SCS)
        else:
            num_slots = 10  # μ=0 (15kHz SCS)

        print(f"Auto-detected {num_slots} slots in signal ({signal_len} samples)")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot resource grid magnitude (single slot)
    im = axes[0, 0].imshow(np.abs(re_grid), aspect='auto', cmap='viridis')
    axes[0, 0].set_title('Resource Grid Magnitude (1 Slot)')
    axes[0, 0].set_xlabel('OFDM Symbols')
    axes[0, 0].set_ylabel('Subcarriers')
    plt.colorbar(im, ax=axes[0, 0])

    # Plot time domain signal (full signal duration)
    sample_idx = np.arange(len(td_signal))
    time_ms = sample_idx / (len(td_signal) / 10)  # Convert to ms assuming 10ms total

    axes[0, 1].plot(time_ms, np.abs(td_signal))
    axes[0, 1].set_title(f'Time Domain Signal ({len(td_signal)} samples)')
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Amplitude')

    # Add markers for slot boundaries
    for slot in range(1, num_slots):
        slot_time_ms = slot * 10 / num_slots
        axes[0, 1].axvline(x=slot_time_ms, color='r', linestyle='--', alpha=0.3)

    # Plot SNR per PRB if available (supporting both old and new key names)
    # Check for new key name first, then fall back to old key name
    snr_prb_data = None
    if 'snr_prb' in measurements and measurements['snr_prb'] is not None:
        snr_prb_data = measurements['snr_prb']
    elif 'snr_prb_est' in measurements and measurements['snr_prb_est'] is not None:
        snr_prb_data = measurements['snr_prb_est']

    if snr_prb_data is not None:
        # Handle the case where snr_prb is a list, numpy array, or has nested structure
        if isinstance(snr_prb_data, list):
            # Convert to numpy array for easier manipulation
            snr_prb_array = np.array(snr_prb_data)
        else:
            # It's already a numpy array
            snr_prb_array = snr_prb_data

        # Process based on dimensions
        if snr_prb_array.ndim > 1:
            # It's a multi-slot array - average across slots
            snr_prb_avg = np.mean(snr_prb_array, axis=0)
            axes[1, 0].bar(range(len(snr_prb_avg)), snr_prb_avg)
            axes[1, 0].set_title(f'Average SNR per PRB (Avg: {measurements.get("snr_est", "N/A"):.2f} dB)')
        else:
            # It's a single slot array
            axes[1, 0].bar(range(len(snr_prb_array)), snr_prb_array)
            axes[1, 0].set_title(f'SNR per PRB (Avg: {measurements.get("snr_est", "N/A"):.2f} dB)')

        axes[1, 0].set_xlabel('PRB Index')
        axes[1, 0].set_ylabel('SNR (dB)')
        axes[1, 0].grid(True)
    else:
        axes[1, 0].set_title('SNR per PRB (Not Available)')

    # Plot the frequency spectrum of the complete signal
    # Using Welch's method for better visualization of the spectrum
    from scipy import signal as sig

    # Full signal spectrum (using Welch's method for smoother visualization)
    f, Pxx = sig.welch(td_signal, fs=len(td_signal)/10e-3, nperseg=1024,
                      scaling='spectrum', return_onesided=False)
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)

    # Convert to MHz for readability
    f_MHz = f / 1e6

    axes[1, 1].semilogy(f_MHz, Pxx)
    axes[1, 1].set_title('Frequency Domain (Full 10ms Signal)')
    axes[1, 1].set_xlabel('Frequency (MHz)')
    axes[1, 1].set_ylabel('Power Spectral Density')
    axes[1, 1].grid(True)

    # Add overall information text box - supporting both old and new key names
    timing_value = None
    if 'timing_est' in measurements:
        timing_value = measurements['timing_est']
    elif 'ta_est' in measurements:
        timing_value = measurements['ta_est']

    # Reference parameters if available
    ref_params = measurements.get('ref_params', {})

    info_text = (
        f"Signal length: {len(td_signal)} samples\n"
        f"Duration: {len(td_signal)/30.72e6*1000:.2f} ms\n"  # Assuming 30.72MHz sampling rate
        f"SNR: {measurements.get('snr_est', 'N/A'):.2f} dB\n"
        f"Timing: {timing_value*1e9 if timing_value is not None else 'N/A'} ns\n"
        f"n_id: {ref_params.get('n_id', 'N/A')}\n"
        f"port: {ref_params.get('tx_port', 'N/A')}\n"
        f"comb: {ref_params.get('comb_size', 'N/A')}"
    )

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
    axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes,
                   fontsize=9, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

    return fig


def generate_srs_example():
    """Example function to demonstrate SRS generation and analysis"""
    # Define parameters with both naming conventions for compatibility
    params = {
        # Core parameters
        'prb_num': 25,         # Number of PRBs
        'n_id': 500,           # Scrambling ID
        'comb': 2,             # SRS comb size
        'comb_offset': 1,      # Comb offset

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

        'sym_offset': 4,       # Symbol offset (original naming)
        'start_pos': 4,        # Symbol offset (alternative naming)

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
        'fft_size': 1024,      # FFT size

        # Analysis parameters
        'snr_prb_en': True,    # Enable SNR per PRB estimation
        'ta_est_en': True,     # Enable timing alignment estimation
        'n_slots': 1           # Total Slots to generate
    }

    # Create SRS generator
    srs_gen = SRS_Generator(params)

    # Generate SRS signal
    re_grid, td_signal, srs_info = srs_gen.generate_srs()

    # Export to binary file
    srs_gen.export_binary(td_signal, "srs_buffer.bin", scale=2**14, endian='little', metadata=srs_info)
    srs_gen.export_binary_resource_grid(re_grid, "grid_in.bin", params, scale_factor=2**14, endian='little')

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

