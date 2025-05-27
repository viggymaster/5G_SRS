import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import struct
class Numerology(Enum):
    """Numerology values for 5G NR"""
    MU0 = 0  # 15 kHz subcarrier spacing
    MU1 = 1  # 30 kHz subcarrier spacing
@dataclass
class SSBParameters:
    """Comprehensive SSB parameters for 5G NR"""
    # Basic numerology parameters
    numerology: Numerology = Numerology.MU1  # μ=0 or μ=1
    subcarrier_spacing_khz: int = 30  # 15 or 30 kHz
    
    # Cell and SSB configuration
    n_id_1: int = 0  # Physical layer cell ID group (0-335)
    n_id_2: int = 0  # Physical layer cell ID (0-2)
    ssb_index: int = 0  # SS/PBCH block index (0-7/0-3 based on Lmax)
    ssb_pattern: str = "Case A"  # SSB pattern (Case A/B/C)
    k_ssb: int = 0  # Frequency domain offset
    
    # System information
    sfn: int = 0  # System frame number (0-1023)
    hrf: int = 0  # Half radio frame bit (0/1)
    dmrs_type_a_position: int = 2  # Position of DM-RS (2/3)
    pdcch_config_sib1: int = 0  # PDCCH configuration for SIB1
    cell_barred: bool = False  # Cell barred status
    intra_freq_reselection: bool = False  # Intra-frequency reselection
    
    # Resource grid parameters
    n_rb_ssb: int = 20  # Number of resource blocks for SSB (20)
    l_max: int = 8  # Maximum number of SSB candidates (4/8/64)
    
    # Frequency parameters
    center_frequency_mhz: float = 3500.0  # Center frequency
    bandwidth_mhz: float = 20.0  # Channel bandwidth
    
    # Sampling parameters
    sample_rate: float = 30.72e6  # Sampling rate (Hz)
    use_fixed_point: bool = True  # Use fixed-point processing
    
    @property
    def cell_id(self) -> int:
        """Calculate cell ID from n_id_1 and n_id_2"""
        return 3 * self.n_id_1 + self.n_id_2
    
    @property
    def fft_size(self) -> int:
        """Calculate FFT size based on numerology"""
        # For 5G NR with 20 RBs for SSB
        if self.numerology == Numerology.MU0:
            return 1024  # For 15 kHz SCS
        else:
            return 512   # For 30 kHz SCS
    
    @property
    def cp_length_normal(self) -> int:
        """Calculate normal CP length for the current numerology"""
        # Based on 3GPP specifications for normal CP
        if self.numerology == Numerology.MU0:
            return 144  # For 15 kHz SCS <sup data-citation="36"><a href="https://publications.eai.eu/index.php/mca/article/download/6933/3549/20167#:~:text=For%20%CE%BC%3D0%2C%20the,144%20samples." target="_blank" title="publications.eai.eu">36</a></sup>
        else:
            return 72   # For 30 kHz SCS <sup data-citation="3"><a href="https://www.researchgate.net/publication/342221540_On_the_design_details_of_SSPBCH_Signal_Generation_and_PRACH_in_5G-NR#:~:text=In%205G-NR%20the,and%20expected%20conditions." target="_blank" title="www.researchgate.net">3</a></sup>
    
    @property
    def cp_length_first_symbol(self) -> int:
        """Calculate first symbol CP length"""
        # First symbol CP is longer to account for slot boundary alignment
        if self.numerology == Numerology.MU0:
            return 160  # For 15 kHz SCS
        else:
            return 80   # For 30 kHz SCS
    
    @property
    def symbol_duration(self) -> int:
        """Total symbol duration including CP"""
        if self.numerology == Numerology.MU0:
            return self.fft_size + self.cp_length_normal  # For normal symbols
        else:
            return self.fft_size + self.cp_length_normal
			
class OFDMModulator:
    """OFDM modulator with proper CP handling for 5G NR"""
    
    def __init__(self, params: SSBParameters):
        self.params = params
        self.fft_size = params.fft_size
    
    def modulate(self, resource_grid: np.ndarray) -> np.ndarray:
        """
        OFDM modulate resource grid to time domain with proper CP
        
        Args:
            resource_grid: Resource grid with shape [num_subcarriers, num_symbols]
            
        Returns:
            Time domain signal with proper CP
        """
        num_subcarriers, num_symbols = resource_grid.shape
        
        # Calculate output signal length with CPs
        output_length = 0
        for symbol_idx in range(num_symbols):
            cp_length = self._get_cp_length(symbol_idx)
            output_length += self.fft_size + cp_length
        
        output_signal = np.zeros(output_length, dtype=complex)
        
        # Process each OFDM symbol
        current_idx = 0
        for symbol_idx in range(num_symbols):
            # Get current symbol from resource grid
            symbol_data = resource_grid[:, symbol_idx]
            
            # Zero pad to FFT size
            padded_data = np.zeros(self.fft_size, dtype=complex)
            padded_data[:num_subcarriers//2] = symbol_data[num_subcarriers//2:]
            padded_data[-num_subcarriers//2:] = symbol_data[:num_subcarriers//2]
            
            # IFFT operation
            time_domain = np.fft.ifft(padded_data) * np.sqrt(self.fft_size)
            
            # Calculate CP length for this symbol
            cp_length = self._get_cp_length(symbol_idx)
            
            # Add CP
            output_signal[current_idx:current_idx + cp_length] = time_domain[-cp_length:]
            output_signal[current_idx + cp_length:current_idx + cp_length + self.fft_size] = time_domain
            
            # Update index
            current_idx += cp_length + self.fft_size
        
        return output_signal
    
    def _get_cp_length(self, symbol_idx: int) -> int:
        """
        Get the correct CP length for a given symbol index
        
        The first symbol in a slot has a longer CP to ensure slot alignment.
        For SSB, we need to handle 4 symbols correctly.
        
        Args:
            symbol_idx: OFDM symbol index
            
        Returns:
            CP length in samples
        """
        # For SSB, the first symbol (PSS) has a longer CP
        if symbol_idx == 0:
            return self.params.cp_length_first_symbol
        else:
            return self.params.cp_length_normal

class SSBGenerator:
    """
    Complete SSB Generator for 5G NR with μ=0 and μ=1 support
    
    This class implements generation of:
    - PSS/SSS sequences
    - PBCH payload with MIB encoding
    - PBCH DMRS generation
    - Resource grid mapping
    - OFDM modulation with proper CP handling
    - Fixed-point conversion to int16
    """
    
    def __init__(self, params: SSBParameters):
        self.params = params
        self.ofdm_modulator = OFDMModulator(params)
        self._initialize_sequences()
    
    def _initialize_sequences(self):
        """Initialize PSS and SSS sequences"""
        # Generate PSS sequence based on n_id_2
        self.pss_sequence = self._generate_pss_sequence(self.params.n_id_2)
        
        # Generate SSS sequence based on n_id_1 and n_id_2
        self.sss_sequence = self._generate_sss_sequence(
            self.params.n_id_1, self.params.n_id_2
        )
    
    def _generate_pss_sequence(self, n_id_2: int) -> np.ndarray:
        """
        Generate PSS sequence for given n_id_2
        
        Args:
            n_id_2: Physical layer identity (0-2)
            
        Returns:
            127-length PSS sequence
        """
        # Initialize m-sequence
        x = np.zeros(127, dtype=int)
        x[0:7] = [1, 1, 1, 0, 1, 1, 0]  # Initial value
        
        # Generate m-sequence
        for i in range(7, 127):
            x[i] = (x[i-7] + x[i-4]) % 2
        
        # Map from {0,1} to {1,-1}
        d_pss = np.zeros(127, dtype=complex)
        for n in range(127):
            # Different m-sequences based on n_id_2
            m = (n + 43 * n_id_2) % 127
            d_pss[n] = 1 - 2 * x[m]
        
        return d_pss
    
    def _generate_sss_sequence(self, n_id_1: int, n_id_2: int) -> np.ndarray:
        """
        Generate SSS sequence for given n_id_1 and n_id_2
        
        Args:
            n_id_1: Cell ID group (0-335)
            n_id_2: Physical layer identity (0-2)
            
        Returns:
            127-length SSS sequence
        """
        # Initialize m-sequences
        x0 = np.zeros(127, dtype=int)
        x1 = np.zeros(127, dtype=int)
        
        # Initial values
        x0[0:7] = [1, 0, 0, 0, 0, 0, 0]
        x1[0:7] = [1, 0, 0, 0, 0, 0, 0]
        
        # Generate m-sequences
        for i in range(7, 127):
            x0[i] = (x0[i-7] + x0[i-4]) % 2
            x1[i] = (x1[i-7] + x1[i-1]) % 2
        
        # Calculate m0 and m1 based on n_id_1 and n_id_2
        m0 = 15 * (n_id_1 // 112) + 5 * n_id_2
        m1 = n_id_1 % 112
        
        # Generate SSS sequence
        d_sss = np.zeros(127, dtype=complex)
        for n in range(127):
            d_sss[n] = (1 - 2 * x0[(n + m0) % 127]) * (1 - 2 * x1[(n + m1) % 127])
        
        return d_sss
    
    def _generate_pbch_dmrs(self) -> np.ndarray:
        """
        Generate PBCH DMRS sequences
        
        Returns:
            PBCH DMRS sequence (144 elements)
        """
        # Initialize DMRS sequence with cell ID scrambling
        cell_id = self.params.cell_id
        
        # Gold sequence initialization
        x1 = np.zeros(1600, dtype=int)
        x1[0:31] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        # Generate x1 sequence
        for i in range(31, 1600):
            x1[i] = (x1[i-31] + x1[i-28]) % 2
        
        # Initialize x2 with cell ID
        c_init = cell_id
        x2 = np.zeros(1600, dtype=int)
        
        # Convert c_init to binary
        for i in range(31):
            x2[i] = (c_init >> i) & 1
        
        # Generate x2 sequence
        for i in range(31, 1600):
            x2[i] = (x2[i-31] + x2[i-30] + x2[i-29] + x2[i-28]) % 2
        
        # Generate DMRS sequence
        c = np.zeros(1600, dtype=int)
        for i in range(1600):
            c[i] = (x1[i] + x2[i]) % 2
        
        # Map to QPSK symbols (DMRS uses QPSK modulation)
        dmrs_sequence = np.zeros(144, dtype=complex)
        for i in range(144):
            # Each QPSK symbol uses 2 bits
            b0 = c[2*i]
            b1 = c[2*i + 1]
            
            # QPSK mapping
            if b0 == 0 and b1 == 0:
                dmrs_sequence[i] = (1 + 1j) / np.sqrt(2)
            elif b0 == 0 and b1 == 1:
                dmrs_sequence[i] = (1 - 1j) / np.sqrt(2)
            elif b0 == 1 and b1 == 0:
                dmrs_sequence[i] = (-1 + 1j) / np.sqrt(2)
            else:  # b0 == 1 and b1 == 1
                dmrs_sequence[i] = (-1 - 1j) / np.sqrt(2)
        
        return dmrs_sequence
    
    def _generate_mib_payload(self) -> np.ndarray:
        """
        Generate MIB payload
        
        Returns:
            32-bit MIB payload
        """
        # Initialize 24-bit MIB content
        mib = np.zeros(24, dtype=int)
        
        # Extract parameters
        sfn_msb = (self.params.sfn >> 4) & 0x3F  # 6 MSBs of SFN
        scs_index = 1 if self.params.numerology == Numerology.MU1 else 0
        dmrs_position = 1 if self.params.dmrs_type_a_position == 3 else 0
        
        # Set MIB fields based on 3GPP TS 38.331
        # SFN MSB [0:5]
        for i in range(6):
            mib[i] = (sfn_msb >> (5-i)) & 1
        
        # Subcarrier spacing [6]
        mib[6] = scs_index
        
        # SSB subcarrier offset [7:10]
        k_ssb = self.params.k_ssb & 0xF  # 4 bits of k_ssb
        for i in range(4):
            mib[7+i] = (k_ssb >> (3-i)) & 1
        
        # DMRS position [11]
        mib[11] = dmrs_position
        
        # PDCCH config SIB1 [12:19]
        pdcch_config = self.params.pdcch_config_sib1 & 0xFF
        for i in range(8):
            mib[12+i] = (pdcch_config >> (7-i)) & 1
        
        # Cell barred [20]
        mib[20] = 1 if self.params.cell_barred else 0
        
        # Intra-frequency reselection [21]
        mib[21] = 1 if self.params.intra_freq_reselection else 0
        
        # Reserved bits [22:23]
        mib[22:24] = 0
        
        # Explicit part (SFN LSB, HRF, SS Block index)
        explicit_bits = np.zeros(8, dtype=int)
        
        # SFN LSB [0:3]
        sfn_lsb = self.params.sfn & 0xF
        for i in range(4):
            explicit_bits[i] = (sfn_lsb >> (3-i)) & 1
        
        # Half frame bit [4]
        explicit_bits[4] = self.params.hrf
        
        # SS Block index MSB bits [5:7]
        # Format depends on L_max
        if self.params.l_max == 64:
            # 3 MSB bits for L=64
            ssb_idx_msb = self.params.ssb_index >> 3
            for i in range(3):
                explicit_bits[5+i] = (ssb_idx_msb >> (2-i)) & 1
        else:
            # 1 MSB bit for L=4/8
            ssb_idx_msb = self.params.ssb_index >> 2
            explicit_bits[5] = ssb_idx_msb
            # Reserved bits
            explicit_bits[6:8] = 0
        
        # Combine MIB and explicit part
        mib_payload = np.concatenate([mib, explicit_bits])
        
        return mib_payload
    
    def _scramble_mib_payload(self, mib_payload: np.ndarray) -> np.ndarray:
        """
        Scramble MIB payload
        
        Args:
            mib_payload: 32-bit MIB payload
            
        Returns:
            Scrambled MIB payload
        """
        # Extract parameters for scrambling
        cell_id = self.params.cell_id
        ssb_idx = self.params.ssb_index
        
        # Generate scrambling sequence
        c_init = cell_id
        if self.params.l_max == 64:
            # v corresponds to LSB 3 bits of SSB index for L=64
            v = ssb_idx & 0x7
            c_init = (c_init << 3) + v
        else:
            # v corresponds to LSB 2 bits of SSB index for L=4/8
            v = ssb_idx & 0x3
            c_init = (c_init << 2) + v
        
        # Generate gold sequence
        x1 = np.zeros(1600, dtype=int)
        x1[0:31] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        # Generate x1 sequence
        for i in range(31, 1600):
            x1[i] = (x1[i-31] + x1[i-28]) % 2
        
        # Initialize x2 with c_init
        x2 = np.zeros(1600, dtype=int)
        for i in range(31):
            x2[i] = (c_init >> i) & 1
        
        # Generate x2 sequence
        for i in range(31, 1600):
            x2[i] = (x2[i-31] + x2[i-30] + x2[i-29] + x2[i-28]) % 2
        
        # Generate scrambling sequence
        c = np.zeros(32, dtype=int)
        for i in range(32):
            c[i] = (x1[i] + x2[i]) % 2
        
        # Apply scrambling (XOR)
        scrambled = (mib_payload + c) % 2
        
        return scrambled
    
    def _apply_polar_encoding(self, input_bits: np.ndarray) -> np.ndarray:
        """
        Apply Polar encoding to input bits
        
        Args:
            input_bits: Input bits (32-bit)
            
        Returns:
            Polar encoded bits (512-bit)
        """
        # Polar encoding parameters for PBCH (K=32, E=864, N=512)
        K = 32  # Information bits
        N = 512  # Mother code size
        E = 864  # Rate-matched output size
        
        # Add CRC (24A)
        crc_poly = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]  # CRC24A polynomial
        msg_with_crc = self._add_crc(input_bits, crc_poly)
        
        # Generate frozen bit pattern (simplified version)
        # Based on 3GPP TS 38.212 Section 5.3.1
        reliability_sequence = self._get_polar_reliability_sequence(N)
        
        # Get information bit positions
        info_bit_positions = reliability_sequence[N-K-24:N]  # Last K+CRC positions
        
        # Create input vector with frozen bits
        u = np.zeros(N, dtype=int)
        for i, pos in enumerate(info_bit_positions):
            if i < len(msg_with_crc):
                u[pos] = msg_with_crc[i]
        
        # Perform polar encoding (simplified)
        encoded = self._polar_encode(u, N)
        
        # Rate matching (simplified - just taking the first E bits after encoding)
        rate_matched = encoded[:E]
        
        return rate_matched
    
    def _add_crc(self, bits: np.ndarray, poly: List[int]) -> np.ndarray:
        """
        Add CRC to input bits
        
        Args:
            bits: Input bit sequence
            poly: CRC polynomial coefficients
            
        Returns:
            Bit sequence with CRC appended
        """
        message = bits.copy()
        crc_len = len(poly) - 1
        
        # Append zeros for CRC
        message_with_crc = np.concatenate([message, np.zeros(crc_len, dtype=int)])
        
        # CRC calculation
        for i in range(len(message)):
            if message_with_crc[i] == 1:
                for j in range(len(poly)):
                    message_with_crc[i+j] = (message_with_crc[i+j] + poly[j]) % 2
        
        return message_with_crc
    
    def _get_polar_reliability_sequence(self, n: int) -> np.ndarray:
        """
        Get reliability sequence for polar code
        
        Args:
            n: Code length
            
        Returns:
            Reliability sequence
        """
        # This is a simplified version - actual implementation should follow 3GPP specs
        # A basic sequence for N=512 (not the exact 5G sequence)
        if n == 512:
            # Simplified reliability sequence (actual sequence from 3GPP is more complex)
            return np.array([
                0, 256, 128, 384, 64, 320, 192, 448, 32, 288, 160, 416, 96, 352, 224, 480,
                16, 272, 144, 400, 80, 336, 208, 464, 48, 304, 176, 432, 112, 368, 240, 496,
                8, 264, 136, 392, 72, 328, 200, 456, 40, 296, 168, 424, 104, 360, 232, 488,
                24, 280, 152, 408, 88, 344, 216, 472, 56, 312, 184, 440, 120, 376, 248, 504,
                4, 260, 132, 388, 68, 324, 196, 452, 36, 292, 164, 420, 100, 356, 228, 484,
                20, 276, 148, 404, 84, 340, 212, 468, 52, 308, 180, 436, 116, 372, 244, 500,
                12, 268, 140, 396, 76, 332, 204, 460, 44, 300, 172, 428, 108, 364, 236, 492,
                28, 284, 156, 412, 92, 348, 220, 476, 60, 316, 188, 444, 124, 380, 252, 508,
                2, 258, 130, 386, 66, 322, 194, 450, 34, 290, 162, 418, 98, 354, 226, 482,
                18, 274, 146, 402, 82, 338, 210, 466, 50, 306, 178, 434, 114, 370, 242, 498,
                10, 266, 138, 394, 74, 330, 202, 458, 42, 298, 170, 426, 106, 362, 234, 490,
                26, 282, 154, 410, 90, 346, 218, 474, 58, 314, 186, 442, 122, 378, 250, 506,
                6, 262, 134, 390, 70, 326, 198, 454, 38, 294, 166, 422, 102, 358, 230, 486,
                22, 278, 150, 406, 86, 342, 214, 470, 54, 310, 182, 438, 118, 374, 246, 502,
                14, 270, 142, 398, 78, 334, 206, 462, 46, 302, 174, 430, 110, 366, 238, 494,
                30, 286, 158, 414, 94, 350, 222, 478, 62, 318, 190, 446, 126, 382, 254, 510,
                1, 257, 129, 385, 65, 321, 193, 449, 33, 289, 161, 417, 97, 353, 225, 481,
                17, 273, 145, 401, 81, 337, 209, 465, 49, 305, 177, 433, 113, 369, 241, 497,
                9, 265, 137, 393, 73, 329, 201, 457, 41, 297, 169, 425, 105, 361, 233, 489,
                25, 281, 153, 409, 89, 345, 217, 473, 57, 313, 185, 441, 121, 377, 249, 505,
                5, 261, 133, 389, 69, 325, 197, 453, 37, 293, 165, 421, 101, 357, 229, 485,
                21, 277, 149, 405, 85, 341, 213, 469, 53, 309, 181, 437, 117, 373, 245, 501,
                13, 269, 141, 397, 77, 333, 205, 461, 45, 301, 173, 429, 109, 365, 237, 493,
                29, 285, 157, 413, 93, 349, 221, 477, 61, 317, 189, 445, 125, 381, 253, 509,
                3, 259, 131, 387, 67, 323, 195, 451, 35, 291, 163, 419, 99, 355, 227, 483,
                19, 275, 147, 403, 83, 339, 211, 467, 51, 307, 179, 435, 115, 371, 243, 499,
                11, 267, 139, 395, 75, 331, 203, 459, 43, 299, 171, 427, 107, 363, 235, 491,
                27, 283, 155, 411, 91, 347, 219, 475, 59, 315, 187, 443, 123, 379, 251, 507,
                7, 263, 135, 391, 71, 327, 199, 455, 39, 295, 167, 423, 103, 359, 231, 487,
                23, 279, 151, 407, 87, 343, 215, 471, 55, 311, 183, 439, 119, 375, 247, 503,
                15, 271, 143, 399, 79, 335, 207, 463, 47, 303, 175, 431, 111, 367, 239, 495,
                31, 287, 159, 415, 95, 351, 223, 479, 63, 319, 191, 447, 127, 383, 255, 511
            ])
        else:
            # For other sizes, we'd need to implement the full algorithm
            raise ValueError(f"Polar reliability sequence for N={n} not implemented")
    
    def _polar_encode(self, u: np.ndarray, n: int) -> np.ndarray:
        """
        Perform polar encoding
        
        Args:
            u: Input vector
            n: Code length
            
        Returns:
            Encoded vector
        """
        # Simplified implementation of polar encoding
        # The actual implementation should follow the 3GPP specifications
        
        # For power of 2 lengths, use recursive structure
        if n == 1:
            return u
        
        n_half = n // 2
        u1 = u[0::2]  # Even indices
        u2 = u[1::2]  # Odd indices
        
        # Recursive encoding
        v1 = self._polar_encode((u1 + u2) % 2, n_half)
        v2 = self._polar_encode(u2, n_half)
        
        # Combine outputs
        return np.concatenate([v1, v2])
    
    def _map_to_resource_grid(self) -> np.ndarray:
        """
        Map PSS, SSS, PBCH, and DMRS to resource grid
        
        Returns:
            Resource grid with mapped elements
        """
        # Create resource grid (240 subcarriers × 4 OFDM symbols)
        n_subcarriers = 240  # 20 RBs × 12 subcarriers
        n_symbols = 4
        grid = np.zeros((n_subcarriers, n_symbols), dtype=complex)
        
        # Map PSS (symbol 0)
        pss_start = (n_subcarriers - 127) // 2
        grid[pss_start:pss_start+127, 0] = self.pss_sequence
        
        # Map SSS (symbol 2)
        sss_start = (n_subcarriers - 127) // 2
        grid[sss_start:sss_start+127, 2] = self.sss_sequence
        
        # Generate PBCH payload
        mib_payload = self._generate_mib_payload()
        scrambled_mib = self._scramble_mib_payload(mib_payload)
        
        # Apply polar encoding to PBCH
        encoded_pbch = self._apply_polar_encoding(scrambled_mib)
        
        # Map PBCH to resource elements (symbols 1, 2, 3)
        # This is a simplified mapping - actual mapping follows 3GPP specifications
        pbch_idx = 0
        pbch_symbols = [1, 2, 3]
        
        # QPSK modulation for PBCH (2 bits per symbol)
        qpsk_symbols = []
        for i in range(0, len(encoded_pbch), 2):
            if i+1 < len(encoded_pbch):
                b0 = encoded_pbch[i]
                b1 = encoded_pbch[i+1]
                
                # QPSK mapping
                if b0 == 0 and b1 == 0:
                    qpsk_symbols.append((1 + 1j) / np.sqrt(2))
                elif b0 == 0 and b1 == 1:
                    qpsk_symbols.append((1 - 1j) / np.sqrt(2))
                elif b0 == 1 and b1 == 0:
                    qpsk_symbols.append((-1 + 1j) / np.sqrt(2))
                else:  # b0 == 1 and b1 == 1
                    qpsk_symbols.append((-1 - 1j) / np.sqrt(2))
        
        # Map PBCH to resource elements (excluding DMRS positions)
        # Simplified mapping - actual mapping follows RE pattern defined in 3GPP
        pbch_idx = 0
        for symbol_idx in pbch_symbols:
            for sc_idx in range(n_subcarriers):
                # Skip positions reserved for PSS/SSS
                if symbol_idx == 2 and sc_idx >= sss_start and sc_idx < sss_start + 127:
                    continue
                
                # Skip DMRS positions (simplification)
                if sc_idx % 4 == 0:
                    continue
                
                if pbch_idx < len(qpsk_symbols):
                    grid[sc_idx, symbol_idx] = qpsk_symbols[pbch_idx]
                    pbch_idx += 1
        
        # Generate and map PBCH DMRS
        dmrs_sequence = self._generate_pbch_dmrs()
        
        # Map DMRS to resource elements (every 4th subcarrier in symbols 1, 2, 3)
        dmrs_idx = 0
        for symbol_idx in pbch_symbols:
            for sc_idx in range(0, n_subcarriers, 4):
                if dmrs_idx < len(dmrs_sequence):
                    grid[sc_idx, symbol_idx] = dmrs_sequence[dmrs_idx]
                    dmrs_idx += 1
        
        return grid
    
    def generate_ssb(self) -> np.ndarray:
        """
        Generate complete SSB signal
        
        Returns:
            Complex time-domain SSB signal
        """
        # Map PSS, SSS, PBCH to resource grid
        resource_grid = self._map_to_resource_grid()
        
        # Apply OFDM modulation with correct CP
        time_domain_signal = self.ofdm_modulator.modulate(resource_grid)
        
        return time_domain_signal
    
    def save_to_file(self, filename: str, time_domain_signal: np.ndarray):
        """
        Save time domain signal to file in big-endian int16 I/Q format
        
        Args:
            filename: Output filename
            time_domain_signal: Complex time-domain signal
        """
        # Scale to int16 range
        max_val = np.max(np.abs(time_domain_signal))
        if max_val > 0:
            scale_factor = 32767 / max_val
        else:
            scale_factor = 1
        
        scaled_signal = time_domain_signal * scale_factor
        
        # Convert to int16
        i_samples = np.round(np.real(scaled_signal)).astype(np.int16)
        q_samples = np.round(np.imag(scaled_signal)).astype(np.int16)
        
        # Write to binary file in big-endian format
        with open(filename, 'wb') as f:
            for i, q in zip(i_samples, q_samples):
                # Write I sample (big-endian)
                f.write(struct.pack('>h', i))
                # Write Q sample (big-endian)
                f.write(struct.pack('>h', q))
    
    def generate_to_int16(self, time_domain_signal: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate SSB signal and convert to int16 I/Q arrays
        
        Args:
            time_domain_signal: Optional time-domain signal (if None, generates new signal)
            
        Returns:
            Tuple of (i_samples, q_samples) as int16 arrays
        """
        if time_domain_signal is None:
            time_domain_signal = self.generate_ssb()
        
        # Scale to int16 range
        max_val = np.max(np.abs(time_domain_signal))
        if max_val > 0:
            scale_factor = 32767 / max_val
        else:
            scale_factor = 1
        
        scaled_signal = time_domain_signal * scale_factor
        
        # Convert to int16
        i_samples = np.round(np.real(scaled_signal)).astype(np.int16)
        q_samples = np.round(np.imag(scaled_signal)).astype(np.int16)
        
        return i_samples, q_samples
		
class SSBAnalyzer:
    """
    SSB Analyzer for 5G NR with μ=0 and μ=1 support
    
    This class implements:
    - AGC processing
    - Decimation chain
    - PSS detection
    - SSS detection
    - PBCH decoding with channel estimation
    - Fixed-point implementation compatibility
    """
    
    def __init__(self, params: SSBParameters):
        self.params = params
        self._initialize_sequences()
        self._initialize_filters()
    
    def _initialize_sequences(self):
        """Initialize PSS and SSS reference sequences"""
        # Generate all possible PSS sequences
        self.pss_sequences = {}
        for n_id_2 in range(3):
            self.pss_sequences[n_id_2] = self._generate_pss_sequence(n_id_2)
        
        # SSS sequences will be generated on demand
        self.sss_cache = {}
    
    def _initialize_filters(self):
        """Initialize decimation filters"""
        # Decimation filter coefficients (7-tap FIR)
        self.decimation_filter = np.array([
            0.04111940976293188, 0.12335805100729215, 
            0.21586890691103866, 0.25698831667397054,
            0.21586890691103866, 0.12335805100729215,
            0.04111940976293188
        ])
        
        self.filter_delay = len(self.decimation_filter) // 2
    
    def _generate_pss_sequence(self, n_id_2: int) -> np.ndarray:
        """Generate PSS sequence for given n_id_2"""
        # Initialize m-sequence
        x = np.zeros(127, dtype=int)
        x[0:7] = [1, 1, 1, 0, 1, 1, 0]  # Initial value
        
        # Generate m-sequence
        for i in range(7, 127):
            x[i] = (x[i-7] + x[i-4]) % 2
        
        # Map from {0,1} to {1,-1}
        d_pss = np.zeros(127, dtype=complex)
        for n in range(127):
            # Different m-sequences based on n_id_2
            m = (n + 43 * n_id_2) % 127
            d_pss[n] = 1 - 2 * x[m]
        
        return d_pss
    
    def _generate_sss_sequence(self, n_id_1: int, n_id_2: int) -> np.ndarray:
        """
        Generate SSS sequence for given n_id_1 and n_id_2
        
        Args:
            n_id_1: Cell ID group (0-335)
            n_id_2: Physical layer identity (0-2)
            
        Returns:
            127-length SSS sequence
        """
        # Check cache
        key = (n_id_1, n_id_2)
        if key in self.sss_cache:
            return self.sss_cache[key]
        
        # Initialize m-sequences
        x0 = np.zeros(127, dtype=int)
        x1 = np.zeros(127, dtype=int)
        
        # Initial values
        x0[0:7] = [1, 0, 0, 0, 0, 0, 0]
        x1[0:7] = [1, 0, 0, 0, 0, 0, 0]
        
        # Generate m-sequences
        for i in range(7, 127):
            x0[i] = (x0[i-7] + x0[i-4]) % 2
            x1[i] = (x1[i-7] + x1[i-1]) % 2
        
        # Calculate m0 and m1 based on n_id_1 and n_id_2
        m0 = 15 * (n_id_1 // 112) + 5 * n_id_2
        m1 = n_id_1 % 112
        
        # Generate SSS sequence
        d_sss = np.zeros(127, dtype=complex)
        for n in range(127):
            d_sss[n] = (1 - 2 * x0[(n + m0) % 127]) * (1 - 2 * x1[(n + m1) % 127])
        
        # Cache result
        self.sss_cache[key] = d_sss
        
        return d_sss
    
    def process_input_agc(self, signal: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Apply input AGC to scale signal into desired dynamic range
        
        Args:
            signal: Input complex signal
            
        Returns:
            Tuple of (scaled_signal, shift_value)
        """
        # Calculate mean absolute value
        mean_val = np.mean(np.abs(signal))
        
        # Calculate required shift to fit into target bit width
        target_bits = 12  # 12-bit ADC
        
        if mean_val > 0:
            shift = -np.ceil(np.log2(mean_val)) + (-6)  # -6 is the exponent shift
        else:
            shift = 0
        
        # Apply scaling
        scaled_signal = signal * (2 ** shift)
        
        # Apply fixed-point clipping if enabled
        if self.params.use_fixed_point:
            # Clip to target bit width
            max_val = 2 ** (target_bits - 1) - 1
            scaled_signal = np.clip(scaled_signal, -max_val, max_val)
            scaled_signal = np.round(scaled_signal)
        
        return scaled_signal, shift
    
    def decimate_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Decimate signal by decimation_ratio
        
        Args:
            signal: Input signal
            
        Returns:
            Decimated signal
        """
        # Determine decimation ratio based on numerology
        if self.params.numerology == Numerology.MU0:
            decimation_ratio = 8  # For 15 kHz SCS <sup data-citation="33"><a href="https://ytd2525.wordpress.com/2019/12/15/5g-nr-cyclic-prefix-cp-design/#:~:text=Cyclic%20prefix%20(CP)%20refers,spacing%20coincide%20with" target="_blank" title="ytd2525.wordpress.com">33</a></sup>
        else:
            decimation_ratio = 4  # For 30 kHz SCS
        
        # Apply filter and decimate
        # Using scipy for simplicity - in a real implementation, you'd use
        # more efficient methods for fixed-point compatibility
        decimated = signal.decimate(signal, decimation_ratio, ftype='fir')
        
        return decimated
    
    def detect_pss(self, signal: np.ndarray) -> Tuple[int, float, int]:
        """
        Detect PSS in time-domain signal
        
        Args:
            signal: Time-domain signal
            
        Returns:
            Tuple of (n_id_2, correlation_peak, timing_offset)
        """
        # Correlation results for each PSS sequence
        correlation_results = {}
        
        # Process each possible PSS sequence
        for n_id_2 in range(3):
            # Get reference sequence
            pss_seq = self.pss_sequences[n_id_2]
            
            # Calculate correlation
            correlation = np.abs(np.correlate(signal, pss_seq, mode='valid'))
            
            # Find peak
            peak_idx = np.argmax(correlation)
            peak_value = correlation[peak_idx]
            
            correlation_results[n_id_2] = (peak_value, peak_idx)
        
        # Find best match
        best_n_id_2 = max(correlation_results, key=lambda x: correlation_results[x][0])
        best_peak_value, best_peak_idx = correlation_results[best_n_id_2]
        
        return best_n_id_2, best_peak_value, best_peak_idx
    
    def detect_sss(self, signal: np.ndarray, n_id_2: int) -> Tuple[int, float]:
        """
        Detect SSS in time-domain signal
        
        Args:
            signal: Time-domain signal
            n_id_2: Physical layer identity (0-2)
            
        Returns:
            Tuple of (n_id_1, correlation_peak)
        """
        # Correlation results for each SSS sequence
        correlation_results = {}
        
        # Process each possible N_ID_1 value
        for n_id_1 in range(336):
            # Generate SSS sequence
            sss_seq = self._generate_sss_sequence(n_id_1, n_id_2)
            
            # Calculate correlation
            correlation = np.abs(np.correlate(signal, sss_seq, mode='valid'))
            
            # Find peak
            peak_idx = np.argmax(correlation)
            peak_value = correlation[peak_idx]
            
            correlation_results[n_id_1] = (peak_value, peak_idx)
        
        # Find best match
        best_n_id_1 = max(correlation_results, key=lambda x: correlation_results[x][0])
        best_peak_value, _ = correlation_results[best_n_id_1]
        
        return best_n_id_1