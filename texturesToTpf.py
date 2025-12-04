"""
Combined script to scan directory for PNG texture files, extract hexadecimal IDs,
and create TPF (TexMod Package File) format directly without intermediate log files.
Operates entirely in-memory using a dictionary to map hash IDs to texture paths.
"""

# Standard library imports (always available, no dependency check needed)
import sys
import subprocess
import importlib.util
import os
import re
import io
import time
import struct
import shutil
import configparser
import zlib
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from zipfile import ZIP_DEFLATED

try:
    import msvcrt
except ImportError:
    msvcrt = None

# ============================================================================
# Configuration Loading
# ============================================================================

def load_config() -> Dict:
    """
    Load configuration from config.ini file, or use defaults if not found.
    Returns a dictionary with configuration values.
    """
    # Default values
    defaults = {
        'input_formats': ['.png', '.jpg', '.jpeg', '.bmp'],
        'generate_mipmaps': True,
        'channel_variance_threshold': 0.001,
        'normal_variance_threshold': 0.01,
        'enable_compression': True
    }
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    config_path = script_dir / 'config.ini'
    
    # Check if config.ini exists
    if not config_path.exists():
        print("No config.ini detected, using defaults")
        return defaults
    
    # Parse config file
    config = configparser.ConfigParser()
    try:
        config.read(config_path)
        
        result = defaults.copy()
        
        # Load FileFormats section
        if config.has_section('FileFormats'):
            if config.has_option('FileFormats', 'input_formats'):
                formats_str = config.get('FileFormats', 'input_formats')
                # Split by comma, strip whitespace, convert to lowercase, and add dot prefix if needed
                formats_list = []
                for f in formats_str.split(','):
                    f = f.strip().lower()
                    if f:
                        # Add dot prefix if not already present
                        if not f.startswith('.'):
                            f = '.' + f
                        formats_list.append(f)
                result['input_formats'] = formats_list
        
        # Load DDS section
        if config.has_section('DDS'):
            if config.has_option('DDS', 'generate_mipmaps'):
                result['generate_mipmaps'] = config.getboolean('DDS', 'generate_mipmaps')
            if config.has_option('DDS', 'channel_variance_threshold'):
                result['channel_variance_threshold'] = config.getfloat('DDS', 'channel_variance_threshold')
            if config.has_option('DDS', 'normal_variance_threshold'):
                result['normal_variance_threshold'] = config.getfloat('DDS', 'normal_variance_threshold')
            if config.has_option('DDS', 'enable_compression'):
                result['enable_compression'] = config.getboolean('DDS', 'enable_compression')
        
        return result
        
    except Exception as e:
        print(f"Error reading config.ini: {e}, using defaults")
        return defaults

# Load configuration once at module level
_config = load_config()

# ============================================================================
# Configuration Constants (User-configurable)
# ============================================================================

# File format settings
INPUT_FORMATS = _config['input_formats']

# DDS compression settings
GENERATE_MIPMAPS = _config['generate_mipmaps']
CHANNEL_VARIANCE_THRESHOLD = _config['channel_variance_threshold']  # Threshold for alpha channel variance (below this = uniform/unused)
NORMAL_VARIANCE_THRESHOLD = _config['normal_variance_threshold']  # Threshold for normal map channel detection (10x higher, below this = unused channel)
ENABLE_COMPRESSION = _config['enable_compression']  # Whether to compress textures to DDS format

# ============================================================================
# Dependency Checking and Installation
# ============================================================================

def _print_colored(message: str, color: str = ''):
    """Print message with optional color (works even without colorama)."""
    print(f"{color}{message}\033[0m" if color else message)

def check_and_install_package(package_name: str, import_name: str = None, pip_name: str = None) -> bool:
    """
    Check if a Python package is installed, and install it if missing.
    
    Args:
        package_name: Display name of the package
        import_name: Name to use for import (defaults to package_name)
        pip_name: Name to use for pip install (defaults to package_name)
        
    Returns:
        True if package is available (after installation attempt), False otherwise
    """
    if import_name is None:
        import_name = package_name
    if pip_name is None:
        pip_name = package_name
    
    # Check if already installed
    spec = importlib.util.find_spec(import_name)
    if spec is not None:
        return True
    
    # Package not found, try to install
    _print_colored(f"\n[Checking] {package_name} is not installed.", '\033[93m')  # Yellow
    _print_colored(f"[Installing] Attempting to install {package_name} via pip...", '\033[96m')  # Cyan
    
    try:
        # Use subprocess to install via pip
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name],
            capture_output=True,
            text=True,
            check=True
        )
        _print_colored(f"[Success] {package_name} installed successfully!", '\033[92m')  # Green
        
        # Verify installation
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            return True
        else:
            _print_colored(f"[Warning] {package_name} was installed but cannot be imported.", '\033[93m')  # Yellow
            return False
            
    except subprocess.CalledProcessError as e:
        _print_colored(f"[Error] Failed to install {package_name}: {e.stderr}", '\033[91m')  # Red
        _print_colored(f"[Manual] Please install manually with: pip install {pip_name}", '\033[93m')  # Yellow
        return False
    except Exception as e:
        _print_colored(f"[Error] Unexpected error installing {package_name}: {e}", '\033[91m')  # Red
        return False


def check_imagemagick() -> bool:
    """
    Check if ImageMagick is installed and available in PATH.
    
    Returns:
        True if ImageMagick is available, False otherwise
    """
    try:
        result = subprocess.run(
            ['magick', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass
    except Exception:
        pass
    
    return False


def check_dependencies():
    """
    Check all required dependencies and install missing pip packages.
    Prompts user for ImageMagick installation if missing.
    
    Returns:
        True if all critical dependencies are available, False otherwise
    """
    _print_colored("\n" + "="*60, '\033[96m')  # Cyan
    _print_colored("Checking Dependencies", '\033[96m')  # Cyan
    _print_colored("="*60, '\033[96m')  # Cyan
    
    # Required pip packages
    required_packages = [
        ("Pillow", "PIL", "Pillow"),
        ("numpy", "numpy", "numpy"),
        ("colorama", "colorama", "colorama"),
    ]
    
    # Optional but recommended packages
    optional_packages = [
        ("numba", "numba", "numba"),
    ]
    
    all_ok = True
    
    # Check and install required packages
    _print_colored("\n[Required Packages]", '\033[96m')  # Cyan
    for display_name, import_name, pip_name in required_packages:
        if not check_and_install_package(display_name, import_name, pip_name):
            _print_colored(f"[Critical] {display_name} is required but could not be installed!", '\033[91m')  # Red
            all_ok = False
    
    # Check and install optional packages
    _print_colored("\n[Optional Packages]", '\033[96m')  # Cyan
    for display_name, import_name, pip_name in optional_packages:
        if not check_and_install_package(display_name, import_name, pip_name):
            _print_colored(f"[Info] {display_name} is optional but recommended for better performance.", '\033[93m')  # Yellow
    
    # Check ImageMagick (optional, only needed for DDS compression)
    _print_colored("\n[External Tools]", '\033[96m')  # Cyan
    if check_imagemagick():
        _print_colored("[Found] ImageMagick is installed and available.", '\033[92m')  # Green
    else:
        _print_colored("[Missing] ImageMagick is not found in your system PATH.", '\033[93m')  # Yellow
        _print_colored("\nImageMagick is required for optional DDS compression feature.", '\033[93m')  # Yellow
        _print_colored("If you want to use DDS compression, please install ImageMagick:", '\033[93m')  # Yellow
        _print_colored("  1. Download from: https://imagemagick.org/script/download.php", '\033[96m')  # Cyan
        _print_colored("  2. Install ImageMagick on your system", '\033[96m')  # Cyan
        _print_colored("  3. Make sure 'magick' command is in your system PATH", '\033[96m')  # Cyan
        _print_colored("  4. Verify installation by running: magick -version", '\033[96m')  # Cyan
        _print_colored("\n[Note] You can still use the script without ImageMagick,", '\033[93m')  # Yellow
        _print_colored("       but DDS compression will not be available.", '\033[93m')  # Yellow
    
    _print_colored("\n" + "="*60, '\033[96m')  # Cyan
    
    if not all_ok:
        _print_colored("\n[Error] Some required dependencies are missing!", '\033[91m')  # Red
        _print_colored("Please install the missing packages and try again.", '\033[91m')  # Red
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    _print_colored("[Success] All required dependencies are available!", '\033[92m')  # Green
    _print_colored("", '')  # Empty line


# Run dependency checks before importing third-party packages that might fail
check_dependencies()

# Import colorama (should be available now after dependency check)
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    # Fallback if colorama is not installed (shouldn't happen after check)
    class Fore:
        GREEN = ''
        YELLOW = ''
        CYAN = ''
        RED = ''
        BLUE = ''
        MAGENTA = ''
        WHITE = ''
        RESET = ''
    class Style:
        RESET_ALL = ''
    def init(**kwargs):
        pass

# Import Pillow (should be available after dependency check)
try:
    from PIL import Image
except ImportError:
    print(f"{Fore.RED}[Error] Pillow package is required but not available.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}This should not happen if dependency check passed.{Style.RESET_ALL}")
    input("\nPress Enter to exit...")
    sys.exit(1)

# Import numpy (should be available after dependency check)
try:
    import numpy as np
except ImportError:
    print(f"{Fore.RED}[Error] numpy package is required but not available.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}This should not happen if dependency check passed.{Style.RESET_ALL}")
    input("\nPress Enter to exit...")
    sys.exit(1)

# Try to use Numba-optimized ZipCrypto, fallback to zipencrypt if Numba unavailable
NUMBA_AVAILABLE = False
try:
    from numba import jit  # noqa: F401
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    # We'll define NumbaZipFile class below, set ZipFile to it
    ZipFile = None  # Will be set after class definition
else:
    # Fallback to zipencrypt
    try:
        from zipencrypt import ZipFile, ZIP_DEFLATED
        # Print warning about numba not being available (non-blocking)
        print(f"{Fore.YELLOW}[Warning] Numba is not installed. Using zipencrypt fallback.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[Info] For better performance, consider installing numba: pip install numba{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[Info] The script will continue using zipencrypt, which works but is slower.{Style.RESET_ALL}\n")
    except ImportError:
        raise ImportError(
            "Either numba or zipencrypt package is required for ZipCrypto encryption. "
            "Install with: pip install numba (preferred) or pip install zipencrypt"
        )

# ============================================================================
# Hardcoded Implementation Constants (Not user-configurable)
# ============================================================================

# ZipCrypto algorithm constants
KEY0_INIT = 0x12345678
KEY1_INIT = 0x23456789
KEY2_INIT = 0x34567890
LCG_MULTIPLIER = 134775813
CRC32_POLY = 0xEDB88320

# ZipCrypto password: 42-byte hardcoded password from TexMod specification
ZIPCRYPTO_PASSWORD = bytes([
    0x73, 0x2A, 0x63, 0x7D, 0x5F, 0x0A, 0xA6, 0xBD, 0x7D, 0x65,
    0x7E, 0x67, 0x61, 0x2A, 0x7F, 0x7F, 0x74, 0x61, 0x67, 0x5B,
    0x60, 0x70, 0x45, 0x74, 0x5C, 0x22, 0x74, 0x5D, 0x6E, 0x6A,
    0x73, 0x41, 0x77, 0x6E, 0x46, 0x47, 0x77, 0x49, 0x0C, 0x4B,
    0x46, 0x6F
])

# XOR obfuscation key
XOR_KEY = 0x3FA43FA4

# Pattern: _0X or _0x followed by hex digits, must end at end of filename (before extension)
ID_PATTERN = re.compile(r'_0[xX]([0-9A-Fa-f]+)$')


# ============================================================================
# Numba-Optimized ZipCrypto Implementation
# ============================================================================

if NUMBA_AVAILABLE:
    # Generate CRC32 table for polynomial 0xEDB88320 (IEEE 802.3)
    # This table is used by ZipCrypto for key initialization and updates
    _crc32_table_list = []
    for i in range(256):
        crc = np.uint32(i)
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
        _crc32_table_list.append(crc)
    _CRC32_TABLE = np.array(_crc32_table_list, dtype=np.uint32)

    @jit(nopython=True, cache=True)
    def _zipcrypto_init_keys(password_bytes):
        """
        Initialize ZipCrypto keys from password bytes.
        
        Args:
            password_bytes: uint8 array of password bytes
            
        Returns:
            uint32 array of [key0, key1, key2]
        """
        k0 = np.uint32(0x12345678)
        k1 = np.uint32(0x23456789)
        k2 = np.uint32(0x34567890)
        
        for i in range(len(password_bytes)):
            byte = password_bytes[i]
            # Update k0 using CRC32
            idx0 = (k0 ^ byte) & 0xFF
            k0 = (k0 >> 8) ^ _CRC32_TABLE[idx0]
            # Update k1
            low_k0 = k0 & 0xFF
            k1 = (k1 + low_k0) & 0xFFFFFFFF
            k1 = (k1 * 134775813) & 0xFFFFFFFF
            k1 = (k1 + 1) & 0xFFFFFFFF
            # Update k2 using CRC32
            idx2 = (k2 ^ (k1 >> 24)) & 0xFF
            k2 = (k2 >> 8) ^ _CRC32_TABLE[idx2]
        
        return np.array([k0, k1, k2], dtype=np.uint32)

    @jit(nopython=True, cache=True)
    def _zipcrypto_process_chunk(data, keys):
        """
        Encrypt a data chunk using ZipCrypto stream cipher.
        
        Args:
            data: uint8 array of plaintext data
            keys: uint32 array of [key0, key1, key2]
            
        Returns:
            Tuple of (encrypted_data, updated_keys)
        """
        k0, k1, k2 = keys[0], keys[1], keys[2]
        n = len(data)
        ciphertext = np.empty(n, dtype=np.uint8)
        
        for i in range(n):
            byte = data[i]  # This is the PLAINTEXT byte
            
            # Generate keystream byte
            temp = (k2 | 2) & 0xFFFFFFFF
            keystream = (((temp * (temp ^ 1)) & 0xFFFFFFFF) >> 8) & 0xFF
            
            # Encrypt byte
            cipher_byte = byte ^ keystream
            ciphertext[i] = cipher_byte
            
            # Update keys
            # FIXED: Use 'byte' (plaintext), not 'cipher_byte'
            # k0 = crc32(k0, plaintext_byte)
            idx0 = (k0 ^ byte) & 0xFF
            k0 = (k0 >> 8) ^ _CRC32_TABLE[idx0]
            
            # k1 = (k1 + (k0 & 0xFF)) * 134775813 + 1
            low_k0 = k0 & 0xFF
            k1 = (k1 + low_k0) & 0xFFFFFFFF
            k1 = (k1 * 134775813) & 0xFFFFFFFF
            k1 = (k1 + 1) & 0xFFFFFFFF
            
            # k2 = crc32(k2, k1 >> 24)
            idx2 = (k2 ^ (k1 >> 24)) & 0xFF
            k2 = (k2 >> 8) ^ _CRC32_TABLE[idx2]
        
        return ciphertext, np.array([k0, k1, k2], dtype=np.uint32)

    class NumbaZipFile:
        """
        High-performance ZipCrypto-enabled ZIP file writer using Numba-optimized encryption.
        Provides minimal API compatible with zipencrypt.ZipFile for writestr() and write() methods.
        """
        
        def __init__(self, fileobj, mode='w', compression=ZIP_DEFLATED):
            """
            Initialize ZIP file writer.
            
            Args:
                fileobj: File-like object (e.g., BytesIO) to write ZIP data to
                mode: Must be 'w' for writing
                compression: Compression method (ZIP_DEFLATED or ZIP_STORED)
            """
            if mode != 'w':
                raise ValueError("Only 'w' mode is supported")
            
            self.fileobj = fileobj
            self.compression = compression
            self.entries = []  # List of (name, crc, size_uncompressed, size_compressed, offset, flags)
            self.offset = 0
            self.closed = False
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()
            return False
        
        def close(self):
            """Write Central Directory and End of Central Directory records, then close."""
            if self.closed:
                return
            
            # Write Central Directory
            cd_offset = self.offset
            for name, crc, size_uncomp, size_comp, offset, flags in self.entries:
                self._write_central_directory_entry(name, crc, size_uncomp, size_comp, offset, flags)
            
            cd_size = self.offset - cd_offset
            
            # Write End of Central Directory
            self._write_end_of_central_directory(len(self.entries), cd_size, cd_offset)
            
            self.closed = True
        
        def writestr(self, filename, data, pwd=None):
            """
            Write string/bytes data to ZIP archive with optional ZipCrypto encryption.
            
            Args:
                filename: Name of file in archive
                data: String or bytes data to write
                pwd: Password bytes for encryption (None = no encryption)
            """
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Calculate CRC32
            crc = zlib.crc32(data) & 0xFFFFFFFF
            
            # Compress data if needed
            if self.compression == ZIP_DEFLATED:
                compressor = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15)
                compressed_data = compressor.compress(data) + compressor.flush()
            else:
                compressed_data = data
            
            # Encrypt if password provided
            flags = 0
            if pwd is not None:
                flags |= 0x1  # Set encryption flag
                compressed_data = self._encrypt_data(compressed_data, pwd, crc)
            
            # Write Local File Header
            lfh_offset = self.offset
            self._write_local_file_header(filename, len(compressed_data), len(data), crc, flags)
            
            # Write file data
            self.fileobj.write(compressed_data)
            self.offset += len(compressed_data)
            
            # Record entry for Central Directory
            self.entries.append((filename, crc, len(data), len(compressed_data), lfh_offset, flags))
        
        def write(self, filepath, arcname=None, pwd=None):
            """
            Write file from disk to ZIP archive with optional ZipCrypto encryption.
            
            Args:
                filepath: Path to file on disk (Path or str)
                arcname: Name of file in archive (defaults to filepath.name)
                pwd: Password bytes for encryption (None = no encryption)
            """
            filepath = Path(filepath)
            if arcname is None:
                arcname = filepath.name
            
            # Read file data
            with open(filepath, 'rb') as f:
                data = f.read()
            
            # Use writestr to handle the rest
            self.writestr(arcname, data, pwd)
        
        def _encrypt_data(self, data, password, crc):
            """
            Encrypt data using ZipCrypto algorithm.
            
            Args:
                data: Bytes to encrypt
                password: Password bytes
                crc: CRC32 of uncompressed data
                
            Returns:
                Encrypted bytes (12-byte header + encrypted body)
            """
            # Initialize keys from password
            pwd_array = np.frombuffer(password, dtype=np.uint8)
            keys = _zipcrypto_init_keys(pwd_array)
            
            # Generate 12-byte encryption header
            # First 11 bytes are random, last byte is CRC check byte
            header_nonce = os.urandom(11)
            check_byte = (crc >> 24) & 0xFF
            header = header_nonce + bytes([check_byte])
            
            # Encrypt header (this updates the keys)
            header_array = np.frombuffer(header, dtype=np.uint8)
            enc_header, keys = _zipcrypto_process_chunk(header_array, keys)
            
            # Encrypt body in chunks for better performance
            chunk_size = 256 * 1024  # 256KB chunks
            encrypted_chunks = []
            
            data_array = np.frombuffer(data, dtype=np.uint8)
            for i in range(0, len(data_array), chunk_size):
                chunk = data_array[i:i + chunk_size]
                enc_chunk, keys = _zipcrypto_process_chunk(chunk, keys)
                encrypted_chunks.append(enc_chunk.tobytes())
            
            # Combine header and body
            return enc_header.tobytes() + b''.join(encrypted_chunks)
        
        def _write_local_file_header(self, filename, compressed_size, uncompressed_size, crc, flags):
            """Write Local File Header (LFH) record."""
            filename_bytes = filename.encode('utf-8')
            filename_len = len(filename_bytes)
            
            # Local File Header structure (30 bytes + filename)
            # Signature (I), version (H), flags (H), compression (H), mod time (H), mod date (H),
            # CRC-32 (I), compressed size (I), uncompressed size (I), filename len (H), extra len (H)
            header = struct.pack('<IHHHHHIIIHH',
                0x04034b50,  # Local file header signature (4 bytes)
                20,          # Version needed to extract (2.0)
                flags,       # General purpose bit flag
                8 if self.compression == ZIP_DEFLATED else 0,  # Compression method
                0,           # Last mod file time
                0,           # Last mod file date
                crc,         # CRC-32
                compressed_size,    # Compressed size
                uncompressed_size,  # Uncompressed size
                filename_len,       # Filename length
                0            # Extra field length
            )
            
            self.fileobj.write(header)
            self.fileobj.write(filename_bytes)
            self.offset += 30 + filename_len
        
        def _write_central_directory_entry(self, filename, crc, uncompressed_size, compressed_size, offset, flags):
            """Write Central Directory File Header."""
            filename_bytes = filename.encode('utf-8')
            filename_len = len(filename_bytes)
            
            # Central Directory File Header structure (46 bytes + filename)
            # Signature (I), version made by (H), version needed (H), flags (H), compression (H),
            # mod time (H), mod date (H), CRC-32 (I), compressed size (I), uncompressed size (I),
            # filename len (H), extra len (H), comment len (H), disk num (H), internal attr (H),
            # external attr (I), offset (I)
            header = struct.pack('<IHHHHHHIIIHHHHHII',
                0x02014b50,  # Central file header signature (4 bytes)
                20,          # Version made by
                20,          # Version needed to extract
                flags,       # General purpose bit flag
                8 if self.compression == ZIP_DEFLATED else 0,  # Compression method
                0,           # Last mod file time
                0,           # Last mod file date
                crc,         # CRC-32
                compressed_size,    # Compressed size
                uncompressed_size,  # Uncompressed size
                filename_len,       # Filename length
                0,           # Extra field length
                0,           # File comment length
                0,           # Disk number start
                0,           # Internal file attributes
                0,           # External file attributes (4 bytes)
                offset       # Relative offset of local header
            )
            
            self.fileobj.write(header)
            self.fileobj.write(filename_bytes)
            self.offset += 46 + filename_len
        
        def _write_end_of_central_directory(self, num_entries, cd_size, cd_offset):
            """Write End of Central Directory Record."""
            # Signature (I), disk num (H), disk with CD (H), entries on disk (H), total entries (H),
            # CD size (I), CD offset (I), comment length (H)
            eocd = struct.pack('<IHHHHIIH',
                0x06054b50,  # End of central dir signature (4 bytes)
                0,           # Number of this disk
                0,           # Number of disk with start of central directory
                num_entries, # Total number of entries in central directory on this disk
                num_entries, # Total number of entries in central directory
                cd_size,     # Size of central directory
                cd_offset,   # Offset of start of central directory
                0            # ZIP file comment length
            )
            
            self.fileobj.write(eocd)
            self.offset += 22
    
    # Set ZipFile to use NumbaZipFile (we're already inside NUMBA_AVAILABLE block)
    ZipFile = NumbaZipFile


# ============================================================================
# Cache Functions
# ============================================================================

def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate MD5 hash of file content.

    Args:
        file_path: Path to the file to hash

    Returns:
        MD5 hash as hexadecimal string
    """
    import hashlib

    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except (OSError, IOError) as e:
        print(f"{Fore.YELLOW}Warning: Could not hash file {file_path.name}: {e}{Style.RESET_ALL}")
        return ""


def load_cache(cache_path: Path) -> Dict[str, dict]:
    """
    Load cache data from JSON file.

    Args:
        cache_path: Path to the cache.json file

    Returns:
        Dictionary containing cache data, empty dict if file doesn't exist or is invalid
    """
    import json

    if not cache_path.exists():
        return {}

    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, IOError) as e:
        print(f"{Fore.YELLOW}Warning: Could not load cache file {cache_path}: {e}{Style.RESET_ALL}")
        return {}


def save_cache(cache_path: Path, cache_data: Dict[str, dict]) -> None:
    """
    Save cache data to JSON file.

    Args:
        cache_path: Path to save the cache.json file
        cache_data: Dictionary containing cache data to save
    """
    import json

    try:
        # Create parent directory if it doesn't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
    except (OSError, IOError) as e:
        print(f"{Fore.YELLOW}Warning: Could not save cache file {cache_path}: {e}{Style.RESET_ALL}")


def compare_with_cache(texture_dict: Dict[str, Path], cache_data: Dict[str, dict], texture_dir: Path) -> Dict[str, str]:
    """
    Compare current texture files against cache to determine status.

    Args:
        texture_dict: Dictionary mapping hash IDs to file paths
        cache_data: Dictionary containing cache data
        texture_dir: Base texture directory for relative path calculation

    Returns:
        Dictionary mapping hash IDs to status ("new", "changed", "unchanged")
    """
    status_dict: Dict[str, str] = {}

    for hash_id, texture_path in texture_dict.items():
        # Calculate relative path for cache key
        try:
            relative_path = texture_path.relative_to(texture_dir)
            cache_key = str(relative_path)
        except ValueError:
            # Fallback to absolute path if relative path calculation fails
            cache_key = str(texture_path)

        # Calculate current file hash
        current_hash = calculate_file_hash(texture_path)
        if not current_hash:
            # Could not hash file, treat as new
            status_dict[hash_id] = "new"
            continue

        # Check if file is in cache
        if cache_key in cache_data:
            cached_hash = cache_data[cache_key].get('hash', '')
            if current_hash == cached_hash:
                status_dict[hash_id] = "unchanged"
            else:
                status_dict[hash_id] = "changed"
        else:
            status_dict[hash_id] = "new"

    return status_dict


# ============================================================================
# Validation Wrapper Functions
# ============================================================================

def is_valid_texture_directory(directory: Path) -> bool:
    """
    Check if directory exists, is readable, and contains PNG files matching the ID pattern.
    
    Args:
        directory: Directory path to validate
        
    Returns:
        True if directory has valid texture files, False otherwise
    """
    if not directory.exists():
        return False
    
    if not directory.is_dir():
        return False
    
    # Check if directory is readable
    if not os.access(directory, os.R_OK):
        return False
    
    # Check if directory contains any matching PNG files
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_file():
                    path = Path(entry)
                    if path.suffix.lower() in INPUT_FORMATS:
                        id_value = extract_id_from_filename(path)
                        if id_value:
                            return True
    except (PermissionError, OSError):
        return False
    
    return False


def validate_texture_file(path: Path) -> bool:
    """
    Validate that a texture file exists and is readable.
    Supports both PNG and DDS files (DDS when compression is enabled).
    
    Args:
        path: Path to texture file
        
    Returns:
        True if file is valid for processing, False otherwise
    """
    if not path.exists():
        return False
    
    if not path.is_file():
        return False
    
    if not os.access(path, os.R_OK):
        return False
    
    # Accept input formats and DDS files
    suffix_lower = path.suffix.lower()
    if suffix_lower not in INPUT_FORMATS and suffix_lower != '.dds':
        return False
    
    return True


def is_valid_target_directory(directory: Path) -> Tuple[bool, str]:
    """
    Check if directory is writable for TPF output.
    
    Args:
        directory: Directory path to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not directory.exists():
        return False, f"Directory does not exist: {directory}"
    
    if not directory.is_dir():
        return False, f"Path is not a directory: {directory}"
    
    # Check if directory is writable
    if not os.access(directory, os.W_OK):
        return False, f"Directory is not writable: {directory}"
    
    # Check available disk space (warn if less than 100MB, but don't fail)
    try:
        stat = shutil.disk_usage(directory)
        free_mb = stat.free / (1024 * 1024)
        if free_mb < 100:
            return True, f"Warning: Low disk space ({free_mb:.1f} MB free)"
    except OSError:
        pass  # Can't check disk space, but continue anyway
    
    return True, ""


# ============================================================================
# Path Resolution Functions
# ============================================================================

def detect_texture_directory(script_dir: Path) -> Optional[Path]:
    """
    Auto-detect texture directory by checking script directory and parent directory.
    
    Args:
        script_dir: Script directory to start search from
        
    Returns:
        Path if valid directory found, None otherwise
    """
    # Check script directory first
    if is_valid_texture_directory(script_dir):
        return script_dir
    
    # Check parent directory
    parent_dir = script_dir.parent
    if parent_dir.exists() and is_valid_texture_directory(parent_dir):
        return parent_dir
    
    return None


def prompt_directory_selection(start_dir: Path) -> Path:
    """
    Interactive folder selection dialog.
    
    Args:
        start_dir: Starting directory for selection
        
    Returns:
        Selected and validated Path object
        
    Raises:
        SystemExit: If user cancels the operation
    """
    print(f"\n{Fore.YELLOW}Please select the directory containing texture files.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Current directory: {start_dir}{Style.RESET_ALL}")
    
    while True:
        try:
            path_input = input(f"\n{Fore.YELLOW}Enter directory path (relative or absolute, or press Enter to cancel): {Style.RESET_ALL}").strip()
            
            if not path_input:
                print(f"{Fore.YELLOW}Operation cancelled.{Style.RESET_ALL}")
                raise SystemExit(0)
            
            # Try as absolute path first
            selected_path = Path(path_input)
            if not selected_path.is_absolute():
                # Resolve relative to start directory
                selected_path = start_dir / path_input
            
            selected_path = selected_path.resolve()
            
            if not selected_path.exists():
                print(f"{Fore.RED}Directory not found: {selected_path}{Style.RESET_ALL}")
                continue
            
            if not selected_path.is_dir():
                print(f"{Fore.RED}Path is not a directory: {selected_path}{Style.RESET_ALL}")
                continue
            
            # Validate it contains texture files
            if not is_valid_texture_directory(selected_path):
                print(f"{Fore.RED}Directory does not contain valid texture files: {selected_path}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Please select a directory containing PNG files matching pattern *_0X*.png{Style.RESET_ALL}")
                continue
            
            print(f"{Fore.GREEN}Selected directory: {selected_path}{Style.RESET_ALL}")
            return selected_path
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Operation cancelled.{Style.RESET_ALL}")
            raise SystemExit(0)
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
            continue


def resolve_texture_directory(script_dir: Path) -> Path:
    """
    Main function combining auto-detection and manual selection.
    
    Args:
        script_dir: Script directory
        
    Returns:
        Validated Path object to texture directory
    """
    # Try auto-detection first
    detected_dir = detect_texture_directory(script_dir)
    
    if detected_dir:
        print(f"{Fore.GREEN}[Auto-detected] Found texture directory: {detected_dir}{Style.RESET_ALL}")
        return detected_dir
    
    # Auto-detection failed, prompt user
    print(f"{Fore.YELLOW}[Auto-detection] No texture files found in script directory or parent.{Style.RESET_ALL}")
    return prompt_directory_selection(script_dir)


def prompt_auto_compress() -> bool:
    """
    Prompt user if they want to auto-compress textures to DDS format.
    Checks for ImageMagick availability first.
    
    Returns:
        True if user wants compression, False otherwise
        
    Raises:
        SystemExit: If user cancels the operation
    """
    # Check if ImageMagick is available
    if not check_imagemagick():
        print(f"\n{Fore.YELLOW}[Warning] ImageMagick is not available.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}DDS compression requires ImageMagick to be installed.{Style.RESET_ALL}")
        print(f"{Fore.CYAN}To install ImageMagick:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  1. Download from: https://imagemagick.org/script/download.php{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  2. Install ImageMagick on your system{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  3. Make sure 'magick' command is in your system PATH{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  4. Restart this script after installation{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Proceeding without DDS compression...{Style.RESET_ALL}")
        return False
    
    # ImageMagick is available, prompt user
    while True:
        try:
            response = input(f"\n{Fore.YELLOW}Do you want to auto-compress textures to DDS format? (y/n): {Style.RESET_ALL}").strip().lower()
            
            if not response:
                print(f"{Fore.YELLOW}Operation cancelled.{Style.RESET_ALL}")
                raise SystemExit(0)
            
            if response in ('yes', 'y'):
                return True
            elif response in ('no', 'n'):
                return False
            else:
                print(f"{Fore.RED}Please enter 'y' or 'n'.{Style.RESET_ALL}")
                continue
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Operation cancelled.{Style.RESET_ALL}")
            raise SystemExit(0)
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
            continue


# ============================================================================
# DDS Compression Functions
# ============================================================================

def _calculate_channel_variance(channel_data: np.ndarray) -> float:
    """Calculate variance of a channel to determine if it contains meaningful data."""
    if channel_data.size == 0:
        return 0.0
    return float(np.var(channel_data))


def _calculate_channel_statistics(channels) -> List[dict]:
    """Calculate mean and variance statistics for RGB channels."""
    stats = []
    for i, channel in enumerate(channels[:3]):  # Only RGB, ignore alpha if present
        channel_array = np.array(channel, dtype=np.float32)
        mean = float(np.mean(channel_array))
        variance = float(np.var(channel_array))
        stats.append({'channel': i, 'mean': mean, 'variance': variance})
    return stats


def _check_outlier_pattern(stats: List[dict]) -> bool:
    """Check if channel statistics show normal map pattern (outlier + two similar channels)."""
    if len(stats) < 3:
        return False
    
    stats_sorted = sorted(stats, key=lambda x: x['variance'])
    outlier = stats_sorted[0]
    other_two = stats_sorted[1:]
    
    outlier_variance = outlier['variance']
    other_variances = [s['variance'] for s in other_two]
    
    if outlier_variance == 0:
        return False
    
    variance_ratio_1 = other_variances[0] / outlier_variance if outlier_variance > 0 else float('inf')
    variance_ratio_2 = other_variances[1] / outlier_variance if outlier_variance > 0 else float('inf')
    
    if variance_ratio_1 < 3.0 or variance_ratio_2 < 3.0:
        return False
    
    mean_diff = abs(other_two[0]['mean'] - other_two[1]['mean'])
    variance_diff = abs(other_two[0]['variance'] - other_two[1]['variance'])
    
    mean_avg = (other_two[0]['mean'] + other_two[1]['mean']) / 2.0
    variance_avg = (other_two[0]['variance'] + other_two[1]['variance']) / 2.0
    
    mean_similar = mean_avg == 0 or (mean_diff / mean_avg) < 0.2
    variance_similar = variance_avg == 0 or (variance_diff / variance_avg) < 0.3
    
    return mean_similar and variance_similar


def is_normal_map_rgb(image_input) -> bool:
    """
    Detect if texture is a normal map using relative RGB channel statistics.
    
    Normal maps typically have:
    - 2 channels (R, G) with similar mean and variance
    - 1 channel (B) that is an outlier with much lower variance than the other two
    
    Args:
        image_input: Path to image file or PIL Image object
        
    Returns:
        True if texture appears to be a normal map, False otherwise
    """
    try:
        if isinstance(image_input, Path):
            img = Image.open(image_input)
            should_close = True
        else:
            img = image_input
            should_close = False
        
        try:
            img_rgb = img.convert('RGB')
            channels = img_rgb.split()
            stats = _calculate_channel_statistics(channels)
            return _check_outlier_pattern(stats)
        finally:
            if should_close:
                img.close()
    except Exception:
        return False


def _load_and_convert_image(image_input):
    """Load image and convert to RGBA format, handling various input types."""
    if isinstance(image_input, Path):
        img = Image.open(image_input)
        should_close = True
        image_name = image_input.name
    else:
        img = image_input
        should_close = False
        image_name = getattr(image_input, 'filename', 'image')
        if isinstance(image_name, Path):
            image_name = image_name.name
    
    if img.mode not in ('RGBA', 'RGB', 'LA', 'L'):
        if img.mode == 'P' and 'transparency' in img.info:
            img = img.convert('RGBA')
        elif img.mode == 'P':
            img = img.convert('RGB')
        else:
            img = img.convert('RGBA')
    
    return img, should_close, image_name


def _check_channel_variance(channels) -> Tuple[bool, bool, bool, bool]:
    """Check which channels have meaningful variance."""
    has_r = len(channels) >= 1
    has_g = len(channels) >= 2
    has_b = len(channels) >= 3
    has_a = len(channels) >= 4
    
    r_variance = _calculate_channel_variance(np.array(channels[0], dtype=np.float32)) if has_r else 0.0
    g_variance = _calculate_channel_variance(np.array(channels[1], dtype=np.float32)) if has_g else 0.0
    b_variance = _calculate_channel_variance(np.array(channels[2], dtype=np.float32)) if has_b else 0.0
    a_variance = _calculate_channel_variance(np.array(channels[3], dtype=np.float32)) if has_a else 0.0
    
    has_r = has_r and r_variance >= NORMAL_VARIANCE_THRESHOLD
    has_g = has_g and g_variance >= NORMAL_VARIANCE_THRESHOLD
    has_b = has_b and b_variance >= NORMAL_VARIANCE_THRESHOLD
    has_a = has_a and a_variance >= CHANNEL_VARIANCE_THRESHOLD
    
    return has_r, has_g, has_b, has_a


def analyze_texture_channels(image_input) -> Tuple[bool, bool, bool, bool]:
    """
    Analyze all channels of an image to determine which contain meaningful data.
    
    Args:
        image_input: Path to image file or PIL Image object
        
    Returns:
        Tuple of (has_r, has_g, has_b, has_a) boolean flags indicating channel usage
    """
    try:
        img, should_close, image_name = _load_and_convert_image(image_input)
        try:
            channels = img.split()
            return _check_channel_variance(channels)
        finally:
            if should_close:
                img.close()
    except Exception as e:
        print(f"{Fore.YELLOW}Warning: Could not analyze channels for {image_name}: {e}{Style.RESET_ALL}")
        return True, True, True, False


def classify_texture_type(image_path: Path) -> str:
    """
    Classify texture type based on channel usage analysis and RGB statistical detection.
    
    Classification:
    - "RGBA": Color map with alpha (all 4 channels used)
    - "RGB Color": Color map without alpha (3 channels used, no alpha)
    - "RGB Normal": Normal map (detected via RGB channel statistics - 2 similar, 1 outlier)
    
    Args:
        image_path: Path to image file
        
    Returns:
        Texture type classification string
    """
    # Load image once and reuse for both analyses
    try:
        with Image.open(image_path) as img:
            # First check channel variance to see if alpha is used
            has_r, has_g, has_b, has_a = analyze_texture_channels(img)
            
            # If alpha is not used, check if it's a normal map using RGB statistics
            if not has_a:
                if is_normal_map_rgb(img):
                    return "RGB Normal"
            
            # Count active channels
            active_channels = sum([has_r, has_g, has_b, has_a])
            
            if active_channels == 4:
                # All 4 channels used - RGBA color map
                return "RGBA"
            elif active_channels == 3:
                # 3 channels used - RGB color map (no alpha or alpha is uniform)
                return "RGB Color"
            elif active_channels == 2:
                # Only 2 channels used - Normal map (fallback detection)
                return "RGB Normal"
            else:
                # Fallback: assume RGB color if we can't determine
                return "RGB Color"
    except Exception:
        # If we can't load the image, default to RGB Color
        return "RGB Color"


def has_alpha_channel(image_path: Path) -> bool:
    """
    Check if an image has an alpha channel with variance.
    If alpha channel exists but is uniform (all pixels have same value),
    returns False to use DXT1 compression instead of DXT5.
    
    Uses the generic channel variance function for consistency.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if image has alpha channel with variance, False otherwise
    """
    _, _, _, has_a = analyze_texture_channels(image_path)
    return has_a


def convert_png_to_dds(png_path: Path, dds_path: Path, has_alpha: bool) -> Path:
    """
    Convert PNG file to DDS format with DXT1 or DXT5 compression and mipmaps.
    
    Args:
        png_path: Path to source PNG file
        dds_path: Path where DDS file should be created
        has_alpha: Whether image has alpha channel (determines DXT1 vs DXT5)
        
    Returns:
        Path to created DDS file
        
    Raises:
        RuntimeError: If ImageMagick conversion fails
    """
    # Determine compression format
    compression_format = 'dxt5' if has_alpha else 'dxt1'
    
    # Build ImageMagick command
    # Format: magick input.png -define dds:compression=dxt5 -define dds:mipmaps=auto output.dds
    cmd = [
        'magick',
        str(png_path),
        '-define', f'dds:compression={compression_format}',
    ]
    
    # Add mipmap generation if enabled
    if GENERATE_MIPMAPS:
        cmd.extend(['-define', 'dds:mipmaps=auto'])
    
    cmd.append(str(dds_path))
    
    try:
        # Run ImageMagick conversion
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Verify DDS file was created
        if not dds_path.exists():
            raise RuntimeError(f"DDS file was not created: {dds_path}")
        
        return dds_path
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        raise RuntimeError(f"ImageMagick conversion failed: {error_msg}")
    except FileNotFoundError:
        raise RuntimeError(
            "ImageMagick 'magick' command not found. "
            "Please ensure ImageMagick is installed and in your system PATH."
        )


def classify_all_textures(texture_dict: Dict[str, Path]) -> Dict[str, str]:
    """
    Classify all textures in the dictionary by analyzing their channels.
    
    Args:
        texture_dict: Dictionary mapping hash IDs to texture file paths
        
    Returns:
        Dictionary mapping hash IDs to texture type classifications
    """
    classifications: Dict[str, str] = {}
    total = len(texture_dict)
    
    print(f"\n{Fore.CYAN}[Analyzing] Classifying texture types...{Style.RESET_ALL}")
    
    for idx, (hash_str, texture_path) in enumerate(texture_dict.items(), 1):
        if total > 10:
            _show_progress(idx, total, texture_path.name, "Classifying")
        texture_type = classify_texture_type(texture_path)
        classifications[hash_str] = texture_type
    
    # Clear progress line if shown
    if total > 10:
        print('\r' + ' ' * 100 + '\r', end='', flush=True)
    
    return classifications


def get_texture_dimensions(texture_path: Path) -> Tuple[int, int]:
    """
    Get texture dimensions (width, height) from image file.
    
    Args:
        texture_path: Path to texture file
        
    Returns:
        Tuple of (width, height) or (0, 0) if unable to read
    """
    try:
        with Image.open(texture_path) as img:
            return img.size  # Returns (width, height)
    except Exception:
        return (0, 0)


def display_texture_classification(texture_dict: Dict[str, Path], classifications: Dict[str, str], status_dict: Optional[Dict[str, str]] = None) -> None:
    """
    Display texture classification results with statistics and dimensions.

    Args:
        texture_dict: Dictionary mapping hash IDs to texture file paths
        classifications: Dictionary mapping hash IDs to texture type classifications
        status_dict: Optional dictionary mapping hash IDs to cache status ("new", "changed", "unchanged")
    """
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Texture Classification Results{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    
    # Count classifications
    rgba_count = sum(1 for t in classifications.values() if t == "RGBA")
    rgb_color_count = sum(1 for t in classifications.values() if t == "RGB Color")
    rgb_normal_count = sum(1 for t in classifications.values() if t == "RGB Normal")
    
    # Calculate file size statistics
    total_size_bytes = 0
    file_sizes = []
    for texture_path in texture_dict.values():
        try:
            size_bytes = texture_path.stat().st_size
            total_size_bytes += size_bytes
            file_sizes.append(size_bytes)
        except (OSError, PermissionError):
            pass
    
    total_size_mb = total_size_bytes / (1024 * 1024)
    avg_size_mb = (total_size_mb / len(file_sizes)) if file_sizes else 0
    
    # Show detailed list if not too many
    if len(classifications) <= 20:
        print(f"\n{Fore.YELLOW}Detailed Classification:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{'-'*90}{Style.RESET_ALL}")

        # Print column headers
        if status_dict:
            print(f"{Fore.CYAN}{'ID':<17} | {'Type':<13} | {'Size (width x height)':<22} | {'File Size':<12} | {'Status':<10} | {'Filename'}{Style.RESET_ALL}")
        else:
            print(f"{Fore.CYAN}{'ID':<17} | {'Type':<13} | {'Size (width x height)':<22} | {'File Size':<12} | {'Filename'}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{'-'*90}{Style.RESET_ALL}")
        
        # Print each texture with dimensions
        for hash_str, texture_path in texture_dict.items():
            texture_type = classifications.get(hash_str, "Unknown")
            type_color = Fore.GREEN if texture_type == "RGBA" else (Fore.BLUE if texture_type == "RGB Color" else Fore.MAGENTA)

            # Get dimensions - numbers in light blue, parentheses and x in white
            width, height = get_texture_dimensions(texture_path)
            if width > 0 and height > 0:
                dim_str = f"{Fore.WHITE}({Fore.CYAN}{width}{Fore.WHITE}x{Fore.CYAN}{height}{Fore.WHITE}){Style.RESET_ALL}"
            else:
                dim_str = "N/A"

            # Get file size in MB
            try:
                file_size_bytes = texture_path.stat().st_size
                file_size_mb = file_size_bytes / (1024 * 1024)
                size_str = f"{Fore.WHITE}{file_size_mb:.1f} MB{Style.RESET_ALL}"
            except (OSError, PermissionError):
                size_str = "N/A"

            # Get status with color coding
            if status_dict:
                status = status_dict.get(hash_str, "unknown")
                if status == "unchanged":
                    status_color = Fore.GREEN
                    status_display = "Unchanged"
                elif status == "changed":
                    status_color = Fore.YELLOW
                    status_display = "Changed"
                elif status == "new":
                    status_color = Fore.CYAN
                    status_display = "New"
                else:
                    status_color = Fore.WHITE
                    status_display = status.capitalize()

                print(f"{Fore.CYAN}{hash_str:<17}{Style.RESET_ALL} | {type_color}{texture_type:<13}{Style.RESET_ALL} | {dim_str:<22} | {size_str:<12} | {status_color}{status_display:<10}{Style.RESET_ALL} | {texture_path.name}")
            else:
                print(f"{Fore.CYAN}{hash_str:<17}{Style.RESET_ALL} | {type_color}{texture_type:<13}{Style.RESET_ALL} | {dim_str:<22} | {size_str:<12} | {texture_path.name}")

        print(f"{Fore.WHITE}{'-'*90}{Style.RESET_ALL}")
    
    # Print summary below the table
    print(f"\n{Fore.CYAN}Classification Summary:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{'-'*90}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'Type':<25} | {'Count':<10} | {'Total Size':<15} | {'Avg Size':<15}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{'-'*90}{Style.RESET_ALL}")
    
    # Calculate sizes per type
    rgba_size = sum(texture_path.stat().st_size for hash_str, texture_path in texture_dict.items() 
                    if classifications.get(hash_str) == "RGBA" and texture_path.exists())
    rgb_color_size = sum(texture_path.stat().st_size for hash_str, texture_path in texture_dict.items() 
                         if classifications.get(hash_str) == "RGB Color" and texture_path.exists())
    rgb_normal_size = sum(texture_path.stat().st_size for hash_str, texture_path in texture_dict.items() 
                          if classifications.get(hash_str) == "RGB Normal" and texture_path.exists())
    
    rgba_size_mb = rgba_size / (1024 * 1024)
    rgb_color_size_mb = rgb_color_size / (1024 * 1024)
    rgb_normal_size_mb = rgb_normal_size / (1024 * 1024)
    
    rgba_avg = (rgba_size_mb / rgba_count) if rgba_count > 0 else 0
    rgb_color_avg = (rgb_color_size_mb / rgb_color_count) if rgb_color_count > 0 else 0
    rgb_normal_avg = (rgb_normal_size_mb / rgb_normal_count) if rgb_normal_count > 0 else 0
    
    print(f"{Fore.GREEN}{'RGBA (Color + Alpha)':<25}{Style.RESET_ALL} | {Fore.WHITE}{rgba_count:<10}{Style.RESET_ALL} | {Fore.WHITE}{rgba_size_mb:>13.1f} MB{Style.RESET_ALL} | {Fore.WHITE}{rgba_avg:>13.1f} MB{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'RGB Color (no alpha)':<25}{Style.RESET_ALL} | {Fore.WHITE}{rgb_color_count:<10}{Style.RESET_ALL} | {Fore.WHITE}{rgb_color_size_mb:>13.1f} MB{Style.RESET_ALL} | {Fore.WHITE}{rgb_color_avg:>13.1f} MB{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'RGB Normal (2 channels)':<25}{Style.RESET_ALL} | {Fore.WHITE}{rgb_normal_count:<10}{Style.RESET_ALL} | {Fore.WHITE}{rgb_normal_size_mb:>13.1f} MB{Style.RESET_ALL} | {Fore.WHITE}{rgb_normal_avg:>13.1f} MB{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{'-'*90}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'Total textures':<25}{Style.RESET_ALL} | {Fore.WHITE}{len(classifications):<10}{Style.RESET_ALL} | {Fore.WHITE}{total_size_mb:>13.1f} MB{Style.RESET_ALL} | {Fore.WHITE}{avg_size_mb:>13.1f} MB{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{'-'*90}{Style.RESET_ALL}")


def compress_textures_to_dds(texture_dict: Dict[str, Path], classifications: Dict[str, str], output_dir: Path, enable_compression: bool, status_dict: Optional[Dict[str, str]] = None, cache_data: Optional[Dict[str, dict]] = None, texture_dir: Optional[Path] = None) -> Tuple[Dict[str, Path], List[Path], dict]:
    """
    Compress textures to DDS format if compression is enabled.
    Uses texture classification to determine compression format.
    Only compresses files that are new or changed according to cache status.

    Args:
        texture_dict: Dictionary mapping hash IDs to PNG file paths
        classifications: Dictionary mapping hash IDs to texture type classifications
        output_dir: Directory where temporary DDS files should be created (legacy parameter, now uses texture_dir/compressed/)
        enable_compression: Whether to enable compression
        status_dict: Optional dictionary mapping hash IDs to cache status ("new", "changed", "unchanged")
        cache_data: Optional dictionary containing cache data for updating
        texture_dir: Base texture directory for cache and compressed file paths

    Returns:
        Tuple of (updated_texture_dict, dds_cleanup_list, compression_stats)
        - updated_texture_dict: Dictionary with DDS paths instead of PNG (or original if disabled)
        - dds_cleanup_list: List of temporary DDS file paths to delete later
        - compression_stats: Dictionary with compression statistics
    """
    if not enable_compression:
        return texture_dict, [], {'enabled': False, 'dxt1_count': 0, 'dxt5_count': 0, 'failed': 0}

    # Determine compressed directory
    if texture_dir:
        compressed_dir = texture_dir / "compressed"
    else:
        compressed_dir = output_dir

    # Ensure compressed directory exists
    compressed_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{Fore.CYAN}[Compressing] Converting textures to DDS format...{Style.RESET_ALL}")

    updated_dict: Dict[str, Path] = {}
    cleanup_list: List[Path] = []  # Will be empty now since we keep DDS files
    dxt1_count = 0
    dxt5_count = 0
    failed_count = 0
    skipped_count = 0

    total = len(texture_dict)

    # Filter files that need compression (new or changed)
    files_to_compress = []
    for hash_str, png_path in texture_dict.items():
        status = status_dict.get(hash_str, "new") if status_dict else "new"
        if status in ("new", "changed"):
            files_to_compress.append((hash_str, png_path))
        else:
            # For unchanged files, try to use cached DDS
            if cache_data and texture_dir:
                try:
                    relative_path = png_path.relative_to(texture_dir)
                    cache_key = str(relative_path)
                    cache_entry = cache_data.get(cache_key, {})
                    dds_rel_path = cache_entry.get('dds_path')
                    if dds_rel_path:
                        dds_path = texture_dir / dds_rel_path
                        if dds_path.exists():
                            updated_dict[hash_str] = dds_path
                            skipped_count += 1
                            continue
                except (ValueError, KeyError):
                    pass
            # If no cached DDS found, treat as new
            files_to_compress.append((hash_str, png_path))

    # Compress only the files that need it
    for idx, (hash_str, png_path) in enumerate(files_to_compress, 1):
        try:
            # Get texture classification
            texture_type = classifications.get(hash_str, "RGB Color")

            # Determine compression format based on classification
            # RGBA and RGB Normal use DXT5, RGB Color (non-normal, no alpha) uses DXT1
            has_alpha = (texture_type == "RGBA" or texture_type == "RGB Normal")
            compression_format = "DXT5" if has_alpha else "DXT1"

            # Show progress bar with format type and texture type
            format_text = f"{compression_format} ({texture_type})"
            _show_progress(idx, len(files_to_compress), png_path.name, "Compressing", format_type=format_text, use_bar=True)

            # Create DDS filename (replace .png with .dds)
            dds_filename = png_path.stem + '.dds'
            dds_path = compressed_dir / dds_filename

            # Convert PNG to DDS
            convert_png_to_dds(png_path, dds_path, has_alpha)

            # Update dictionary to point to DDS file
            updated_dict[hash_str] = dds_path

            # Update cache entry if we have cache data and texture_dir
            if cache_data is not None and texture_dir:
                try:
                    relative_path = png_path.relative_to(texture_dir)
                    cache_key = str(relative_path)
                    dds_rel_path = f"compressed/{dds_filename}"

                    # Calculate file info
                    current_hash = calculate_file_hash(png_path)
                    file_size = png_path.stat().st_size if png_path.exists() else 0
                    timestamp = png_path.stat().st_mtime if png_path.exists() else time.time()

                    cache_data[cache_key] = {
                        'hash': current_hash,
                        'timestamp': timestamp,
                        'compression_type': compression_format,
                        'file_size': file_size,
                        'dds_path': dds_rel_path
                    }
                except (ValueError, OSError):
                    pass

            # Update statistics
            if has_alpha:
                dxt5_count += 1
            else:
                dxt1_count += 1

        except Exception as e:
            failed_count += 1
            # Clear progress line before showing error
            print('\r' + ' ' * 100 + '\r', end='', flush=True)
            print(f"{Fore.YELLOW}Warning: Failed to compress {png_path.name}: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Using original PNG file instead.{Style.RESET_ALL}")
            # Fall back to original PNG file
            updated_dict[hash_str] = png_path
            # Show progress again after error
            _show_progress(idx, len(files_to_compress), png_path.name, "Compressing", format_type="Failed", use_bar=True)
    
    # Clear progress line
    print('\r' + ' ' * 100 + '\r', end='', flush=True)
    
    # Calculate total compressed texture file size (DDS + PNG fallbacks)
    compressed_total_size = 0
    for texture_path in updated_dict.values():
        try:
            if texture_path.exists():
                compressed_total_size += texture_path.stat().st_size
        except (OSError, PermissionError):
            pass

    # Save cache if we have cache data and texture directory
    if cache_data is not None and texture_dir:
        cache_path = texture_dir / "compressed" / "cache.json"
        save_cache(cache_path, cache_data)

    stats = {
        'enabled': True,
        'dxt1_count': dxt1_count,
        'dxt5_count': dxt5_count,
        'failed': failed_count,
        'skipped': skipped_count,
        'total': total,
        'compressed_total_size': compressed_total_size
    }

    print(f"{Fore.GREEN}Compression complete: {dxt1_count} DXT1, {dxt5_count} DXT5, {failed_count} failed, {skipped_count} skipped{Style.RESET_ALL}")

    return updated_dict, cleanup_list, stats


def cleanup_temp_dds_files(dds_files: List[Path]) -> None:
    """
    Delete temporary DDS files after TPF creation.
    
    Args:
        dds_files: List of DDS file paths to delete
    """
    if not dds_files:
        return
    
    print(f"\n{Fore.CYAN}[Cleanup] Removing temporary DDS files...{Style.RESET_ALL}")
    
    deleted_count = 0
    failed_count = 0
    
    for dds_path in dds_files:
        try:
            if dds_path.exists():
                dds_path.unlink()
                deleted_count += 1
        except Exception as e:
            failed_count += 1
            print(f"{Fore.YELLOW}Warning: Could not delete {dds_path.name}: {e}{Style.RESET_ALL}")
    
    if failed_count == 0:
        print(f"{Fore.GREEN}Cleaned up {deleted_count} temporary DDS file(s).{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}Cleaned up {deleted_count} file(s), {failed_count} failed.{Style.RESET_ALL}")


# ============================================================================
# Scanning Functions
# ============================================================================

def extract_id_from_filename(path: Path) -> Optional[str]:
    """
    Extract and validate hexadecimal ID from filename.
    
    Filename pattern: *_0X[hex_digits].png or *_0x[hex_digits].png
    The hex ID must end immediately before the file extension with no extra characters.
    
    Valid examples:
    - ACBSP_T_0X3263C677.png ✓
    - file_0xABC123.png ✓
    
    Invalid examples:
    - ACBSP_T_0X3263C677 - Copy.png ✗ (has " - Copy" before extension)
    - file_0xABC123_extra.png ✗ (has "_extra" before extension)
    
    Args:
        path: Path object to the file
        
    Returns:
        Uppercase ID string (e.g., "0X3263C677") if valid, None otherwise
    """
    filename = path.stem  # Get filename without extension
    
    match = ID_PATTERN.search(filename)
    if match:
        # Ensure the match ends at the end of the filename (no extra characters after hex ID)
        if match.end() == len(filename):
            hex_part = match.group(1)
            # Validate it's actually hexadecimal and not empty
            if hex_part and len(hex_part) > 0:
                try:
                    int(hex_part, 16)
                    return f"0X{hex_part.upper()}"
                except ValueError:
                    return None
    
    return None


def scan_directory_for_textures(directory: Path) -> Dict[str, Path]:
    """
    Scan directory for PNG files matching ID pattern and build in-memory dictionary.
    
    Args:
        directory: Directory path to scan
        
    Returns:
        Dictionary mapping hash IDs to file paths: {hash_id: file_path}
    """
    texture_dict: Dict[str, Path] = {}
    
    # First pass: collect all matching files for progress calculation
    all_files = []
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_file():
                    path = Path(entry)
                    if path.suffix.lower() in INPUT_FORMATS:
                        all_files.append(path)
    except (PermissionError, OSError) as e:
        print(f"{Fore.RED}Error scanning directory: {e}{Style.RESET_ALL}")
        return texture_dict
    
    total_files = len(all_files)
    
    if total_files == 0:
        return texture_dict
    
    # Second pass: process files with progress indication
    for idx, path in enumerate(all_files, 1):
        _show_progress(idx, total_files, path.name, "Scanning")
        
        id_value = extract_id_from_filename(path)
        if id_value:
            # Check for duplicate IDs
            if id_value in texture_dict:
                print(f"\n{Fore.YELLOW}Warning: Duplicate ID {id_value} found. Keeping first occurrence: {texture_dict[id_value].name}{Style.RESET_ALL}")
            else:
                texture_dict[id_value] = path
    
    # Clear progress line
    if total_files > 0:
        print('\r' + ' ' * 100 + '\r', end='', flush=True)
    
    return texture_dict


# ============================================================================
# TPF Creation Functions
# ============================================================================

def validate_texture_files(texture_dict: Dict[str, Path]) -> Tuple[Dict[str, Path], Dict[str, Path], dict]:
    """
    Validate that all texture files exist.
    
    Args:
        texture_dict: Dictionary mapping hash IDs to file paths
        
    Returns:
        Tuple of (valid_dict, missing_dict, stats_dict)
    """
    print(f"\n{Fore.CYAN}[Validating] Checking texture files...{Style.RESET_ALL}")
    
    valid_dict: Dict[str, Path] = {}
    missing_dict: Dict[str, Path] = {}
    total = len(texture_dict)
    
    if total == 0:
        stats = {'total': 0, 'valid': 0, 'missing': 0}
        return valid_dict, missing_dict, stats
    
    for idx, (hash_str, texture_path) in enumerate(texture_dict.items(), 1):
        if validate_texture_file(texture_path):
            valid_dict[hash_str] = texture_path
            status = "Found"
        else:
            missing_dict[hash_str] = texture_path
            status = "Missing"
        
        # Show progress for operations with more than 10 items
        if total > 10:
            _show_progress(idx, total, texture_path.name, "Validating", status=status, use_bar=True)
    
    # Clear progress line if shown
    if total > 10:
        print('\r' + ' ' * 100 + '\r', end='', flush=True)
    
    stats = {
        'total': total,
        'valid': len(valid_dict),
        'missing': len(missing_dict)
    }
    
    if missing_dict:
        print(f"{Fore.YELLOW}Warning: {len(missing_dict)} texture file(s) not found:{Style.RESET_ALL}")
        for hash_str, path in list(missing_dict.items())[:10]:  # Show first 10
            print(f"  {Fore.YELLOW}✗{Style.RESET_ALL} {hash_str} | {path}")
        if len(missing_dict) > 10:
            print(f"  {Fore.YELLOW}... and {len(missing_dict) - 10} more{Style.RESET_ALL}")
    
    print(f"{Fore.GREEN}Validation complete: {len(valid_dict)}/{total} files found.{Style.RESET_ALL}")
    
    return valid_dict, missing_dict, stats


def generate_texmod_def(texture_dict: Dict[str, Path]) -> str:
    """
    Generate texmod.def content from texture dictionary.
    
    Format: 0XHASH|filename (just filename, not full path)
    
    Args:
        texture_dict: Dictionary mapping hash IDs to file paths
        
    Returns:
        String content for texmod.def file
    """
    print(f"{Fore.CYAN}[Generating] Creating texmod.def...{Style.RESET_ALL}")
    
    lines = []
    for hash_str, texture_path in texture_dict.items():
        filename = texture_path.name
        # Ensure hash is uppercase with 0X prefix
        hash_upper = hash_str.upper()
        if not hash_upper.startswith('0X'):
            hash_upper = '0X' + hash_upper.lstrip('0xX')
        lines.append(f"{hash_upper}|{filename}")
    
    content = '\n'.join(lines)
    if lines:
        content += '\n'  # Add final newline
    
    print(f"{Fore.GREEN}Generated texmod.def with {len(lines)} entries.{Style.RESET_ALL}")
    return content


def create_zip_archive(texture_dict: Dict[str, Path], texmod_def: str, password: bytes) -> bytes:
    """
    Create ZIP archive in memory with texmod.def and texture files.
    Encrypts all entries with ZipCrypto using the provided password.
    
    Args:
        texture_dict: Dictionary mapping hash IDs to file paths
        texmod_def: Content for texmod.def file
        password: ZipCrypto password bytes
        
    Returns:
        Encrypted ZIP archive as bytes
    """
    print(f"\n{Fore.CYAN}[ZIP] Creating archive with ZipCrypto encryption...{Style.RESET_ALL}")
    
    zip_buffer = io.BytesIO()
    
    with ZipFile(zip_buffer, 'w', ZIP_DEFLATED) as zip_file:
        # Add texmod.def with ZipCrypto encryption
        zip_file.writestr('texmod.def', texmod_def.encode('utf-8'), pwd=password)
        
        # Add texture files with ZipCrypto encryption
        total = len(texture_dict)
        for idx, (_, texture_path) in enumerate(texture_dict.items(), 1):
            # Show progress for operations with more than 10 items
            if total > 10:
                _show_progress(idx, total, texture_path.name, "Adding", use_bar=True)
            
            try:
                # Read file and add to ZIP with filename only (flat structure)
                # Encrypt with ZipCrypto using the password
                zip_file.write(texture_path, texture_path.name, pwd=password)
            except Exception as e:
                if total > 10:
                    print('\r' + ' ' * 100 + '\r', end='', flush=True)
                print(f"{Fore.RED}Error adding {texture_path.name}: {e}{Style.RESET_ALL}")
                raise
    
    # Clear progress line if shown
    if total > 10:
        print('\r' + ' ' * 100 + '\r', end='', flush=True)
    
    zip_bytes = zip_buffer.getvalue()
    zip_buffer.close()
    
    print(f"{Fore.GREEN}ZIP archive created with ZipCrypto encryption{Style.RESET_ALL}")
    
    return zip_bytes


def apply_xor_obfuscation(encrypted_bytes: bytes, xor_key: int = XOR_KEY) -> bytes:
    """
    Apply XOR obfuscation to encrypted ZIP bytes.
    
    XOR key: 0x3FA43FA4 = [0xA4, 0x3F, 0xA4, 0x3F] (little-endian)
    Process in 4-byte chunks, handling remainder bytes cyclically.
    
    Args:
        encrypted_bytes: Encrypted ZIP archive bytes
        xor_key: XOR key (default: 0x3FA43FA4)
        
    Returns:
        XOR-obfuscated bytes
    """
    print(f"{Fore.YELLOW}[Obfuscating] Applying XOR mask 0x{XOR_KEY:08X}...{Style.RESET_ALL}")
    
    # Convert key to bytes (little-endian)
    key_bytes = struct.pack('<I', xor_key)  # [0xA4, 0x3F, 0xA4, 0x3F]
    
    # Convert to bytearray for mutation
    result = bytearray(encrypted_bytes)
    
    # Process in 4-byte chunks
    for i in range(0, len(result), 4):
        chunk_size = min(4, len(result) - i)
        for j in range(chunk_size):
            result[i + j] ^= key_bytes[j]
    
    print(f"{Fore.GREEN}XOR obfuscation complete.{Style.RESET_ALL}")
    return bytes(result)


def write_tpf_file(tpf_bytes: bytes, output_path: Path) -> None:
    """
    Write final TPF file to disk.
    
    Args:
        tpf_bytes: Final TPF file bytes (XOR-obfuscated)
        output_path: Path where TPF file should be written
    """
    print(f"{Fore.CYAN}[Writing] Saving TPF file...{Style.RESET_ALL}")
    
    try:
        output_path.write_bytes(tpf_bytes)
        size_mb = len(tpf_bytes) / (1024 * 1024)
        print(f"{Fore.GREEN}TPF file written: {output_path.name} ({size_mb:.2f} MB){Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error writing TPF file: {e}{Style.RESET_ALL}")
        raise


# ============================================================================
# Display Functions
# ============================================================================

def calculate_total_file_size(texture_dict: Dict[str, Path]) -> int:
    """
    Calculate total size of all files in texture dictionary.
    
    Args:
        texture_dict: Dictionary mapping hash IDs to file paths
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    for path in texture_dict.values():
        try:
            if path.exists():
                total_size += path.stat().st_size
        except (OSError, PermissionError):
            pass  # Skip files that can't be accessed
    return total_size


def _show_progress(current: int, total: int, filename: str, operation: str, 
                   status: str = None, format_type: str = None, use_bar: bool = False) -> None:
    """
    Unified progress display function.
    
    Args:
        current: Current file number
        total: Total number of files
        filename: Name of file being processed
        operation: Operation name (e.g., "Scanning", "Validating", "Compressing", "Adding")
        status: Optional status string (e.g., "Found", "Missing")
        format_type: Optional format type string (e.g., "DXT5 (RGBA)")
        use_bar: Whether to show progress bar (default: False for simple progress)
    """
    if use_bar:
        percentage = int((current / total) * 100) if total > 0 else 0
        bar_length = 30
        filled = int(bar_length * current / total) if total > 0 else 0
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Build operation text
        if status:
            status_color = Fore.GREEN if status == "Found" else Fore.YELLOW
            op_text = f"{status_color}{status}{Style.RESET_ALL}"
        elif format_type:
            op_text = f"{Fore.CYAN}{operation} {format_type}{Style.RESET_ALL}"
        else:
            op_text = f"{Fore.CYAN}{operation}{Style.RESET_ALL}"
        
        # Determine filename length based on context
        max_filename_len = 35 if format_type else 40
        
        progress_text = (
            f"\r{Fore.CYAN}[{current}/{total}]{Style.RESET_ALL} "
            f"{op_text} | "
            f"{Fore.CYAN}{bar}{Style.RESET_ALL} {percentage}% | "
            f"{filename[:max_filename_len]}"
        )
    else:
        progress_text = f"{Fore.CYAN}[{current}/{total}] {operation}: {filename}{Style.RESET_ALL}"
    
    print(f'\r{progress_text}', end='', flush=True)


def display_scan_summary(texture_dict: Dict[str, Path], directory: Path) -> None:
    """
    Display formatted summary of scanned textures.
    
    Args:
        texture_dict: Dictionary mapping hash IDs to file paths
        directory: Directory that was scanned
    """
    file_count = len(texture_dict)
    
    # Header
    print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Texture Scan Complete{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}Directory:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
    print(f"{directory}")
    
    print(f"\n{Fore.CYAN}Statistics:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
    print(f"Textures found: {Fore.GREEN}{file_count}{Style.RESET_ALL}")
    
    if file_count > 0:
        print(f"\n{Fore.YELLOW}Found textures:{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'-'*60}{Style.RESET_ALL}")
        
        for hash_id, path in texture_dict.items():
            filename = path.name
            print(f"ID: {Fore.CYAN}{hash_id:<15}{Style.RESET_ALL} | File: {filename}")
        
        print(f"{Fore.YELLOW}{'-'*60}{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.YELLOW}No matching texture files found.{Style.RESET_ALL}")


def display_build_summary(stats: dict, tpf_path: Path, build_time: float, compression_stats: Optional[dict] = None, original_file_size: Optional[int] = None, compressed_file_size: Optional[int] = None) -> None:
    """
    Display formatted summary of the build process.
    
    Args:
        stats: Statistics dictionary with 'total', 'valid', 'missing' keys
        tpf_path: Path to created TPF file
        build_time: Build time in seconds
        compression_stats: Optional compression statistics dictionary
        original_file_size: Optional original PNG file size in bytes
        dds_file_size: Optional total DDS file size in bytes
    """
    print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}TPF Build Complete{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}Statistics:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
    print(f"Total textures scanned: {Fore.YELLOW}{stats['total']}{Style.RESET_ALL}")
    print(f"Valid textures found: {Fore.GREEN}{stats['valid']}{Style.RESET_ALL}")
    
    if stats['missing'] > 0:
        print(f"Missing textures: {Fore.YELLOW}{stats['missing']}{Style.RESET_ALL}")
    
    # Display compression statistics if available
    if compression_stats and compression_stats.get('enabled'):
        print(f"\n{Fore.CYAN}Compression:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
        print(f"DXT1 (no alpha): {Fore.GREEN}{compression_stats['dxt1_count']}{Style.RESET_ALL}")
        print(f"DXT5 (with alpha): {Fore.GREEN}{compression_stats['dxt5_count']}{Style.RESET_ALL}")
        if compression_stats.get('failed', 0) > 0:
            print(f"Failed conversions: {Fore.YELLOW}{compression_stats['failed']}{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}Output:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
    print(f"TPF file: {Fore.GREEN}{tpf_path}{Style.RESET_ALL}")
    
    # Display file sizes together for comparison
    if tpf_path.exists():
        tpf_size = tpf_path.stat().st_size
        tpf_mb = tpf_size / (1024 * 1024)
        
        if compressed_file_size is not None:
            # Show all three: PNG vs compressed vs TPF
            original_mb = original_file_size / (1024 * 1024) if original_file_size else 0
            compressed_mb = compressed_file_size / (1024 * 1024)
            print(f"Original PNG files size: {Fore.CYAN}{original_mb:.2f} MB{Style.RESET_ALL}")
            print(f"Compressed texture files size: {Fore.CYAN}{compressed_mb:.2f} MB{Style.RESET_ALL}")
            print(f"Final TPF file size: {Fore.GREEN}{tpf_mb:.2f} MB{Style.RESET_ALL}")
        elif original_file_size is not None:
            # Show only PNG vs TPF (no compression)
            original_mb = original_file_size / (1024 * 1024)
            print(f"Original PNG files size: {Fore.CYAN}{original_mb:.2f} MB{Style.RESET_ALL}")
            print(f"Final TPF file size: {Fore.GREEN}{tpf_mb:.2f} MB{Style.RESET_ALL}")
        else:
            # Show only TPF
            print(f"Final TPF file size: {Fore.GREEN}{tpf_mb:.2f} MB{Style.RESET_ALL}")
    
    print(f"Build time: {Fore.CYAN}{build_time:.2f} seconds{Style.RESET_ALL}")
    
    if stats['missing'] > 0:
        print(f"\n{Fore.YELLOW}Warning: Some textures were missing and were not included in the TPF.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}The TPF may not work correctly if those textures are required.{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.GREEN}All textures found and included successfully!{Style.RESET_ALL}")


# ============================================================================
# Main Function
# ============================================================================

def _resolve_and_validate_directory(script_dir: Path) -> Optional[Path]:
    """Resolve and validate texture directory."""
    texture_dir = resolve_texture_directory(script_dir)
    
    is_valid, error_msg = is_valid_target_directory(texture_dir)
    if not is_valid:
        print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Cannot write TPF file to this directory.{Style.RESET_ALL}")
        return None
    
    if error_msg and "Warning" in error_msg:
        print(f"{Fore.YELLOW}{error_msg}{Style.RESET_ALL}")
    
    return texture_dir


def _scan_and_validate_textures(texture_dir: Path) -> Tuple[Optional[Dict[str, Path]], Optional[dict]]:
    """Scan directory and validate texture files."""
    print(f"\n{Fore.CYAN}[Scanning] Processing texture files in: {texture_dir}{Style.RESET_ALL}")
    texture_dict = scan_directory_for_textures(texture_dir)
    
    if not texture_dict:
        print(f"{Fore.RED}No valid texture files found in directory.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Expected PNG files matching pattern: *_0X*.png{Style.RESET_ALL}")
        return None, None
    
    display_scan_summary(texture_dict, texture_dir)
    
    valid_dict, missing_dict, stats = validate_texture_files(texture_dict)
    
    if not valid_dict:
        print(f"{Fore.RED}No valid texture files found. Cannot create TPF.{Style.RESET_ALL}")
        return None, None
    
    return valid_dict, stats


def _classify_and_compress_textures(valid_dict: Dict[str, Path], texture_dir: Path) -> Tuple[Dict[str, Path], List[Path], Optional[dict], Optional[int], Optional[int]]:
    """Classify textures and compress if requested."""
    classifications = classify_all_textures(valid_dict)

    # Load cache data
    cache_path = texture_dir / "compressed" / "cache.json"
    cache_data = load_cache(cache_path)

    # Compare with cache to get status
    status_dict = compare_with_cache(valid_dict, cache_data, texture_dir)

    # Display classifications with status
    display_texture_classification(valid_dict, classifications, status_dict)

    # Get compression setting from config (defaults to True)
    enable_compression = ENABLE_COMPRESSION
    
    # Check ImageMagick availability if compression is enabled
    if enable_compression and not check_imagemagick():
        print(f"\n{Fore.YELLOW}[Warning] ImageMagick is not available.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}DDS compression requires ImageMagick to be installed.{Style.RESET_ALL}")
        print(f"{Fore.CYAN}To install ImageMagick:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  1. Download from: https://imagemagick.org/script/download.php{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  2. Install ImageMagick on your system{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  3. Make sure 'magick' command is in your system PATH{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Proceeding without DDS compression...{Style.RESET_ALL}")
        enable_compression = False
    
    original_file_size = calculate_total_file_size(valid_dict)

    dds_cleanup_list = []
    compression_stats = None
    compressed_file_size = None

    if enable_compression:
        valid_dict, dds_cleanup_list, compression_stats = compress_textures_to_dds(
            valid_dict, classifications, texture_dir, enable_compression,
            status_dict=status_dict, cache_data=cache_data, texture_dir=texture_dir
        )
        if compression_stats.get('compressed_total_size'):
            compressed_file_size = compression_stats['compressed_total_size']
            original_mb = original_file_size / (1024 * 1024)
            compressed_mb = compressed_file_size / (1024 * 1024)
            print(f"\n{Fore.CYAN}Original PNG files size: {Fore.CYAN}{original_mb:.2f} MB{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Compressed texture files size: {Fore.CYAN}{compressed_mb:.2f} MB{Style.RESET_ALL}")

    return valid_dict, dds_cleanup_list, compression_stats, original_file_size, compressed_file_size


def _build_tpf_file(valid_dict: Dict[str, Path], texture_dir: Path) -> Path:
    """Build and write TPF file."""
    texmod_def = generate_texmod_def(valid_dict)
    zip_bytes = create_zip_archive(valid_dict, texmod_def, ZIPCRYPTO_PASSWORD)
    tpf_bytes = apply_xor_obfuscation(zip_bytes)
    
    tpf_filename = f"{texture_dir.name}.tpf"
    tpf_path = texture_dir / tpf_filename
    write_tpf_file(tpf_bytes, tpf_path)
    
    return tpf_path


def main() -> None:
    """Main execution function."""
    script_dir = Path(__file__).resolve().parent
    start_time = time.time()
    dds_cleanup_list: List[Path] = []
    compression_stats: Optional[dict] = None
    compressed_file_size: Optional[int] = None
    
    try:
        texture_dir = _resolve_and_validate_directory(script_dir)
        if not texture_dir:
            return

        # Ensure compressed directory exists
        compressed_dir = texture_dir / "compressed"
        compressed_dir.mkdir(parents=True, exist_ok=True)

        result = _scan_and_validate_textures(texture_dir)
        if result[0] is None:
            return
        valid_dict, stats = result
        
        result = _classify_and_compress_textures(valid_dict, texture_dir)
        valid_dict, dds_cleanup_list, compression_stats, original_file_size, compressed_file_size = result
        
        tpf_path = _build_tpf_file(valid_dict, texture_dir)

        build_time = time.time() - start_time
        display_build_summary(stats, tpf_path, build_time, compression_stats, original_file_size, compressed_file_size)
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Operation cancelled by user.{Style.RESET_ALL}")
    except SystemExit:
        raise
    except Exception as e:
        print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
        raise


if __name__ == "__main__":
    main()

