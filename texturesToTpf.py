"""
Combined script to scan directory for PNG texture files, extract hexadecimal IDs,
and create TPF (TexMod Package File) format directly without intermediate log files.
Operates entirely in-memory using a dictionary to map hash IDs to texture paths.
"""

import os
import re
import io
import time
import struct
import shutil
from pathlib import Path
from typing import Optional, Dict, Tuple

try:
    import msvcrt
except ImportError:
    msvcrt = None

try:
    from zipencrypt import ZipFile, ZIP_DEFLATED
except ImportError:
    raise ImportError(
        "zipencrypt package is required for ZipCrypto encryption. "
        "Install it with: pip install zipencrypt"
    )

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    # Fallback if colorama is not installed
    class Fore:
        GREEN = ''
        YELLOW = ''
        CYAN = ''
        RED = ''
        RESET = ''
    class Style:
        RESET_ALL = ''
    def init(**kwargs):
        pass

# Constants
INPUT_FILE_FORMAT = ".png"
EXIT_DELAY_SECONDS = 10
XOR_KEY = 0x3FA43FA4
# Pattern: _0X or _0x followed by hex digits, must end at end of filename (before extension)
ID_PATTERN = re.compile(r'_0[xX]([0-9A-Fa-f]+)$')
# ZipCrypto password: 42-byte hardcoded password from TexMod specification
ZIPCRYPTO_PASSWORD = bytes([
    0x73, 0x2A, 0x63, 0x7D, 0x5F, 0x0A, 0xA6, 0xBD, 0x7D, 0x65,
    0x7E, 0x67, 0x61, 0x2A, 0x7F, 0x7F, 0x74, 0x61, 0x67, 0x5B,
    0x60, 0x70, 0x45, 0x74, 0x5C, 0x22, 0x74, 0x5D, 0x6E, 0x6A,
    0x73, 0x41, 0x77, 0x6E, 0x46, 0x47, 0x77, 0x49, 0x0C, 0x4B,
    0x46, 0x6F
])


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
                    if path.suffix.lower() == INPUT_FILE_FORMAT.lower():
                        id_value = extract_id_from_filename(path)
                        if id_value:
                            return True
    except (PermissionError, OSError):
        return False
    
    return False


def validate_texture_file(path: Path) -> bool:
    """
    Validate that a texture file exists and is readable.
    
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
    
    if path.suffix.lower() != INPUT_FILE_FORMAT.lower():
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
                    if path.suffix.lower() == INPUT_FILE_FORMAT.lower():
                        all_files.append(path)
    except (PermissionError, OSError) as e:
        print(f"{Fore.RED}Error scanning directory: {e}{Style.RESET_ALL}")
        return texture_dict
    
    total_files = len(all_files)
    
    if total_files == 0:
        return texture_dict
    
    # Second pass: process files with progress indication
    for idx, path in enumerate(all_files, 1):
        show_progress(idx, total_files, path.name, "Scanning")
        
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
            show_validation_progress(idx, total, texture_path.name, status)
    
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
                show_zip_progress(idx, total + 1, texture_path.name)  # +1 for texmod.def
            
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
    
    size_mb = len(zip_bytes) / (1024 * 1024)
    print(f"{Fore.GREEN}ZIP archive created with ZipCrypto encryption: {size_mb:.2f} MB{Style.RESET_ALL}")
    
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

def show_progress(current: int, total: int, filename: str, operation: str) -> None:
    """
    Display progress indicator during file operations.
    
    Args:
        current: Current file number
        total: Total number of files
        filename: Name of file being processed
        operation: Operation name (e.g., "Scanning", "Validating")
    """
    progress_text = f"{Fore.CYAN}[{current}/{total}] {operation}: {filename}{Style.RESET_ALL}"
    print(f'\r{progress_text}', end='', flush=True)


def show_validation_progress(current: int, total: int, filename: str, status: str) -> None:
    """
    Display progress indicator during texture validation with progress bar.
    
    Args:
        current: Current file number
        total: Total number of files
        filename: Name of file being checked
        status: Status string (Found/Missing)
    """
    percentage = int((current / total) * 100) if total > 0 else 0
    bar_length = 30
    filled = int(bar_length * current / total) if total > 0 else 0
    bar = '█' * filled + '░' * (bar_length - filled)
    
    status_color = Fore.GREEN if status == "Found" else Fore.YELLOW
    progress_text = (
        f"\r{Fore.CYAN}[{current}/{total}]{Style.RESET_ALL} "
        f"{status_color}{status}{Style.RESET_ALL} | "
        f"{Fore.CYAN}{bar}{Style.RESET_ALL} {percentage}% | "
        f"{filename[:40]}"
    )
    print(progress_text, end='', flush=True)


def show_zip_progress(current: int, total: int, filename: str) -> None:
    """
    Display progress indicator during ZIP creation with progress bar.
    
    Args:
        current: Current file number
        total: Total number of files
        filename: Name of file being added
    """
    percentage = int((current / total) * 100) if total > 0 else 0
    bar_length = 30
    filled = int(bar_length * current / total) if total > 0 else 0
    bar = '█' * filled + '░' * (bar_length - filled)
    
    progress_text = (
        f"\r{Fore.CYAN}[{current}/{total}]{Style.RESET_ALL} "
        f"{Fore.CYAN}Adding{Style.RESET_ALL} | "
        f"{Fore.CYAN}{bar}{Style.RESET_ALL} {percentage}% | "
        f"{filename[:40]}"
    )
    print(progress_text, end='', flush=True)


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


def display_build_summary(stats: dict, tpf_path: Path, build_time: float) -> None:
    """
    Display formatted summary of the build process.
    
    Args:
        stats: Statistics dictionary with 'total', 'valid', 'missing' keys
        tpf_path: Path to created TPF file
        build_time: Build time in seconds
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
    
    print(f"\n{Fore.CYAN}Output:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
    print(f"TPF file: {Fore.GREEN}{tpf_path}{Style.RESET_ALL}")
    
    if tpf_path.exists():
        size_mb = tpf_path.stat().st_size / (1024 * 1024)
        print(f"File size: {Fore.CYAN}{size_mb:.2f} MB{Style.RESET_ALL}")
    
    print(f"Build time: {Fore.CYAN}{build_time:.2f} seconds{Style.RESET_ALL}")
    
    if stats['missing'] > 0:
        print(f"\n{Fore.YELLOW}Warning: Some textures were missing and were not included in the TPF.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}The TPF may not work correctly if those textures are required.{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.GREEN}All textures found and included successfully!{Style.RESET_ALL}")


def interactive_countdown(duration: int) -> None:
    """
    Interactive countdown with pause/resume functionality.
    
    Press SPACE to pause/unpause, any other key to exit immediately.
    
    Args:
        duration: Countdown duration in seconds
    """
    if msvcrt is None:
        print(f"\n{Fore.YELLOW}Keyboard input not available on this platform.{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Closing in {duration} seconds...{Style.RESET_ALL}")
        time.sleep(duration)
        return
    
    remaining = duration
    paused = False
    start_time = time.time()
    
    print(f"\n{Fore.YELLOW}Press SPACE to pause/unpause, or any other key to exit immediately.{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Closing in {remaining} seconds...{Style.RESET_ALL}", end='', flush=True)
    
    while remaining > 0:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b' ':  # Space key
                paused = not paused
                if paused:
                    print(f"\r{Fore.YELLOW}PAUSED - Press SPACE to resume, or any other key to exit.{Style.RESET_ALL}", 
                          end='', flush=True)
                else:
                    start_time = time.time() - (duration - remaining)  # Adjust start time
                    print(f"\r{Fore.CYAN}Closing in {remaining} seconds...{Style.RESET_ALL}", 
                          end='', flush=True)
            else:  # Any other key
                print(f"\r{Fore.YELLOW}Exiting immediately...{Style.RESET_ALL}")
                return
        
        if not paused:
            elapsed = time.time() - start_time
            new_remaining = max(0, duration - int(elapsed))
            
            if new_remaining != remaining:
                remaining = new_remaining
                if remaining > 0:
                    print(f"\r{Fore.CYAN}Closing in {remaining} seconds...{Style.RESET_ALL}", 
                          end='', flush=True)
        
        time.sleep(0.1)
    
    print(f"\r{Fore.GREEN}Closing now...{Style.RESET_ALL}")


# ============================================================================
# Main Function
# ============================================================================

def main() -> None:
    """Main execution function."""
    script_dir = Path(__file__).resolve().parent
    start_time = time.time()
    
    try:
        # Step 1: Resolve texture directory (auto-detect or prompt)
        texture_dir = resolve_texture_directory(script_dir)
        
        # Step 2: Validate target directory is writable
        is_valid, error_msg = is_valid_target_directory(texture_dir)
        if not is_valid:
            print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Cannot write TPF file to this directory.{Style.RESET_ALL}")
            return
        if error_msg and "Warning" in error_msg:
            print(f"{Fore.YELLOW}{error_msg}{Style.RESET_ALL}")
        
        # Step 3: Scan directory and build texture dictionary
        print(f"\n{Fore.CYAN}[Scanning] Processing texture files in: {texture_dir}{Style.RESET_ALL}")
        texture_dict = scan_directory_for_textures(texture_dir)
        
        if not texture_dict:
            print(f"{Fore.RED}No valid texture files found in directory.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Expected PNG files matching pattern: *_0X*.png{Style.RESET_ALL}")
            return
        
        # Step 4: Display scan summary
        display_scan_summary(texture_dict, texture_dir)
        
        # Step 5: Validate texture files exist
        valid_dict, missing_dict, stats = validate_texture_files(texture_dict)
        
        if not valid_dict:
            print(f"{Fore.RED}No valid texture files found. Cannot create TPF.{Style.RESET_ALL}")
            return
        
        # Step 6: Generate texmod.def
        texmod_def = generate_texmod_def(valid_dict)
        
        # Step 7: Create ZIP archive with ZipCrypto encryption
        zip_bytes = create_zip_archive(valid_dict, texmod_def, ZIPCRYPTO_PASSWORD)
        
        # Step 8: Apply XOR obfuscation
        tpf_bytes = apply_xor_obfuscation(zip_bytes)
        
        # Step 9: Write TPF file to texture directory
        tpf_filename = f"{texture_dir.name}.tpf"
        tpf_path = texture_dir / tpf_filename
        write_tpf_file(tpf_bytes, tpf_path)
        
        # Step 10: Display build summary
        build_time = time.time() - start_time
        display_build_summary(stats, tpf_path, build_time)
        
        # Step 11: Interactive countdown before exit
        interactive_countdown(EXIT_DELAY_SECONDS)
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Operation cancelled by user.{Style.RESET_ALL}")
    except SystemExit:
        raise  # Re-raise SystemExit to allow clean exit
    except Exception as e:
        print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
        raise


if __name__ == "__main__":
    main()

