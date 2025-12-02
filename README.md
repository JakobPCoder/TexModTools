# TexModTools

A collection of Python scripts and tools for creating texture mods for DirectX 9 games using OpenTexMod.

## About

TexModTools provides a complete workflow for extracting, processing, and packaging game textures. The tools support the modern OpenTexMod implementation, allowing you to create texture replacement mods for classic DirectX 9 games.

## Copyright Notice

- **TexModTools**
- First published 2022 - Copyright, Jakob Wapenhensch (2025 updates)
- Check LICENSE.md for full license text
- https://creativecommons.org/licenses/by-nc/4.0/
- https://creativecommons.org/licenses/by-nc/4.0/legalcode

## Contents

- **OpenTexMod** - Modern OpenTexMod beta_v1_r21 with enhanced compatibility
- **Python Scripts** - Automated tools for texture processing and TPF creation

## Requirements

### Python Dependencies

Install the required Python packages using pip:

```bash
pip install Pillow numpy zipencrypt colorama
```

**Package Details:**
- `Pillow` - Image processing for alpha channel detection
- `numpy` - Numerical operations for alpha variance calculation
- `zipencrypt` - ZipCrypto encryption for TPF format
- `colorama` - Colored terminal output (optional, falls back gracefully if not installed)

**Note:** For the optional DDS compression feature in `TexturesToTpf.py`, ImageMagick must be installed separately on your system (download from [ImageMagick's official website](https://imagemagick.org/script/download.php)) and the `magick` command must be in your system PATH.

## Scripts Overview

### 1. `1toPng.py` - DDS to PNG Converter

Converts `.dds` texture files extracted from games to `.png` format while preserving the alpha channel.

**Usage:**
1. Place `1toPng.py` in a folder containing `.dds` files
2. Run: `python 1toPng.py`
3. The script converts all `.dds` files to `.png` and removes the original `.dds` files

**Features:**
- Preserves alpha channel (BGRA → RGBA conversion)
- Batch processes all DDS files in the directory
- Automatically removes source files after conversion

### 2. `splitAlpha.py` - Alpha Channel Separator

Separates alpha channels from PNG textures, creating separate alpha mask files when needed.

**Usage:**
1. Place `splitAlpha.py` in a folder containing `.png` files with alpha channels
2. Run: `python splitAlpha.py`
3. For textures with non-constant alpha:
   - RGB version overwrites the original file
   - Alpha channel saved as `*_a.png` (grayscale)

**Features:**
- Detects constant alpha channels (discards if uniform)
- Creates separate alpha mask files (`*_a.png`) for textures with varying transparency
- Preserves RGB data while separating alpha for editing workflows

### 3. `TexturesToTpf.py` - TPF Package Creator

Automatically scans a directory for PNG texture files, extracts hexadecimal IDs from filenames, and creates a TPF (TexMod Package File) format archive.

**Usage:**
1. Place `TexturesToTpf.py` in a folder containing PNG textures with `*_0X[hex].png` naming pattern
2. Run: `python TexturesToTpf.py` or double-click `Run_TexturesToTpf.bat`
3. The script will:
   - Auto-detect the texture directory (or prompt for selection)
   - **Prompt: Ask if you want to auto-compress textures to DDS format (y/n) for smaller file sizes**
   - Scan for valid texture files matching the ID pattern
   - Validate all files exist
   - Generate `texmod.def` definition file
   - Create encrypted ZIP archive with ZipCrypto
   - Apply XOR obfuscation
   - Output a `.tpf` file ready for use

**Filename Pattern:**
Textures must follow the pattern: `*_0X[hexadecimal].png` or `*_0x[hexadecimal].png`

**Examples:**
- `ACBSP_T_0X3263C677.png` ✓
- `texture_0xABC123.png` ✓
- `file_0X12345678.png` ✓

**Features:**
- Auto-detection of texture directories
- **Optional DDS compression: Converts PNG to DXT1 (no alpha) or DXT5 (with alpha variance) format using ImageMagick; generates mipmaps by default for better performance**
- **Alpha channel detection: Uses variance threshold to decide compression type (avoids DXT5 for uniform alpha to save space)**
- Progress indicators for large texture sets (>10 files)
- Validation of texture files before packaging
- Duplicate ID detection and warnings
- Full TPF format compliance (XOR obfuscation + ZipCrypto encryption)
- Detailed build summary with file size comparisons (PNG vs. DDS vs. TPF)
- Interactive countdown before exit (press SPACE to pause, any key to exit)
- Automatic cleanup of temporary DDS files
- Fallback to original PNG if compression fails

## Complete Workflow

### Step 1: Extract Textures
Use **OpenTexMod** to extract `.dds` texture files from your game:
1. Launch `OpenTexMod.exe`
2. Select Main -> Use Global hook
3. Start your game
4. Use the hotkeys from OpenTexMod to dump textures
4. Textures are saved as `.dds` files

### Step 2: Convert to PNG
1. Place `1toPng.py` in the folder with extracted `.dds` files
2. Run: `python 1toPng.py`
3. All `.dds` files are converted to `.png` format

### Step 3: Edit Textures (Optional)
Edit your textures using Photoshop, GIMP, or any image editor. **Important:** Keep the `_0X[hex].png` suffix intact from the underscore to the end of the filename!

### Step 4: Split Alpha Channels (Optional)
If you need to edit alpha channels separately:
1. Place `splitAlpha.py` in your texture folder
2. Run: `python splitAlpha.py`
3. Alpha masks are saved as `*_a.png` files

### Step 5: Create TPF Package
1. Ensure all textures follow the `*_0X[hex].png` naming pattern
2. Place `TexturesToTpf.py` in the texture folder (or run from anywhere)
3. Run: `python TexturesToTpf.py` or double-click `Run_TexturesToTpf.bat`
4. **The script will prompt: Do you want to auto-compress to DDS? (y/n)**
5. The script creates a `.tpf` file ready for use

### Step 6: Use Your Mod
1. Launch **OpenTexMod.exe**
2. Select Main -> Use Global hook
3. Start your game
4. Select Open texture/package and select your `.tpf` file
6. Select Main -> Set as Template
7. Press Main -> Update (Reload)

From now on for every game start:
1. Start openTexMod
2. Start your game
That's it!


## Technical Details

### TPF Format
The TPF (TexMod Package File) format is a specialized archive format:
- **Outer Layer:** XOR obfuscation with key `0x3FA43FA4`
- **Container:** Standard ZIP archive
- **Encryption:** Legacy ZipCrypto with hardcoded password
- **Contents:** `texmod.def` definition file + texture images

The `TexturesToTpf.py` script handles all format requirements automatically, ensuring compatibility with both TexMod and OpenTexMod.

### Texture ID Format
Texture IDs are extracted from filenames using the pattern `_0X[hex]` or `_0x[hex]` at the end of the filename (before the extension). These IDs correspond to CRC32 hashes computed by TexMod/OpenTexMod during texture extraction.

## Troubleshooting

### ImageMagick/Wand Issues
If `TexturesToTpf.py` fails during DDS compression with 'magick' errors:
- Ensure ImageMagick is installed and in your system PATH
- Verify installation: `magick -version`
- If compression fails for specific files, the script falls back to using the original PNG (check the console output for warnings)

### DDS Compression Issues
- **Command not found:** Install ImageMagick if prompted during compression.
- **Conversion failures:** Some PNGs may not convert due to format issues; the script skips them and uses PNG fallback. Review the build summary for failed counts.
- **Verify compression:** Check the final summary for DXT1/DXT5 counts and size reductions. Test the TPF in OpenTexMod to ensure textures load correctly.

### Missing Textures
If `TexturesToTpf.py` reports missing textures:
- Verify all PNG files exist in the scanned directory
- Check file permissions (ensure files are readable)
- Ensure filenames match the required pattern `*_0X[hex].png`

### TPF Not Loading
If your TPF file doesn't load in TexMod/OpenTexMod:
- Verify texture IDs match the original game textures
- Ensure filenames follow the exact pattern required
- Check that all referenced textures are included in the package

## Updates

### Version 0.4
- Enhanced `TexturesToTpf.py` with optional DDS compression for smaller TPF files
- Added intelligent alpha channel variance detection (DXT1 for uniform/no alpha, DXT5 for varying alpha)
- Automatic mipmap generation for improved texture performance
- Improved compression statistics and file size reporting (PNG vs compressed vs TPF)
- Fixed size calculation bug when compression fails (now includes fallback PNGs in totals)


### Version 0.3
- Added `1toPng.py` - Improved DDS to PNG conversion
- Added `splitAlpha.py` - Alpha channel separation tool
- Added `TexturesToTpf.py` - Automated TPF creation with full format compliance
- Added `Run_TexturesToTpf.bat` - Convenient batch launcher
- Replaced legacy `DdsToPng.py` and `CreateLog.py` scripts
- Enhanced workflow documentation

### Version 0.2
- Added OpenTexMod files (no maintained download source available)
- Fixed various issues

### Version 0.1
- Initial release
- Added DdsToPng.py
- Added CreateLog.py

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. See LICENSE.md for details.

## Credits

- **OpenTexMod** - Open-source implementation by the modding community
- **TexModTools Scripts** - Jakob Wapenhensch (2022, 2025 updates)
