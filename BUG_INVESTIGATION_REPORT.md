# TPF Format Bug Investigation Report

## Git History Analysis

### Timeline of Changes

1. **Commit `ddc365d`** (Dec 2024) - **LAST WORKING VERSION**
   - Used `zipencrypt` package (third-party library)
   - Proper ZipCrypto implementation
   - TPF files worked with OpenTexMod ✅

2. **Commit `3d906e3`** (Dec 2024) - **BUGS INTRODUCED**
   - "Refactor to integrate Numba-accelerated ZipCrypto encryption"
   - Removed dependency on `zipencrypt` package
   - Implemented custom `EncryptedZipWriter` class
   - **Introduced 3 critical bugs** ❌

3. **Commit `95f5727`** (Recent)
   - Added config.ini support
   - Enhanced texture classification
   - Did NOT fix ZipCrypto bugs

4. **Commit `15138eb`** (Current HEAD)
   - Removed user input prompt
   - Did NOT fix ZipCrypto bugs

---

## Critical Bugs Introduced in Commit 3d906e3

### 🔴 BUG #1: Wrong CRC32 for Encryption Header (CRITICAL)

**Location:** `EncryptedZipWriter.writestr()` line 715 + `encrypt_data()` line 633

**The Problem:**
```python
# In writestr() - line 706-715:
crc32 = zlib.crc32(data) & 0xFFFFFFFF  # Calculate CRC32 of UNCOMPRESSED data
if plaintext_compress_type == ZIP_DEFLATED:
    compressed_data = zlib.compress(data, zlib.Z_DEFAULT_COMPRESSION)  # Compress
else:
    compressed_data = data

encrypted_data = encrypt_data(compressed_data, pwd)  # Pass COMPRESSED data

# Then in encrypt_data() - line 633:
crc32 = zlib.crc32(data) & 0xFFFFFFFF  # ❌ Recalculates CRC32 of COMPRESSED data!
header = _generate_encryption_header(crc32)  # ❌ Uses wrong CRC32 in header!
```

**Why This Breaks OpenTexMod:**
- ZipCrypto encryption header contains 2 check bytes derived from the **uncompressed** file's CRC32
- OpenTexMod decrypts the header, extracts the check bytes, and verifies against the file's CRC32
- If CRC32 is from compressed data instead of uncompressed, verification fails
- OpenTexMod rejects the file as corrupted/invalid password

**Fix Required:**
`encrypt_data()` should accept both the data to encrypt AND the correct CRC32 value:
```python
def encrypt_data(data: bytes, password: bytes, crc32: int) -> bytes:
    # Don't recalculate CRC32, use the provided one
    header = _generate_encryption_header(crc32)
    # ... rest of encryption
```

---

### 🔴 BUG #2: Wrong Compression Format (CRITICAL)

**Location:** `EncryptedZipWriter.writestr()` line 710

**The Problem:**
```python
compressed_data = zlib.compress(data, zlib.Z_DEFAULT_COMPRESSION)
```

**Issue:**
- `zlib.compress()` creates **zlib format** = [2-byte header] + [deflate data] + [4-byte Adler32]
- ZIP format requires **raw DEFLATE** (no zlib wrapper)
- OpenTexMod expects standard PKZip DEFLATE, gets zlib format instead
- Decompression fails or produces garbage data

**Fix Required:**
```python
# Use raw DEFLATE format (-15 = negative wbits for raw)
compressor = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15)
compressed_data = compressor.compress(data) + compressor.flush()
```

---

### 🟡 BUG #3: ZIP Header Mismatch (Compatibility Issue)

**Location:** `EncryptedZipWriter.writestr()` lines 733-748

**The Problem:**
```python
# Write with STORED compression
zinfo.compress_type = ZIP_STORED
self.zipfile.writestr(zinfo, encrypted_data)

# Then hack the central directory to say DEFLATED
if hasattr(self.zipfile, 'filelist') and self.zipfile.filelist:
    last_file = self.zipfile.filelist[-1]
    if last_file.filename == zinfo.filename:
        last_file.compress_type = original_compress_type  # Change to DEFLATED!
```

**Issue:**
- Local file header says: `compression = STORED (0)`
- Central directory says: `compression = DEFLATED (8)`
- This violates PKZip specification (headers should match)
- OpenTexMod might reject as malformed ZIP

**Why This Hack Exists:**
The code manually encrypts and compresses, then needs to write pre-encrypted data.
It uses `ZIP_STORED` to prevent Python's zipfile from compressing again, but then
manually patches the metadata.

---

## Comparison: Old (Working) vs New (Broken)

### Old Version (commit ddc365d) - **WORKED** ✅
```python
from zipencrypt import ZipFile, ZIP_DEFLATED

with ZipFile(zip_buffer, 'w', ZIP_DEFLATED) as zip_file:
    zip_file.writestr('texmod.def', texmod_def.encode('utf-8'), pwd=password)
    zip_file.write(texture_path, texture_path.name, pwd=password)
```

- `zipencrypt` package handles all the complexity correctly
- Proper CRC32 handling
- Proper DEFLATE compression
- No header mismatches
- OpenTexMod accepts the files

### New Version (commit 3d906e3+) - **BROKEN** ❌
```python
# Custom EncryptedZipWriter class
# - Calculates wrong CRC32
# - Uses zlib format instead of raw DEFLATE
# - Creates header mismatches
# Result: OpenTexMod rejects the files
```

---

## Root Cause

The refactor in commit **3d906e3** tried to eliminate the `zipencrypt` dependency by
implementing ZipCrypto encryption from scratch. However, the implementation has bugs
in critical areas:

1. CRC32 calculation timing (uses compressed instead of uncompressed)
2. Compression format (zlib wrapper instead of raw DEFLATE)
3. ZIP metadata integrity (header mismatch hack)

---

## Recommended Solutions

### Option 1: Fix the Custom Implementation (More Work)
- Fix all 3 bugs in `EncryptedZipWriter`
- Extensive testing required
- May find more subtle bugs

### Option 2: Revert to `zipencrypt` Package (Safer)
- Revert to commit `ddc365d` approach
- Use proven, working `zipencrypt` package
- Keep all other improvements (DDS compression, config.ini, etc.)
- Just add `zipencrypt` back to dependencies

### Option 3: Use Python's Built-in Encryption (Simplest?)
- Python 3.7+ has `zipfile.ZipFile` with encryption support
- However, may not support legacy ZipCrypto properly
- Needs testing

---

## Testing Required After Fix

1. Build a TPF file with the fixed code
2. Try to load it in OpenTexMod
3. Verify textures actually appear in-game
4. Test with both PNG and DDS texture inputs
5. Compare hex dump of working vs new TPF files

---

## Files Affected

- `TexModTools/texturesToTpf.py`
  - Lines 619-771: `encrypt_data()` and `EncryptedZipWriter` class
  - Line 1802: `create_zip_archive()` function using the broken class

---

## Conclusion

**The TPF files build successfully but don't work with OpenTexMod because the
custom ZipCrypto implementation has critical bugs that corrupt the encryption
headers and compression format.**

The safest fix is to revert to using the `zipencrypt` package while keeping
all the other improvements (DDS compression, config system, etc.).

