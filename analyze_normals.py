"""
Analysis script to examine normal map texture characteristics.
Loads normal textures and analyzes them in various color spaces to identify patterns.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Normal texture paths to analyze
NORMAL_PATHS = [
    r"G:\Program Files (x86)\Ubisoft\Ubisoft Game Launcher\games\Assassin's Creed Brotherhood HD Fixup\hd\ground\ACBSP_T_0XD47D60ED.png",
    r"G:\Program Files (x86)\Ubisoft\Ubisoft Game Launcher\games\Assassin's Creed Brotherhood HD Fixup\hd\ground\ACBSP_T_0X797B674C.png",
    r"G:\Program Files (x86)\Ubisoft\Ubisoft Game Launcher\games\Assassin's Creed Brotherhood HD Fixup\hd\ground\ACBSP_T_0XF9D99162.png",
]


def analyze_channel_stats(channel_data: np.ndarray, channel_name: str) -> dict:
    """Calculate comprehensive statistics for a single channel."""
    if channel_data.size == 0:
        return {}
    
    channel_flat = channel_data.flatten().astype(np.float32)
    
    return {
        'name': channel_name,
        'min': float(np.min(channel_flat)),
        'max': float(np.max(channel_flat)),
        'mean': float(np.mean(channel_flat)),
        'median': float(np.median(channel_flat)),
        'std': float(np.std(channel_flat)),
        'variance': float(np.var(channel_flat)),
        'range': float(np.max(channel_flat) - np.min(channel_flat)),
    }


def analyze_image_rgb(image_path: Path) -> dict:
    """Analyze image in RGB color space."""
    with Image.open(image_path) as img:
        img_rgb = img.convert('RGB')
        channels = img_rgb.split()
        
        stats = {
            'mode': 'RGB',
            'size': img_rgb.size,
            'channels': {}
        }
        
        for i, (channel, name) in enumerate(zip(channels, ['R', 'G', 'B'])):
            channel_array = np.array(channel, dtype=np.float32)
            stats['channels'][name] = analyze_channel_stats(channel_array, name)
        
        return stats


def analyze_image_hsv(image_path: Path) -> dict:
    """Analyze image in HSV color space."""
    with Image.open(image_path) as img:
        img_rgb = img.convert('RGB')
        img_hsv = img_rgb.convert('HSV')
        channels = img_hsv.split()
        
        stats = {
            'mode': 'HSV',
            'size': img_hsv.size,
            'channels': {}
        }
        
        for i, (channel, name) in enumerate(zip(channels, ['H', 'S', 'V'])):
            channel_array = np.array(channel, dtype=np.float32)
            stats['channels'][name] = analyze_channel_stats(channel_array, name)
        
        return stats


def analyze_image_hsl(image_path: Path) -> dict:
    """Analyze image in HSL color space."""
    with Image.open(image_path) as img:
        img_rgb = img.convert('RGB')
        # PIL doesn't have HSL directly, convert RGB to HSL manually
        rgb_array = np.array(img_rgb, dtype=np.float32)
        
        r = rgb_array[:, :, 0] / 255.0
        g = rgb_array[:, :, 1] / 255.0
        b = rgb_array[:, :, 2] / 255.0
        
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        delta = max_val - min_val
        
        # Lightness
        l = (max_val + min_val) / 2.0
        
        # Saturation
        s = np.zeros_like(l)
        mask = delta != 0
        s[mask] = delta[mask] / (1 - np.abs(2 * l[mask] - 1))
        
        # Hue
        h = np.zeros_like(l)
        mask_r = (max_val == r) & (delta != 0)
        mask_g = (max_val == g) & (delta != 0)
        mask_b = (max_val == b) & (delta != 0)
        
        h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
        h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
        h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)
        
        # Normalize to 0-255 range for consistency
        h_norm = (h / 360.0 * 255.0).astype(np.float32)
        s_norm = (s * 255.0).astype(np.float32)
        l_norm = (l * 255.0).astype(np.float32)
        
        stats = {
            'mode': 'HSL',
            'size': img_rgb.size,
            'channels': {
                'H': analyze_channel_stats(h_norm, 'H'),
                'S': analyze_channel_stats(s_norm, 'S'),
                'L': analyze_channel_stats(l_norm, 'L'),
            }
        }
        
        return stats


def print_stats(stats: dict, title: str):
    """Print statistics in a formatted way."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"Color Space: {stats['mode']}")
    print(f"Image Size: {stats['size'][0]}x{stats['size'][1]}")
    print(f"\n{'Channel':<10} {'Min':<10} {'Max':<10} {'Mean':<10} {'Median':<10} {'Std':<10} {'Variance':<12} {'Range':<10}")
    print(f"{'-'*80}")
    
    for channel_name, channel_stats in stats['channels'].items():
        print(f"{channel_stats['name']:<10} "
              f"{channel_stats['min']:<10.2f} "
              f"{channel_stats['max']:<10.2f} "
              f"{channel_stats['mean']:<10.2f} "
              f"{channel_stats['median']:<10.2f} "
              f"{channel_stats['std']:<10.2f} "
              f"{channel_stats['variance']:<12.4f} "
              f"{channel_stats['range']:<10.2f}")


def analyze_normal_texture(image_path: str):
    """Analyze a single normal texture file."""
    path = Path(image_path)
    
    if not path.exists():
        print(f"ERROR: File not found: {image_path}")
        return
    
    print(f"\n{'#'*80}")
    print(f"# Analyzing: {path.name}")
    print(f"{'#'*80}")
    
    # Analyze in RGB
    rgb_stats = analyze_image_rgb(path)
    print_stats(rgb_stats, f"RGB Analysis - {path.name}")
    
    # Analyze in HSV
    hsv_stats = analyze_image_hsv(path)
    print_stats(hsv_stats, f"HSV Analysis - {path.name}")
    
    # Analyze in HSL
    hsl_stats = analyze_image_hsl(path)
    print_stats(hsl_stats, f"HSL Analysis - {path.name}")


def main():
    """Main analysis function."""
    print("="*80)
    print("Normal Map Texture Analysis")
    print("="*80)
    print(f"Analyzing {len(NORMAL_PATHS)} normal texture(s)...")
    
    for normal_path in NORMAL_PATHS:
        try:
            analyze_normal_texture(normal_path)
        except Exception as e:
            print(f"\nERROR analyzing {normal_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("Analysis Complete")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

