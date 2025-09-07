import os
import argparse
import sys
try:
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"--> Error: A required library is missing: {e.name}")
    print("--> Please install the necessary libraries by running this command:")
    print("pip install opencv-python numpy matplotlib")
    sys.exit(1)

# --- Constants ---
BG_COLOR = "#2e2e2e"
FRAME_COLOR = "#3c3c3c"
TEXT_COLOR = "#dcdcdc"
ERROR_COLOR = "#e74c3c"
DEFAULT_OUTPUT_DIR = "cli_output"

def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Applies Histogram Equalization to a grayscale image."""
    print("--> Applying Histogram Equalization...")
    return cv2.equalizeHist(image)

def apply_gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """Applies Power-Law (Gamma) transformation to a grayscale image."""
    if gamma <= 0:
        raise ValueError("Gamma value must be greater than zero.")
    
    print(f"--> Applying Gamma Correction with gamma={gamma:.2f}...")
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def save_histogram_to_file(image_data: np.ndarray, file_path: str, title: str):
    """Calculates and saves the histogram of an image to a file."""
    print(f"--> Generating and saving histogram to: {file_path}")
    
    counts, bins = np.histogram(image_data.ravel(), bins=256, range=[0, 256])

    fig, ax = plt.subplots(facecolor=FRAME_COLOR, figsize=(6, 4))
    fig.suptitle(title, color=TEXT_COLOR, fontsize=12)
    ax.set_facecolor(BG_COLOR)
    ax.bar(bins[:-1], counts, width=1.0, color=ERROR_COLOR)
    ax.set_xlim([0, 255])
    ax.tick_params(colors=TEXT_COLOR, which='both')
    ax.set_xlabel("Pixel Intensity", color=TEXT_COLOR)
    ax.set_ylabel("Frequency", color=TEXT_COLOR)
    ax.grid(True, linestyle='--', alpha=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    fig.savefig(file_path, facecolor=FRAME_COLOR, dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="A command-line tool for medical image enhancement.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("filepath", type=str, help="Path to the input image file.")
    parser.add_argument("-t", "--technique", type=str, required=True, choices=["hist_eq", "gamma"], help="The enhancement technique to apply:\n'hist_eq': Histogram Equalization\n'gamma':   Power-Law (Gamma) Correction")
    parser.add_argument("-g", "--gamma", type=float, help="The gamma value for the 'gamma' technique. Required if technique is 'gamma'.")
    parser.add_argument("-o", "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"The directory to save output files. (Default: {DEFAULT_OUTPUT_DIR})")

    args = parser.parse_args()
    
    if args.technique == 'gamma' and not args.gamma:
        parser.error("The --gamma (-g) argument is REQUIRED when using the 'gamma' technique.")
    print(f"Step 1: Validating input file path...")
    absolute_filepath = os.path.abspath(args.filepath)
    
    if not os.path.exists(absolute_filepath):
        print(f"\n--- FATAL ERROR ---")
        print(f"Input file not found at the specified path.")
        print(f"Attempted to read: {absolute_filepath}")
        print(f"Please check the filename for typos and ensure it's in the correct folder.")
        return

    print(f"--> File found: {absolute_filepath}")

    try:
        # --- Main Logic ---
        print("\nStep 2: Preparing output directory...")
        # Get the directory of the script to create the output folder there
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, args.output_dir)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"--> Created output directory: '{output_path}'")
        else:
            print(f"--> Output directory already exists: '{output_path}'")

        print("\nStep 3: Processing image...")
        original_image = cv2.imread(absolute_filepath, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            print(f"--- FATAL ERROR ---")
            print(f"The file was found, but OpenCV could not read it. It may be corrupted or in an unsupported format.")
            return

        if args.technique == "hist_eq":
            processed_image = apply_histogram_equalization(original_image)
            suffix = "hist_eq"
        else: # Must be gamma, already validated
            processed_image = apply_gamma_correction(original_image, args.gamma)
            suffix = f"gamma_{args.gamma:.2f}"
        
        print("\nStep 4: Saving output files...")
        base_name, _ = os.path.splitext(os.path.basename(absolute_filepath))
        new_image_filename = f"{base_name}_{suffix}.png"
        new_hist_filename = f"{base_name}_{suffix}_hist.png"
        
        image_save_path = os.path.join(output_path, new_image_filename)
        hist_save_path = os.path.join(output_path, new_hist_filename)

        cv2.imwrite(image_save_path, processed_image)
        print(f"--> Saved enhanced image to: {image_save_path}")
        
        save_histogram_to_file(processed_image, hist_save_path, f"Enhanced Histogram ({suffix})")

        print("\n--- Enhancement Complete! ---")

    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---")
        print(f"Error details: {e}")

if __name__ == "__main__":
    main()