import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import tensor_train
from tensorly.tt_tensor import tt_to_tensor

# --- 1. Configuration ---
IMAGE_SIZE = 128
# A list of Tensor-Train ranks to test.
# Lower rank = more compression, but lower quality.
# Higher rank = less compression, but higher quality.
TT_RANKS_TO_TEST = [1, 2, 5, 10, 20, 32, 64, 512]

def calculate_compression_ratio(original_tensor, tt_cores):
    """Calculates the compression ratio."""
    original_size = original_tensor.size
    compressed_size = sum(core.size for core in tt_cores)
    ratio = compressed_size / original_size
    return ratio

def main():
    """
    Main function to create, compress, reconstruct, and display the image.
    """
    # --- 2. Create a Random RGB Image ---
    # Create a 3D tensor (height, width, channels) with random integer values (0-255)
    original_image = np.random.randint(
        0, 256, 
        size=(IMAGE_SIZE, IMAGE_SIZE, 3), 
        dtype=np.uint8
    )
    
    # For numerical stability during decomposition, it's best to work with floats (0.0 to 1.0)
    image_tensor_float = original_image.astype(np.float64) / 255.0
    
    print(f"Original image tensor shape: {image_tensor_float.shape}")
    print(f"Total elements in original image: {image_tensor_float.size}\n")

    reconstructed_images = []
    compression_ratios = []

    # --- 3. Loop Through Ranks, Compress and Reconstruct ---
    for rank in TT_RANKS_TO_TEST:
        print(f"--- Processing for Rank={rank} ---")

        # Perform Tensor Train decomposition
        # The image (H, W, C) is decomposed into a list of 3 cores.
        # Core 1 shape: (1, H, r)
        # Core 2 shape: (r, W, r)
        # Core 3 shape: (r, C, 1)
        # where r is the rank.
        tt_cores = tensor_train(image_tensor_float, rank=rank)

        # Reconstruct the tensor from the compressed TT-cores
        reconstructed_tensor = tt_to_tensor(tt_cores)

        # Post-process the reconstructed tensor to be a valid image
        # 1. Clip values to be in the [0, 1] range as reconstruction can have minor overflows
        reconstructed_tensor = np.clip(reconstructed_tensor, 0, 1)
        # 2. Scale back to [0, 255] and convert to an 8-bit integer type for display
        reconstructed_image = (reconstructed_tensor * 255).astype(np.uint8)
        
        reconstructed_images.append(reconstructed_image)

        # Calculate and store compression info
        ratio = calculate_compression_ratio(image_tensor_float, tt_cores)
        compression_ratios.append(ratio)
        
        print(f"Compressed size (sum of elements in cores): {sum(c.size for c in tt_cores)}")
        print(f"Compression Ratio: {ratio:.4f} (Compressed is {ratio*100:.2f}% of Original size)\n")

    # --- 4. Display the Results ---
    num_images = len(TT_RANKS_TO_TEST) + 1
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 4, 5))
    
    # Display the original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Display each reconstructed image
    for i, (img, rank, ratio) in enumerate(zip(reconstructed_images, TT_RANKS_TO_TEST, compression_ratios)):
        ax = axes[i + 1]
        ax.imshow(img)
        title = f"Reconstructed (Rank={rank})\nCR: {ratio*100:.1f}%"
        ax.set_title(title)
        ax.axis('off')

    plt.suptitle("Tensor Train (TT) Image Compression Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    main()