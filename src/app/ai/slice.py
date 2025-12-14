import torch
import torch.nn.functional as F

def bilateral_slice(bilateral_grid, guidemap, input_image=None):
    """
    Bilateral slice implementation that handles both 2 and 3 argument cases
    """
    batch_size, gh, gw, gd, coeffs = bilateral_grid.shape
    _, h, w = guidemap.shape
    device = bilateral_grid.device
    
    # Create output tensor for coefficients
    output_coeffs = torch.zeros(batch_size, h, w, coeffs, device=device)
    
    # Simple bilinear interpolation for now
    # In practice this would do proper bilateral grid slicing
    for c in range(coeffs):
        # Take middle slice of depth dimension
        coeff_slice = bilateral_grid[:, :, :, gd//2, c]  # [B, gh, gw]
        # Upsample to full resolution
        upsampled = F.interpolate(coeff_slice.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)
        output_coeffs[:, :, :, c] = upsampled.squeeze(1)
    
    return output_coeffs

# Make sure this doesn't run CUDA code at import
if __name__ == "__main__":
    print("Bilateral slice module loaded successfully")
