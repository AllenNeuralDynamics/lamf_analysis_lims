import numpy as np
import cupy as cp
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple, List, Optional, Dict, Any

class MariusMotionEstimator:
    def __init__(self, ref_zstack: np.ndarray, zs: List[int]):
        """
        Initialize the motion estimator with reference data
        
        Args:
            ref_zstack: Reference z-stack (np.ndarray, (y, x, z))
            zs: Z positions
        """
        self.eps0 = np.float32(1e-10)
        self.restrict_zs = []
        self.sub_pixel = 10

        assert ref_zstack.ndim == 3, "Reference z-stack must be 3D"
        assert ref_zstack.shape[2] == len(zs), "Reference z-stack must have the same number of z-slices as zs"
        
        # Extract ROI information
        self.zs = zs
             
        # Preprocess the reference volume
        self.preprocess_volume(ref_zstack)
        

    def preprocess_volume(self, ref_zstack: np.ndarray) -> None:
        """
        Preprocess the reference volume for motion estimation
        
        Args:
            reference_roi_data: Dictionary containing reference ROI data
        """
        mask_slope = 2  # for tapering mask
        smooth_sigma = 1.15  # for smoothing filter
        
        # Get reference images
        Z = ref_zstack.astype(np.float32)
        
        # Store dimensions
        self.sz = Z.shape
        n_zs = self.sz[2]
        
        # Z-score the reference stack
        Z = (Z - np.mean(Z, axis=(0,1), keepdims=True)) / np.std(Z, axis=(0,1), keepdims=True)
        
        # Calculate optimal FFT sizes
        self.sz_pad = [
            self._next_cufft_size(self.sz[0], 1, 'even')[0],
            self._next_cufft_size(self.sz[1], 1, 'even')[0],
            self.sz[2]
        ]
        self.sz_pad_half = [self.sz_pad[0]//2 + 1, self.sz_pad[1], self.sz_pad[2]]
        
        # Create sliding average of z-stack
        ref_img = np.zeros_like(Z)
        for z_idx in range(n_zs):
            itarget = np.arange(z_idx-2, z_idx+3)
            itarget = itarget[(itarget >= 0) & (itarget < n_zs)]
            ref_img[:,:,z_idx] = np.mean(Z[:,:,itarget], axis=2)
        
        # Create multiplicative mask
        mm, nn = np.meshgrid(np.arange(self.sz[0]), np.arange(self.sz[1]), indexing='ij')
        mm = np.abs(mm - np.mean(mm))
        nn = np.abs(nn - np.mean(nn))
        Mmax = np.max(mm) - 4
        Nmax = np.max(nn) - 4
        mask_mul = 1.0 / (1 + np.exp((mm - Mmax)/mask_slope)) / (1 + np.exp((nn - Nmax)/mask_slope))
        mask_mul = mask_mul.astype(np.float32)
        
        # Create smoothing filter
        hgm = np.exp(-(((np.arange(self.sz[0]) - self.sz[0]//2)/smooth_sigma)**2))
        hgn = np.exp(-(((np.arange(self.sz[1]) - self.sz[1]//2)/smooth_sigma)**2))
        hg = np.outer(hgm, hgn)
        hg = hg.astype(np.float32)
        
        # FFT of reference image
        fhg = np.fft.fft2(np.fft.ifftshift(hg/np.sum(hg)), s=(self.sz_pad[0], self.sz_pad[1]))
        fhg = np.real(fhg)
        
        # Apply FFT to each z-slice individually
        cf_ref_img = np.zeros((self.sz_pad[0], self.sz_pad[1], self.sz[2]), dtype=np.complex64)
        for z in range(self.sz[2]):
            slice_fft = np.fft.fft2(ref_img[:,:,z], s=(self.sz_pad[0], self.sz_pad[1]))
            cf_ref_img[:,:,z] = np.conj(slice_fft)
        
        cf_ref_img = cf_ref_img / (self.eps0 + np.abs(cf_ref_img))
        
        # Expand fhg to match dimensions for proper broadcasting
        fhg_expanded = np.repeat(fhg[:,:,np.newaxis], self.sz[2], axis=2)
        cf_ref_img = cf_ref_img * fhg_expanded
        
        # Transfer to GPU
        self.cf_ref_img_gpu = cp.asarray(cf_ref_img[:self.sz_pad_half[0],:,:])
        self.mask_mul_gpu = cp.asarray(mask_mul)

               
    def estimate_motion(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Core motion estimation function using phase correlation
        
        Args:
            image: Current frame image
            
        Returns:
            Tuple containing motion vector, confidence, and correlation data
        """
        # Transfer image to GPU
        image = image.astype(np.float32)
        image_gpu = cp.asarray(image)
        
        # Subtract mean and apply mask
        image_gpu = image_gpu - cp.mean(image_gpu)
        image_gpu = self.mask_mul_gpu * image_gpu
        
        # FFT and phase correlation
        image_fft_gpu = cp.fft.fft2(image_gpu, s=(self.sz_pad[0], self.sz_pad[1]))
        image_fft_gpu = image_fft_gpu[:self.sz_pad_half[0],:]
        image_fft_gpu = image_fft_gpu / (self.eps0 + cp.abs(image_fft_gpu))
        image_fft_gpu = image_fft_gpu[:, :, np.newaxis].repeat(self.sz_pad[2], axis=2)
       
        # Cross correlation - element-wise multiplication
        corr_map_gpu = self.cf_ref_img_gpu * image_fft_gpu
        
        # Apply IFFT separately for each z-slice to preserve dimensions
        result = []
        for z in range(self.sz[2]):
            slice_ifft = cp.fft.ifft2(corr_map_gpu[:,:,z], s=(self.sz_pad[0], self.sz_pad[1]))
            result.append(slice_ifft)
        
        # Stack the results back together
        corr_map_gpu = cp.stack(result, axis=2)
        corr_map_gpu = cp.real(corr_map_gpu)

        # Find correlation peak
        cmax_gpu = cp.max(corr_map_gpu)
        cmax_idx_gpu = cp.argmax(corr_map_gpu)
        cmax = float(cmax_gpu)
        cmax_idx = int(cmax_idx_gpu)

        Imax, Jmax, Zmax = np.unravel_index(cmax_idx, self.sz_pad)
        
        # Sub-pixel refinement
        if self.sub_pixel > 0:
            spline_base = 5
            bb = np.arange(-spline_base, spline_base + 1)
            
            II = np.mod(Imax + bb - 1, self.sz_pad[0])
            JJ = np.mod(Jmax + bb - 1, self.sz_pad[1])
            ZZ = np.mod(Zmax + bb - 1, self.sz_pad[2])
            
            corr_clip = cp.asnumpy(corr_map_gpu[II,  :, :][:, JJ, :][:, :, ZZ])
            
            # Upsample volume around peak
            IIg, JJg, ZZg = np.meshgrid(bb, bb, bb, indexing='ij')
            sub_pixels = np.linspace(-1, 1, 2*self.sub_pixel + 3)
            IIu, JJu, ZZu = np.meshgrid(sub_pixels, sub_pixels, sub_pixels, indexing='ij')
            
            F = RegularGridInterpolator((bb, bb, bb), corr_clip, method='cubic')
            corr_clip_up = F((IIu, JJu, ZZu))
            
            # Find peak in upsampled volume
            idx = np.argmax(corr_clip_up)
            Imax_u, Jmax_u, Zmax_u = np.unravel_index(idx, corr_clip_up.shape)

            Isub_pix = IIu[Imax_u, Jmax_u, Zmax_u]
            Jsub_pix = JJu[Imax_u, Jmax_u, Zmax_u]
            Zsub_pix = ZZu[Imax_u, Jmax_u, Zmax_u]
        else:
            Isub_pix = Jsub_pix = Zsub_pix = 0
            
        # Calculate motion vector
        dI = Imax - self.sz_pad[0] - 1 if Imax > self.sz_pad[0]//2 else Imax - 1
        dJ = Jmax - self.sz_pad[1] - 1 if Jmax > self.sz_pad[1]//2 else Jmax - 1
        Zmax = Zmax + Zsub_pix
        
        dx = dI + Isub_pix
        dy = dJ + Jsub_pix
        
        if len(self.zs) > 1:
            z_idx = len(self.zs) //2 + 1
            dz = self.zs[z_idx] - np.interp(Zmax, np.arange(len(self.zs)), self.zs)
        else:
            dz = 0
            
        dr = np.array([dx, dy, dz])
        
        # Calculate correlation projections
        cii_gpu = cp.max(corr_map_gpu, axis=1, keepdims=True)
        cjj_gpu = cp.max(corr_map_gpu, axis=0, keepdims=True)
        
        cii = cp.asnumpy(cii_gpu)
        cjj = cp.asnumpy(cjj_gpu)

        cii = np.transpose(cii, (0, 2, 1))
        cjj = np.transpose(cjj, (1, 2, 0))

        czz = np.max(cii, axis=0)
        corr_zmax_idx = min(max(0, int(round(Zmax))), cii.shape[1]-1)

        cii = cii[:,corr_zmax_idx]
        cjj = cjj[:,corr_zmax_idx]
        
        cii = np.fft.fftshift(cii, axes=0)
        cjj = np.fft.fftshift(cjj, axes=0)

        cii = self._unpad_correlation(cii, self.sz[0])
        cjj = self._unpad_correlation(cjj, self.sz[1])

        
        confidence = np.full(3, cmax)
        correlation = [cii, cjj, czz]
        # correlation_3d = cp.asnumpy(corr_map_gpu)
        
        return dr, confidence, correlation

        
    def _unpad_correlation(self, cc: np.ndarray, sz: int) -> np.ndarray:
        """Helper function to unpad correlation data"""
        dpad = len(cc) - sz
        if dpad > 0:
            return cc[dpad//2:-(dpad-dpad//2)]
        else:
            return np.pad(cc, (dpad//2, -(dpad-dpad//2)), mode='constant', constant_values=np.nan)
            


    def _prime_factors(n: int, primes=(2, 3, 5, 7)) -> Tuple[bool, List[int]]:
        powers = []
        for p in primes:
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            powers.append(count)
        return (n == 1), powers


    def _next_cufft_size(self, n: int, direction: int = 1, evenflag: Optional[str] = None) -> Tuple[int, List[int], List[int]]:
        primes = [2, 3, 5, 7]
        direction = 1 if direction >= 0 else -1

        while True:
            is_valid, powers = _prime_factors(n, primes)

            # Check if it's smooth and satisfies even/odd constraint
            if is_valid:
                if evenflag == 'even' and n % 2 != 0:
                    pass
                elif evenflag == 'odd' and n % 2 == 0:
                    pass
                else:
                    return n, powers, primes
            n += direction