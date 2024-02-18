import numpy as np

from scipy.ndimage import center_of_mass

from .modeling.pseudo_voigt2d import PseudoVoigt2DFitter


class PeakFitter:
    def __init__(self):
        pass


    @staticmethod
    def initial_guess_for_peak(peak_image):
        # Estimate centroid (cy, cx)
        cy, cx = center_of_mass(peak_image)

        # Estimate amplitude (amp)
        amp = np.max(peak_image)

        # Estimate width parameters (sigma_x, sigma_y)
        # Assuming a symmetric peak for initial guess
        y, x = np.indices(peak_image.shape)
        weighted_distances = peak_image * ((x - cx)**2 + (y - cy)**2)
        sigma = np.sqrt(np.sum(weighted_distances) / np.sum(peak_image))

        # Estimate background (c)
        # Assuming the background is the average of the corners of the image
        corners = [peak_image[0,0], peak_image[0,-1], peak_image[-1,0], peak_image[-1,-1]]
        c = np.mean(corners)

        # Balance between Gaussian and Lorentzian (eta)
        eta = 0.5

        # For gamma, start with the same value as sigma
        gamma = sigma

        return {
            'amp'    : amp,
            'cy'     : cy,
            'cx'     : cx,
            'sigma_y': sigma,
            'sigma_x': sigma,
            'gamma_y': gamma,
            'gamma_x': gamma,
            'eta'    : eta,
            'a'      : 0,
            'b'      : 0,
            'c'      : c
        }


    @staticmethod
    def perform_fitting(image, initial_params, max_nfev = 2000):
        residual = PseudoVoigt2DFitter(initial_params)
        result = residual.fit(image, max_nfev = max_nfev)
        return result


    @staticmethod
    def fit(image, max_nfev = 2000):
        initial_params = PeakFitter.initial_guess_for_peak(image)
        return PeakFitter.perform_fitting(image, initial_params, max_nfev = max_nfev)
