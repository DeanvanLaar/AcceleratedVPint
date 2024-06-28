"""Module for MRP-based interpolation.
"""

import numpy as np
from math import log10, sqrt


class MRP:
    """
    Basic MRP class containing common functionality of SMRP and STMRP.

    Attributes
    ----------
    original_grid : 2D numpy array
        the original grid supplied to be interpolated
    pred_grid : 2D numpy array
        interpolated version of original_grid
    _dims : the number of dimensions for this MRP (2 for spatial, 3 for temporal)

    Methods
    -------
    get_pred_grid():
        Returns pred_grid

    get_pred_grid():
        Reset pred_grid

    init_pred_grid():
        Initialises pred_grid given an input grid and an initialisation strategy

    r_squared():
        Computes the r^2 of pred_grid compared to a given ground truth grid

    MAE():
        Computes the mean absolute error of pred_grid compared to a given ground truth grid

    RMSE():
        Computes the root mean squared error of pred_grid compared to a given ground truth grid

    PSNR():
        Computes the peak signal-to-noise ratio of pred_grid compared to a given ground truth grid

    SSIM():
        Computes the structural similarity index of pred_grid compared to a given ground truth grid
    """

    def __init__(self, grid, init_strategy='mean', mask=None):
        if (mask is not None):
            grid = self.apply_mask(grid, mask)
        self.original_grid = grid.copy()
        self.dims = len(grid.shape)
        self.init_pred_grid(init_strategy=init_strategy)

    def __str__(self):
        return (str(self.pred_grid))

    def reset(self):
        """Set pred_grid to its original state (original grid)"""
        self.pred_grid = self.original_grid

    def init_pred_grid(self, init_strategy='mean'):
        """Initialise pred_grid with mean/random/zero values as initial values for missing cells.

        :param init_strategy: method for initialising unknown values. Options: 'zero', 'random', 'mean'"""

        pred_grid = self.original_grid.copy()
        shp = pred_grid.shape
        size = np.product(pred_grid.shape)
        if (init_strategy == 'mean'):
            mean = np.nanmean(pred_grid)
            pred_vec = pred_grid.reshape(size)
            pred_vec[np.isnan(pred_vec)] = mean
            pred_grid = pred_vec.reshape(shp)
        elif (init_strategy == 'zero'):
            pred_vec = pred_grid.reshape(size)
            pred_vec[np.isnan(pred_vec)] = 0
            pred_grid = pred_vec.reshape(shp)
        elif (init_strategy == 'random'):
            pred_vec = pred_grid.reshape(size)
            num_nan = len(pred_vec[np.isnan(pred_vec)])
            random_vec = np.random.normal(np.nanmean(pred_vec), np.nanstd(pred_vec), size=num_nan)
            pred_vec[np.isnan(pred_vec)] = random_vec
            pred_grid = pred_vec.reshape(shp)
        else:
            raise VPintError("Invalid initialisation strategy: " + str(init_strategy))

        self.pred_grid = pred_grid

    def get_pred_grid(self):
        """Return pred_grid

        :returns: pred_grid"""
        # self.update_grid()
        return (self.pred_grid)

    def apply_mask(self, grid, mask):
        """Apply a supplied binary mask of missing values (1 denotes missing values) to the input grid.

        :param grid: the target grid to apply the mask to
        :param mask: binary mask (0/1) of the same shape as grid, where 1 denotes missing values
        :returns: target grid with missing values as specified by the mask"""
        shp = grid.shape
        grid_vec = grid.reshape(np.product(grid.shape))
        mask_vec = grid.reshape(np.product(mask.shape))
        if (grid.shape != mask.shape):
            raise VPintError(
                "Target and mask grids have different shapes: " + str(grid.shape) + " and " + str(mask.shape))
        grid_vec[mask_vec == 1] = np.nan
        grid = grid_vec.reshape(shp)
        return (grid)

    def r_squared(self, true_grid):
        """
        Compute the r^2 of pred_grid given true_grid as ground truth

        :param true_grid: ground truth for all grid cells
        :returns: r^2
        """

        # In case true grid contains nans
        true_grid_mean = np.nanmean(true_grid)
        true_grid = np.nan_to_num(true_grid, nan=true_grid_mean)

        height = self.pred_grid.shape[0]
        width = self.pred_grid.shape[1]
        if (self.dims == 3):
            depth = self.pred_grid.shape[2]

        m = np.mean(true_grid)

        res = 0
        tot = 0

        for i in range(0, height):
            for j in range(0, width):
                if (self.dims == 3):
                    # Spatio-temporal
                    for t in range(0, depth):
                        if (np.isnan(self.original_grid[i][j][t])):
                            resval = (true_grid[i][j][t] - self.pred_grid[i][j][t]) ** 2
                            totval = (true_grid[i][j][t] - m) ** 2

                            res += resval
                            tot += totval

                else:
                    # Spatial/temporal
                    if (np.isnan(self.original_grid[i][j])):
                        resval = (true_grid[i][j] - self.pred_grid[i][j]) ** 2
                        totval = (true_grid[i][j] - m) ** 2

                        res += resval
                        tot += totval
        r_squared = 1 - (res / tot)

        return (r_squared)

    def MAE(self, true):
        """
        Compute the mean absolute error of pred_grid given true as ground truth.

        :param true: ground truth for all grid cells
        :returns: mean absolute error
        """
        # In case true grid contains nans
        true_mean = np.nanmean(true)
        true = np.nan_to_num(true, nan=true_mean)

        pred = self.pred_grid.copy()
        mask = self.original_grid.copy()

        diff = np.absolute(true - pred)

        flattened_mask = mask.copy().reshape((np.prod(mask.shape)))
        flattened_diff = diff.reshape((np.prod(diff.shape)))[np.isnan(flattened_mask)]

        mae = np.nanmean(flattened_diff)
        return (mae)

    def RMSE(self, true):
        """
        Compute the root mean squared error of pred_grid given true as ground truth.

        :param true: ground truth for all grid cells
        :returns: root mean squared error
        """
        # In case true grid contains nans
        true_mean = np.nanmean(true)
        true = np.nan_to_num(true, nan=true_mean)

        pred = self.pred_grid.copy()
        mask = self.original_grid.copy()

        diff = true - pred

        flattened_mask = mask.copy().reshape((np.prod(mask.shape)))
        flattened_diff = diff.reshape((np.prod(diff.shape)))[np.isnan(flattened_mask)]

        rmse = np.mean(np.square(flattened_diff))
        return (rmse)

    def PSNR(self, true):
        """
        Compute the peak signal-to-noise ratio of pred_grid given true as ground truth.

        :param true: ground truth for all grid cells
        :returns: peak signal-to-noise ratio
        """
        # In case true grid contains nans
        true_mean = np.nanmean(true)
        true = np.nan_to_num(true, nan=true_mean)

        # Based on https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
        pred = self.pred_grid.copy()
        mask = self.original_grid.copy()

        flattened_mask = mask.copy().reshape((np.prod(mask.shape)))
        flattened_true = true.reshape((np.prod(true.shape)))[np.isnan(flattened_mask)]
        flattened_pred = pred.reshape((np.prod(pred.shape)))[np.isnan(flattened_mask)]

        mse = np.nanmean((flattened_true - flattened_pred) ** 2) + 0.001  # 0.001 for smoothing

        if (mse == 0):
            return (1)
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse)) / 100  # /100 because I want 0-1
        return (psnr)

    def SSIM(self, true):
        """
        Compute the structural similarity index of pred_grid given true as ground truth.

        :param true: ground truth for all grid cells
        :returns: structural similarity index
        """
        # In case true grid contains nans
        true_mean = np.nanmean(true)
        true = np.nan_to_num(true, nan=true_mean)

        pred = self.pred_grid.copy()
        mask = self.original_grid.copy()

        flattened_mask = mask.copy().reshape((np.prod(mask.shape)))
        flattened_true = true.reshape((np.prod(true.shape)))[np.isnan(flattened_mask)]
        flattened_pred = pred.reshape((np.prod(pred.shape)))[np.isnan(flattened_mask)]

        try:
            from skimage.measure import compare_ssim
            (s, d) = compare_ssim(flattened_true, flattened_pred, full=True)
        except:
            print("Error running compare_ssim. For this functionality, please ensure that scikit-image is installed.")
            s = np.nan
        return (s)


class SMRP(MRP):
    """
    Basic class implementing basic functions used by SD-MRP and WP-MRP.

    Attributes
    ----------
    original_grid : 2D numpy array
        the original grid supplied to be interpolated
    pred_grid : 2D numpy array
        interpolated version of original_grid

    Methods
    -------

    get_pred_grid():
        Returns pred_grid
    """

    def __init__(self, grid, init_strategy='mean', mask=None):
        super().__init__(grid, init_strategy=init_strategy, mask=mask)

    def mean_absolute_error(self, true_grid, scale_factor=1, gridded=False):
        """
        Compute the mean absolute error of pred_grid given true_grid as ground truth. Old code.

        :param true_grid: ground truth for all grid cells
        :param gridded: optional Boolean specifying whether to return an error grid with the MAE
        :returns: mean absolute error, optionally a tuple of MAE and per-cell error
        """

        # Rescale; naive approach for now
        height = self.pred_grid.shape[0]
        width = self.pred_grid.shape[1]

        new_height = int(height / scale_factor)
        new_width = int(width / scale_factor)

        new_pred_grid = self.pred_grid.copy().reshape(
            (new_height, int(height / new_height), new_width, int(width / new_width))).sum(3).sum(1)
        new_original_grid = self.original_grid.copy().reshape(
            (new_height, int(height / new_height), new_width, int(width / new_width))).sum(3).sum(1)
        new_true_grid = true_grid.copy().reshape(
            (new_height, int(height / new_height), width, int(new_width / new_width))).sum(3).sum(1)
        error_grid = np.zeros((new_height, new_width))

        e = 0
        c = 0

        for i in range(0, new_height):
            for j in range(0, new_width):

                if (np.isnan(new_original_grid[i][j])):
                    err = abs(new_pred_grid[i][j] - new_true_grid[i][j])
                    error_grid[i][j] = err
                    e += err
                    c += 1
                else:
                    error_grid[i][j] = 0
        mae = e / c
        if (gridded):
            result = (mae, error_grid)
        else:
            result = mae

        return (result)




class VPintError(Exception):
    pass