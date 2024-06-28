"""Module for MRP-based spatio-temporal interpolation.
"""

import numpy as np
import cupy as cp

from .cupy_MRP import SMRP, VPintError
from .cupy_SD_MRP import SD_SMRP

import math


class WP_SMRP(SMRP):
    """
    Class for WP-SMRP, extending SMRP.

    Attributes
    ----------
    original_grid : 2D numpy array
        the original grid supplied to be interpolated
    pred_grid : 2D numpy array
        interpolated version of original_grid
    feature_grid : 2D or 3D numpy array
        grid corresponding to original_grid, with feature vectors on the z-axis
    model : sklearn-based prediction model
        optional user-supplied machine learning model used to predict weights

    Methods
    -------
    run():
        Runs WP-MRP

    predict_weight():
        Predict the weight between two cells based on their features

    find_beta():
        Automatically determine the best value for beta

    contrast_map():
        Create a contrast map for a given input image (used in find_beta)

    get_weight_grid():
        Returns a grid of weights in a particular direction (can be useful for visualisations and debugging)

    get_weights():
        Returns the weights towards a particular cell based in its and its neighbours' features

    train():
        Train supplied prediction model on subsampled data or a training set

    compute_confidence():
        compute an indication of uncertainty per pixel in pred_grid
    """

    # clip_val = np.inf
    def __init__(self, grid, feature_grid, model=None, init_strategy='mean', max_gamma=np.inf, min_gamma=0, mask=None):
        # Check shapes
        if (grid.shape[0] != feature_grid.shape[0] or grid.shape[1] != feature_grid.shape[1]):
            raise VPintError("Target and feature grids have different shapes: " + str(grid.shape) + " and " + str(
                feature_grid.shape))
        if (len(grid.shape) > 2):
            # Reshape [None,None,1] to [None,None]
            if (grid.shape[2] == 1):
                grid = grid[:, :, 0]
            else:
                raise VPintError("Input grid needs to be two-dimensional, got " + str(grid.shape))
        super().__init__(grid, init_strategy=init_strategy, mask=mask)
        if (len(feature_grid.shape) == 3):
            self.feature_grid = feature_grid.copy().astype(float)
        elif (len(feature_grid.shape) == 2):
            self.feature_grid = feature_grid.copy().astype(float).reshape((feature_grid.shape[0],
                                                                           feature_grid.shape[1],
                                                                           1))
        else:
            raise VPintError("Improper feature dimensions; expected 2 or 3, got " + str(len(feature_grid.shape)))
        self.model = model
        self.max_gamma = max_gamma
        self.min_gamma = min_gamma
        self._run_state = False
        self._run_method = "predict"


    def run(self, iterations=-1, method='exact', auto_terminate=True, auto_terminate_threshold=1e-4, track_delta=False,
            confidence=False, confidence_model=None,
            save_gif=False, gif_path="convergence.gif", gif_ms_per_frame=100, store_gif_source=False, gif_size_up=20,
            gif_clip_iter=100,
            auto_adapt=False, auto_adaptation_epochs=100, auto_adaptation_proportion=0.5,
            auto_adaptation_strategy='random', auto_adaptation_max_iter=-1,
            auto_adaptation_subsample_strategy='max_contrast',
            auto_adaptation_verbose=False,
            prioritise_identity=False, priority_intensity=1, known_value_bias=0,
            resistance=False, epsilon=0.01, mu=1.0):
        """
        Runs WP-SMRP for the specified number of iterations. Creates a 3D (h,w,4) tensor val_grid, where the z-axis corresponds to a neighbour of each cell, and a 3D (h,w,4) weight tensor weight_grid, where the z-axis corresponds to the weights of every neighbour in val_grid's z-axis. The x and y axes of both tensors are stacked into 2D (h*w,4) matrices (one of which is transposed), after which the dot product is taken between both matrices, resulting in a (h*w,h*w) matrix. As we are only interested in multiplying the same row numbers with the same column numbers, we take the diagonal entries of the computed matrix to obtain a 1D (h*w) vector of updated values (we use numpy's einsum to do this efficiently, without wasting computation on extra dot products). This vector is then divided element-wise by a vector (flattened 2D grid) counting the number of neighbours of each cell, and we use the object's original_grid to replace wrongly updated known values to their original true values. We finally reshape this vector back to the original 2D pred_grid shape of (h,w).

        :param iterations: number of iterations used for the state value update function. If not specified, default to 10000, which functions as the maximum number of iterations in case of non-convergence
        :param method: method for computing weights. Options: "predict" (using self.model), "cosine_similarity" (based on feature similarity), "exact" (compute average weight exactly for features)
        :param auto_terminate: if True, automatically terminate once the mean change in values after calling the update rule converges to a value under the auto_termination_threshold. Capped at 10000 iterations by default, though it usually takes under 100 iterations to converge
        :param auto_terminate_threshold: threshold for the amount of change as a proportion of the mean value of the grid, after which the algorithm automatically terminates (this is always relative/scaled to values)
        :param track_delta: if True, return a vector containing the evolution of delta (mean proportion of change per iteration) along with the interpolated grid
        :param confidence: return highly experimental confidence grid with predictions
        :param confidence_model: model used to create confidence grid
        :param save_gif: experimental code to save gif (not currently working)
        :param gif_path: file to save convergence gif to
        :param prioritise_identity: if True, predictions made using weights close to 1 will be weighted more heavily in the prediction dot product. For example, if a cell has 4 neighbours, 1 of which has a spatial weight of 1, and the others spatial weights of 0.1, 123 and 0.005, the predicted value will be largely based on the prediction resulting from the spatial weight of 1. If False, the predicted value will simply be the mean prediction of all four neighbours. Weights up to 1 are copied directly, and weights>1 are copied as 1/weight. This effect can be amplified by the priority_intensity parameter.
        :param priority_intensity: intensity of the identity prioritisation function. Set to 'auto' to automatically set this value using grid search on a subsampled proportion of the grid.
        :param auto_intensity_epochs: number of epochs for the random search if priority_intensity=='auto'
        :param auto_intensity_proportion: proportion of cells to subsample for the random search if priority_intensity=='auto'
        :param known_value_bias: if prioritise_identity==True, also give higher weight to predictions derived from cells (close to) known values. Determined using SD-MRP.
        :returns: interpolated grid pred_grid
        """

        if (iterations > -1):
            auto_terminate = False
        else:
            iterations = 5000



        # Setup all this once

        height = self.pred_grid.shape[0]
        width = self.pred_grid.shape[1]

        h = height - 1
        w = width - 1

        neighbour_count_grid = cp.zeros((height, width))
        weight_grid = cp.zeros((height, width, 4))
        val_grid = cp.zeros((height, width, 4))

        # Compute weight grid once (vectorise at some point if possible)

        # Ideally this if wouldn't be hardcodedhere , ML would also be vectorised, and predict_weight
        # would take entire grids as input
        if (method == 'exact'):
            feature_size = self.feature_grid.shape[2]

            shp = self.feature_grid.shape
            size = np.product(shp)
            f_grid = self.feature_grid.reshape(size)
            f_grid[f_grid == 0] = 0.01
            f_grid = f_grid.reshape(shp)

            # Every matrix contains feature vectors for the neighbour in some direction

            up_grid = cp.ones((height, width, feature_size))
            right_grid = cp.ones((height, width, feature_size))
            down_grid = cp.ones((height, width, feature_size))
            left_grid = cp.ones((height, width, feature_size))

            up_grid[1:, :, :] = f_grid[0:-1, :, :]
            right_grid[:, 0:-1, :] = f_grid[:, 1:, :]
            down_grid[0:-1, :, :] = f_grid[1:, :, :]
            left_grid[:, 1:, :] = f_grid[:, 0:-1, :]

            # Compute weights exacly (coming in from direction to source, e.g., up means going from neighbour above to source)

            up_weights = cp.mean(f_grid / up_grid, axis=2)
            up_weights[0, :] = 0
            right_weights = cp.mean(f_grid / right_grid, axis=2)
            right_weights[:, -1] = 0
            down_weights = cp.mean(f_grid / down_grid, axis=2)
            down_weights[-1, :] = 0
            left_weights = cp.mean(f_grid / left_grid, axis=2)
            left_weights[:, 0] = 0

            weight_grid = cp.stack([up_weights, right_weights, down_weights, left_weights], axis=-1)


        weight_matrix = weight_grid.reshape((height * width, 4)).transpose()

        # Set neighbour count grid

        neighbour_count_grid = cp.ones(self.pred_grid.shape) * 4

        neighbour_count_grid[:, 0] = neighbour_count_grid[:, 0] - cp.ones(neighbour_count_grid.shape[0])
        neighbour_count_grid[:, width - 1] = neighbour_count_grid[:, width - 1] - cp.ones(neighbour_count_grid.shape[0])

        neighbour_count_grid[0, :] = neighbour_count_grid[0, :] - cp.ones(neighbour_count_grid.shape[1])
        neighbour_count_grid[height - 1, :] = neighbour_count_grid[height - 1, :] - cp.ones(
            neighbour_count_grid.shape[1])

        neighbour_count_vec = neighbour_count_grid.reshape(width * height)

        # Search for best parameters
        if (auto_adapt):
            params = []
            if (prioritise_identity):
                params.append('beta')
            if (resistance):
                params.append('epsilon')
                params.append('mu')

            # In case some configurations cause, e.g., overflow issues
            fail_counter = 0
            while True:
                if (fail_counter >= 3):
                    print("Auto-adaptation failed 3 times; proceeding with default parameters")
                    break
                try:
                    params_opt = self.auto_adapt(params, auto_adaptation_epochs, auto_adaptation_proportion,
                                                 search_strategy=auto_adaptation_strategy,
                                                 max_sub_iter=auto_adaptation_max_iter,
                                                 subsample_strategy=auto_adaptation_subsample_strategy)
                    if (auto_adaptation_verbose):
                        print("Best found params: " + str(params_opt))
                    for k, v in params_opt.items():
                        if (k == 'beta'):
                            priority_intensity = params_opt[k]
                        elif (k == 'epsilon'):
                            epsilon = params_opt[k]
                        elif (k == 'mu'):
                            mu = params_opt[k]
                    break
                except:
                    fail_counter += 1

        # if(priority_intensity==0):
        #    prioritise_identity=False
        if (prioritise_identity):
            # Prioritise weights close to 1, under the assumption they will be
            # more informative/constant.

            if (priority_intensity == 0):
                # Same as no priority
                priority_grid = cp.ones(weight_grid.shape)
                # Since we later divide by sum of priority weights (replacing neighbour count grid),
                # we need to set some 0s for edges
                priority_grid[0, :, 0] = 0.0  # up
                priority_grid[:, -1, 1] = 0.0  # right
                priority_grid[-1, :, 2] = 0.0  # down
                priority_grid[:, 0, 3] = 0.0  # left
            else:
                priority_grid = weight_grid.copy()
                priority_grid[priority_grid > 1] = 1 / priority_grid[priority_grid > 1] * (
                            1 / priority_grid[priority_grid > 1] / (
                                1 / priority_grid[priority_grid > 1] * priority_intensity))
                priority_grid[priority_grid < 1] = priority_grid[priority_grid < 1] * (
                            priority_grid[priority_grid < 1] / (
                                (priority_grid[priority_grid < 1] + 0.001) * priority_intensity))
            self.priority_grid = priority_grid  # for debugging access


        # Track amount of change over iterations
        if(track_delta):
            delta_vec = np.zeros(iterations)


        # Main loop

        for it in range(0, iterations):
            # Set val_grid

            # Up
            val_grid[1:, :, 0] = self.pred_grid[0:-1, :]
            val_grid[0, :, 0] = cp.zeros((width))

            # Right
            val_grid[:, 0:-1, 1] = self.pred_grid[:, 1:]
            val_grid[:, -1, 1] = cp.zeros((height))

            # Down
            val_grid[0:-1, :, 2] = self.pred_grid[1:, :]
            val_grid[-1, :, 2] = cp.zeros((width))

            # Left
            val_grid[:, 1:, 3] = self.pred_grid[:, 0:-1]
            val_grid[:, 0, 3] = cp.zeros((height))

            # Compute new values, update pred grid

            val_matrix = val_grid.reshape((height * width, 4))  # To do a dot product width weight matrix
            if (prioritise_identity):
                # element wise multiplication weight+vals
                individual_predictions = cp.multiply(weight_matrix.transpose(), val_matrix)
                # dot product with priority weights
                new_grid = cp.einsum('ij,ji->i', individual_predictions,
                                     priority_grid.reshape(height * width, 4).transpose())
                # divide by sum of priority weights
                new_grid = new_grid / cp.sum(priority_grid, axis=2).reshape(height * width)

            else:
                new_grid = cp.einsum('ij,ji->i', val_matrix, weight_matrix)
                new_grid = new_grid / neighbour_count_vec  # Correct for neighbour count

            flattened_original = self.original_grid.copy().reshape(
                (height * width))  # can't use argwhere with 2D indexing
            new_grid[cp.argwhere(~cp.isnan(flattened_original))] = flattened_original[
                cp.argwhere(~cp.isnan(flattened_original))]  # Keep known values from original
            new_grid = new_grid.reshape((height, width))  # Return to 2D grid

            # Apply 'resistance' to make it harder to deviate further from the mean
            if (resistance):
                # Add 'elastic band resistance'
                # Up until threshold (band length), increase as much as desired (band is slack)
                # After threshold, add resistance scaled by how far away from threshold we are

                # Flatten arrays
                shp = new_grid.shape
                size = np.product(shp)
                new_grid_vec = new_grid.reshape(size)
                pred_grid_vec = self.pred_grid.reshape(size)

                # Compute delta_y for all pixels
                delta_y = new_grid_vec - pred_grid_vec

                # Update pixels < threshold as old + delta # TODO: check if just doing all is faster
                inds = cp.where(new_grid_vec <= mu)
                new_grid_vec[inds] = pred_grid_vec[inds] + delta_y[inds]

                # Update pixels > threshold as old + delta - k*old (or old+delta?)
                inds = cp.where(new_grid_vec > mu)
                new_grid_vec[inds] = pred_grid_vec[inds] + delta_y[inds] - (epsilon * pred_grid_vec[inds])

            # Compute delta, save to vector and/or auto-terminate where relevant
            if (track_delta or auto_terminate):
                delta = cp.nanmean(cp.absolute(new_grid - self.pred_grid)) / cp.nanmean(self.pred_grid)
                if (track_delta):
                    delta_vec[it] = delta
                if (auto_terminate):
                    if (delta <= auto_terminate_threshold):
                        self.pred_grid = new_grid
                        if (track_delta):
                            delta_vec = delta_vec[0:it + 1]
                        break

            self.pred_grid = new_grid
        #self.pred_grid = cp.asnumpy(new_grid)
        self.run_state = True
        self.run_method = method




        if (track_delta):
            return (self.pred_grid, delta_vec)
        else:
            return (self.pred_grid)



    def auto_adapt(self, params, search_epochs, subsample_proportion, search_strategy='random',
                   subsample_strategy='max_contrast', ranges={}, max_sub_iter=-1, hill_climbing_threshold=5):
        """
        Automatically sets the identity priority intensity parameter to the best found value. Currently
        only supports random search.

        :param params: dict of parameters to search for (supported: beta, epsilon, mu)
        :param search_epochs: number of epochs used by the random search
        :param subsample_proportion: proportion of training data used to compute errors
        :returns: best found value for identity priority intensity
        """

        # Subsample
        if (subsample_strategy == 'max_contrast'):
            sub_grid = self.original_grid.copy()
            contrast_grid = self.contrast_map(sub_grid)

            shp = sub_grid.shape
            size = np.product(shp)
            contrast_vec = contrast_grid.reshape(size)
            sub_vec = sub_grid.reshape(size)

            # Get indices of sorted array
            num_pixels = int(subsample_proportion * len(sub_vec[~np.isnan(sub_vec)]))
            contrast_vec = np.nan_to_num(contrast_vec, nan=-1.0)
            temp = cp.argpartition(-contrast_vec, num_pixels)
            result_args = temp[:num_pixels]
            # Replace most different pixels by nan
            sub_vec[result_args] = np.nan
            sub_grid = sub_vec.reshape(shp)


        else:
            raise VPintError("Invalid subsample strategy: " + str(subsample_strategy))

        # Set bounds/distribution if specified, default otherwise
        bounds = {'beta': (0, 4), 'epsilon': (0.1, 0.01),
                  'mu': (cp.nanmean(self.original_grid), 3 * cp.nanmean(self.original_grid))}
        for k, v in ranges:
            bounds[k] = v

        # Initialise params, best stuff
        best_loss = np.inf
        best_val = {}
        for k in params:
            if (k not in ['beta', 'epsilon', 'mu']):
                raise VPintError("Invalid parameter to optimise: " + str(k))
            best_val[k] = -1

        if (search_strategy == 'random'):

            ran_default = False
            for ep in range(0, search_epochs):
                # Random search for best val for search_epochs iterations

                temp_MRP = WP_SMRP(sub_grid, self.feature_grid.copy())

                # Make sure to check defaults first, explore randomly otherwise
                if (not (ran_default)):
                    vals = {}
                    for k in params:
                        if (k == 'beta'):
                            vals[k] = 1
                        if (k == 'epsilon'):
                            vals[k] = 0
                        if (k == 'mu'):
                            vals[k] = cp.nanmean(self.original_grid) + 2 * cp.nanstd(self.original_grid)
                    ran_default = True
                else:
                    vals = {}
                    for k in params:
                        if (k == 'beta'):
                            vals[k] = cp.random.randint(low=bounds[k][0], high=bounds[k][1])
                        if (k == 'epsilon'):
                            # Technically not min/max, but mean/std. Bounded by 0-1
                            # vals[k] = min(max(0,np.random.normal(bounds[k][0],bounds[k][1])),1)
                            vals[k] = min(max(0, cp.random.uniform(low=0, high=0.3)), 1)
                        if (k == 'mu'):
                            vals[k] = cp.random.uniform(low=bounds[k][0], high=bounds[k][1])

                # Kind of hacky, for different combinations of parameters
                # Currently mu can only be optimised if epsilon is too
                if ('beta' in vals and 'epsilon' in vals):
                    if ('mu' in vals):
                        mu = vals['mu']
                    else:
                        mu = cp.nanmean(self.original_grid) + 2 * cp.nanstd(self.original_grid)
                    pred_grid = temp_MRP.run(prioritise_identity=True, priority_intensity=vals['beta'],
                                             iterations=max_sub_iter, auto_adapt=False,
                                             resistance=True, epsilon=vals['epsilon'], mu=mu)

                elif ('beta' in vals and not ('epsilon' in vals)):
                    pred_grid = temp_MRP.run(prioritise_identity=True, priority_intensity=vals['beta'],
                                             iterations=max_sub_iter, auto_adapt=False)

                elif (not ('beta' in vals) and 'epsilon' in vals):
                    if ('mu' in vals):
                        mu = vals['mu']
                    else:
                        mu = cp.nanmean(self.original_grid) + 2 * cp.nanstd(self.original_grid)
                    pred_grid = temp_MRP.run(resistance=True, epsilon=vals['epsilon'], mu=mu, auto_adapt=False)

                # Compute MAE of subsampled predictions
                mae = cp.nanmean(cp.absolute(
                    pred_grid.reshape(np.product(pred_grid.shape)) - self.original_grid.reshape(
                        np.product(self.original_grid.shape))))

                # Update where necessary
                if (mae < best_loss):
                    best_vals = vals
                    best_loss = mae

                temp_MRP.reset()






        else:
            raise VPintError("Invalid search strategy: " + str(search_strategy))

        for k, v in best_vals.items():
            if (k in params and v == -1):
                print("WARNING: no " + k + " better than dummy, please check your code (defaulting to 1)")
                best_vals[k] = 1
            if (not (k in params)):
                best_vals.pop(k)  # Remove parameters that were not optimised

        return (best_vals)




    def contrast_map(self, grid):
        """
        Create a contrast map of the feature grid, which can be used by find_beta to select pixels to sample. Contrast is computed as the mean average distance between a pixel and its neighbours, normalised to a 0-1 range.

        :param grid: input grid to create a contrast map for
        :returns: contrast map
        """
        height = grid.shape[0]
        width = grid.shape[1]

        # Create neighbour count grid
        neighbour_count_grid = cp.ones(grid.shape) * 4

        neighbour_count_grid[:, 0] = neighbour_count_grid[:, 0] - cp.ones(
            neighbour_count_grid.shape[0])  # Full row, so shape[0]
        neighbour_count_grid[:, width - 1] = neighbour_count_grid[:, width - 1] - cp.ones(neighbour_count_grid.shape[0])

        neighbour_count_grid[0, :] = neighbour_count_grid[0, :] - cp.ones(
            neighbour_count_grid.shape[1])  # Full col, so shape[0]
        neighbour_count_grid[height - 1, :] = neighbour_count_grid[height - 1, :] - cp.ones(
            neighbour_count_grid.shape[1])

        # Create (h*w*4) value grid
        val_grid = cp.zeros((height, width, 4))

        up_grid = cp.zeros((height, width))
        right_grid = cp.zeros((height, width))
        down_grid = cp.zeros((height, width))
        left_grid = cp.zeros((height, width))

        up_grid[1:-1, :] = grid[0:-2, :]
        right_grid[:, 0:-2] = grid[:, 1:-1]
        down_grid[0:-2, :] = grid[1:-1, :]
        left_grid[:, 1:-1] = grid[:, 0:-2]

        val_grid[:, :, 0] = up_grid
        val_grid[:, :, 1] = right_grid
        val_grid[:, :, 2] = down_grid
        val_grid[:, :, 3] = left_grid

        # Compute contrast as average absolute distance
        temp_grid = cp.repeat(grid[:, :, cp.newaxis], 4, axis=2)
        diff = cp.absolute(val_grid - temp_grid)
        sum_diff = cp.nansum(diff, axis=-1)
        avg_contrast = sum_diff / neighbour_count_grid

        min_val = cp.nanmin(avg_contrast)
        max_val = cp.nanmax(avg_contrast)
        avg_contrast = cp.asnumpy(cp.clip((avg_contrast - min_val) / (max_val - min_val), 0, 1))

        return (avg_contrast)



