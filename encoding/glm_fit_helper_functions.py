import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from scipy.stats import t

'''
design matrix "X" and session information "session" is assumed to be loaded using
session, fit, design = GLM_fit_tools.load_fit_experiment(oeid, alex_run_param)
GLM_fit_tools is a module in the visual_behavior_glm.
'''

def fit_full_dropout_models(session, X, num_folds=5,
                            feature_drop_prefix = {'image': ['image'],
                                                   'change': ['hits', 'misses'],
                                                   'omission': ['omissions'],
                                                   'behavior': ['running', 'pupil', 'licks'],
                                                   'running': ['running'],
                                                   'pupil': ['pupil'],
                                                   'licks': ['licks'],
                                                   'hits': ['hits'],
                                                   'misses': ['misses'],
                  }):
    interp_events = get_interpolated_events(session, X)
    ridge_splits_inds, fit_splits_inds = get_split_inds(X, num_folds=num_folds)
    optimal_lam = find_optimal_lam(interp_events, X, ridge_splits_inds)
    optimal_lam = optimal_lam.assign_coords(feature_drop='none')
    var_exp, predicted_events = fit_cell(interp_events, X, fit_splits_inds, optimal_lam)
    var_exp = var_exp.assign_coords(feature_drop='none')
    predicted_events = predicted_events.assign_coords(feature_drop='none')

    for prefix_key in feature_drop_prefix.keys():
        prefix_list = feature_drop_prefix[prefix_key]
        weight_names = []
        for prefix in prefix_list:
            weight_names += [n for n in X.weights.values if prefix in n]
        Xd = X.drop_sel(weights=weight_names)
        optimal_lam_d = find_optimal_lam(interp_events, Xd, ridge_splits_inds)
        optimal_lam_d = optimal_lam_d.assign_coords(feature_drop=prefix_key)
        optimal_lam = xr.concat([optimal_lam, optimal_lam_d], dim='feature_drop')

        var_exp_d, predicted_events_d = fit_cell(interp_events, Xd, fit_splits_inds, optimal_lam_d)
        var_exp_d = var_exp_d.assign_coords(feature_drop=prefix_key)    
        var_exp = xr.concat([var_exp, var_exp_d], dim='feature_drop')
        predicted_events_d = predicted_events_d.assign_coords(feature_drop=prefix_key)
        predicted_events = xr.concat([predicted_events, predicted_events_d], dim='feature_drop')

    return var_exp, predicted_events, optimal_lam


def get_interpolated_events(session, X):
    num_cells = len(session.events)
    coords = list(X.coords)
    assert coords[1] == 'timestamps'
    X_timestamps = X.coords[coords[1]].values
    interp_events = np.zeros((len(X_timestamps), num_cells))
    for ci in range(num_cells):
        trace = session.events.events.values[ci]
        f = interp1d(session.ophys_timestamps, trace, kind='linear')
        interp_events[:, ci] = f(X_timestamps)
    interp_events = xr.DataArray(interp_events, coords=[X.coords[coords[1]].values, session.events.index.values], dims=[coords[1], 'cell_specimen_id'])
    return interp_events


def get_split_inds(X, num_folds=5):
    time_inds = np.arange(len(X.timestamps.values))
    np.random.shuffle(time_inds)
    ridge_splits_inds = np.array_split(time_inds, num_folds)
    np.random.shuffle(time_inds)
    fit_splits_inds = np.array_split(time_inds, num_folds)
    assert len(np.unique(np.concatenate(ridge_splits_inds))) == len(np.unique(np.concatenate(fit_splits_inds))) == len(time_inds)
    assert len(ridge_splits_inds) == len(fit_splits_inds) == num_folds
    assert np.abs(np.diff([len(x) for x in ridge_splits_inds])).max() <= 1
    assert np.abs(np.diff([len(x) for x in fit_splits_inds])).max() <= 1

    return ridge_splits_inds, fit_splits_inds


def find_optimal_lam(interp_events, X, ridge_split_inds, lam_range=np.geomspace(1e-3, 1e5, 200)):
    num_folds = len(ridge_split_inds)
    num_cells = interp_events.shape[1]
    mse_all = np.zeros((num_cells, len(lam_range), num_folds))
    for fi in range(num_folds):
        train_inds = np.concatenate([ridge_split_inds[i] for i in range(num_folds) if i != fi])
        test_inds = ridge_split_inds[fi]
        X_train = X.isel(timestamps=train_inds)
        X_test = X.isel(timestamps=test_inds)
        interp_events_train = interp_events.isel(timestamps=train_inds)
        interp_events_test = interp_events.isel(timestamps=test_inds)
        for li, lam in enumerate(lam_range):
            W = fit_regularized(interp_events_train, X_train, lam)
            pred = np.dot(X_test.values, W.values)
            mse = np.mean((pred - interp_events_test.values)**2, axis=0)
            mse_all[:, li, fi] = mse
    mean_mse = np.mean(mse_all, axis=-1)
    optimal_lam = xr.DataArray(lam_range[np.argmin(mean_mse, axis=1)],
                                 dims='cell_specimen_id',
                                 coords={'cell_specimen_id': interp_events.coords['cell_specimen_id'].values})
    return optimal_lam


def fit_regularized(interp_events, X, lam):
    '''
    Analytical OLS solution with added L2 regularization penalty. 

    traces: xarray with shape (n_timestamps * n_cells)
    X: xarray with shape (n_timestamps * n_kernel_params)
    lam (float): Strength of L2 regularization (hyperparameter to tune)

    Returns: XArray
    '''
    # Compute the weights
    W = np.dot(np.linalg.inv(np.dot(X.T, X) + lam * np.eye(X.shape[-1])),
               np.dot(X.T, interp_events.values))

    # Make xarray
    cellids = interp_events.coords['cell_specimen_id'].values[:]
    W_xarray= xr.DataArray(
            W, 
            dims =('weights','cell_specimen_id'), 
            coords = {  'weights':X.weights.values, 
                        'cell_specimen_id':cellids}
            )
    return W_xarray


def fit_cell(interp_events, X, fit_split_inds, optimal_lam):
    assert interp_events.shape[1] == len(optimal_lam)
    assert (interp_events.cell_specimen_id.values == optimal_lam.cell_specimen_id.values).all()
    predicted_events = xr.DataArray(np.zeros(interp_events.shape),
                                    dims=('timestamps', 'cell_specimen_id'),
                                    coords={'timestamps': interp_events.coords['timestamps'].values,
                                            'cell_specimen_id': interp_events.coords['cell_specimen_id'].values})
    
    num_folds = len(fit_split_inds)
    num_cells = interp_events.shape[1]
    var_exp_values = np.zeros((num_cells, num_folds))
    for fi in range(num_folds):
        train_inds = np.concatenate([fit_split_inds[i] for i in range(num_folds) if i != fi])
        test_inds = fit_split_inds[fi]
        X_train = X.isel(timestamps=train_inds)
        X_test = X.isel(timestamps=test_inds)
        interp_events_train = interp_events.isel(timestamps=train_inds)
        interp_events_test = interp_events.isel(timestamps=test_inds)
        for ci in range(interp_events.shape[1]):
            W = fit_regularized_single(interp_events_train.isel(cell_specimen_id=ci).values,
                                X_train, optimal_lam.values[ci])
            pred = np.dot(X_test.values, W)
            y = interp_events_test.values[:,ci]
            var_exp_values[ci, fi] = 1 - np.mean((y - pred)**2) / np.mean((y - np.mean(y))**2)
            # assign the predicted values to the predicted_events array
            predicted_events.isel(cell_specimen_id=ci).values[test_inds] = pred
    var_exp = xr.DataArray(var_exp_values,
                           dims=('cell_specimen_id', 'fold'),
                           coords={'cell_specimen_id': interp_events.coords['cell_specimen_id'].values,
                                   'fold': [f'fold_{i}' for i in np.arange(num_folds)]})
    return var_exp, predicted_events


def fit_regularized_single(trace, X, lam):
    '''
    Analytical OLS solution with added L2 regularization penalty. 

    traces: 1d array (n_timestamps)
    X: xarray with shape (n_timestamps * n_kernel_params)
    lam (float): Strength of L2 regularization (hyperparameter to tune)

    Returns: XArray
    '''
    # Compute the weights
    W = np.dot(np.linalg.inv(np.dot(X.T, X) + lam * np.eye(X.shape[-1])),
               np.dot(X.T, trace))

    return W


def t_test(mean, std, n):
    t_value = (mean - 0) / (std / np.sqrt(n))
    p_value = t.sf(np.abs(t_value), n-1) * 2
    return p_value


