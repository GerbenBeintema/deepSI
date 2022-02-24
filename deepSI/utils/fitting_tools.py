


def fit_with_early_stopping(model, fit_kwargs, stop_frac=2/3, step0=3000, max_step=None, verbose=1):
    '''A very simple function which some heuristics for early stopping as
    such that if for a fraction of the training no new validation error is obtained the training is stopped.
    The condition used is 
        (stop_frac*step0 + best_id)/last_id < stop_frac
    such that if best_id = 0 and last_id=step0 the loop will be broken. (step0 will be the minimum number of steps)
    '''
    if verbose:
        print(f'Starting training with early stopping with settings: stop_frac={stop_frac:.2%} step0={step0:,} max_step={max_step}',)
        print()

    it = 0
    while True:
        model.fit(**fit_kwargs)
        best_id = model.batch_id[-1]
        best_val = model.Loss_val[-1]
        model.checkpoint_load_system('_last')
        last_id = model.batch_id[-1]
        last_val = model.Loss_val[-1]
        last_epoch_id = int(round(model.epoch_id[-1]))
        if verbose:
            it += 1
            print(f'######## Early stopping iteration check: {it} ########')
            print(f'\t epochs done: {last_epoch_id:.1f} steps done: {last_id:,} last best val loss at step: {best_id:,}')
            print(f'\t Current val: {last_val:.6f} Lowest val: {best_val:.6f}')
            print(f'\t stopping condition: (stop_frac*step0+best_id)/last_id={(stop_frac*step0+best_id)/last_id:.3%} < {stop_frac:.3%} = stop_frac\n')
        if (stop_frac*step0+best_id)/last_id < stop_frac or (isinstance(max_step,int) and last_id>=max_step):
            model.checkpoint_load_system('_best')
            return model

if __name__=='__main__':
    import deepSI
    train, test = deepSI.datasets.Silverbox()
    sys = deepSI.fit_systems.SS_encoder(nx=4)
    fit_with_early_stopping(sys, dict(train_sys_data=train, val_sys_data=test, epochs=10, verbose=0))