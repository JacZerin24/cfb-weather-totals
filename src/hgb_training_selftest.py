from __future__ import annotations

from .fcs_model import build_fcs_model
from .model_bakeoff import reg_models


def main() -> None:
    general = reg_models(['closing_total'], [])['hist_gradient_boosting'].named_steps['model']
    fcs = build_fcs_model().named_steps['model']

    for name, estimator in [('general', general), ('fcs', fcs)]:
        assert estimator.max_iter == 250, f'{name} HGB max_iter drifted: {estimator.max_iter}'
        assert estimator.early_stopping is False, (
            f'{name} HGB must pin early_stopping=False; got {estimator.early_stopping!r}. '
            'Leaving sklearn auto behavior enabled makes the training regime change when the sample crosses 10,000 rows.'
        )
    print('HGB training-regime self-test passed.')


if __name__ == '__main__':
    main()
