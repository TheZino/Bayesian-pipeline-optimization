from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.study._study_direction import StudyDirection
from optuna.trial import TrialState


def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figMngs = matplotlib._pylab_helpers.Gcf.get_all_fig_managers()
        for figManager in figMngs:
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return


class plotCallback:

    def __init__(self):
        matplotlib.use("TkAgg")
        plt.ion()
        self.fig = plt.figure(2)
        plt.show()
        plt.title("Optimization")

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:

        plt.figure(2)
        plt.cla()

        trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

        if study.direction == StudyDirection.MINIMIZE:
            best_values = np.minimum.accumulate(
                [cast(float, t.value) for t in trials])
            bv = best_values.min()
        else:
            best_values = np.maximum.accumulate(
                [cast(float, t.value) for t in trials])
            bv = best_values.max()

        plt.plot(
            [t.number for t in trials],
            [t.value for t in trials],
            marker=".",
            color='royalblue',
            label='current value',
            linewidth=0
        )
        plt.plot([t.number for t in trials],
                 best_values,
                 marker=".",
                 color='firebrick',
                 label='best value {:.4}'.format(bv),
                 alpha=0.5)

        plt.title(study.study_name)
        plt.legend()

        # limits plot height when outliers are too bad
        top_limit = trials[0].value * 3 / 2
        if top_limit > 7000:
            top_limit = 7000
        plt.ylim(0, top_limit)

        self.fig.canvas.draw()
        mypause(0.0000001)


class plotCallbackValid:

    def __init__(self):

        self.valid_psnrs = []
        self.train_psnrs = []
        self.valid_ssims = []
        self.train_ssims = []
        self.upper_bound_train = 0
        self.curr_psnr = None
        self.curr_ssim = None

        self._cbv_index = None

        matplotlib.use("TkAgg")
        plt.ion()
        self.fig = plt.figure(1)
        plt.show()
        plt.title("Optimization")
        self.fig2 = plt.figure(2, figsize=(12, 6))
        plt.show()
        plt.title("Validation")

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:

        plt.figure(1)
        plt.cla()

        trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

        if study.direction == StudyDirection.MINIMIZE:
            best_values = np.minimum.accumulate(
                [cast(float, t.value) for t in trials])
            bv = best_values.min()
            bv_index = np.argmin(best_values)
        else:
            best_values = np.maximum.accumulate(
                [cast(float, t.value) for t in trials])
            bv = best_values.max()
            bv_index = np.argmax(best_values)

        plt.plot(
            [t.number for t in trials],
            [t.value for t in trials],
            marker=".",
            color='royalblue',
            label='current value',
            linewidth=0
        )
        plt.plot([t.number for t in trials],
                 best_values,
                 marker=".",
                 color='firebrick',
                 label='best trial {} value {:.4}'.format(bv_index, bv),
                 alpha=0.5)

        plt.title(study.study_name)
        plt.legend()

        if self._cbv_index != bv_index:
            self._cbv_index = bv_index
            self.train_psnrs.append(self.curr_psnr)
            self.train_ssims.append(self.curr_ssim)
        else:
            self.train_psnrs.append(self.train_psnrs[-1])
            self.train_ssims.append(self.train_ssims[-1])

        if study.direction == StudyDirection.MINIMIZE:

            if len(self.valid_psnrs) == 0 or self.all_psnrs[-1] > self.curr_psnr:
                self.valid_psnrs.append(self.curr_psnr)
            else:
                self.valid_psnrs.append(self.valid_psnrs[-1])

            if len(self.valid_ssims) == 0 or self.train_ssims[-1] > self.curr_ssim:
                self.valid_ssims.append(self.curr_ssim)
            else:
                self.valid_ssims.append(self.valid_ssims[-1])

        else:

            if len(self.valid_psnrs) == 0 or self.valid_psnrs[-1] < self.curr_psnr:
                self.valid_psnrs.append(self.curr_psnr)
            else:
                self.valid_psnrs.append(self.valid_psnrs[-1])

            if len(self.valid_ssims) == 0 or self.valid_ssims[-1] < self.curr_ssim:
                self.valid_ssims.append(self.curr_ssim)
            else:
                self.valid_ssims.append(self.valid_ssims[-1])

        plt.figure(2)
        plt.clf()

        plt.subplot(1, 2, 1)
        plt.plot(
            range(0, len(self.valid_psnrs)),
            self.valid_psnrs,
            marker=".",
            color='royalblue',
            label='actual best PSNR'
        )

        plt.plot(
            self.valid_psnrs.index(max(self.valid_psnrs)),
            max(self.valid_psnrs),
            marker=".",
            color='firebrick',
            label='Best trial valid {} value {:.4f}'.format(self.valid_psnrs.index(max(self.valid_psnrs)),
                                                            max(self.valid_psnrs)),
        )

        plt.plot(
            range(0, len(self.train_psnrs)),
            self.train_psnrs,
            marker=".",
            color='goldenrod',
            label='train best PSNR'
        )
        plt.plot(
            bv_index,
            self.train_psnrs[-1],
            marker=".",
            color='forestgreen',
            label='Best trial train {} value {:.4f}'.format(bv_index,
                                                            self.train_psnrs[-1]),
        )

        plt.title("PSNR")
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)

        plt.plot(
            range(0, len(self.valid_ssims)),
            self.valid_ssims,
            marker=".",
            color='royalblue',
            label='actual best SSIM'
        )
        plt.plot(
            self.valid_ssims.index(max(self.valid_ssims)),
            max(self.valid_ssims),
            marker=".",
            color='firebrick',
            label='Best trial {} value {:.4f}'.format(self.valid_ssims.index(max(self.valid_ssims)),
                                                      max(self.valid_ssims)),
        )
        plt.plot(
            range(0, len(self.train_ssims)),
            self.train_ssims,
            marker=".",
            color='goldenrod',
            label='train best SSIM'
        )
        plt.plot(
            bv_index,
            self.train_ssims[-1],
            marker=".",
            color='forestgreen',
            label='Best trial train {} value {:.4f}'.format(bv_index,
                                                            self.train_ssims[-1]),
        )

        plt.title("SSIM")
        plt.legend(loc='lower right')

        mypause(0.0000000000000001)


class plotCallbackValidMultyObj:

    def __init__(self):

        self.valid_psnrs = []
        self.train_psnrs = []
        self.valid_ssims = []
        self.train_ssims = []
        self.upper_bound_train = 0
        self.curr_psnr = None
        self.curr_ssim = None

        self._cbv_index = None

        matplotlib.use("TkAgg")
        plt.ion()
        self.fig = plt.figure(1)
        plt.show()
        plt.title("Optimization")
        self.fig2 = plt.figure(2, figsize=(12, 6))
        plt.show()
        plt.title("Validation")

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:

        plt.figure(1)
        plt.cla()

        trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

        if study.directions[0] == StudyDirection.MINIMIZE:
            best_values = np.minimum.accumulate(
                [cast(float, t.values[0]) for t in trials])
            bv = best_values.min()
            bv_index = np.argmin(best_values)
        else:
            best_values = np.maximum.accumulate(
                [cast(float, t.values[0]) for t in trials])
            bv = best_values.max()
            bv_index = np.argmax(best_values)

        plt.plot(
            [t.number for t in trials],
            [t.values[0] for t in trials],
            marker=".",
            color='royalblue',
            label='current value',
            linewidth=0
        )
        plt.plot([t.number for t in trials],
                 best_values,
                 marker=".",
                 color='firebrick',
                 label='best trial {} value {:.4}'.format(bv_index, bv),
                 alpha=0.5)

        plt.title(study.study_name)
        plt.legend()

        if self._cbv_index != bv_index:
            self._cbv_index = bv_index
            self.train_psnrs.append(self.curr_psnr)
            self.train_ssims.append(self.curr_ssim)
        else:
            self.train_psnrs.append(self.train_psnrs[-1])
            self.train_ssims.append(self.train_ssims[-1])

        if study.directions[0] == StudyDirection.MINIMIZE:

            if len(self.valid_psnrs) == 0 or self.all_psnrs[-1] > self.curr_psnr:
                self.valid_psnrs.append(self.curr_psnr)
            else:
                self.valid_psnrs.append(self.valid_psnrs[-1])

            if len(self.valid_ssims) == 0 or self.train_ssims[-1] > self.curr_ssim:
                self.valid_ssims.append(self.curr_ssim)
            else:
                self.valid_ssims.append(self.valid_ssims[-1])

        else:

            if len(self.valid_psnrs) == 0 or self.valid_psnrs[-1] < self.curr_psnr:
                self.valid_psnrs.append(self.curr_psnr)
            else:
                self.valid_psnrs.append(self.valid_psnrs[-1])

            if len(self.valid_ssims) == 0 or self.valid_ssims[-1] < self.curr_ssim:
                self.valid_ssims.append(self.curr_ssim)
            else:
                self.valid_ssims.append(self.valid_ssims[-1])

        plt.figure(2)
        plt.clf()

        plt.subplot(1, 2, 1)
        plt.plot(
            range(0, len(self.valid_psnrs)),
            self.valid_psnrs,
            marker=".",
            color='royalblue',
            label='actual best PSNR'
        )

        plt.plot(
            self.valid_psnrs.index(max(self.valid_psnrs)),
            max(self.valid_psnrs),
            marker=".",
            color='firebrick',
            label='Best trial valid {} value {:.4f}'.format(self.valid_psnrs.index(max(self.valid_psnrs)),
                                                            max(self.valid_psnrs)),
        )

        plt.plot(
            range(0, len(self.train_psnrs)),
            self.train_psnrs,
            marker=".",
            color='goldenrod',
            label='train best PSNR'
        )
        plt.plot(
            bv_index,
            self.train_psnrs[-1],
            marker=".",
            color='forestgreen',
            label='Best trial train {} value {:.4f}'.format(bv_index,
                                                            self.train_psnrs[-1]),
        )

        plt.title("PSNR")
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)

        plt.plot(
            range(0, len(self.valid_ssims)),
            self.valid_ssims,
            marker=".",
            color='royalblue',
            label='actual best SSIM'
        )
        plt.plot(
            self.valid_ssims.index(max(self.valid_ssims)),
            max(self.valid_ssims),
            marker=".",
            color='firebrick',
            label='Best trial {} value {:.4f}'.format(self.valid_ssims.index(max(self.valid_ssims)),
                                                      max(self.valid_ssims)),
        )
        plt.plot(
            range(0, len(self.train_ssims)),
            self.train_ssims,
            marker=".",
            color='goldenrod',
            label='train best SSIM'
        )
        plt.plot(
            bv_index,
            self.train_ssims[-1],
            marker=".",
            color='forestgreen',
            label='Best trial train {} value {:.4f}'.format(bv_index,
                                                            self.train_ssims[-1]),
        )

        plt.title("SSIM")
        plt.legend(loc='lower right')

        mypause(0.0000000000000001)
