import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Learning code plots retrieved from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

def plot_learning_curve(
    train_scores,
    val_scores,
    train_sizes,
    title,
    y_axis,
    axes=None,
):

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    axes.set_title(title,fontsize=20)
    axes.set_xlabel("Training examples",fontsize=16)
    axes.set_ylabel(y_axis,fontsize=16)

    axes.grid()
    # axes.fill_between(
    #     train_sizes,
    #     train_scores_mean - train_scores_std,
    #     train_scores_mean + train_scores_std,
    #     alpha=0.1,
    #     color="r",
    # )
    # axes.fill_between(
    #     train_sizes,
    #     val_scores_mean - val_scores_std,
    #     val_scores_mean + val_scores_std,
    #     alpha=0.1,
    #     color="b",
    # )
    axes.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training"
    )
    axes.plot(
        train_sizes, val_scores_mean, "o-", color="b", label="Cross-validation"
    )
    axes.legend(loc="best")


def plot_fit_time(
    fit_times,
    train_sizes,
    title,
    y_axis,
    axes=None,
):
    fit_times = fit_times*1000
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    axes.set_title(title,fontsize=20)
    axes.set_xlabel("Training examples",fontsize=16)
    axes.set_ylabel("Fit Time (ms)",fontsize=16)

    axes.grid()
    # axes.fill_between(
    #     train_sizes,
    #     fit_times_mean - fit_times_std,
    #     fit_times_mean + fit_times_std,
    #     alpha=0.1,
    #     color="r",
    # )

    axes.plot(
        train_sizes, fit_times_mean, "o-", color="r"
    )

    
def plot_predict_time(
    predict_times,
    train_sizes,
    title,
    y_axis,
    axes=None,
):

    predict_times = predict_times * 1000
    predict_times_mean = np.mean(predict_times, axis=1)
    predict_times_std = np.std(predict_times, axis=1)

    axes.set_title(title,fontsize=20)
    axes.set_xlabel("Training examples",fontsize=16)
    axes.set_ylabel("Predict Time (ms)",fontsize=16)

    axes.grid()
    # axes.fill_between(
    #     train_sizes,
    #     predict_times_mean - predict_times_std,
    #     predict_times_mean + predict_times_std,
    #     alpha=0.1,
    #     color="r",
    # )

    axes.plot(
        train_sizes, predict_times_mean, "o-", color="r"
    )

def plot_validation_curve(
    train_scores,
    val_scores,
    val_values,
    title,
    y_axis,
    x_axis,
    axes=None,
    x_log=None
):

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    axes.set_title(title,fontsize=20)
    axes.set_xlabel(x_axis,fontsize=16)
    axes.set_ylabel(y_axis,fontsize=16)

    if x_log:
        axes.set_xscale('log')

    axes.grid()
    # axes.fill_between(
    #     val_values,
    #     train_scores_mean - train_scores_std,
    #     train_scores_mean + train_scores_std,
    #     alpha=0.1,
    #     color="r",
    # )
    # axes.fill_between(
    #     val_values,
    #     val_scores_mean - val_scores_std,
    #     val_scores_mean + val_scores_std,
    #     alpha=0.1,
    #     color="b",
    # )
    axes.plot(
        val_values, train_scores_mean, "o-", color="r", label="Training"
    )
    axes.plot(
        val_values, val_scores_mean, "o-", color="b", label="Cross-validation"
    )
    axes.legend(loc="best")

def plot_loss_curve(
    loss_curve,
    title,
    y_axis,
    x_axis,
    axes=None,
    x_log=None
):

    axes.set_title(title,fontsize=20)
    axes.set_xlabel(x_axis,fontsize=16)
    axes.set_ylabel(y_axis,fontsize=16)

    if x_log:
        axes.set_xscale('log')

    axes.grid()

    axes.plot(
        loss_curve, color="r", label="Training Loss"
    )

    axes.legend(loc="best")


def plot_learning_curve(
    train_scores,
    val_scores,
    train_sizes,
    title,
    y_axis,
    axes=None,
):

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    axes.set_title(title,fontsize=20)
    axes.set_xlabel("Training examples",fontsize=16)
    axes.set_ylabel(y_axis,fontsize=16)

    axes.grid()
    # axes.fill_between(
    #     train_sizes,
    #     train_scores_mean - train_scores_std,
    #     train_scores_mean + train_scores_std,
    #     alpha=0.1,
    #     color="r",
    # )
    # axes.fill_between(
    #     train_sizes,
    #     val_scores_mean - val_scores_std,
    #     val_scores_mean + val_scores_std,
    #     alpha=0.1,
    #     color="b",
    # )
    axes.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training"
    )
    axes.plot(
        train_sizes, val_scores_mean, "o-", color="b", label="Cross-validation"
    )
    axes.legend(loc="best")


def plot_fit_time(
    fit_times,
    train_sizes,
    title,
    y_axis,
    axes=None,
):
    fit_times = fit_times*1000
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    axes.set_title(title,fontsize=20)
    axes.set_xlabel("Training examples",fontsize=16)
    axes.set_ylabel("Fit Time (ms)",fontsize=16)

    axes.grid()
    # axes.fill_between(
    #     train_sizes,
    #     fit_times_mean - fit_times_std,
    #     fit_times_mean + fit_times_std,
    #     alpha=0.1,
    #     color="r",
    # )

    axes.plot(
        train_sizes, fit_times_mean, "o-", color="r"
    )

    
def plot_predict_time(
    predict_times,
    train_sizes,
    title,
    y_axis,
    axes=None,
):

    predict_times = predict_times * 1000
    predict_times_mean = np.mean(predict_times, axis=1)
    predict_times_std = np.std(predict_times, axis=1)

    axes.set_title(title,fontsize=20)
    axes.set_xlabel("Training examples",fontsize=16)
    axes.set_ylabel("Predict Time (ms)",fontsize=16)

    axes.grid()
    # axes.fill_between(
    #     train_sizes,
    #     predict_times_mean - predict_times_std,
    #     predict_times_mean + predict_times_std,
    #     alpha=0.1,
    #     color="r",
    # )

    axes.plot(
        train_sizes, predict_times_mean, "o-", color="r"
    )