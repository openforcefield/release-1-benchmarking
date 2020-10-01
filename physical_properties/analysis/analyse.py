"""A script to analyze the outputs of benchmarking each force field of interest against
the test set of physical properties and produce relevant plots and tables.
"""
import os
from typing import Dict, List, Tuple

import numpy
import pandas
import seaborn
from matplotlib import pyplot
from openff.evaluator import properties
from openff.evaluator.datasets import PhysicalPropertyDataSet

seaborn.set_context("paper", font_scale=2.2)

try:
    pyplot.rc('text', usetex=True)
except:
    pass
pyplot.rcParams["mathtext.fontset"] = "cm"

# Define properties whose density values don't seem to be correct. In particular,
# it seems that the experimental gas phase density may have been included in the test
# set, while the liquid phase density was computed by simulation.
OUTLIERS = [
    "9111b26a-a5e2-4b38-a2e8-d9c4cbbec88d",
    "acd8823e-a58f-4290-ad98-9901e1f1ffce",
]

# Define templates for LATEX output
PROPERTY_TYPE_TO_TITLE = {
    "Density_1": fr"$\rho \left({properties.Density.default_unit():P~}\right)$",
    "Density_2": (
        fr"$\rho\(x\right) \left({properties.Density.default_unit():P~}\right)$"
    ),
    "EnthalpyOfVaporization_1": (
        fr"$H_{{vap}} "
        fr"\left({properties.EnthalpyOfVaporization.default_unit():P~}\right)$"
    ),
    "EnthalpyOfMixing_2": (
        fr"$H_{{mix}}\left(x\right) "
        fr"\left({properties.EnthalpyOfMixing.default_unit():P~}\right)$"
    ),
    "ExcessMolarVolume_2": (
        fr"$V_{{excess}}\left(x\right) "
        fr"\left("
        fr"{properties.ExcessMolarVolume.default_unit():P~}\right)$"
    ).replace("³", "^3"),
    "DielectricConstant_1": fr"$\epsilon$",
}


def results_to_pandas(force_fields: List[str]) -> pandas.DataFrame:
    """Imports the experimental and estimated data sets and stores them in a
    pandas data frame.
    """

    # Load in the experimental data set.
    training_set = {
        physical_property.id: physical_property
        for physical_property in PhysicalPropertyDataSet.from_json(
            os.path.join("raw_data_v2", "curated_data_set.json")
        )
    }

    # Load in the results.
    estimated_results = {
        force_field: {
            physical_property.id: physical_property
            for physical_property in PhysicalPropertyDataSet.from_json(
                os.path.join("raw_data_v2", f"{force_field}.json")
            )
        }
        for force_field in force_fields
    }

    # Refactor the experimental and estimated data into a single data frame.
    data_rows = []

    for property_id in training_set:

        experimental_property = training_set[property_id]

        estimated_properties = {
            force_field: estimated_results[force_field].get(property_id, None)
            for force_field in force_fields
        }

        if (
            any(
                estimated_property is None
                for estimated_property in estimated_properties.values()
            )
            or property_id in OUTLIERS
        ):
            print(f"Skipping property {property_id}")
            continue

        data_rows.extend(
            {
                "Id": property_id,
                "Type": (
                    f"{experimental_property.__class__.__name__}_"
                    f"{len(experimental_property.substance)}"
                ),
                "Force Field": force_field,
                "NIST ThermoML": experimental_property.value.to(
                    experimental_property.default_unit()
                ).magnitude,
                "Estimated": estimated_properties[force_field]
                .value.to(experimental_property.default_unit())
                .magnitude,
                "Estimated Uncertainty": estimated_properties[force_field]
                .uncertainty.to(experimental_property.default_unit())
                .magnitude,
            }
            for force_field in force_fields
        )

    return pandas.DataFrame(data_rows)


def plot_estimated_vs_experimental(results_frame: pandas.DataFrame):
    """Produces scatter plots of the estimated values (X) vs the experimental values
    (Y) of the test set physical properties, one per property type."""

    os.makedirs("plots", exist_ok=True)

    property_types = results_frame["Type"].unique()

    for property_type in property_types:

        property_data = results_frame[results_frame["Type"] == property_type]

        plot = seaborn.FacetGrid(
            property_data,
            col="Force Field",
            sharex=True,
            sharey=True,
            height=3.0,
            aspect=0.90,
            col_wrap=2,
        )
        plot.map(
            seaborn.scatterplot,
            "Estimated",
            "NIST ThermoML",
            s=80,
            alpha=0.8,
        )

        min_limit = numpy.min(
            [
                numpy.minimum(axis.get_xlim(), axis.get_ylim())
                for axis in plot.axes.ravel()
            ]
        )
        max_limit = numpy.max(
            [
                numpy.maximum(axis.get_xlim(), axis.get_ylim())
                for axis in plot.axes.ravel()
            ]
        )

        for axis in plot.axes.ravel():
            axis.set_xlim(min_limit, max_limit)
            axis.set_ylim(min_limit, max_limit)

        for axis in plot.axes.ravel():
            axis.plot(
                [0, 1], [0, 1], transform=axis.transAxes, color="darkgrey", zorder=-1
            )

        plot.set_titles("{col_name}")

        pyplot.subplots_adjust(top=0.86)
        plot.fig.suptitle(PROPERTY_TYPE_TO_TITLE[property_type])

        file_name = {
            "Density_1": "physprop-benchmark-density",
            "DielectricConstant_1": "physprop-benchmark-dielectric",
            "EnthalpyOfMixing_2": "physprop-benchmark-hmix",
            "EnthalpyOfVaporization_1": "physprop-benchmark-hvap",
            "ExcessMolarVolume_2": "physprop-benchmark-vexcess",
        }[property_type]

        plot.savefig(os.path.join("plots", f"{file_name}.pdf"))


def _compute_statistics(
    measured_values: numpy.ndarray, estimated_values: numpy.ndarray
) -> Dict[str, float]:
    """Computes a collection of common statistics comparing a set of measured
    and estimated values.

    Parameters
    ----------
    measured_values
        The experimentally measured values with shape=(n_data_points)
    estimated_values: numpy.ndarray
        The computationally estimated values with shape=(number of data points)

    Returns
    -------
        A dictionary of the computed statistics which include the
        "Slope", "Intercept", "R", "R^2", "p", "RMSE", "MSE", "MUE", "Tau"
    """
    import scipy.stats

    statistics = {}

    (
        statistics["Slope"],
        statistics["Intercept"],
        statistics["R"],
        statistics["p"],
        _,
    ) = scipy.stats.linregress(measured_values, estimated_values)

    statistics["R^2"] = statistics["R"] ** 2
    statistics["RMSE"] = numpy.sqrt(
        numpy.mean((estimated_values - measured_values) ** 2)
    )
    statistics["MSE"] = numpy.mean(estimated_values - measured_values)
    statistics["MUE"] = numpy.mean(numpy.absolute(estimated_values - measured_values))
    statistics["Tau"], _ = scipy.stats.kendalltau(measured_values, estimated_values)

    return statistics


def _compute_bootstrapped_statistics(
    measured_values: numpy.ndarray,
    estimated_values: numpy.ndarray,
    estimated_stds: numpy.ndarray,
    percentile=0.95,
    bootstrap_iterations=2000,
) -> Dict[str, Tuple[float, Tuple[float, float]]]:
    """Compute the bootstrapped mean and confidence interval for a set
    of common error statistics.

    Notes
    -----
    Bootstrapped samples are generated with replacement from the full
    original data set.

    Parameters
    ----------
    measured_values
        The experimentally measured values with shape=(n_data_points)
    estimated_values
        The computationally estimated values with shape=(n_data_points)
    estimated_stds
        The standard deviations in the computationally estimated values with
        shape=(n_data_points)
    percentile: float
        The percentile of the confidence interval to calculate.
    bootstrap_iterations: int
        The number of bootstrap iterations to perform.
    """

    sample_count = len(measured_values)

    # Compute the mean of the statistics.
    mean_statistics = _compute_statistics(measured_values, estimated_values)
    statistic_types = sorted(mean_statistics)

    # Generate the bootstrapped statistics samples.
    sample_statistics = numpy.zeros((bootstrap_iterations, len(statistic_types)))

    for sample_index in range(bootstrap_iterations):

        samples_indices = numpy.random.randint(
            low=0, high=sample_count, size=sample_count
        )

        sample_measured_values = measured_values[samples_indices]
        sample_estimated_values = estimated_values[samples_indices]

        if estimated_stds is not None:
            sample_estimated_values += numpy.random.normal(0.0, estimated_stds)

        sample_statistics_dict = _compute_statistics(
            sample_measured_values, sample_estimated_values
        )

        sample_statistics[sample_index] = numpy.array(
            [
                sample_statistics_dict[statistic_type]
                for statistic_type in statistic_types
            ]
        )

    # Compute the confidence intervals.
    lower_percentile_index = int(bootstrap_iterations * (1 - percentile) / 2)
    upper_percentile_index = int(bootstrap_iterations * (1 + percentile) / 2)

    bootstrapped_statistics = dict()

    for statistic_index, statistic_type in enumerate(statistic_types):

        sorted_samples = numpy.sort(sample_statistics[:, statistic_index])

        bootstrapped_statistics[statistic_type] = (
            mean_statistics[statistic_type],
            (
                sorted_samples[lower_percentile_index],
                sorted_samples[upper_percentile_index],
            ),
        )

    return bootstrapped_statistics


def produce_statistics_table(force_fields: List[str], results_frame: pandas.DataFrame):
    """Produces a table containing the RMSE, R^2 and Tau for each benchmarked force
    field and property."""

    property_types = results_frame["Type"].unique()

    statistic_rows = []

    for property_type in property_types:

        property_data = results_frame[results_frame["Type"] == property_type]

        for force_field_index, force_field in enumerate(force_fields):
            benchmark_data = property_data[property_data["Force Field"] == force_field]

            experimental_values = benchmark_data["NIST ThermoML"].values

            estimated_values = benchmark_data["Estimated"].values
            estimated_std = benchmark_data["Estimated Uncertainty"].values

            statistics = _compute_bootstrapped_statistics(
                experimental_values, estimated_values, estimated_std
            )

            del statistics["Intercept"]
            del statistics["Slope"]
            del statistics["R"]
            del statistics["MSE"]
            del statistics["MUE"]
            del statistics["p"]

            statistics[r"$R^2$"] = statistics["R^2"]
            del statistics["R^2"]

            statistics[r"$\tau$"] = statistics["Tau"]
            del statistics["Tau"]

            statistic_rows.append(
                {
                    "Property": (
                        PROPERTY_TYPE_TO_TITLE[property_type].replace("³", "$^3$")
                        if force_field_index == 0
                        else ""
                    ),
                    "Force Field": force_field,
                    **{
                        statistic_type: (
                            f"${statistic[0]:.2f}"
                            f"^{{{statistic[1][1]:.2f}}}"
                            f"_{{{statistic[1][0]:.2f}}}$"
                        )
                        for statistic_type, statistic in statistics.items()
                    },
                }
            )

    os.makedirs("statistics", exist_ok=True)

    statistics_frame = pandas.DataFrame(statistic_rows)
    statistics_frame.to_csv(os.path.join("statistics", "physprop-benchmark-summary.csv"))

    with open(os.path.join("statistics", "physprop-benchmark-summary.tex"), "w") as file:

        file.write(
            statistics_frame.to_latex(index=False, escape=False, column_format="llcccc")
        )


def main():

    force_fields = [
        "smirnoff99frosst 1.1.0",
        "parsley 1.0.0",
        "gaff 1.81",
        "gaff 2.11",
    ]

    results_frame = results_to_pandas(force_fields)

    # Plot the results.
    plot_estimated_vs_experimental(results_frame)

    # Produce a table of statistics.
    # produce_statistics_table(force_fields, results_frame)


if __name__ == "__main__":
    main()
