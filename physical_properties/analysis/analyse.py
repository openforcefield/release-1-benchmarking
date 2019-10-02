import json
import os
import re
import sys
from io import StringIO
from collections import defaultdict

import numpy as np

import matplotlib
from matplotlib import gridspec, pyplot
from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.utils import UndefinedStereochemistryError

from propertyestimator import unit
from propertyestimator.utils.serialization import TypedJSONDecoder

preferred_units = {
    'Density': unit.kilogram / unit.meter**3,
    'DielectricConstant': unit.dimensionless,
    'EnthalpyOfVaporization': unit.kilojoule / unit.mole,
    'EnthalpyOfMixing': unit.kilojoule / unit.mole,
    'ExcessMolarVolume': unit.meter**3 / unit.mole
}


axis_bounds = {
    'Density': (500.0, 3000.0),
    'DielectricConstant': (0.0, 50.0),
    'EnthalpyOfVaporization': (30, 90.0),
    'EnthalpyOfMixing': (-4.0, 3.0),
    'ExcessMolarVolume': (-1.0e-6, 1.0e-6)
}


cached_smirks_parameters = {}


matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['b', 'r', 'g', 'k', 'c'])


def load_results(base_name, measured_data_set, save_tracebacks=False):

    # Load the results
    with open(f'{base_name}.json', 'r') as file:
        json_results = json.load(file, cls=TypedJSONDecoder)

    properties_by_type = defaultdict(list)

    for substance_id in json_results.properties:

        measured_properties = measured_data_set.properties[substance_id]
        estimated_properties = json_results.properties[substance_id]

        for estimated_property in estimated_properties:
            property_type = estimated_property.__class__.__name__

            measured_property = next(x for x in measured_properties if x.id == estimated_property.id)
            properties_by_type[property_type].append((measured_property, estimated_property))

    if save_tracebacks:

        for index, exception in enumerate(json_results.exceptions):

            with open(f'{base_name}_error_{index}.txt', 'w') as file:
                file.write(f'{exception.directory}\n')
                file.write(exception.message.replace('\\n', '\n'))

    return properties_by_type


def compute_mse(property_tuples):

    average_mse = 0.0
    mse_standard_deviation = 0.0

    for measured_property, estimated_property in property_tuples:

        error = (estimated_property.value -
                 measured_property.value).to(preferred_units[type(measured_property).__name__]).magnitude

        average_mse += error ** 2

    average_mse /= len(property_tuples)

    for measured_property, estimated_property in property_tuples:

        error = (estimated_property.value -
                 measured_property.value).to(preferred_units[type(measured_property).__name__]).magnitude

        mse_standard_deviation += (average_mse - error ** 2) ** 2

    mse_standard_deviation = np.sqrt(mse_standard_deviation / len(property_tuples))

    return average_mse, mse_standard_deviation


def find_smirks_parameters(parameter_tag='vdW', *smiles_patterns):
    """Finds those force field parameters with a given tag which
    would be assigned to a specified set of molecules defined by
    the their smiles patterns.

    Parameters
    ----------
    parameter_tag: str
        The tag of the force field parameters to find.
    smiles_patterns: str
        The smiles patterns to assign the force field parameters
        to.

    Returns
    -------
    dict of str and list of str
        A dictionary with keys of parameter smirks patterns, and
        values of lists of smiles patterns which would utilize
        those parameters.
    """

    stdout_ = sys.stdout  # Keep track of the previous value.
    stderr_ = sys.stderr  # Keep track of the previous value.

    stream = StringIO()
    sys.stdout = stream
    sys.stderr = stream
    force_field = ForceField('smirnoff99Frosst-1.1.0.offxml')
    sys.stdout = stdout_  # restore the previous stdout.
    sys.stderr = stderr_

    parameter_handler = force_field.get_parameter_handler(parameter_tag)

    smiles_by_parameter_smirks = {}

    # Initialize the array with all possible smirks pattern
    # to make it easier to identify which are missing.
    for parameter in parameter_handler.parameters:

        if parameter.smirks in smiles_by_parameter_smirks:
            continue

        smiles_by_parameter_smirks[parameter.smirks] = set()

    # Populate the dictionary using the open force field toolkit.
    for smiles in smiles_patterns:

        if smiles not in cached_smirks_parameters or parameter_tag not in cached_smirks_parameters[smiles]:

            try:
                molecule = Molecule.from_smiles(smiles)
            except UndefinedStereochemistryError:
                # Skip molecules with undefined stereochemistry.
                continue

            topology = Topology.from_molecules([molecule])

            if smiles not in cached_smirks_parameters:
                cached_smirks_parameters[smiles] = {}

            if parameter_tag not in cached_smirks_parameters[smiles]:
                cached_smirks_parameters[smiles][parameter_tag] = []

            cached_smirks_parameters[smiles][parameter_tag] = [
                parameter.smirks for parameter in force_field.label_molecules(topology)[0][parameter_tag].values()
            ]

        parameters_with_tag = cached_smirks_parameters[smiles][parameter_tag]

        for smirks in parameters_with_tag:
            smiles_by_parameter_smirks[smirks].add(smiles)

    return smiles_by_parameter_smirks


def substance_to_smiles_tuples(substance):
    """Converts a `Substance` object to a tuple of smiles
    patterns sorted alphabetically.

    Parameters
    ----------
    substance: Substance
        The substance to convert.

    Returns
    -------
    tuple of str
        The tuple of smiles patterns.
    """
    return tuple(sorted([component.smiles for component in substance.components]))


def smiles_to_png(smiles, file_path, image_size=200):
    """Creates a png image of the 2D representation of
    a given smiles pattern.

    Parameters
    ----------
    smiles: str
        The smiles pattern to generate the png of.
    file_path: str
        The path of the output png file.
    image_size: int
        The size in pixels of the square image.
    """

    from openeye import oedepict
    from openforcefield.topology import Molecule

    if os.path.isfile(file_path):
        return

    off_molecule = Molecule.from_smiles(smiles)
    oe_molecule = off_molecule.to_openeye()
    # oe_molecule.SetTitle(off_molecule.to_smiles())

    oedepict.OEPrepareDepiction(oe_molecule)

    options = oedepict.OE2DMolDisplayOptions(image_size, image_size, oedepict.OEScale_AutoScale)

    display = oedepict.OE2DMolDisplay(oe_molecule, options)
    oedepict.OERenderMolecule(file_path, display)


def plot_estimated_vs_experiment(properties_by_type, figure_size=6.5, dots_per_inch=400,
                                 font=None, marker_size='7'):

    if font is None:
        font = {'size': 18}

    matplotlib.rc('font', **font)

    for property_type in properties_by_type:

        preferred_unit = preferred_units[property_type]
        pyplot.figure(figsize=(figure_size, figure_size), dpi=dots_per_inch)

        for results_name in properties_by_type[property_type]:

            measured_values = []
            estimated_values = []

            estimated_uncertainties = []

            for measured_property, estimated_property in properties_by_type[property_type][results_name]:

                measured_values.append(measured_property.value.to(preferred_unit).magnitude)

                estimated_values.append(estimated_property.value.to(preferred_unit).magnitude)
                estimated_uncertainties.append(estimated_property.uncertainty.to(preferred_unit).magnitude)

            pyplot.errorbar(x=estimated_values,
                            y=measured_values,
                            xerr=estimated_uncertainties,
                            fmt='x',
                            label=results_name,
                            markersize=marker_size)

        pyplot.legend(bbox_to_anchor=(1.04, 0.5), loc='center left', borderaxespad=0)

        pyplot.xlim(*axis_bounds[property_type])
        pyplot.ylim(*axis_bounds[property_type])

        pyplot.xlabel('Estimated')
        pyplot.ylabel('NIST ThermoML')

        if ((1.0e-2 > abs(axis_bounds[property_type][0]) > 0.0) or
            (1.0e-2 > abs(axis_bounds[property_type][1]) > 0.0)):
            pyplot.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        else:
            pyplot.ticklabel_format(style='plain', axis='both')

        pyplot.gca().grid(axis='both', linestyle='--', lw=1.0, color='black', alpha=0.2)
        pyplot.gca().set_aspect('equal')

        pyplot.draw()

        title = ' '.join(re.sub('([A-Z][a-z]+)', r' \1',
                         re.sub('([A-Z]+)', r' \1', property_type)).split()).title()

        if preferred_unit != unit.dimensionless:
            title = f'{title} ({str(preferred_unit)})'

        pyplot.title(title)
        pyplot.savefig(f'{property_type}.pdf', bbox_inches='tight')


def plot_per_property_mse(properties_by_type, figure_size=6.5, dots_per_inch=400, font=None):

    if font is None:
        font = {'size': 18}

    matplotlib.rc('font', **font)

    for property_type in properties_by_type:

        preferred_unit = preferred_units[property_type]
        pyplot.figure(figsize=(figure_size, figure_size), dpi=dots_per_inch)

        bar_indices = np.arange(len(properties_by_type[property_type]))
        bar_labels = [results_name for results_name in properties_by_type[property_type]]

        bar_values = []
        bar_errors = []

        for results_name in properties_by_type[property_type]:

            mse, mse_std = compute_mse(properties_by_type[property_type][results_name])

            bar_values.append(mse)
            bar_errors.append(mse_std)

        pyplot.bar(x=bar_indices,
                   height=bar_values,
                   yerr=bar_errors,
                   align='center')

        pyplot.xticks(bar_indices, bar_labels, rotation=90)

        pyplot.xlabel('Force Fields')
        pyplot.ylabel('MSE')

        title = ' '.join(re.sub('([A-Z][a-z]+)', r' \1',
                         re.sub('([A-Z]+)', r' \1', property_type)).split()).title()

        if preferred_unit != unit.dimensionless:
            title = f'{title} ({str(preferred_unit)})'

        pyplot.title(title)
        pyplot.savefig(f'{property_type}_MSE.pdf', bbox_inches='tight')


def print_per_property_mse(results_paths, properties_by_type):

    header_string = f'Property Type'

    for results_path in results_paths:
        header_string = ','.join([header_string, results_path, f'{results_path} MSE'])

    print('Property Type,' + ','.join(results_paths))

    for property_type in properties_by_type:

        row_string = f'{property_type}'

        for results_path in results_paths:

            mse, mse_std = compute_mse(properties_by_type[property_type][results_path])
            row_string = ','.join([row_string, f'{mse:.6e}', f'{mse_std:.6e}'])


def plot_per_substance_rmse(properties_by_type):

    os.makedirs('images', exist_ok=True)

    for property_type in properties_by_type:

        preferred_unit = preferred_units[property_type]

        properties_by_smiles = defaultdict(lambda: defaultdict(list))

        for results_name in properties_by_type[property_type]:

            for measured_property, estimated_property in properties_by_type[property_type][results_name]:

                smiles_tuple = substance_to_smiles_tuples(measured_property.substance)
                properties_by_smiles[smiles_tuple][results_name].append((measured_property, estimated_property))

        pyplot.figure(figsize=(8.5, 1.5 * len(properties_by_smiles)))

        title = ' '.join(re.sub('([A-Z][a-z]+)', r' \1',
                                re.sub('([A-Z]+)', r' \1', property_type)).split()).title()

        if preferred_unit != unit.dimensionless:
            title = f'{title} ({str(preferred_unit ** 2)}) MSE'

        pyplot.title(title)

        subplot_grid_spec = gridspec.GridSpec(len(properties_by_smiles), 2, width_ratios=[2, 5])

        # Determine the axis bounds
        minimum_value = 1e100
        maximum_value = -1e100

        for row_index, smiles_tuple in enumerate(properties_by_smiles):

            for results_name in properties_by_smiles[smiles_tuple]:

                mse, _ = compute_mse(properties_by_smiles[smiles_tuple][results_name])

                minimum_value = np.minimum(minimum_value, mse)
                maximum_value = np.maximum(maximum_value, mse)

        for row_index, smiles_tuple in enumerate(properties_by_smiles):

            bar_values = []
            bar_errors = []

            bar_indices = np.arange(len(properties_by_smiles[smiles_tuple]))
            bar_labels = [results_name for results_name in properties_by_smiles[smiles_tuple]]

            for results_name in properties_by_smiles[smiles_tuple]:

                mse, mse_std = compute_mse(properties_by_smiles[smiles_tuple][results_name])

                bar_values.append(mse)
                bar_errors.append(mse_std)

            pyplot.subplot(subplot_grid_spec[row_index * 2 + 1])

            pyplot.xlim(minimum_value, maximum_value)
            pyplot.xscale('log')

            pyplot.barh(y=bar_indices, width=bar_values, tick_label=bar_labels,
                        height=0.85, color='C0', align='center', xerr=bar_errors)

        pyplot.tight_layout()

        plot_size = pyplot.gcf().get_size_inches() * pyplot.gcf().dpi
        base_image_height = plot_size[1] / len(properties_by_smiles)

        for row_index, smiles_tuple in enumerate(properties_by_smiles):

            pyplot.subplot(subplot_grid_spec[row_index * 2])
            pyplot.axis('off')

            image_size = int(base_image_height / len(smiles_tuple))

            for index, smiles in enumerate(smiles_tuple):

                image_base_name = smiles.replace('/', '').replace('\\', '')

                image_path = os.path.join('images', f'{image_base_name}_{image_size}.png')
                smiles_to_png(smiles, image_path, image_size)

                molecule_image = pyplot.imread(image_path)

                image_y_height = (len(properties_by_smiles) - row_index - 1) * base_image_height
                image_y_height += base_image_height / 2 - image_size / 2

                pyplot.figimage(molecule_image, image_size * index, image_y_height)

        pyplot.savefig(f'{property_type}_MSE_per_substance.pdf', bbox_inches='tight')
        print('Done')


def print_mse_per_smirks(property_tuples):

    vdw_smirks_of_interest = [
        '[#1:1]-[#6X4]',
        '[#1:1]-[#6X4]-[#7,#8,#9,#16,#17,#35]',
        '[#1:1]-[#6X3]',
        '[#1:1]-[#6X3]~[#7,#8,#9,#16,#17,#35]',
        '[#1:1]-[#8]',
        '[#6:1]',
        '[#6X4:1]',
        '[#8:1]',
        '[#8X2H0+0:1]',
        '[#8X2H1+0:1]',
        '[#7:1]',
        '[#16:1]',
        '[#9:1]',
        '[#17:1]',
        '[#35:1]'
    ]

    property_tuples_by_smirks = defaultdict(list)

    for measured_property, estimated_property in property_tuples:

        smiles = [component.smiles for component in measured_property.substance.components]

        all_smirks = find_smirks_parameters('vdW', *smiles)
        smirks = [smirks_pattern for smirks_pattern in all_smirks.keys() if
                  smirks_pattern in vdw_smirks_of_interest and len(all_smirks[smirks_pattern]) > 0]

        for smirks_pattern in smirks:
            property_tuples_by_smirks[smirks_pattern].append((measured_property, estimated_property))

    results = dict()

    for smirks_pattern in property_tuples_by_smirks:
        results[smirks_pattern] = compute_mse(property_tuples_by_smirks[smirks_pattern])

    # for property_type in all_results_by_type:
    #
    #     print(f'\n{property_type}\n')
    #
    #     print('VDW SMIRKS|' + '|'.join(smirnoff_results_paths))
    #
    #     mse_by_smirks_path = defaultdict(dict)
    #
    #     for results_path in smirnoff_results_paths:
    #         for smirks in mse_by_type_path_smirks[property_type][results_path]:
    #             mse_by_smirks_path[smirks][results_path] = mse_by_type_path_smirks[property_type][results_path][smirks]
    #
    #     for smirks in mse_by_smirks_path:
    #
    #         output = f'{smirks}|'
    #
    #         for results_path in smirnoff_results_paths:
    #             output += ('-' if results_path not in mse_by_smirks_path[smirks] else
    #                        str(mse_by_smirks_path[smirks][results_path])) + '|'
    #
    #         print(output)
    #
    #     print(f'\n\n')

    return results


def main():

    # Load the original data set.
    with open('curated_data_set.json', 'r') as file:
        measured_data_set = json.load(file, cls=TypedJSONDecoder)

    results_paths = ['smirnoff99frosst 1.1.0', 'parsley 0.0.9', 'parsley rc 1', 'gaff 1.81', 'gaff 2.11']
    # smirnoff_results_paths = ['smirnoff99frosst 1.1.0', 'parsley 0.0.9', 'parsley rc 1']

    all_results_by_type = defaultdict(lambda: defaultdict(list))

    for results_path in results_paths:

        properties_by_type = load_results(results_path, measured_data_set)

        for property_type in properties_by_type:
            all_results_by_type[property_type][results_path] = properties_by_type[property_type]

    # print_per_property_mse(results_paths, all_results_by_type)

    plot_estimated_vs_experiment(all_results_by_type, dots_per_inch=200)
    plot_per_property_mse(all_results_by_type, dots_per_inch=200, font={'size': 16})

    plot_per_substance_rmse(all_results_by_type)
    pyplot.show()


if __name__ == '__main__':
    main()
