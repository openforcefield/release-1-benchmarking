import json
import re
import sys
from io import StringIO
from collections import defaultdict

import numpy as np

import matplotlib
from matplotlib import pyplot
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

        mse_standard_deviation = (average_mse - error ** 2) ** 2

    mse_standard_deviation = np.sqrt(mse_standard_deviation / len(property_tuples))

    return average_mse, mse_standard_deviation


def plot_collated_data(properties_by_type, figure_size=6.5, dots_per_inch=400, font=None, marker_size='7'):

    if font is None:
        font = {'size': 18}

    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['r', 'b', 'g', 'k'])

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


def print_absolute_deviations(all_results_by_type, shared_ids_by_type):

    for property_type in all_results_by_type:

        propery_deviations = defaultdict(dict)
        results_paths = []

        substances_by_id = dict()

        for results_path in all_results_by_type[property_type]:

            results_paths.append(results_path)

            for property_tuple in all_results_by_type[property_type][results_path]:

                if property_tuple[0].id not in shared_ids_by_type[property_type]:
                    continue

                measured_property, estimated_property = property_tuple

                measured_property = measured_property.value.to(preferred_units[property_type]).magnitude
                estimated_property = estimated_property.value.to(preferred_units[property_type]).magnitude

                deviation = (estimated_property - measured_property) ** 2
                propery_deviations[property_tuple[0].id][results_path] = deviation

                substances_by_id[property_tuple[0].id] = property_tuple[0].substance.identifier

        print(f'\n{property_type}\n')
        print('substance,' + ','.join(results_paths))

        for property_id in propery_deviations:

            output = substances_by_id[property_id]

            for results_path in results_paths:
                output = f'{output},{propery_deviations[property_id][results_path]}'

            print(output)


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

    return results


def main():

    # Load the original data set.
    with open('curated_data_set.json', 'r') as file:
        measured_data_set = json.load(file, cls=TypedJSONDecoder)

    results_paths = ['smirnoff99frosst 1.1.0', 'parsley 0.0.9', 'parsley rc 1', 'gaff 1.81', 'gaff 2.11']
    # smirnoff_results_paths = ['smirnoff99frosst 1.1.0', 'parsley 0.0.9', 'parsley rc 1']

    all_results_by_type = defaultdict(lambda: defaultdict(list))

    shared_ids_by_type = defaultdict(set)

    mse_by_type_path = defaultdict(dict)
    # mse_by_type_path_smirks = defaultdict(dict)

    for results_path in results_paths:

        properties_by_type = load_results(results_path, measured_data_set)

        for property_type in properties_by_type:

            mse, mse_std = compute_mse(properties_by_type[property_type])
            mse_by_type_path[property_type][results_path] = f'{mse:.6e}', f'{mse_std:.6e}'

            # if results_path in smirnoff_results_paths:
            #     mse_per_smirks = print_mse_per_smirks(properties_by_type[property_type])
            #     mse_by_type_path_smirks[property_type][results_path] = mse_per_smirks

            property_ids = set([property_tuple[0].id for property_tuple in properties_by_type[property_type]])

            if len(shared_ids_by_type[property_type]) == 0:
                shared_ids_by_type[property_type] = property_ids

            shared_ids_by_type[property_type] = shared_ids_by_type[property_type].intersection(property_ids)

        for property_type in properties_by_type:
            all_results_by_type[property_type][results_path] = properties_by_type[property_type]

    print('Property Type,' + ','.join(results_paths))

    for property_type in all_results_by_type:

        print(property_type + ',' + ','.join([' +/- '.join(mse_by_type_path[property_type][results_path]) for
                                              results_path in results_paths]))

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

    # print_absolute_deviations(all_results_by_type, shared_ids_by_type)
    plot_collated_data(all_results_by_type, dots_per_inch=200)


if __name__ == '__main__':
    main()
