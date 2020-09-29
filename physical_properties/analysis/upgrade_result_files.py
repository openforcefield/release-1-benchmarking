"""A script to upgrade the results files generated using `propertyestimator ==0.0.5`
to files compatible with the latest `openff-evaluator =0.2.1`.
"""
import functools
import json
import os

import requests
from openff.evaluator import properties, unit
from openff.evaluator.datasets import (
    CalculationSource,
    MeasurementSource,
    PhysicalPropertyDataSet,
    PropertyPhase,
)
from openff.evaluator.substances import Component, ExactAmount, MoleFraction, Substance
from openff.evaluator.thermodynamics import ThermodynamicState


@functools.lru_cache(2000)
def correct_doi(file_name: str):
    """Attempt extract a DOI from a filename which contains a DOI."""

    if file_name.startswith("acs.jced") or file_name.startswith("je"):
        doi = f"10.1021/{file_name}"
    elif file_name.startswith("j.jct"):
        doi = f"10.1016/{file_name}"
    elif file_name.startswith("j.fluid"):
        doi = f"10.1016/{file_name}"
    elif file_name.startswith("j.tca"):
        doi = f"10.1016/{file_name}"
    elif file_name.startswith("s"):
        doi = f"10.1007/{file_name}"
    else:
        raise NotImplementedError()

    doi = doi.replace(".xml", "")

    doi_request = requests.get(
        f"https://doi.org/{doi}", headers={"Accept": "application/x-bibtex"}
    )
    doi_request.raise_for_status()

    return doi


def main():

    os.makedirs("raw_data_v2", exist_ok=True)

    for data_set_name in [
        "curated_data_set",
        "gaff 1.81",
        "gaff 2.11",
        "parsley 1.0.0",
        "smirnoff99frosst 1.1.0",
    ]:

        with open(os.path.join("raw_data", f"{data_set_name}.json")) as file:
            raw_data_set = json.load(file)

        assert (
            raw_data_set["@type"]
            == "propertyestimator.datasets.datasets.PhysicalPropertyDataSet"
        )

        physical_properties = []

        for raw_data_set_entries in raw_data_set["properties"].values():

            for raw_data_set_entry in raw_data_set_entries:

                # Extract the substance this entry was measured for.
                substance = Substance()

                for raw_component in raw_data_set_entry["substance"]["components"]:

                    component = Component(
                        smiles=raw_component["smiles"],
                        role=Component.Role[raw_component["role"]["value"]],
                    )

                    raw_amounts = raw_data_set_entry["substance"]["amounts"][
                        raw_component["smiles"]
                    ]

                    for raw_amount in raw_amounts["value"]:

                        if (
                            raw_amount["@type"]
                            == "propertyestimator.substances.Substance->MoleFraction"
                        ):

                            substance.add_component(
                                component, MoleFraction(raw_amount["value"])
                            )

                        elif (
                            raw_amount["@type"]
                            == "propertyestimator.substances.Substance->ExactAmount"
                        ):

                            substance.add_component(
                                component, ExactAmount(raw_amount["value"])
                            )

                        else:
                            raise NotImplementedError()

                # Extract the source of the property
                if (
                    raw_data_set_entry["source"]["@type"]
                    == "propertyestimator.properties.properties.CalculationSource"
                ):
                    source = CalculationSource(
                        fidelity=raw_data_set_entry["source"]["fidelity"]
                    )
                elif (
                    raw_data_set_entry["source"]["@type"]
                    == "propertyestimator.properties.properties.MeasurementSource"
                ):
                    source = MeasurementSource(
                        doi=correct_doi(raw_data_set_entry["source"]["reference"])
                    )
                else:
                    raise NotImplementedError()

                # Generate the new property object.
                property_class = getattr(
                    properties, raw_data_set_entry["@type"].split(".")[-1]
                )

                physical_property = property_class(
                    thermodynamic_state=ThermodynamicState(
                        temperature=(
                            raw_data_set_entry["thermodynamic_state"]["temperature"][
                                "value"
                            ]
                            * unit.Unit(
                                raw_data_set_entry["thermodynamic_state"][
                                    "temperature"
                                ]["unit"]
                            )
                        ),
                        pressure=(
                            raw_data_set_entry["thermodynamic_state"]["pressure"][
                                "value"
                            ]
                            * unit.Unit(
                                raw_data_set_entry["thermodynamic_state"]["pressure"][
                                    "unit"
                                ]
                            )
                        ),
                    ),
                    phase=PropertyPhase(raw_data_set_entry["phase"]),
                    substance=substance,
                    value=(
                        raw_data_set_entry["value"]["value"]
                        * unit.Unit(raw_data_set_entry["value"]["unit"])
                    ),
                    uncertainty=(
                        None
                        if isinstance(source, MeasurementSource)
                        else (
                            raw_data_set_entry["uncertainty"]["value"]
                            * unit.Unit(raw_data_set_entry["uncertainty"]["unit"])
                        )
                    ),
                    source=source,
                )
                physical_property.id = raw_data_set_entry["id"]

                physical_properties.append(physical_property)

        data_set = PhysicalPropertyDataSet()
        data_set.add_properties(*physical_properties)

        data_set.json(os.path.join("raw_data_v2", f"{data_set_name}.json"), format=True)
        data_set.to_pandas().to_csv(os.path.join("raw_data_v2", f"{data_set_name}.csv"))


if __name__ == "__main__":
    main()
