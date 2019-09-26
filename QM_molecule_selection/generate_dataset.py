import pandas as pd
import qcportal as ptl

PRODUCTION = True

data = [
    [
        "primary_torsiondrive_benchmark_1.txt",
        "OpenFF Primary TorsionDrive Benchmark 1"
    ],
    [
        "full_torsiondrive_benchmark_1.txt",
        "OpenFF Full TorsionDrive Benchmark 1"
    ],
    [
        "primary_optimization_benchmark_1.txt",
        "OpenFF Primary Optimization Benchmark 1"
    ],
    [
        "full_optimization_benchmark_1.txt",
        "OpenFF Full Optimization Benchmark 1"
    ],
]

if PRODUCTION:
    client = ptl.FractalClient.from_file()
else:
    client = ptl.FractalClient()

collection_map = {
    "OptimizationDataset": ptl.collections.OptimizationDataset,
    "TorsionDriveDataset": ptl.collections.TorsionDriveDataset,
}

for fn, ds_name in data:
    indices = pd.read_csv(fn, header=None, names=["Type", "Name", "Index"])
    ds_type = indices.iloc[0, 0]
    print(f"\n{ds_name}")

    ds = collection_map[ds_type](ds_name, client=client)

    specs = True
    for ds_name, df in indices.groupby("Name"):
        ds_name = ds_name.strip()
        print(f"Data Source: {ds_name}")

        old_ds = client.get_collection(ds_type, ds_name)
        if specs:
            ds.data.__dict__["specs"] = old_ds.data.specs
            ds.data.__dict__["history"] = old_ds.data.history
            specs = False

        # print(ds.data.specs)

        for idx, row in df.iterrows():
            idx = str(row["Index"]).strip().lower()
            if idx in ds.data.records:
                print("!WARNING! Found duplicate record")

            try:
                ds.data.records[idx] = old_ds.data.records[idx]
            except KeyError:
                print(f"Missing: {idx}")

    print(f"Data Shape: {len(ds.data.records)}")
    if PRODUCTION:
        ds.save()

#    exit()
