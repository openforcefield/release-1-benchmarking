This directory contains the raw results of the benchmarks, scripts to analyze them, 
and summary plots and statistics tables.

This directory is split into three key subdirectories:

- The ``raw_data`` directory contains the original experimental test set and the 
  corresponding estimated data sets.
- The ``raw_data_v2`` directory contains *exactly* the same data as the ``raw_data``
  directory, however the files have been updated to use the file format of the latest 
  available version of the ``openff-evaluator`` framework (``0.2.1``). This conversion
  was performed by the ``upgrade_result_files.py`` script.
- The ``plots`` directory contains scatter plots comparing the experimental and estimated 
  data sets. These were generated by the ``analyse.py`` script.
- The ``statistics`` directory contains summary tables of statistics comparing the 
  experimental and estimated data sets. These were generated by the ``analyse.py`` script.

### Notes

- The physical properties with ids
    
    - ``1a803034-7366-4a0c-9636-8ce3470f1bb1``
    - ``1e3999e0-59e8-41f2-991c-e7ec6c7c7bee``
    - ``a8229131-40d7-47d5-88a7-5910ca0a5421``
    - ``c6121e84-6082-4976-b21d-2faa32119080``
    - ``9111b26a-a5e2-4b38-a2e8-d9c4cbbec88d``
    - ``acd8823e-a58f-4290-ad98-9901e1f1ffce``
    - ``5bb33791-c951-4384-8e68-762ee0e77a26``

were omitted from the analysis as they were not successfully estimtated for 
all of the benchmarked force fields.
