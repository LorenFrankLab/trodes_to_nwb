# spikegadgets_to_nwb

Converts data from SpikeGadgets to the NWB Data Format

This will supersede [rec_to_nwb](https://github.com/LorenFrankLab/rec_to_nwb) once it is operational.

## Notes

### Unit Test

Unit test are quick-running test to ensure that no code is broken as new code is added. It is recommended you run them frequently and before and after adding new code. To run the unit test, run -

```bash
pytest src/spikegadgets_to_nwb/tests/unit_tests/
```

### Integration Test

Integration test are to ensure the system works. Functionality and third-party libraries are tested together and noting is mocked. They are best ran prior to pushing a commit as they can be slow. To run the test, run -

```bash
pytest src/spikegadgets_to_nwb/tests/integration_tests/
```
