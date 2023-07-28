# franklabnwb

Example notebook and related code to create NWB files from trodes .rec and associated files assuming franklab
compatible naming system

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

### Sub tree

franklabnwb_git_subtree is a [git subtree](https://www.atlassian.com/git/tutorials/git-subtree) from [franklabnwb](https://github.com/LorenFrankLab/franklabnwb). Git subtree is essentially a way to pull down one git repository into another. It is a means of code-reuse. Do not update franklabnwb_git_subtree directly. Rather, update [franklabnwb](https://github.com/LorenFrankLab/franklabnwb) and pull it down into spikegadget_to_nwb. To update this, run -

```bash
git subtree pull --prefix=src/spikegadgets_to_nwb/franklabnwb_git_subtree https://github.com/LorenFrankLab/franklabnwb main --squash
```
