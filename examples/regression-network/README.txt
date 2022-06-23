To Compile this example:
  $ easea -gp regression.ez
  $ cmake . && cmake --build . --config Release
or CUDA version:
  $ easea -cuda_gp regression.ez
  $ cmake . && cmake --build . --config Release

To test it without communications:
  $ ./regression
To test it with network communications:
  $ ./cmd.sh regression

To clean easea related file:
 $ make easeaclean
