# Thanks for using EASENA

## Requirement
This project required you to have flex, bison, wget, gunzip and cmake installed:
```
$ sudo apt-get install flex bison wget gunzip cmake
```
In order to have a working grapher, java jre 1.6 is required. Without it, an error appears at the start of easea's compiled programs but can be safely ignored.

EASEA had been compiled and tested with those following compilers:
* Gcc 4.4 to 4.8.2
* Clang 3.0 to 3.3
* Mingw-gcc 4.8.2
* CUDA SDK > 4.1


## Easea can be build by following the next steps :

- cmake ./
- make
- make install 
- Export and setting environent variables (EZ_PATH="/usr/local/easena/" and PATH="$PATH:/usr/local/easena/bin")

"." is equivalent to "source".

Easena can either be installed locally without special permission or in /usr/local/ with root permissions.

Once installed, one can test its installation by compiling easena test programsin example/ :
```
$ cd example/weierstrass
$ easena weierstrass.ez
$ make 
$ ./weierstrass
```

To print all options available type:
```
$ ./weierstrass --help
```

To test the CUDA version (Need the CUDA developper kit found at https://developer.nvidia.com/cuda-downloads ):
```
$ easena -cuda weierstrass.ez
```

For additional help and documentation about the EASEA platform, please refers to the main wiki http://easea.unistra.fr/

To get the latest version of EASEnA, git clone the sourceforge repository:
```
git clone git://git.code.sf.net/p/easea/code easea-code
```

That's all !
