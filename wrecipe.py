"""
The wrecipe-module contains the build instructions for each particular
FlowSimulator Package, in particular it defines
- The C++-Libraries to compile (which have to be laid out with
  Headers in <Name>/include, and
  Sources in <Name>/src and are compile to lib<Name>.<so-extension>).
- The Python-Interfaces to generate via SWIG and compile, which have to be named
  <Name>/include/_<Name>.i for the top-level interface-file.
- Additional Python-Modules.
"""

VERSION='0.0.1'
APPNAME='FSMeshPyTreeConversion'

# For which files to call doxygen
DoxygenConfigs = []

# External libraries (note: the ones ending in '*' are optional) and their
# dependencies. If available WITH_<NAME> will be defined.
ExternalLibraries = {
    'fsdm': [],
    'cassiopee': []
}

def build(bld):
    # install examples
    exdir = bld.path.find_dir('example')
    if exdir:
        bld.install_files('${PREFIX}/examples/' + APPNAME, exdir.ant_glob('**/*.py'), cwd=exdir, relative_trick=True, chmod=0o755)
        bld.install_files('${PREFIX}/examples/' + APPNAME, exdir.ant_glob('**/*', excl=['**/*.py', 'tmp/**', 'Summary*']), cwd=exdir, relative_trick=True)


def detect_fsdm(conf, env,*args, **kw):
    versionViaPkgConfig = conf.check_cfg(modversion='fsdm',
                                         msg='Checking if \'fsdm\' is available via pkg-config',
                                         env=env, *args, **kw)
    if versionViaPkgConfig:
        conf.check_cfg(package='fsdm', args=['--cflags', '--libs'],
                       msg='Checking for \'fsdm\' using pkg-config',
                       env=env, *args, **kw)
        env.VERSION_fsdm = versionViaPkgConfig
        return True
    else:
        return False

TestFragment_fsdm = """
#include "FSCommon.h"
#include "FSTypes.h"
int main(void)
{
    FS_floatT a = 3.0;
    FS_floatT b = -4.0;
    FSMin(a, b);
    return 0;
}"""

def detect_cassiopee(conf, env,*args, **kw):
    try: import KCore as K
    except: return False
    env.VERSION_cassiopee = K.__version__
    print("Found Cassiopée version: " + env.VERSION_cassiopee)
    return True

