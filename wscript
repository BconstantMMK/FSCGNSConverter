# encoding: utf-8
# Daniel Vollmer, 2012-2016

import os

from waflib import Build, Logs

import wrecipe as br

# these variables are mandatory ('/' are converted automatically)
top = '.'
out = 'build'

# the following two variables are used by the target "waf dist"
VERSION = br.VERSION
APPNAME = br.APPNAME
APPDESC = getattr(br, "APPDESC", "")

# the prefix for defines for used dependencies
_DefinePrefix = getattr(br, "DEFINE_PREFIX", "WITH_")

_FilteredPythonDefs = ('PYTHONDIR', 'PYTHONARCHDIR', 'HAVE_PYTHON_H', 'HAVE_PYEXT')

# The configure options to add for each external library, mapped to (wafname, help-text)
_LibOpts = {
    'defs': ('defines', 'defines for %s (comma-separated)'),
    'incpaths': ('includes', 'include paths to add for %s (comma-separated)'),
    'libs': ('lib', 'libraries to link for %s (comma-separated)'),
    'libpaths': ('libpath', 'library paths to add for %s (comma-separated)'),
    'linkflags': ('linkflags', 'linker flags to add for %s (whitespace-separated)'),
    'uses': ('use', 'dependencies to add for %s (comma-separated)'),
    'pkgconfigname': ('pkgconfigname', 'name of %s in pkg-config'),
    'version': ('version', 'version of %s'),
}

_OutputPkgConfigFile = APPNAME.lower() + '.pc'


def _expand(value):
    result = value
    try:
        result = os.path.expandvars(value)
    finally:
        return result


def _expandValue(key, value):
    import fsconfig

    if not value:
        return value
    if key in fsconfig.KEYS_LIST:  # expand each item in list
        return type(value)([_expand(v) for v in value])
    elif key in fsconfig.KEYS_DICT:  # expand each value in dict
        result = type(value)()
        for k, v in value.items():
            result[k] = _expand(v)
        return result
    else:  # it's a value
        return _expand(value)


def _packageToWafSettings(fsr, pname):
    """Converts values from an FSPackage to a settings dict that is waf-compatible.
    Should only be called if fsconfig is available"""
    import fsconfig

    result = {}
    for opt, (wafname, _) in _LibOpts.items():
        if opt in fsr[pname]:
            val = fsr.getValueInherited(pname, opt)
            if opt in fsconfig.KEYS_DICT:  # need to convert to list
                result[wafname] = ['%s=%s' % (_expand(k), _expand(v)) if v else _expand(k) for (k, v) in val.items()]
            else:
                result[wafname] = _expandValue(opt, val)
    return result


def _wafSettingsToPackage(s):
    """Converts values from an waf-settings dict into an FSPackage.
    Should only be called if fsconfig is available"""
    import fsconfig

    result = {}  # note: not creating FSPackage, because we don't want to care about order
    for k, v in s.items():
        for opt, (wafname, _) in _LibOpts.items():  # do reverse lookup
            if k == wafname:
                if opt in fsconfig.KEYS_DICT:  # need to convert from list to dict
                    r = {}
                    for define in v:
                        defAndVal = define.split('=')
                        val = None
                        if len(defAndVal) > 1:
                            val = defAndVal[-1]
                        r[defAndVal[0]] = val
                    v = r
                result[opt] = v
    return result


def _settingsToPkgConfigSettings(env, listOfUsedExternalLibraries):
    """Converts values from a list of used libraries into the settings for pkg config.

    Args:
        env: The dict of environment variables used in the waf context.
        listOfUsedLibraries: List of used libraries.
    Returns:
        Lists of used pkg-config dependencies, include flags, cflags, lib flags"""

    dependencies = []
    includes = []
    cflags = []
    libs = []

    for lib in listOfUsedExternalLibraries:
        # if the lib was discovered via pkg-config, add it as a dependency
        if 'VERSION_{}'.format(lib) in env and env['VERSION_{}'.format(lib)]:
            name = env['PKGCONFIGNAME_{}'.format(lib)] if 'PKGCONFIGNAME_{}'.format(lib) in env else lib
            version = env['VERSION_{}'.format(lib)] if 'VERSION_{}'.format(lib) in env else 'yes'

            dependencies += [name] if version == 'yes' else ['{} = {}'.format(name, version)]
        # if the lib is of unknown origin, add include and link flags
        else:
            includes += ['-I{}'.format(directory) for directory in env['INCLUDES_{}'.format(lib)]]
            cflags += ['-D{}'.format(cppflag) for cppflag in env['DEFINES_{}'.format(lib)]]
            libs += ['-L{}'.format(libpath) for libpath in env['LIBPATH_{}'.format(lib)]]
            libs += ['-l{}'.format(lib) for lib in env['LIB_{}'.format(lib)]]

    return dependencies, includes, cflags, libs


def _getRegistry(filename=None):
    """None => default path, '' => don't use"""
    result = None
    if filename != '':
        try:
            import fsconfig
            result = fsconfig.FSRegistry()
            result.read(filename)
        except ImportError:
            pass
    return result


def options(opt):
    opt.load('compiler_cxx python')

    from sysconfig import get_platform
    platform = get_platform().replace('/', '_')
    defaultRegistry = os.path.join('config', platform + '.ini')
    # debug information
    opt.parser.set_defaults(debug=False)
    opt.add_option('--disable-debug', action='store_false', dest='debug',
                   help='Disable generation of debug information')
    opt.add_option('--enable-debug', action='store_true', dest='debug',
                   help='Enable generation of debug information [default: \'%default\']')
    # fsconfig
    opt.add_option('-r', '--registry', type='string', dest='registry', default=defaultRegistry,
                   help='location of the registry [default: %default]')
    opt.add_option('--disable-fsconfig', action='store_const', const='', dest='registry',
                   help='disable use of fsconfig for reading / writing settings')
    # doxygen
    opt.add_option('--disable-doxygen', action='store_false', default=True, dest='enable_doxygen',
                   help='disable use of doxygen for creating documentation')
    # python-bindings
    opt.add_option('--disable-swig', action='store_false', default=True, dest='enable_swig',
                   help='disable use of swig for generating python bindings')

    gr = opt.add_option_group("configure options for external libraries", "These settings override any settings from fsconfig!")
    # auto-detect libs
    gr.add_option('--disable-autodetect', action='store_false', default=True, dest='enable_libdetect',
                  help='disable use of auto-detection for libraries (if available)')
    gr.add_option('--disable-libtest', action='store_false', default=True, dest='enable_libtest',
                  help='disable use of configure tests for libraries (if available)')

    # and now let's add options for all the ExternalLibraries
    def split_comma(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))

    for lib in getattr(br, 'ExternalLibraries', {}):
        isLibOptional = lib.endswith('*')
        lib = lib.rstrip('*')
        for optname, (_, helpstr) in _LibOpts.items():
            rname = '--' + lib + '-' + optname
            vname = lib + '_' + optname
            rhelp = helpstr % lib
            if optname != 'linkflags':
                gr.add_option(rname, dest=vname, help=rhelp, type='string', action='callback', callback=split_comma)
            else:
                gr.add_option(rname, dest=vname, help=rhelp, type='string')

            if isLibOptional:
                gr.add_option('--{0}-enabled'.format(lib), dest='{0}_enabled'.format(lib), action='store_true',
                              help='Compile with {0}. Abort configuration if {0} is not available'.format(lib))
                gr.add_option('--{0}-disabled'.format(lib), dest='{0}_disabled'.format(lib), action='store_true',
                              help='Compile without {0}.'.format(lib))

    # add our recipe's options
    if hasattr(br, 'options'):
        br.options(opt)


def configure(conf):
    conf.msg('Using registry file', conf.options.registry or 'None', color='CYAN')
    fsr = _getRegistry(conf.options.registry)
    if conf.options.registry is not None:
        conf.env.FSREGISTRY = conf.options.registry

    conf.load('compiler_cxx')
    conf.load('python')  # we load this even if we don't use swig for installing py-files

    conf.check_python_version((3, 6, 0))  # check early so we can do stuff depending on Python version
    if conf.options.enable_swig:
        conf.check_python_headers(features='pyext')

    defs = []  # global defines
    pyDefs = []  # python specific defines (only relevant for swig)

    # clean up defines from the above check_python methods "polluting" our global defines
    for define in conf.env.DEFINES:
        defName = define.split('=')[0]
        if defName in _FilteredPythonDefs:
            pyDefs.append(define)
            conf.env.define_key.remove(defName)  # prevent it showing up as 'undef'
            del conf.env.DEFINE_COMMENTS[defName]  # no need to keep any comments
        else:
            defs.append(define)

    conf.env.DEFINES = defs
    if conf.options.enable_swig:  # we only keep the Python defines if we actually want them
        conf.env.DEFINES_PYEXT += pyDefs

    # debug information
    if conf.options.debug:
        conf.env.append_unique('CXXFLAGS', '-g')
        conf.env.append_unique('LINKFLAGS', '-g')

    # our recipe's configure (which we run before our tests because it might add flags / settings)
    if hasattr(br, 'configure'):
        br.configure(conf)

    # external libraries
    def processLib(lib, lib2Dependencies={}, alreadyDone={}, isOptional=None, workingOn=None, env=None):
        """
        Process the given library to determine its settings (and - recursively - all dependencies).
        :param lib: name of lib to be processed
        :param lib2Dependencies: dict from library name to dependency names; its keys are the direct dependencies
        :param alreadyDone: dictionary containing settings for already processed libs
        :param isOptional: is the given lib optional (name ends with asterisk)
        :param workingOn: set containing all libs we are currently working on (to avoid circular uses of libs)
        :param env: current waf configuration environment (used when checking for dependent libs)
        """
        if workingOn is None:
            workingOn = set()
        if env is None:
            env = conf.env.derive().detach()
        if isOptional is None:
            isOptional = lib.endswith('*')
        lib = lib.rstrip('*')

        if lib in alreadyDone:
            return alreadyDone[lib]

        # whether the lib is a direct dependency (or an implicit one, e.g. from fsconfig)
        isDirectDependency = (lib in lib2Dependencies) or (lib + '*' in lib2Dependencies)

        # gather settings
        settings = {}
        givenOpts = [opt for opt in _LibOpts if getattr(conf.options, lib + '_' + opt, None) is not None]
        if givenOpts:  # if any option is given, take all from options
            for opt in givenOpts:
                settings[_LibOpts[opt][0]] = getattr(conf.options, lib + '_' + opt)
            conf.msg('Searching for ' + lib, 'cmd-line', color='CYAN')
        elif fsr and lib in fsr:  # try fsconfig
            settings = _packageToWafSettings(fsr, lib)
            if settings:
                conf.msg('Searching for ' + lib, 'fsconfig', color='CYAN')

        # gather dependencies from passed list and/or gathered settings
        dependsOn = set(lib2Dependencies.get(lib, lib2Dependencies.get(lib + '*', [])))  # explicitly given dependencies
        dependsOn.update(settings.get('use', []))  # and any extra ones
        # remove double entries (i.e. if both optional and non-optional, then it isn't optional after all)
        dependsOn = set(x for x in dependsOn if not (x.endswith('*') and (x[:-1] in dependsOn)))

        workingOn.add(lib)
        allDeps = {}
        for dep in dependsOn:
            if dep in workingOn:
                conf.fatal('Circular \'uses\' reference for library \'%s\'' % lib)
            # get settings for each dependency
            depSettings = processLib(dep, lib2Dependencies=lib2Dependencies, alreadyDone=alreadyDone,
                                     isOptional=isOptional, workingOn=workingOn, env=env)
            if depSettings is None:
                if not dep.endswith('*'):
                    conf.msg('Checking dependencies for ' + lib, dep + ' failed', color='YELLOW')
                    alreadyDone[lib] = None  # we can't build ourselves if dep is missing
                    return None
            else:  # transfer depSettings to env
                allDeps[dep] = depSettings
                if dep.endswith('*'):
                    dep = dep[:-1]
                for k, v in depSettings.items():
                    env[k.upper() + "_" + dep] = v
        workingOn.remove(lib)

        if not isDirectDependency:
            # whatever we have now is all we'll get (as the recipe only has tests etc. for its direct deps)
            if not settings:
                conf.msg('Searching for ' + lib, 'no settings found', color='YELLOW')
            alreadyDone[lib] = settings or None
            return alreadyDone[lib]

        if not settings:
            # try auto-detect
            if conf.options.enable_libdetect and hasattr(br, 'detect_' + lib):
                env.stash()
                conf.env.stash()
                if getattr(br, 'detect_' + lib)(conf, env=env, uselib_store=lib, deps=allDeps, mandatory=False):
                    # transfer into settings
                    for opt, (wafname, _) in _LibOpts.items():
                        if wafname.upper() + '_' + lib in env:
                            settings[wafname] = env[wafname.upper() + '_' + lib]
                conf.env.revert()
                env.revert()
            else:
                conf.msg('Searching for ' + lib, 'no settings found', color='YELLOW')

        result = alreadyDone[lib] = None
        # perform test
        if conf.options.enable_libtest and hasattr(br, 'TestFragment_' + lib):
            envCopy = env.derive().detach()  # check_cxx merges env with the given settings
            if conf.check_cxx(mandatory=False, execute=False, fragment=getattr(br, 'TestFragment_' + lib), msg='Testing %s' % lib, okmsg='ok', errmsg='failed', uselib_store=lib, env=envCopy, **settings):
                result = alreadyDone[lib] = settings
        else:
            if not settings:
                if not isOptional:
                    conf.fatal('Settings for required dependency %s not found (and no test available)' % lib)
            else:
                conf.msg('No test for ' + lib, 'praying', color='GREEN')
                result = alreadyDone[lib] = settings
        return result

    alreadyDone = {}
    updatedLibs = []
    allDefines = {}
    cleanEnv = conf.env.derive().detach()
    lib2Dependencies = getattr(br, 'ExternalLibraries', {})
    for lib in lib2Dependencies.keys():
        libName = lib.rstrip('*')
        if not getattr(conf.options, '{0}_disabled'.format(libName), False):
            libSettings = processLib(lib, lib2Dependencies=lib2Dependencies, alreadyDone=alreadyDone, env=cleanEnv.derive().detach())
        else:
            if getattr(conf.options, '{0}_enabled'.format(libName), False):
                conf.fatal('Contradictory settings: --{0}-enabled and --{0}-disabled are set!'.format(libName))
            conf.msg('Testing for {0}:'.format(libName), 'explicitly disabled', color='YELLOW')
            alreadyDone[libName] = None
            libSettings = None

        if libSettings is not None:  # add corresponding WITH_
            # check defines
            for d in libSettings.get('defines', []):
                dl = d.split('=')
                dk = dl[0]
                dv = None
                if len(dl) > 1:
                    dv = dl[-1]
                if (dk in allDefines) and (dv != allDefines[dk]):
                    conf.fatal('Incompatible define of \'%s\' (%s != %s) for dependency \'%s\'!' % (dk, allDefines[dk], dv, libName))
                allDefines[dk] = dv
            # transfer to build env
            for k, v in libSettings.items():
                conf.env[k.upper() + "_" + libName] = v
            conf.define(_DefinePrefix + libName.upper(), True)
            # and update fsconfig
            if fsr is not None:
                import fsconfig
                op = fsr.get(libName, fsconfig.FSPackage())
                p = _wafSettingsToPackage(libSettings)
                if op != p:
                    # try to keep order
                    for k, v in p.items():
                        op[k] = v
                    if libName not in fsr:
                        fsr[libName] = op
                    updatedLibs.append(libName)
        elif not lib.endswith('*'):
            conf.fatal('Essential dependency \'%s\' failed!' % lib)
        elif getattr(conf.options, '{0}_enabled'.format(libName), False):
            conf.fatal('Optional but explicitly enabled dependency \'%s\' failed!' % lib.rstrip('*'))

    enabledLibs = [k for k, v in alreadyDone.items() if v is not None]
    disabledLibs = [k for k, v in alreadyDone.items() if v is None]
    enabledDirectLibs = [i for i in enabledLibs if (i in lib2Dependencies) or (i + '*' in lib2Dependencies)]
    disabledDirectLibs = [i for i in disabledLibs if (i in lib2Dependencies) or (i + '*' in lib2Dependencies)]
    enabledIndirectLibs = [i for i in enabledLibs if (i not in lib2Dependencies) and (i + '*' not in lib2Dependencies)]
    disabledIndirectLibs = [i for i in disabledLibs if (i not in lib2Dependencies) and (i + '*' not in lib2Dependencies)]
    if enabledDirectLibs:
        conf.msg('Enabled (external) libraries', ' '.join(enabledDirectLibs))
        # save them for registering them as dependencies in the _register step
        conf.env.used_external_libraries = enabledDirectLibs
    if enabledIndirectLibs:
        conf.msg('Enabled indirect dependencies', ' '.join(enabledIndirectLibs))
    if disabledDirectLibs:
        conf.msg('Disabled (external) libraries', ' '.join(disabledDirectLibs), color='YELLOW')
    if disabledIndirectLibs:
        conf.msg('Disabled indirect dependencies', ' '.join(disabledIndirectLibs), color='YELLOW')

    configHeader = getattr(br, 'ConfigHeader', None)
    if configHeader is not None:
        # write every non-Python define into a header
        conf.write_config_header(configHeader, remove=True)

    # python bindings and headers (uses waf built-in)
    conf.options.enable_swig = conf.options.enable_swig and bool(getattr(br, 'PythonBindings', []))
    if conf.options.enable_swig:
        # the earlier check for the PYEXT settings may include potential NDEBUG defines from Python
        # we remove the NDEBUG because we don't want it to take precedence over our settings
        conf.env.DEFINES_PYEXT = [d for d in conf.env.DEFINES_PYEXT if d != 'NDEBUG']

        # allow for setting of swig args from fsconfig
        swigArgs = None
        if fsr and 'swig' in fsr:
            swig = fsr['swig']
            if 'bin' in swig:
                os.environ['SWIG'] = swig['bin']
            if 'args' in swig:
                swigArgs = swig['args']
        conf.load('swig')

        # set the swig arguments if they were not specified via fsconfig
        if swigArgs is None:
            swigArgs = ['-c++', '-fcompact', '-fvirtual', '-python']
            swigVersion = conf.check_swig_version((1, 3, 31))

            if swigVersion < (4, 0, 0):
                swigArgs += ['-modern']

            if swigVersion >= (4, 0, 1):
                # generate Python docstrings for swig-wrapped methods for swig 4.0.1+
                # at version 4.0.1 swig's doxygen parser fails at parsing some param[FOO] statements
                # disable the warning message associated with it
                swigArgs += ['-doxygen', '-w560']

            if swigVersion < (4, 1, 0):
                swigArgs += ['-py3'] # deprecated since 4.1.0

        conf.env.SWIG_ARGS = swigArgs

        # the shared objects should go into the same installation folder as the modules, e.g. for FSPlugin
        # ${PREFIX}/lib/pythonX.Y/site-packages/FSPlugin/fsplugin.so
        # for backward compatibility, the FSDataManager.so is installed in ${PREFIX}/lib/pythonX.Y/site-packages/
        conf.env.python_extension_directory = conf.env.PYTHONDIR + '/' + (br.APPNAME + '/' if br.APPNAME != 'FSDM' else '')

    conf.env.enable_swig = conf.options.enable_swig  # propagate the option for the build

    # doxygen (uses waf build-in)
    conf.options.enable_doxygen = conf.options.enable_doxygen and bool(getattr(br, 'DoxygenConfigs', []))
    if conf.options.enable_doxygen:
        conf.load('doxygen')
        if 'DOXYGEN' not in conf.env:
            conf.options.enable_doxygen = False
    conf.env.enable_doxygen = conf.options.enable_doxygen  # propagate the option for the build

    # run a post configure test that allows to check interaction between libraries (e.g. matching int types).
    postConfigureTestFragment = getattr(br, 'TestFragment_PostConfigure', None)
    if postConfigureTestFragment:
        conf.check_cxx(msg='Post configuration test for %s' % APPNAME,
                       okmsg='ok', errmsg='failed',
                       use=enabledLibs,
                       fragment=postConfigureTestFragment)

    # generate a pkg-config file to enable version checks and forwarding of defines
    _generatePkgConfigFile(conf, enabledDirectLibs)


def _register(ctx):
    """registered as post-build function, adds this plugin to registry"""
    if ctx.cmd == 'install':
        fname = ctx.options.registry
        if fname is None and 'FSREGISTRY' in ctx.env:
            fname = ctx.env.FSREGISTRY
        fsr = _getRegistry(fname)
        if fsr is not None:
            import fsconfig

            p = fsr.get(APPNAME.lower(), fsconfig.FSPackage())

            # populate uses and defines
            defs = {}
            uses = ctx.env.used_external_libraries

            # in case no config header is written, add defines
            for d in ctx.env.DEFINES:
                ds = d.split('=')
                k = ds[0]
                v = None
                if len(ds) > 0:
                    v = ds[-1]
                defs[k] = v

                uses.append(k[len(_DefinePrefix):].lower())

            if defs:
                p[fsconfig.DEFS] = defs
            if uses:
                p[fsconfig.USES] = uses

            # library and include paths
            if getattr(br, 'SharedLibraries', []):
                p[fsconfig.LIBS] = list(br.SharedLibraries.keys())
                p[fsconfig.LIBPATHS] = [ctx.env.LIBDIR]
                p[fsconfig.INCPATHS] = [os.path.join(ctx.env.PREFIX, 'include', m) for m in br.SharedLibraries]

                if getattr(br, 'THIRD_PARTY_BASE', None):
                  p[fsconfig.INCPATHS].append(os.path.join(ctx.env.PREFIX, 'include', getattr(br, 'THIRD_PARTY_BASE')))

            # python path
            if (ctx.options.enable_swig and ctx.env.enable_swig) or getattr(br, 'PythonModules', []):
                p[fsconfig.PYPATHS] = [os.path.join(ctx.env.PREFIX, 'py')]

            if APPNAME.lower() not in fsr:
                fsr[APPNAME.lower()] = p
            fsr.write(fname)
            Logs.info('Registered \'%s\' with fsconfig.' % APPNAME)


def _generatePkgConfigFile(bld, usedExternalLibraries):
    """Generates a config file that allows for accessing the includes, preprocessor definitions etc.

    Args:
        bld : The waf context.
        usedExternalLibraries : The list of external libraries used during the build process."""

    pkgConfigFileTemplate = """
Description: {APPDESC}

prefix={PREFIX}

Name: {APPNAME}
Version: {VERSION}
Cflags: {CFLAGS}
Libs: {LFLAGS}

Requires: {REQUIRES}
"""

    # get the lists of pkg-config dependencies, includes, cflags, libs from the external libraries
    dependencies, includes, cflags, libs = _settingsToPkgConfigSettings(bld.env, usedExternalLibraries)

    # add stuff for FSDM
    libs += ['-L{}'.format(bld.env['LIBDIR'])]
    libs += ['-l{}'.format(lib) for lib in getattr(br, 'SharedLibraries', {}).keys()]
    includes += [r'-I${prefix}/include/' + lib for lib in getattr(br, 'SharedLibraries', {}).keys()]

    if getattr(br, 'THIRD_PARTY_BASE', None):
        includes += [r'-I${prefix}/include/' + getattr(br, 'THIRD_PARTY_BASE')]

    cflags += ['-D' + define for define in bld.env.DEFINES] + includes

    fileContent = pkgConfigFileTemplate.format(APPDESC=APPDESC, APPNAME=APPNAME.lower(), VERSION=VERSION,
                                               CFLAGS=' '.join(cflags), LFLAGS=' '.join(libs),
                                               PREFIX=bld.options.prefix,
                                               REQUIRES=', '.join(dependencies))

    # largely copied from waf's config-header
    node = bld.path.get_bld().make_node(_OutputPkgConfigFile)
    node.write(fileContent)
    bld.env.append_unique(Build.CFG_FILES, [node.abspath()])


def build(bld):
    # doxygen
    if bld.env.enable_doxygen and bld.options.enable_doxygen:
        for df in getattr(br, 'DoxygenConfigs', []):
            # when installing into the same prefix, the name of the library should be included => add APPNAME
            installPath = '${PREFIX}/share/doc/' + APPNAME + '/cpp/' + os.path.basename(os.path.splitext(df)[0])
            pars = {}
            if hasattr(br, 'VERSION'):
                pars['PROJECT_NUMBER'] = getattr(br, 'VERSION')

            bld(features='doxygen', doxyfile=df, install_path=installPath, pars=pars)

    configIncludes = []
    configHeader = getattr(br, 'ConfigHeader', None)
    if configHeader is not None:
        configNode = bld.path.get_bld().find_node(configHeader)
        configIncludes.append(configNode.parent)
        bld.install_files('${PREFIX}/include/' + os.path.dirname(configHeader), configNode)

    # build the shared libraries
    for m, deps in getattr(br, 'SharedLibraries', {}).items():
        # all .cpp files in m/src except for the ones ending in Test.cpp
        cppFiles = bld.path.ant_glob(m + '/src/*.cpp', excl=[m + '/src/*Test.cpp'])
        includes = configIncludes + [bld.path.find_dir(m + '/include')]
        exportIncludes = configIncludes + ['.', bld.path.find_dir(m + '/include')]
        # build use flags from ExtraLibs from recipe + deps + external
        useLibs = getattr(br, 'ExtraLibraries', {}).get(m, []) + deps + bld.env.used_external_libraries
        bld.shlib(target=m, source=cppFiles, includes=includes, export_includes=exportIncludes, use=useLibs)
        # and install the headers
        headers = bld.path.find_dir(m + '/include')
        if headers:
            bld.install_files('${PREFIX}/include/' + m, headers.ant_glob('**/*', excl='*Test.[hi]'), cwd=headers, relative_trick=True)

    # build the extensions
    if bld.env.enable_swig and bld.options.enable_swig:
        for m in getattr(br, 'PythonBindings', []):
            interface = bld.path.find_node(m + '/include/_' + m + '.i')
            if interface is None:
                Log.fatal("Could not find interface file for Python extension " + m)

            extName = 'py' + '/' + (br.APPNAME + '/' if br.APPNAME != 'FSDM' else '') + '_' + m
            targetNode = bld.path.find_or_declare(extName)

            bld(target=extName, features='cxx cxxshlib pyext', source=interface,
                swig_flags=' '.join(bld.env.SWIG_ARGS  + ['-outdir', targetNode.get_bld().parent.relpath()]),
                use=[m], install_path=bld.env.python_extension_directory)

    # add the function for registering ourselves with fsconfig when we install
    if bld.cmd == 'install':
        bld.add_post_fun(_register)

    # add our recipe's build
    if hasattr(br, 'build'):
        br.build(bld)

    # Run the tests if requested by the user command.
    # We need to call it here to ensure that we have the right build context.
    if bld.cmd == 'test' and hasattr(br, 'test'):
        br.test(bld)

    # install python modules
    pydir = bld.path.find_dir('py')
    if pydir:
        pyFiles = pydir.ant_glob('**/*.py')
        if pyFiles:
            bld(features='py', source=pyFiles, install_from=pydir)
        # link all files of the pydir into the build folder to ensure that it can be used as installation
        for file in pydir.ant_glob('**/*'):
            # use the relative path from the source to the target to allow for nested Python packages
            bld(rule="ln -sf ${SRC[0].path_from(tsk.outputs[0].parent)} ${TGT}", source=file, target=file.get_bld(),
                shell=False)

    # install the swig generated python-files (which are not automatically installed)
    if bld.env.enable_swig and bld.options.enable_swig:
        # make sure that all python files (i.e. the swig generated ones) are already done
        bld.add_group()

        for m in getattr(br, 'PythonBindings', []):
            pyfile = bld.path.get_bld().find_or_declare('py' + '/' + (br.APPNAME + '/' if br.APPNAME != 'FSDM' else '') + '/' + m + '.py')

            # install into prefix Python site-packages folder using the name of the plugin as module name
            bld(features='py', source=pyfile, install_from=pyfile.parent, install_path=bld.env.python_extension_directory)

    # install the pkgconfig
    bld.install_files('${LIBDIR}/pkgconfig', [_OutputPkgConfigFile])

    # Generate the Python documentation if requested
    if hasattr(br, 'pydoc') and (bld.cmd in ('pydoc', 'install')):
        bld.add_group()  # ensure that everything is correctly built before generating the Python documentation
        br.pydoc(bld)


# Definition of the custom 'test' command which
# is activated in the build command if required.
class Test(Build.BuildContext):
    """builds and executes tests"""

    # As we define no handling function and inherit from BuildContext,
    # the build function is called and takes care of building / executing the tests.
    cmd = 'test'


def pydoc(ctx):
    """builds source and documentation"""

    commands = ['build', 'pydoc']
    Options.commands = commands + Options.commands


class PyDoc(Build.BuildContext):
    """builds the source code documentation."""

    cmd = 'pydoc'
