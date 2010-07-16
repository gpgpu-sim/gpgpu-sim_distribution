# (c) 2007 The Board of Trustees of the University of Illinois.

# This module takes care of parsing options for the Parboil driver.
#
# The main option parsing routine is parse_options().  Option parsing
# may print messages and terminate the program, but should not cause
# any other action to be taken.

from sys import stdout
from optparse import OptionParser

import actions
import globals

def invalid_option_message(progname, cmd, args):
    print "Unrecognized command '" + cmd + "'"
    print "'" + progname + " help' for options"

# Parsers for each mode.  These take the command line as parameters
# and return an OptionGetter.
#
# The 'help' field prints a help message.
#
# The 'run' field does the actual parsing.  If there is an error in
# the command line, it prints an error message and returns None;
# otherwise, it returns an action which will carry out the commands.

class OptionGetter:
    def __init__(self, help, run):
        self.help = help
        self.run = run

def help_options(progname, cmd, args):
    help_string = "usage: " + progname + " help [COMMAND]\nWithout parameters: list commands\nWith a parameter: Get help on COMMAND\n"
    get_help = lambda: stdout.write(help_string)
    
    def run():
        if args:
            try: helpcmd = parse_mode_options[args[0]]
            except KeyError:
                print "No help available for unrecognized command '" + args[0] + "'"
                return None

            helpcmd(progname, cmd, args).help()
        else:
            print "Commands: "
            print "  help      Display this help message"
            print "  list      List benchmarks"
            print "  describe  Show details on a benchmark"
            print "  clean     Clean up generated files in a benchmark"
            print "  compile   Compile a benchmark"
            print "  run       Run a benchmark"
            print ""
            print "To get help on a command: " + progname + " help COMMAND"

        return None

    return OptionGetter(get_help, run)

def list_options(progname, cmd, args):
    help_string = "usage: " + progname + " list\nList available benchmarks\n"
    get_help = lambda: stdout.write(help_string)

    def run():
        if args:
            print "Unexpected parameter or option after 'list'"
            return None
        else:
            return actions.list_benchmarks

    return OptionGetter(get_help, run)

def describe_options(progname, cmd, args):
    usage_string = progname + " describe [BENCHMARK]\nWithout parameters: describe all benchmarks in detail\nWith a parameter: describe BENCHMARK in detail"
    parser = OptionParser(usage=usage_string)

    def run():
        (opts, pos) = parser.parse_args(args)
        if len(pos) > 1:
            print "Too many parameters after 'describe'"
            return None
        elif len(pos) == 0:
            return actions.describe_benchmarks
        else:
            bmkname = pos[0]
            return lambda: actions.with_benchmark_named(bmkname,
                                                        actions.describe_benchmark)

    return OptionGetter(parser.print_help, run)

def clean_options(progname, cmd, args):
    usage_string = progname + " clean BENCHMARK [VERSION]\nDelete the object code and executable of BENCHMARK version VERSION;\nif no version is given, remove the object code and executable of all versions"
    parser = OptionParser(usage=usage_string)
    parser.add_option('-v', "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="Produce verbose status messages")

    def run():
        (opts, pos) = parser.parse_args(args)
        globals.verbose = opts.verbose

        if len(pos) == 0:
            print "Expecting one or two parameters after 'clean'"
            return None
        elif len(pos) == 1:
            bmkname = pos[0]
            return lambda: actions.with_benchmark_named(bmkname, actions.clean_benchmark)
        elif len(pos) == 2:
            bmkname = pos[0]
            ver = pos[1]
            return lambda: actions.with_benchmark_named(bmkname, lambda b: actions.clean_benchmark(b, ver))
        else:
            print "Too many parameters after 'clean'"
            return None

    return OptionGetter(parser.print_help, run)

def compile_options(progname, cmd, args):
    help_string = "usage :" + progname + " compile BENCHMARK VERSION\nCompile version VERSION of BENCHMARK"
    parser = OptionParser(usage = help_string)
    parser.add_option('-v', "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="Produce verbose status messages")

    def run():
        (opts, pos) = parser.parse_args(args)
        globals.verbose = opts.verbose
        
        if len(pos) != 2:
            print "Expecting two parameters after 'compile'"
            return None
        else:
            bmkname = pos[0]
            ver = pos[1]
            return lambda: actions.with_benchmark_named(bmkname, lambda b: actions.compile_benchmark(b, ver))

    return OptionGetter(parser.print_help, run)

def run_options(progname, cmd, args):
    usage_string = progname + " run BENCHMARK VERSION INPUT\nRun version VERSION of BENCHMARK with data set INPUT"
    parser = OptionParser(usage=usage_string)
    parser.add_option('-C', "--no-check",
                      action="store_false", dest="check", default=True,
                      help="Skip the output check for this benchmark")
    parser.add_option('-S', "--synchronize",
                      action="store_true", dest="synchronize", default=False,
                      help="Synchronize after GPU calls; necessary for accurate run time accounting on CUDA benchmarks")
    parser.add_option('-v', "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="Produce verbose status messages")

    def run():
        (opts, pos) = parser.parse_args(args)
        globals.verbose = opts.verbose
        if len(pos) != 3:
            print "Expecting three parameters after 'run'"
            return None
        else:
            bmkname = pos[0]
            ver = pos[1]
            inp = pos[2]
            ck=opts.check
            extra_opts = []
            if opts.synchronize:
                extra_opts.append('-S')
            return lambda: actions.with_benchmark_named(bmkname, lambda b: actions.run_benchmark(b, ver, inp, check=ck, extra_opts=extra_opts))

    return OptionGetter(parser.print_help, run)

# Dictionary from option name to function from command-line parameters
# to pair of help thunk and option processor thunk
parse_mode_options = {
    'help'     : help_options,
    '-h'       : help_options,
    '--help'   : help_options,
    'list'     : list_options,
    'describe' : describe_options,
    'clean'    : clean_options,
    'compile'  : compile_options,
    'run'      : run_options
    }

def parse_options(args):
    """Parse a list of command-line options.  If there is an error in
    the options, then the function will print a message and either call
    sys.exit() or return None.  If options were parsed successfully
    then it will return a thunk which represents the action to take,
    or None if no action need be taken.

    Generally, the caller should call the return value, unless None is
    returned."""
    # Parse arguments; skip the program name
    prog_name = args[0]

    # Get the command name
    try: cmd = args[1]
    except IndexError: cmd = 'help'

    # Dispatch
    try: mode = parse_mode_options[cmd]
    except KeyError:
        invalid_option_message(prog_name, cmd, args[2:])
        return

    # Set up and run the option parser
    return mode(prog_name, cmd, args[2:]).run()

    
