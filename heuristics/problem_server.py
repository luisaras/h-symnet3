# ASNet code.
import os, sys, signal
import numpy as np
import rpyc
import ctypes
import getpass
import uuid
import builtins
import threading
import traceback
from copy import deepcopy
from multiprocessing import Process
from time import sleep, time
from types import SimpleNamespace
try:
    import kernprof
except ImportError:
    kernprof = None

from .ssipp_interface import PlannerExtensions


def _has_profile():
    """Check whether we have kernprof & kernprof has given us global 'profile'
    object."""
    return kernprof is not None and hasattr(builtins, 'profile')


class ProblemServiceConfig(object):
    def __init__(
            self,
            pddl_files,
            instance_name,
            heuristics):
        self.pddl_files = pddl_files
        self.init_problem_name = instance_name
        self.heuristics = heuristics

def make_problem_service(config):
    """Construct Service class for a particular problem. Note that we must
    construct classes, not instances (unfortunately), as there is no way of
    passing arguments to the service's initialisation code (AFAICT).

    The extra set_proc_title arg can be set to True if you want the
    ProblemService to figure out a descriptive name for the current process in
    top/htop/etc. It's mostly useful when you're starting a single subprocess
    per environment, and you want to know which subprocess corresponds to which
    environment."""

    lock = threading.Lock()

    class ProblemService(rpyc.Service):
        """Spools up a new Python interpreter and uses it to sandbox SSiPP and
        MDPSim. Can interact with this to train a Q-network."""

        def exposed_initialise(self):
            with lock:
                if self.initialised:
                 return "Can't double-init"
                try:
                    self.p = PlannerExtensions(config.pddl_files, config.init_problem_name, config.heuristics)
                    self.initialised = True
                    return None
                except:
                    return "Couldn't initialize Planner for " + config.init_problem_name

        def exposed_compute_heuristics(self, atoms):
            with lock:
                try:
                    if not self.initialised:
                        raise Exception(f"Planner for pddl {config.init_problem_name} was not initialised")
                    return self.p.compute_heuristics(atoms)
                except Exception as e:
                    return str(traceback.format_exc())
                    #return "".join(traceback.format_exception(type(e), e, e.__traceback__))

        def on_connect(self, conn):
            # we let the initialiser run later, so that it can execute
            # asynchronously (starting up PlannerExtensions & Planner is
            # expensive because it requires grounding the relevant problem)
            with lock:
                if not hasattr(self, "p"):
                    self.initialised = False

    return ProblemService

def parent_death_pact(signal=signal.SIGINT):
    """Commit to kill current process when parent process dies."""
    assert sys.platform == 'linux', \
        "this fn only works on Linux right now"
    libc = ctypes.CDLL("libc.so.6")
    # see include/uapi/linux/prctl.h in kernel
    PR_SET_PDEATHSIG = 1
    # last three args are unused for PR_SET_PDEATHSIG
    retcode = libc.prctl(PR_SET_PDEATHSIG, signal, 0, 0, 0)
    if retcode != 0:
        raise Exception("prctl() returned nonzero retcode %d" % retcode)

def start_server(service_args, socket_path):
    # avoid import cycle
    #parent_death_pact(signal=signal.SIGKILL)
    new_service = make_problem_service(service_args)
    server = rpyc.utils.server.ThreadedServer(new_service, socket_path=socket_path, backlog=1000)
    print('Child process starting ThreadedServer %s' % server)
    try:
        server.start()
    finally:
        # save kernprof profile for this subprocess if we can
        try_save_profile()

def try_save_profile():
    """If there's a profiler, this tries to save a profile with appropriate
    filename. Relies on arguments being passed correctly."""
    # avoids flake8 warnings
    if _has_profile():
        options = _kernprof_options()
        pid = os.getpid()
        if options.outfile is not None:
            # append PID to destination name and save that
            real_dest = options.outfile + '.%d' % pid
            print("Subprocess %d saving stats to '%s'" % (pid, real_dest))
            builtins.profile.dump_stats(real_dest)
        if options.view is not None:
            print("Profiler stats for subprocess %d:" % pid)
            builtins.profile.print_stats()


def to_local(obj):
    """Convert a NetRef to an object to something that's DEFINITELY local."""
    # can probably smarter here (e.g. not copying netrefs, using joblib for
    # efficient Numpy support); oh well
    # TODO: try using encode/decode with joblib instead! Could be much, much
    # faster.
    # TODO: make sure that you're transmitting observations as byte tensors
    # whenever possible (or at most float32s).
    return deepcopy(obj)

def wait_exists_polling(file_path, max_wait, *, delta=0.05):
    """Check if file exists every `delta` seconds. I'm using this to wait for a
    socket to get created by a subprocess."""
    start_time = time()
    while not os.path.exists(file_path):
        sleep(delta)
        if time() - start_time > max_wait:
            return False
    return True

class ProblemServer(object):
    """Spools up another process to host a ProblemService."""
    # how long we need to wait for the connection to spool up
    MAX_WAIT_TIME = 15.0

    def __init__(self, service_conf, verbose=True):
        # Sockets go in /tmp rather than cwd because Linux limits socket paths
        # (not filenames!) to 108 chars, and cwd might be too long (yes,
        # seriously!). The username is just in there to avoid case where
        # somebody else makes the dir & stops us from writing to it.
        user = getpass.getuser()
        sock_dir = f'/tmp/hsymnet3-sockets-{user}/'
        os.makedirs(sock_dir, exist_ok=True)
        self._unix_sock_path = os.path.join(sock_dir,
                                            'socket.' + uuid.uuid4().hex)
        self._serve_proc = Process(
            target=start_server, args=(
                service_conf,
                self._unix_sock_path,
            ))
        self._serve_proc.start()
        self._start_time = time()

        self._thread_conns = SimpleNamespace()#threading.local()
        self._conns = []
        self._verbose = verbose

    def stop(self):
        print('\033[31mCleaning up server process\033[0m')
        try:
            os.unlink(self._unix_sock_path)
        except FileNotFoundError:
            pass

        for conn in self._conns:
            if conn is not None:
                conn.close()

        if self._serve_proc is not None:
            self._serve_proc.terminate()
            try:
                self._serve_proc.join(5)
            except Exception:
                print('Process is being difficult.')
                pid = self._serve_proc.pid
                if pid is not None and self._serve_proc.is_alive():
                    print('I know how to handle difficult processes.')
                    os.kill(pid, signal.SIGKILL)
                    self._serve_proc.join(5)
            self._serve_proc = None

    def __del__(self):
        if hasattr(self, '_serve_proc') and self._serve_proc is not None:
            if self._verbose: print('Stop server in destructor')
            self.stop()

    def _get_rpyc_conn(self):
        if not hasattr(self._thread_conns, 'conn'):
            if not self._serve_proc.is_alive():
                print('\033[31mServer process is dead!\033[0m')
            to_wait = max(0, self.MAX_WAIT_TIME - (time() - self._start_time))
            if to_wait > 0:
                # It actually takes a few seconds for the background worker to
                # spool up and start accepting connections. Obviously it could
                # be more than self.MAX_WAIT_TIME, but I don't really have a
                # better way of doing things than this (mostly because all the
                # socket binding in RPyC happens in a monolithic "run
                # everything" method which I can't break up).
                if self._verbose: print('Waiting at most %.2fs for rpyc connection' % to_wait)
                # ignore return value; we'll get an error later if the file
                # doesn't exist
                has_sock = wait_exists_polling(
                    self._unix_sock_path, max_wait=to_wait)
                if self._verbose: print(f"Wait time up, got has_sock={has_sock}")
            sleep_time = 1.0
            if self._verbose: print(f"Sleeping an extra {sleep_time}s to make sure conn is up")
            sleep(sleep_time)
            conn = rpyc.utils.factory.unix_connect(
                path=self._unix_sock_path)
            self._conns.append(conn)
            self._thread_conns.conn = conn
        return self._thread_conns.conn

    @property
    def conn(self):
        return self._get_rpyc_conn()

    @property
    def service(self):
        # return handle on root service for connection, which in this case is a
        # ProblemService
        return self.conn.root

def make_planner_server(ppddl_file, instance_name, heuristics):
    config = ProblemServiceConfig([ppddl_file], instance_name, heuristics)
    server = ProblemServer(config)
    result = server.service.initialise()
    if result is not None:
        print("\033[31m" + result + "\033[0m")
        raise Exception("Server error")
    return server